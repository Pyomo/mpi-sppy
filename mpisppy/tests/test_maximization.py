###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Serial (single-rank) tests that the solver algorithms genuinely work for a
# MAXIMIZATION problem, not merely run without error on one.
#
# The vehicle is the farmer example, which is a cost-MINIMIZATION model by
# default; with sense=pyo.maximize it negates the cost expression. Over the
# same feasible region the maximize optimum therefore equals the negative of
# the minimize optimum (same solution, flipped objective sign). Each test
# solves BOTH senses and checks
#   (a) the maximize answer is the negation of the minimize answer, and
#   (b) any bound brackets the optimum on the correct, sense-dependent side
#       (outer bounds are LOWER for min but UPPER for max; inner/incumbent
#       bounds are the reverse).
# A min-biased code path can still "run" on a max problem; (a) and (b) are
# what actually distinguish correct max handling from a flipped sign.
#
# Cylinder (hub-and-spoke) maximize coverage lives in test_with_cylinders.py
# because it requires mpiexec.

import unittest
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.opt.ph
from mpisppy.opt.subgradient import Subgradient
from mpisppy.opt.lshaped import LShapedMethod
from mpisppy.opt.fwph import FWPH
from mpisppy.tests.examples import farmer
from mpisppy.tests.utils import get_solver, limit_solver_threads

__version__ = 0.1

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

# Known farmer expected profit (== -1 times the minimize-cost optimum).
FARMER_OPT = 108390.0
NUM_SCENS = 3


class Test_maximization(unittest.TestCase):
    """ Confirm the solver algorithms handle sense=pyo.maximize correctly. """

    def setUp(self):
        self.names = farmer.scenario_names_creator(NUM_SCENS)

    def _kw(self, sense):
        return {"sense": sense, "crops_multiplier": 1, "num_scens": NUM_SCENS}

    def _ph_options(self, extra=None):
        options = {
            "solver_name": solver_name,
            "PHIterLimit": 50,
            "defaultPHrho": 1.0,
            "convthresh": 1e-8,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "iter0_solver_options": None,
            "iterk_solver_options": None,
        }
        if extra:
            options.update(extra)
        return options

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ef_max(self):
        def solve(sense):
            ef = sputils.create_EF(
                self.names,
                farmer.scenario_creator,
                scenario_creator_kwargs=self._kw(sense),
                suppress_warnings=True,
            )
            solver = pyo.SolverFactory(solver_name)
            limit_solver_threads(solver, solver_name)
            if "_persistent" in solver_name:
                solver.set_instance(ef)
            results = solver.solve(ef, tee=False)
            pyo.assert_optimal_termination(results)
            return pyo.value(ef.EF_Obj)

        min_obj = solve(pyo.minimize)
        max_obj = solve(pyo.maximize)
        self.assertAlmostEqual(max_obj, -min_obj, places=2)
        self.assertAlmostEqual(max_obj, FARMER_OPT, delta=5.0)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ph_max(self):
        def run(sense):
            ph = mpisppy.opt.ph.PH(
                self._ph_options(),
                self.names,
                farmer.scenario_creator,
                farmer.scenario_denouement,
                scenario_creator_kwargs=self._kw(sense),
            )
            conv, obj, tbound = ph.ph_main()
            return obj, tbound

        min_obj, min_bound = run(pyo.minimize)
        max_obj, max_bound = run(pyo.maximize)
        # PH converges to the EF optimum on this (continuous) model.
        self.assertAlmostEqual(max_obj, FARMER_OPT, delta=5.0)
        self.assertAlmostEqual(max_obj, -min_obj, places=1)
        # The PH outer bound is a LOWER bound for min, an UPPER bound for max.
        self.assertLessEqual(min_bound, min_obj + 1e-4)
        self.assertGreaterEqual(max_bound, max_obj - 1e-4)
        self.assertAlmostEqual(max_bound, -min_bound, places=1)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_subgradient_max(self):
        def run(sense):
            sg = Subgradient(
                self._ph_options({"smoothed": 0}),
                self.names,
                farmer.scenario_creator,
                farmer.scenario_denouement,
                scenario_creator_kwargs=self._kw(sense),
            )
            conv, obj, tbound = sg.subgradient_main()
            return tbound

        min_bound = run(pyo.minimize)
        max_bound = run(pyo.maximize)
        # The trivial/outer bound brackets the optimum: below for min (optimum
        # is -FARMER_OPT), above for max (optimum is +FARMER_OPT).
        self.assertLessEqual(min_bound, -FARMER_OPT + 1e-4)
        self.assertGreaterEqual(max_bound, FARMER_OPT - 1e-4)
        self.assertAlmostEqual(max_bound, -min_bound, places=1)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_lshaped_max(self):
        def run(sense):
            options = {
                "root_solver": solver_name,
                "sp_solver": solver_name,
                "sp_solver_options": {},
                # supply valid (loose) outer bounds to skip the eta-bound solve
                "valid_eta_lb": {n: -1e6 for n in self.names},
            }
            ls = LShapedMethod(
                options,
                self.names,
                farmer.scenario_creator,
                farmer.scenario_denouement,
                scenario_creator_kwargs=self._kw(sense),
            )
            ls.lshaped_algorithm()
            self.assertEqual(ls.is_minimizing, sense == pyo.minimize)
            # L-shaped negates max->min internally, then negates the bound back;
            # the reported bound must come out with the natural sign per sense.
            return ls._LShaped_bound

        min_bound = run(pyo.minimize)
        max_bound = run(pyo.maximize)
        self.assertAlmostEqual(min_bound, -FARMER_OPT, delta=5.0)
        self.assertAlmostEqual(max_bound, FARMER_OPT, delta=5.0)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_fwph_max(self):
        def run(sense):
            options = self._ph_options({
                "smoothed": 0,
                "FW_iter_limit": 20,
                "FW_weight": 0.0,
                "FW_conv_thresh": 1e-5,
                "stop_check_tol": 1e-5,
                "FW_LP_start_iterations": 0,
                "FW_verbose": False,
                "mip_solver_options": {},
                "qp_solver_options": {},
            })
            fw = FWPH(
                options,
                self.names,
                farmer.scenario_creator,
                farmer.scenario_denouement,
                scenario_creator_kwargs=self._kw(sense),
            )
            fw.fwph_main()
            return fw.best_bound_obj_val, fw.best_solution_obj_val

        min_bound, min_inner = run(pyo.minimize)
        max_bound, max_inner = run(pyo.maximize)
        # min: outer (dual) bound <= optimum <= inner (incumbent).
        self.assertLessEqual(min_bound, -FARMER_OPT + 1.0)
        self.assertGreaterEqual(min_inner, -FARMER_OPT - 1.0)
        # max: inner (incumbent) <= optimum <= outer (dual) bound.
        self.assertGreaterEqual(max_bound, FARMER_OPT - 1.0)
        self.assertLessEqual(max_inner, FARMER_OPT + 1.0)


if __name__ == "__main__":
    unittest.main()
