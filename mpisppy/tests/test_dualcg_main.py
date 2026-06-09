###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Test DCG.dualcg_main() directly (not through the cylinder hub system).


import unittest

import pyomo.environ as pyo

import mpisppy.MPI as mpi
import mpisppy.opt.dualcg
import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.sputils as sputils
from mpisppy.tests.utils import get_solver, limit_solver_threads


solver_available, solver_name, persistent_available, persistent_solver_name = get_solver(persistent_OK=False)

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()


def _solve_farmer_ef(scenario_names, creator_kwargs):
    ef = sputils.create_EF(
        scenario_names,
        farmer.scenario_creator,
        scenario_creator_kwargs=creator_kwargs,
    )
    solver = pyo.SolverFactory(solver_name)
    limit_solver_threads(solver, solver_name)
    solver.solve(ef)
    return pyo.value(ef.EF_Obj)


class TestDCGMainFarmer(unittest.TestCase):
    """Test DCG.dualcg_main() with the farmer model."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "CGIterLimit": 10,
            "convthresh": 1e-8,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "sp_solver_options": { },
            "mp_solver_options": { },
            "relaxed_nonant": False,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]

    def _copy_options(self):
        return dict(self.options)

    def _run_dcg_bound(self, creator_kwargs):
        dcg = mpisppy.opt.dualcg.DCG(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=creator_kwargs,
        )
        conv, obj = dcg.dualcg_main(finalize=False)
        return conv, obj, dcg.best_bound_obj_val

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_minimize_bound(self):
        """DCG on a minimization farmer should approach the EF optimal bound."""
        creator_kwargs = {"crops_multiplier": 1}
        conv, obj, bound = self._run_dcg_bound(creator_kwargs)

        if global_rank == 0:
            ef_obj = _solve_farmer_ef(self.scenario_names, creator_kwargs)
            self.assertIsNone(obj)
            self.assertIsNotNone(conv)
            self.assertAlmostEqual(bound, ef_obj, delta=abs(ef_obj*0.01))

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_maximize_bound(self):
        """DCG on a maximization farmer should approach the EF optimal bound."""
        creator_kwargs = {"crops_multiplier": 1, "sense": pyo.maximize}
        conv, obj, bound = self._run_dcg_bound(creator_kwargs)

        if global_rank == 0:
            ef_obj = _solve_farmer_ef(self.scenario_names, creator_kwargs)
            self.assertIsNone(obj)
            self.assertIsNotNone(conv)
            self.assertAlmostEqual(bound, ef_obj, delta=abs(ef_obj*0.01))


if __name__ == "__main__":
    unittest.main()
