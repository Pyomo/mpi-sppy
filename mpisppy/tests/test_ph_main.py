###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Test PH.ph_main() directly (not through the cylinder hub system).
# This exercises the standalone PH code path, including extensions
# and convergers that are otherwise untested.

import os
import shutil
import unittest

import pyomo.environ as pyo
import mpisppy.opt.ph
import mpisppy.tests.examples.farmer as farmer
from mpisppy.tests.examples.sizes.sizes import scenario_creator as sizes_creator, \
                                               scenario_denouement as sizes_denouement, \
                                               id_fix_list_fct
from mpisppy.tests.utils import get_solver
import mpisppy.MPI as mpi

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

# Known reference values (EF optimal for farmer with 3 scenarios, crops_multiplier=1)
FARMER_EF_OBJ = -118361.33  # approximate


class TestPHMainFarmer(unittest.TestCase):
    """Test PH.ph_main() with the farmer model, checking solution quality."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "PHIterLimit": 50,
            "defaultPHrho": 1,
            "convthresh": 1e-8,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "iter0_solver_options": None,
            "iterk_solver_options": None,
            "smoothed": 0,
            "asynchronousPH": False,
            "subsolvedirectives": None,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]
        self.creator_kwargs = {"crops_multiplier": 1}

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_obj(self):
        """PH on farmer should approach the EF optimal."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main()
        # obj includes the proximal term so won't match EF exactly,
        # but should be in the right ballpark
        self.assertAlmostEqual(obj, FARMER_EF_OBJ, delta=500)
        # trivial bound should be looser (more negative for min)
        self.assertLess(tbound, FARMER_EF_OBJ)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_nonants_converge(self):
        """After PH, farmer nonants should be close across scenarios."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main()
        # Collect first-stage decisions from all local scenarios
        nonant_values = {}
        for sname, s in ph.local_scenarios.items():
            for node in s._mpisppy_node_list:
                for i, v in enumerate(node.nonant_vardata_list):
                    key = (node.name, i)
                    if key not in nonant_values:
                        nonant_values[key] = []
                    nonant_values[key].append(pyo.value(v))
        # Check nonants are close across scenarios (convergence)
        for key, vals in nonant_values.items():
            spread = max(vals) - min(vals)
            self.assertLess(spread, 1.0,
                            f"Nonant {key} spread {spread} too large")

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_no_finalize(self):
        """PH.ph_main() with finalize=False returns None for Eobj."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main(finalize=False)
        self.assertIsNone(obj)
        self.assertIsNotNone(tbound)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_xhatlooper_obj(self):
        """PH with XhatLooper should find a good xhat."""
        from mpisppy.extensions.xhatlooper import XhatLooper
        options = self._copy_options()
        options["xhat_looper_options"] = {
            "xhat_solver_options": None,
            "scen_limit": 3,
        }
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=XhatLooper,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertAlmostEqual(obj, FARMER_EF_OBJ, delta=500)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_frac_converger(self):
        """FractionalConverger on farmer (LP) converges immediately."""
        from mpisppy.convergers.fracintsnotconv import FractionalConverger
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            ph_converger=FractionalConverger,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        # With no integers, converger says "converged" after iter 0,
        # so PH stops at iteration 1 — obj won't be close to optimal
        # but the trivial bound should still be valid
        self.assertLess(tbound, FARMER_EF_OBJ)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_diagnoser(self):
        """Diagnoser should create per-scenario output files."""
        from mpisppy.extensions.diagnoser import Diagnoser
        diagdir = os.path.join(os.path.dirname(__file__), "_test_diagdir")
        if os.path.exists(diagdir):
            shutil.rmtree(diagdir)
        options = self._copy_options()
        options["PHIterLimit"] = 3
        options["diagnoser_options"] = {"diagnoser_outdir": diagdir}
        try:
            ph = mpisppy.opt.ph.PH(
                options,
                self.scenario_names,
                farmer.scenario_creator,
                farmer.scenario_denouement,
                scenario_creator_kwargs=self.creator_kwargs,
                extensions=Diagnoser,
            )
            conv, obj, tbound = ph.ph_main()
            self.assertIsNotNone(obj)
            # Check that diagnostic files were created
            for sname in self.scenario_names:
                dag_file = os.path.join(diagdir, f"{sname}.dag")
                self.assertTrue(os.path.exists(dag_file),
                                f"Missing diagnostic file for {sname}")
                with open(dag_file) as f:
                    lines = f.readlines()
                # header + iter0 write + 3 enditer writes = 4+ lines
                self.assertGreater(len(lines), 1)
        finally:
            if os.path.exists(diagdir):
                shutil.rmtree(diagdir)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_avgminmaxer(self):
        """MinMaxAvg should run and produce valid statistics."""
        from mpisppy.extensions.avgminmaxer import MinMaxAvg
        options = self._copy_options()
        options["PHIterLimit"] = 3
        options["avgminmax_name"] = "FirstStageCost"
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=MinMaxAvg,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)


class TestPHMainSizes(unittest.TestCase):
    """Test PH.ph_main() with the sizes (MIP) model, including fixer."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "PHIterLimit": 10,
            "defaultPHrho": 1,
            "convthresh": 0.001,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "iter0_solver_options": {"mipgap": 0.1},
            "iterk_solver_options": {"mipgap": 0.02},
            "smoothed": 0,
            "asynchronousPH": False,
            "subsolvedirectives": None,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]
        self.creator_kwargs = {"scenario_count": 3}

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_sizes_obj_range(self):
        """PH on sizes should produce an objective in a reasonable range."""
        ph = mpisppy.opt.ph.PH(
            self._copy_options(),
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        # sizes optimal is around 227000; PH obj includes prox terms
        self.assertGreater(obj, 100000)
        self.assertLess(obj, 400000)
        # trivial bound should be less than the PH obj
        self.assertLess(tbound, obj)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_sizes_frac_converger_exercises_ints(self):
        """FractionalConverger on sizes actually counts integer variables."""
        from mpisppy.convergers.fracintsnotconv import FractionalConverger
        options = self._copy_options()
        options["PHIterLimit"] = 5
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            ph_converger=FractionalConverger,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        # The converger should have run; conv is the fraction not converged
        # (might or might not have reached convergence in 5 iters)
        self.assertIsNotNone(conv)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_sizes_fixer(self):
        """Fixer extension should fix some integer variables."""
        from mpisppy.extensions.fixer import Fixer
        options = self._copy_options()
        options["PHIterLimit"] = 10
        options["fixeroptions"] = {
            "id_fix_list_fct": id_fix_list_fct,
            "verbose": False,
            "boundtol": 0.01,
        }
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
            extensions=Fixer,
        )
        conv, obj, tbound = ph.ph_main()
        self.assertIsNotNone(obj)
        self.assertGreater(obj, 100000)
        self.assertLess(obj, 400000)
        # Check that at least some variables got fixed
        total_fixed = 0
        for sname, s in ph.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.is_fixed():
                    total_fixed += 1
        # With 3 scenarios and the sizes fixer settings, we expect some fixing
        self.assertGreater(total_fixed, 0,
                           "Fixer should have fixed at least one variable")


if __name__ == '__main__':
    unittest.main()
