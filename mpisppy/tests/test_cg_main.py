###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Test CG.cg_main() directly (not through the cylinder hub system).
# This exercises the standalone CG code path


import unittest

import mpisppy.opt.cg
import mpisppy.tests.examples.farmer as farmer
from mpisppy.tests.examples.sizes.sizes import scenario_creator as sizes_creator, \
                                               scenario_denouement as sizes_denouement
from mpisppy.tests.utils import get_solver
import mpisppy.MPI as mpi

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver(persistent_OK=False)

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

# Known reference values (EF optimal for farmer with 3 scenarios, crops_multiplier=1)
FARMER_EF_OBJ = -118361.33  # approximate


class TestCGMainFarmer(unittest.TestCase):
    """Test CG.cg_main() with the farmer model, checking solution quality."""

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
        self.creator_kwargs = {"crops_multiplier": 1}

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_iter0_creates_initial_columns(self):
        """After Iter0, farmer should have at least one column per scenario."""
        cg = mpisppy.opt.cg.CG(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        cg.CG_Prep()
        cg.Iter0()

        for sname in cg.all_scenario_names:
            self.assertGreater(cg.next_col[sname], 0)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_master_problem_exists(self):
        """CG on farmer should build the master problem."""
        cg = mpisppy.opt.cg.CG(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        cg.CG_Prep()

        if cg.cylinder_rank == 0:
            self.assertIsNotNone(cg.mp)
            self.assertTrue(hasattr(cg.mp, "obj"))
            self.assertTrue(hasattr(cg.mp, "Convexity"))
            self.assertTrue(hasattr(cg.mp, "NonAnt"))
    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_duplicate_column_rejected(self):
        """CG should reject duplicate columns for the same scenario."""
        cg = mpisppy.opt.cg.CG(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        cg.CG_Prep()
        cg.Iter0()

        if cg.cylinder_rank == 0:
            sname = cg.all_scenario_names[0]
            x_vec = {i: 0.0 for i in cg.nonant_indices}
            cg.add_column_for_scenario(sname, 1.0, x_vec)
            added_again = cg.add_column_for_scenario(sname, 1.0, x_vec)
            self.assertFalse(added_again)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_obj(self):
        """CG on farmer should approach the EF optimal."""
        cg = mpisppy.opt.cg.CG(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj = cg.cg_main()

        if cg.cylinder_rank == 0:
            self.assertIsNotNone(obj)
            self.assertAlmostEqual(obj, FARMER_EF_OBJ, delta=abs(FARMER_EF_OBJ*0.01))

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_farmer_no_finalize(self):
        """CG.cg_main() with finalize=False returns None for Eobj."""
        cg = mpisppy.opt.cg.CG(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj = cg.cg_main(finalize=False)
        self.assertIsNone(obj)
        self.assertIsNotNone(conv)

class TestCGMainSizes(unittest.TestCase):
    """Test CG.cg_main() with the sizes model, including integer behavior."""

    def setUp(self):
        self.options = {
            "solver_name": solver_name,
            "CGIterLimit": 10,
            "convthresh": 0.001,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "sp_solver_options": {"mipgap": 0.02},
            "mp_solver_options": {"mipgap": 0.02},
            "relaxed_nonant": False,
            "toc": False,
        }
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]
        self.creator_kwargs = {"scenario_count": 3}

    def _copy_options(self):
        return dict(self.options)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_sizes_obj_range(self):
        """CG on sizes should produce an objective in a reasonable range."""
        cg = mpisppy.opt.cg.CG(
            self._copy_options(),
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj = cg.cg_main()
        # sizes optimal is around 227000
        if cg.cylinder_rank == 0:
            self.assertIsNotNone(obj)
            self.assertGreater(obj, 100000)
            self.assertLess(obj, 400000)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_sizes_no_finalize(self):
        """CG.cg_main() with finalize=False returns None for Eobj."""
        cg = mpisppy.opt.cg.CG(
            self._copy_options(),
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        conv, obj = cg.cg_main(finalize=False)
        self.assertIsNone(obj)
        self.assertIsNotNone(conv)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solver_name,))
    def test_sizes_iter0_creates_initial_columns(self):
        """After Iter0, sizes should have at least one column per scenario."""
        cg = mpisppy.opt.cg.CG(
            self._copy_options(),
            self.scenario_names,
            sizes_creator,
            sizes_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        cg.CG_Prep()
        cg.Iter0()

        for sname in cg.all_scenario_names:
            self.assertGreater(cg.next_col[sname], 0)


if __name__ == '__main__':
    unittest.main()
