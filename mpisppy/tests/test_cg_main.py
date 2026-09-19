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

import pyomo.environ as pyo
import mpisppy.opt.cg
import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.sputils as sputils
from mpisppy.tests.examples.sizes.sizes import scenario_creator as sizes_creator, \
                                               scenario_denouement as sizes_denouement
from mpisppy.tests.utils import get_solver, limit_solver_threads
from mpisppy.cgbase import CGBase
import mpisppy.MPI as mpi

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver(persistent_OK=False)

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

# Known reference values (EF optimal for farmer with 3 scenarios, crops_multiplier=1)
FARMER_EF_OBJ = -118361.33  # approximate


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
    def test_farmer_maximize_obj(self):
        """CG on a maximization farmer should approach the EF optimal."""
        creator_kwargs = {"crops_multiplier": 1, "sense": pyo.maximize}
        cg = mpisppy.opt.cg.CG(
            self._copy_options(),
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=creator_kwargs,
        )
        conv, obj = cg.cg_main()

        if cg.cylinder_rank == 0:
            ef_obj = _solve_farmer_ef(self.scenario_names, creator_kwargs)
            self.assertIsNotNone(obj)
            self.assertAlmostEqual(obj, ef_obj, delta=abs(ef_obj*0.01))

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



class TestRedCostSumming(unittest.TestCase):
    """add_columns_to_mp_from_results with a subproblem that reported no bound.

    build_columns_from_subproblem_solutions already leaves red_cost at None
    when a subproblem's solve produced no outer bound -- it tests the value for
    finiteness precisely because that can happen. The consumer used to add it
    unconditionally, which raised

        TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'

    the moment the case actually occurred (seen in CI, where the size-limited
    cplex build fails solves that succeed on a full license).

    Silently skipping the missing term would be worse than the crash: the caller
    adds the sum to rmp_obj_val and treats the result as an outer bound, so
    dropping a term overstates it. The sum reports itself unavailable instead.

    No solver needed -- the method only calls self.add_column_for_scenario, so a
    recorder stands in for the CG object.
    """

    class _Recorder:
        def __init__(self):
            self.added = []

        def add_column_for_scenario(self, sname, scen_cost, xvec):
            self.added.append(sname)

    def _sum(self, all_results):
        rec = self._Recorder()
        total = CGBase.add_columns_to_mp_from_results(rec, all_results)
        return total, rec.added

    def test_all_bounds_present_sums_them(self):
        results = [[("s0", 1.5, 10.0, {}), ("s1", 2.5, 20.0, {})]]
        total, added = self._sum(results)
        self.assertEqual(total, 4.0)
        self.assertEqual(added, ["s0", "s1"])

    def test_a_missing_bound_makes_the_sum_unavailable(self):
        results = [[("s0", 1.5, 10.0, {}), ("s1", None, 20.0, {})]]
        total, added = self._sum(results)
        self.assertIsNone(total)
        # The columns are still added: they are useful whether or not a bound
        # came with them.
        self.assertEqual(added, ["s0", "s1"])

    def test_single_tuple_per_rank_is_handled_too(self):
        # Ranks may report one tuple rather than a list of them.
        self.assertEqual(self._sum([("s0", 3.0, 10.0, {})])[0], 3.0)
        self.assertIsNone(self._sum([("s0", None, 10.0, {})])[0])

    def test_none_rank_results_are_skipped(self):
        total, added = self._sum([None, [("s0", 1.0, 10.0, {})]])
        self.assertEqual(total, 1.0)
        self.assertEqual(added, ["s0"])



if __name__ == '__main__':
    unittest.main()
