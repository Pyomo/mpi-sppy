###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Serial unit tests for the subproblem index bookkeeping in
# LShapedCutGeneratorData.set_ls (regression tests for issue #551: scenarios
# built into the root problem have no eta variable and no Benders subproblem,
# so the global subproblem indexing must skip them).

import unittest

import pyomo.contrib.benders.benders_cuts as bc
import pyomo.environ as pyo

from mpisppy.utils.lshaped_cuts import LShapedCutGenerator


class _FakeLShaped:
    """Just enough of LShapedMethod for LShapedCutGeneratorData.set_ls:
    one MPI rank's view of the scenario assignment and the root model.
    """
    def __init__(self, all_scenario_names, local_scenario_names,
                 root_scenarios=None):
        self.all_scenario_names = all_scenario_names
        self.local_scenario_names = local_scenario_names
        self.has_root_scens = root_scenarios is not None
        self.root_scenarios = root_scenarios
        self.root = pyo.ConcreteModel()
        eta_names = [s for s in all_scenario_names
                     if root_scenarios is None or s not in root_scenarios]
        self.root.eta = pyo.Var(eta_names)


def _make_bender(ls):
    m = pyo.ConcreteModel()
    m.bender = LShapedCutGenerator()
    m.bender.set_input(root_vars=[], tol=1e-8)
    m.bender.set_ls(ls)
    return m.bender


@unittest.skipUnless(bc.mpi4py_available and bc.numpy_available,
                     "LShapedCutGenerator requires mpi4py and numpy")
class TestSetLSIndexing(unittest.TestCase):

    _all_names = ["scen0", "scen1", "scen2"]

    def test_no_root_scenarios_single_rank(self):
        ls = _FakeLShaped(self._all_names, list(self._all_names))
        bender = _make_bender(ls)
        self.assertEqual(bender.global_num_subproblems(), 3)
        self.assertEqual(bender._subproblem_ndx_map, {0: 0, 1: 1, 2: 2})
        self.assertEqual(bender.all_root_etas, list(ls.root.eta.values()))

    def test_no_root_scenarios_rank_view(self):
        # the middle rank of a three-rank hub
        ls = _FakeLShaped(self._all_names, ["scen1"])
        bender = _make_bender(ls)
        self.assertEqual(bender.global_num_subproblems(), 3)
        self.assertEqual(bender._subproblem_ndx_map, {0: 1})

    def test_no_has_root_scens_attribute(self):
        # cross_scen_spoke.py calls set_ls with an opt object that has
        # no has_root_scens attribute; all scenarios get subproblems
        ls = _FakeLShaped(self._all_names, list(self._all_names))
        del ls.has_root_scens
        bender = _make_bender(ls)
        self.assertEqual(bender.global_num_subproblems(), 3)
        self.assertEqual(bender._subproblem_ndx_map, {0: 0, 1: 1, 2: 2})

    def test_root_scenario_global_count(self):
        ls = _FakeLShaped(self._all_names, list(self._all_names),
                          root_scenarios=["scen1"])
        bender = _make_bender(ls)
        self.assertEqual(bender.global_num_subproblems(), 2)
        self.assertEqual(len(bender.all_root_etas), 2)
        # the root scenario contributes no subproblem, and the scenario
        # after it maps to the second (not third) global slot
        self.assertEqual(bender._subproblem_ndx_map, {0: 0, 1: 1})

    def test_root_scenario_rank_views(self):
        # issue #551: three-rank hub, one scenario per rank, the middle
        # scenario built into the root problem; the rank owning scen2
        # must map its subproblem inside range(global_num_subproblems())
        rank_locals = [["scen0"], ["scen1"], ["scen2"]]
        expected_maps = [{0: 0}, {}, {0: 1}]
        seen_global_ndxs = []
        for local_names, expected in zip(rank_locals, expected_maps):
            ls = _FakeLShaped(self._all_names, local_names,
                              root_scenarios=["scen1"])
            bender = _make_bender(ls)
            self.assertEqual(bender.global_num_subproblems(), 2)
            self.assertEqual(bender._subproblem_ndx_map, expected)
            for global_ndx in bender._subproblem_ndx_map.values():
                self.assertLess(global_ndx, len(bender.all_root_etas))
            seen_global_ndxs.extend(bender._subproblem_ndx_map.values())
        # across the ranks, every global slot is covered exactly once
        self.assertEqual(sorted(seen_global_ndxs), [0, 1])

    def test_root_scenario_shared_rank_view(self):
        # two-rank hub: the rank owning both scen0 and the root scenario
        # has exactly one subproblem
        ls = _FakeLShaped(self._all_names, ["scen0", "scen1"],
                          root_scenarios=["scen1"])
        bender = _make_bender(ls)
        self.assertEqual(bender.global_num_subproblems(), 2)
        self.assertEqual(bender._subproblem_ndx_map, {0: 0})


@unittest.skipUnless(bc.mpi4py_available and bc.numpy_available,
                     "LShapedCutGenerator requires mpi4py and numpy")
class TestSolverWithoutName(unittest.TestCase):
    """The APPSI solver interfaces have no name attribute (issue #954)."""

    def _make_bender(self, solver_name):
        ls = _FakeLShaped(["scen0"], ["scen0"])
        ls.root.x = pyo.Var(initialize=1.0)
        ls.root.eta["scen0"].set_value(0.0)
        ls.root.bender = LShapedCutGenerator()
        ls.root.bender.set_input(root_vars=[ls.root.x], tol=1e-8)
        ls.root.bender.set_ls(ls)

        def subproblem_fn(scenario_name):
            m = pyo.ConcreteModel()
            m.x = pyo.Var()
            m.obj = pyo.Objective(expr=m.x)
            return m, pyo.ComponentMap([(ls.root.x, m.x)])

        ls.root.bender.add_subproblem(
            subproblem_fn, {"scenario_name": "scen0"}, ls.root.eta["scen0"],
            subproblem_solver=solver_name)
        return ls.root.bender

    def test_solver_is_given_the_name_it_was_requested_by(self):
        bender = self._make_bender("appsi_highs")
        self.assertEqual(bender.subproblem_solvers[0].name, "appsi_highs")

    def test_solver_with_no_sign_convention_gets_a_clear_error(self):
        # The sign check comes before the solve, so appsi_cbc need not be
        # installed.
        bender = self._make_bender("appsi_cbc")
        with self.assertRaisesRegex(NotImplementedError, "appsi_cbc"):
            bender.generate_cut()


if __name__ == "__main__":
    unittest.main()
