###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the ``feasible_xhat_creator`` convention and its
``average_xhat_nonants`` helper.

Two prototypes are exercised: netdes (LP-relax + ceil, via the helper)
and sslp (LP-relax + average + round, rolls own). The convention's
contract is that the returned candidate must be feasible to fix in
every real scenario's per-scenario subproblem; the netdes test
verifies this directly.
"""

import os
import sys
import unittest

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import mpisppy.utils.sputils as sputils
from mpisppy.utils.xhat_helpers import average_xhat_nonants
from mpisppy.tests.utils import get_solver

_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
)
sys.path.insert(0, os.path.join(_EXAMPLES_DIR, "farmer"))
sys.path.insert(0, os.path.join(_EXAMPLES_DIR, "netdes"))
sys.path.insert(0, os.path.join(_EXAMPLES_DIR, "sslp"))

import farmer  # noqa: E402
import farmer_auxiliary  # noqa: E402
import netdes  # noqa: E402
import netdes_auxiliary  # noqa: E402
import sslp_auxiliary  # noqa: E402

solver_available, solver_name, _, _ = get_solver()

_NETDES_DATA = os.path.join(
    _EXAMPLES_DIR, "netdes", "data", "network-10-20-L-01.dat"
)
_SSLP_DATA_DIR = os.path.join(
    _EXAMPLES_DIR, "sslp", "data", "sslp_15_45_5", "scenariodata"
)


class TestAverageXhatNonantsContract(unittest.TestCase):
    """Solver-free contract checks for the helper."""

    def test_rejects_no_node_list(self):
        def bad_creator(name):
            m = pyo.ConcreteModel()
            m.x = pyo.Var()
            m.obj = pyo.Objective(expr=m.x)
            return m

        with self.assertRaises(RuntimeError) as ctx:
            average_xhat_nonants(bad_creator, solver_name="cplex")
        self.assertIn("_mpisppy_node_list", str(ctx.exception))

    def test_rejects_multi_stage(self):
        def two_node_creator(name):
            m = pyo.ConcreteModel()
            m.x = pyo.Var()
            m.y = pyo.Var()
            m.fsc = pyo.Expression(expr=m.x)
            m.obj = pyo.Objective(expr=m.x + m.y)
            sputils.attach_root_node(m, m.fsc, [m.x])
            # tack on a fake second node
            from mpisppy.scenario_tree import ScenarioNode
            m._mpisppy_node_list.append(
                ScenarioNode("STAGE2", 1.0, 2, m.fsc, [m.y], m,
                             parent_name="ROOT")
            )
            return m

        with self.assertRaises(RuntimeError) as ctx:
            average_xhat_nonants(two_node_creator, solver_name="cplex")
        self.assertIn("two-stage only", str(ctx.exception))


@unittest.skipIf(not solver_available, "no solver available")
class TestAverageXhatNonantsOnFarmer(unittest.TestCase):
    """End-to-end on farmer (whose first stage is continuous)."""

    def test_returns_array_of_correct_length(self):
        arr = average_xhat_nonants(
            farmer.average_scenario_creator,
            solver_name=solver_name,
            scenario_creator_kwargs={"num_scens": 6},
        )
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (3,))  # DEVOTED_ACRES over 3 crops

    def test_relax_integrality_is_no_op_on_continuous_first_stage(self):
        a = average_xhat_nonants(
            farmer.average_scenario_creator,
            solver_name=solver_name,
            scenario_creator_kwargs={"num_scens": 6},
        )
        b = average_xhat_nonants(
            farmer.average_scenario_creator,
            solver_name=solver_name,
            scenario_creator_kwargs={"num_scens": 6},
            relax_integrality=True,
        )
        np.testing.assert_allclose(a, b, atol=1e-6)


@unittest.skipIf(not solver_available, "no solver available")
class TestFarmerFeasibleXhatCreator(unittest.TestCase):
    """Continuous first-stage: convention is satisfied by the
    average-scenario optimum unchanged."""

    def test_returns_root_array(self):
        cache = farmer_auxiliary.feasible_xhat_creator(
            solver_name=solver_name, num_scens=6,
        )
        self.assertIn("ROOT", cache)
        arr = cache["ROOT"]
        self.assertEqual(arr.shape, (3,))  # DEVOTED_ACRES over 3 crops
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertTrue(np.all(arr >= -1e-9))

    def test_kwargs_thread_through(self):
        a = farmer_auxiliary.feasible_xhat_creator(
            solver_name=solver_name, num_scens=6, seedoffset=0,
        )
        b = farmer_auxiliary.feasible_xhat_creator(
            solver_name=solver_name, num_scens=6, seedoffset=7,
        )
        self.assertFalse(np.allclose(a["ROOT"], b["ROOT"]),
                         "seedoffset did not change the candidate")


@unittest.skipIf(not solver_available, "no solver available")
class TestNetdesFeasibleXhatCreator(unittest.TestCase):
    """End-to-end check: netdes_auxiliary.feasible_xhat_creator returns
    a candidate that is integer-valued and feasible to pin in every
    real scenario."""

    def setUp(self):
        from parse import parse
        full = parse(_NETDES_DATA, scenario_ix=None)
        self.K = full["K"]
        self.cache = netdes_auxiliary.feasible_xhat_creator(
            solver_name=solver_name, path=_NETDES_DATA,
        )

    def test_returns_root_dict(self):
        self.assertIn("ROOT", self.cache)
        self.assertEqual(set(self.cache.keys()), {"ROOT"})

    def test_values_are_integer(self):
        arr = self.cache["ROOT"]
        self.assertTrue(np.allclose(arr, np.round(arr), atol=1e-9),
                        f"ceil output is not integer-valued: {arr}")

    def test_pins_are_feasible_in_every_real_scenario(self):
        arr = self.cache["ROOT"]
        for k in range(self.K):
            sname = f"Scenario{k}"
            scen = netdes.scenario_creator(sname, path=_NETDES_DATA)
            root = scen._mpisppy_node_list[0]
            for v, val in zip(root.nonant_vardata_list, arr):
                v.fix(val)
            solver = pyo.SolverFactory(solver_name)
            if sputils.is_persistent(solver):
                solver.set_instance(scen)
                results = solver.solve(tee=False)
            else:
                results = solver.solve(scen, tee=False)
            tc = results.solver.termination_condition
            self.assertIn(
                tc, (TerminationCondition.optimal, TerminationCondition.feasible),
                f"netdes pin infeasible on {sname}: tc={tc}, xhat={arr}",
            )


@unittest.skipIf(not solver_available, "no solver available")
@unittest.skipIf(not os.path.isdir(_SSLP_DATA_DIR), "sslp data not found")
class TestSslpFeasibleXhatCreator(unittest.TestCase):

    def test_returns_integer_root_dict(self):
        cache = sslp_auxiliary.feasible_xhat_creator(
            solver_name=solver_name,
            num_scens=5,
            data_dir=_SSLP_DATA_DIR,
        )
        self.assertIn("ROOT", cache)
        arr = cache["ROOT"]
        self.assertEqual(arr.shape, (15,))  # 15 servers in sslp_15_45_5
        self.assertTrue(np.allclose(arr, np.round(arr), atol=1e-9))
        self.assertTrue(np.all((arr >= 0) & (arr <= 1)),
                        f"FacilityOpen out of [0,1]: {arr}")


if __name__ == "__main__":
    unittest.main()
