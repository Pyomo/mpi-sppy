###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the Value of the Stochastic Solution (--vss) report.

Covers vss_prep's fail-fast guards, the sign/bracket helpers, and end-to-end
EF runs (minimize and maximize) plus the infeasible-mean-value-solution path,
all on the two-stage farmer example.
"""

import math
import os
import sys
import types
import unittest

import numpy as np
import pyomo.environ as pyo

import mpisppy.opt.ef
import mpisppy.utils.sputils as sputils
import mpisppy.generic.parsing as parsing
from mpisppy.generic import vss
from mpisppy import MPI
from mpisppy.tests.utils import get_solver

# make the farmer example importable (it defines average_scenario_creator)
_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples")
sys.path.insert(0, os.path.join(_EXAMPLES_DIR, "farmer"))
import farmer  # noqa: E402

solver_available, solver_name, _, _ = get_solver()


class _FakeCfg:
    """Minimal cfg exposing only .get(key, ifmissing=...) for vss_prep."""
    def __init__(self, d):
        self._d = d

    def get(self, key, ifmissing=None):
        return self._d.get(key, ifmissing)


def _make_cfg(extra_argv):
    """Build a real, fully-parsed Config for the farmer module."""
    argv = ["prog", "--module-name", "farmer", "--num-scens", "3",
            "--solver-name", solver_name or "gurobi"] + extra_argv
    old = sys.argv
    sys.argv = argv
    try:
        return parsing.parse_args(farmer)
    finally:
        sys.argv = old


class TestVssPrep(unittest.TestCase):
    """vss_prep should fail fast on unsupported configurations."""

    def test_clean_two_stage_passes(self):
        vss.vss_prep(farmer, _FakeCfg({}))  # farmer has average_scenario_creator

    def test_missing_average_scenario_creator(self):
        dummy = types.SimpleNamespace()  # no average_scenario_creator
        with self.assertRaises(RuntimeError) as ctx:
            vss.vss_prep(dummy, _FakeCfg({}))
        self.assertIn("average_scenario_creator", str(ctx.exception))

    def test_multistage_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            vss.vss_prep(farmer, _FakeCfg({"branching_factors": [2, 3]}))
        self.assertIn("two-stage", str(ctx.exception))

    def test_cvar_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            vss.vss_prep(farmer, _FakeCfg({"cvar": True}))
        self.assertIn("cvar", str(ctx.exception).lower())

    def test_admm_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            vss.vss_prep(farmer, _FakeCfg({"admm": True}))
        self.assertIn("ADMM", str(ctx.exception))

    def test_proper_bundles_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            vss.vss_prep(farmer, _FakeCfg({"scenarios_per_bundle": 2}))
        self.assertIn("bundles", str(ctx.exception))


class TestVssHelpers(unittest.TestCase):
    """Sign convention and bound-reduction helpers (no solver needed)."""

    def test_vss_value_min(self):
        # minimize: VSS = EEV - RP
        self.assertAlmostEqual(vss._vss_value(True, rp=-100.0, eev=-90.0), 10.0)

    def test_vss_value_max(self):
        # maximize: VSS = RP - EEV
        self.assertAlmostEqual(vss._vss_value(False, rp=100.0, eev=90.0), 10.0)

    def test_vss_value_infinite_eev(self):
        self.assertTrue(math.isinf(vss._vss_value(True, rp=-100.0, eev=math.inf)))

    def test_reduce_bounds_min(self):
        # min: inner is the tightest (lowest) incumbent, outer the highest bound
        inner, outer = vss._reduce_bounds(MPI.COMM_WORLD, 5.0, 3.0, is_min=True)
        self.assertEqual((inner, outer), (5.0, 3.0))

    def test_reduce_bounds_max(self):
        inner, outer = vss._reduce_bounds(MPI.COMM_WORLD, 5.0, 7.0, is_min=False)
        self.assertEqual((inner, outer), (5.0, 7.0))


@unittest.skipIf(not solver_available, "no solver is available")
class TestVssEndToEnd(unittest.TestCase):
    """End-to-end VSS on farmer (3 scenarios)."""

    def _ef(self, kwargs):
        snames = farmer.scenario_names_creator(3)
        ef = mpisppy.opt.ef.ExtensiveForm(
            {"solver": solver_name}, snames, farmer.scenario_creator,
            scenario_creator_kwargs=kwargs)
        ef.solve_extensive_form(tee=False)
        return ef

    def test_ef_minimize(self):
        cfg = _make_cfg(["--EF", "--EF-solver-name", solver_name])
        kwargs = farmer.kw_creator(cfg)
        ef = self._ef(kwargs)
        res = vss.do_vss(farmer, cfg, farmer.scenario_creator, kwargs,
                         farmer.scenario_denouement, ef=ef)
        self.assertTrue(res["is_min"])
        self.assertAlmostEqual(res["RP"], ef.get_objective_value(), places=3)
        # VSS = EEV - RP, and VSS >= 0 (the mean-value solution can't beat RP)
        self.assertAlmostEqual(res["VSS"], res["EEV"] - res["RP"], places=3)
        self.assertGreaterEqual(res["VSS"], -1e-6)
        # regression on the known farmer-3 values
        self.assertAlmostEqual(res["RP"], -108390.0, delta=1.0)
        self.assertAlmostEqual(res["EV"], -118600.0, delta=1.0)
        self.assertAlmostEqual(res["EEV"], -107240.0, delta=1.0)
        self.assertAlmostEqual(res["VSS"], 1150.0, delta=1.0)

    def test_ef_maximize(self):
        cfg = _make_cfg(["--EF", "--EF-solver-name", solver_name])
        kwargs = dict(farmer.kw_creator(cfg), sense=pyo.maximize)
        ef = self._ef(kwargs)
        res = vss.do_vss(farmer, cfg, farmer.scenario_creator, kwargs,
                         farmer.scenario_denouement, ef=ef)
        self.assertFalse(res["is_min"])
        # maximize: VSS = RP - EEV, still >= 0
        self.assertAlmostEqual(res["VSS"], res["RP"] - res["EEV"], places=3)
        self.assertGreaterEqual(res["VSS"], -1e-6)

    def test_decomposition_bracket(self):
        # A stub wheel exercises the incumbent+bracket branch without cylinders.
        # Bounds chosen to bracket the true RP (~-108390) for minimization.
        cfg = _make_cfg([])
        kwargs = farmer.kw_creator(cfg)
        fake_wheel = types.SimpleNamespace(BestInnerBound=-108389.0,
                                           BestOuterBound=-108508.0)
        res = vss.do_vss(farmer, cfg, farmer.scenario_creator, kwargs,
                         farmer.scenario_denouement, wheel=fake_wheel)
        self.assertEqual(res["RP"], -108389.0)     # incumbent
        self.assertEqual(res["inner"], -108389.0)
        self.assertEqual(res["outer"], -108508.0)
        self.assertIsNotNone(res["vss_bracket"])
        lo, hi = res["vss_bracket"]
        self.assertAlmostEqual(lo, res["EEV"] - res["inner"], places=4)
        self.assertAlmostEqual(hi, res["EEV"] - res["outer"], places=4)
        self.assertLessEqual(lo, hi)

    def test_eev_infeasible_gives_inf(self):
        # Fixing the first stage x=5 is infeasible in the scenario whose
        # demand is 50, so EEV (hence VSS) is +inf.
        def _infeas_creator(sname, **kw):
            m = pyo.ConcreteModel()
            m.x = pyo.Var(bounds=(0, 100))
            demand = 5.0 if sputils.extract_num(sname) == 0 else 50.0
            m.meet = pyo.Constraint(expr=m.x >= demand)
            m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
            m._mpisppy_probability = 0.5
            sputils.attach_root_node(m, m.x, [m.x])
            return m

        cfg = types.SimpleNamespace(solver_name=solver_name)
        eev, names = vss._compute_eev(
            cfg, _infeas_creator, {}, ["Scenario0", "Scenario1"],
            np.array([5.0]))
        self.assertTrue(math.isinf(eev))
        self.assertIn("Scenario1", names)


if __name__ == "__main__":
    unittest.main()
