###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for IIS-on-xhatter-infeasible (issue #356).

Covers the surfaces touched by the feature:

  1. Config.popular_args() registers xhatter_write_iis / xhatter_iis_method /
     xhatter_iis_dir with the right defaults.
  2. cfg_vanilla.shared_options forwards them into the options dict spokes see.
  3. SPOpt.write_iis_on_xhatter_infeasible control flow: option gating, the
     run-once guard, target selection, and fail-soft -- exercised against a
     light stub so no MPI/solver is needed.
  4. SPOpt._resolve_iis_method / _base_solver_name / _subproblem_file_stem.
  5. (solver-gated) the real write_iis (.ilp) and compute_infeasibility_
     explanation (.iis.log) paths end to end.

The stub mixes the real SPOpt methods onto a tiny object so we test the actual
control flow without standing up SPBase/MPI infrastructure (the same approach
used by test_incumbent_writing.py).
"""

import os
import types
import tempfile
import unittest
from unittest.mock import MagicMock

import pyomo.environ as pyo

from mpisppy.spopt import SPOpt
from mpisppy.utils.config import Config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.tests.utils import get_solver

# Non-persistent base name keeps the solver-gated paths simple (the
# explanation path builds the solver via SolverFactory and solves directly).
solver_available, solver_name, _, _ = get_solver(persistent_OK=False)


def _any_iis_persistent_available():
    # write_iis needs a *persistent* commercial interface; which one is
    # present varies by environment, so probe all three.
    for s in ("cplex_persistent", "gurobi_persistent", "xpress_persistent"):
        try:
            if pyo.SolverFactory(s).available(exception_flag=False):
                return True
        except Exception:
            pass
    return False


iis_persistent_available = _any_iis_persistent_available()


class _StubOpt:
    """Borrow the real SPOpt methods under test onto a minimal object."""

    write_iis_on_xhatter_infeasible = SPOpt.write_iis_on_xhatter_infeasible
    _emit_iis = SPOpt._emit_iis
    _resolve_iis_method = SPOpt._resolve_iis_method
    _base_solver_name = SPOpt._base_solver_name
    _subproblem_file_stem = SPOpt._subproblem_file_stem

    def __init__(self, options, local_scenarios=None):
        self.options = options
        self.local_scenarios = local_scenarios or {}

    def _get_cylinder_name(self):
        return "StubCyl"


def _scen(solution_available):
    s = types.SimpleNamespace()
    s._mpisppy_data = types.SimpleNamespace(
        solution_available=solution_available)
    return s


def _infeasible_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None))
    m.c_lo = pyo.Constraint(expr=m.x >= 1)
    m.c_hi = pyo.Constraint(expr=m.x <= 0)
    m.obj = pyo.Objective(expr=m.x)
    return m


class TestConfigRegistration(unittest.TestCase):
    def test_options_registered_with_defaults(self):
        cfg = Config()
        cfg.popular_args()
        self.assertIn("xhatter_write_iis", cfg)
        self.assertIn("xhatter_iis_method", cfg)
        self.assertIn("xhatter_iis_dir", cfg)
        self.assertFalse(cfg.xhatter_write_iis)
        self.assertEqual(cfg.xhatter_iis_method, "auto")
        self.assertIsNone(cfg.xhatter_iis_dir)


class TestSharedOptionsForwarding(unittest.TestCase):
    def _cfg(self, **overrides):
        cfg = Config()
        cfg.popular_args()
        cfg.solver_name = "gurobi"
        for k, v in overrides.items():
            cfg[k] = v
        return cfg

    def test_defaults_forwarded(self):
        opts = vanilla.shared_options(self._cfg(), is_hub=False)
        self.assertFalse(opts["xhatter_write_iis"])
        self.assertEqual(opts["xhatter_iis_method"], "auto")
        self.assertIsNone(opts["xhatter_iis_dir"])

    def test_set_values_forwarded(self):
        opts = vanilla.shared_options(
            self._cfg(xhatter_write_iis=True,
                      xhatter_iis_method="explanation",
                      xhatter_iis_dir="/tmp/iis"),
            is_hub=False)
        self.assertTrue(opts["xhatter_write_iis"])
        self.assertEqual(opts["xhatter_iis_method"], "explanation")
        self.assertEqual(opts["xhatter_iis_dir"], "/tmp/iis")


class TestControlFlow(unittest.TestCase):
    """write_iis_on_xhatter_infeasible gating / once-guard / fail-soft."""

    def test_disabled_is_noop(self):
        opt = _StubOpt({"xhatter_write_iis": False},
                       {"Scen1": _scen(False)})
        opt._emit_iis = MagicMock()
        opt.write_iis_on_xhatter_infeasible()
        opt._emit_iis.assert_not_called()
        self.assertFalse(getattr(opt, "_xhatter_iis_written", False))

    def test_enabled_emits_for_first_infeasible_scenario(self):
        scen = _scen(False)
        opt = _StubOpt({"xhatter_write_iis": True},
                       {"Feas": _scen(True), "Scen1": scen})
        opt._emit_iis = MagicMock()
        opt.write_iis_on_xhatter_infeasible()
        opt._emit_iis.assert_called_once_with(scen, "Scen1")
        self.assertTrue(opt._xhatter_iis_written)

    def test_runs_only_once(self):
        opt = _StubOpt({"xhatter_write_iis": True},
                       {"Scen1": _scen(False)})
        opt._emit_iis = MagicMock()
        opt.write_iis_on_xhatter_infeasible()
        opt.write_iis_on_xhatter_infeasible()
        self.assertEqual(opt._emit_iis.call_count, 1)

    def test_no_infeasible_local_does_not_trip_guard(self):
        opt = _StubOpt({"xhatter_write_iis": True},
                       {"Feas": _scen(True)})
        opt._emit_iis = MagicMock()
        opt.write_iis_on_xhatter_infeasible()
        opt._emit_iis.assert_not_called()
        # guard NOT set: a later infeasibility on this rank can still emit
        self.assertFalse(getattr(opt, "_xhatter_iis_written", False))

    def test_explicit_model_and_label(self):
        opt = _StubOpt({"xhatter_write_iis": True}, {})
        opt._emit_iis = MagicMock()
        sentinel = object()
        opt.write_iis_on_xhatter_infeasible(model=sentinel, label="ROOT_0")
        opt._emit_iis.assert_called_once_with(sentinel, "ROOT_0")

    def test_failure_is_swallowed_and_guard_set(self):
        opt = _StubOpt({"xhatter_write_iis": True},
                       {"Scen1": _scen(False)})
        opt._emit_iis = MagicMock(side_effect=RuntimeError("boom"))
        # must not raise
        opt.write_iis_on_xhatter_infeasible()
        self.assertTrue(opt._xhatter_iis_written)
        # and never retries
        opt.write_iis_on_xhatter_infeasible()
        self.assertEqual(opt._emit_iis.call_count, 1)


class TestHelpers(unittest.TestCase):
    def test_base_solver_name_strips_suffix(self):
        self.assertEqual(
            _StubOpt({"solver_name": "gurobi_persistent"})._base_solver_name(),
            "gurobi")
        self.assertEqual(
            _StubOpt({"solver_name": "gurobi_direct"})._base_solver_name(),
            "gurobi")
        self.assertEqual(
            _StubOpt({"solver_name": "cplex"})._base_solver_name(), "cplex")
        self.assertEqual(_StubOpt({})._base_solver_name(), "")

    def test_resolve_method_auto_commercial(self):
        for name in ("cplex", "gurobi", "xpress",
                     "gurobi_persistent", "cplex_direct"):
            opt = _StubOpt({"solver_name": name, "xhatter_iis_method": "auto"})
            self.assertEqual(opt._resolve_iis_method(), "ilp", name)

    def test_resolve_method_auto_noncommercial(self):
        for name in ("glpk", "cbc", "appsi_highs", "ipopt", ""):
            opt = _StubOpt({"solver_name": name, "xhatter_iis_method": "auto"})
            self.assertEqual(opt._resolve_iis_method(), "explanation", name)

    def test_resolve_method_explicit_passthrough(self):
        opt = _StubOpt({"solver_name": "glpk", "xhatter_iis_method": "ilp"})
        self.assertEqual(opt._resolve_iis_method(), "ilp")
        opt = _StubOpt({"solver_name": "gurobi",
                        "xhatter_iis_method": "explanation"})
        self.assertEqual(opt._resolve_iis_method(), "explanation")

    def test_file_stem(self):
        opt = _StubOpt({})
        self.assertEqual(opt._subproblem_file_stem("Scen1"), "StubCyl_Scen1")
        self.assertEqual(opt._subproblem_file_stem("ROOT_0"), "StubCyl_ROOT_0")

    def test_unknown_method_raises_in_emit(self):
        # _emit_iis raises on a bad method; the public wrapper swallows it,
        # but here we assert the dispatch error directly.
        opt = _StubOpt({"xhatter_iis_method": "bogus"})
        with self.assertRaises(ValueError):
            opt._emit_iis(_infeasible_model(), "Scen1")


@unittest.skipUnless(solver_available,
                     "no commercial solver for end-to-end IIS")
class TestEndToEnd(unittest.TestCase):
    """Real IIS emission, gated on a commercial solver being present."""

    @unittest.skipUnless(iis_persistent_available,
                         "write_iis needs a persistent commercial interface")
    def test_ilp_file_written_once(self):
        with tempfile.TemporaryDirectory(dir=".") as d:
            opt = _StubOpt({"xhatter_write_iis": True,
                            "xhatter_iis_method": "ilp",
                            "xhatter_iis_dir": d,
                            "solver_name": solver_name})
            opt.write_iis_on_xhatter_infeasible(
                model=_infeasible_model(), label="Scen1")
            self.assertTrue(
                os.path.exists(os.path.join(d, "StubCyl_Scen1.ilp")))
            self.assertTrue(opt._xhatter_iis_written)

    def test_explanation_file_written(self):
        with tempfile.TemporaryDirectory(dir=".") as d:
            opt = _StubOpt({"xhatter_write_iis": True,
                            "xhatter_iis_method": "explanation",
                            "xhatter_iis_dir": d,
                            "solver_name": solver_name})
            opt.write_iis_on_xhatter_infeasible(
                model=_infeasible_model(), label="Scen1")
            self.assertTrue(
                os.path.exists(os.path.join(d, "StubCyl_Scen1.iis.log")))


if __name__ == "__main__":
    unittest.main()
