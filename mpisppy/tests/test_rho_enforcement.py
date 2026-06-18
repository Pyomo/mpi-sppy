###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for consistent enforcement of rho > 0 (GitHub issue #560).

Progressive Hedging requires a strictly positive rho for every non-anticipative
variable (rho multiplies the proximal/consensus term). These tests cover:

  * the central guard ``check_rhos_positive`` (raises on rho <= 0);
  * the no-spam fallback reporter ``report_zero_rho_fallback`` (aggregated,
    report-only-on-change);
  * validation of the ``default_rho`` command-line option;
  * end-to-end PH runs that reject a non-positive ``defaultPHrho`` and a
    ``rho_setter`` that returns a non-positive value (solver required).
"""

import io
import types
import contextlib
import unittest

import pyomo.environ as pyo

import mpisppy.opt.ph
import mpisppy.MPI as MPI
from mpisppy.utils.rho_utils import check_rhos_positive, report_zero_rho_fallback
from mpisppy.generic.decomp import _get_rho_setter

from mpisppy.tests.utils import get_solver
from mpisppy.tests.examples.farmer import scenario_creator, scenario_denouement

solver_available, solver_name, *_ = get_solver()

NDN = "ROOT"


class _Holder:
    """Stand-in for a mutable Pyomo Param data object (only ._value is used)."""
    def __init__(self, value):
        self._value = value


def _make_opt(rho_values, cylinder_rank=0):
    """Minimal SPBase-like object with one scenario carrying the given rhos."""
    sm = pyo.ConcreteModel()
    sm.v = pyo.Var(range(len(rho_values)))
    for i in range(len(rho_values)):
        sm.v[i]._value = 1.0
    data = types.SimpleNamespace(
        nonant_indices={(NDN, i): sm.v[i] for i in range(len(rho_values))},
    )
    model = types.SimpleNamespace(
        rho={(NDN, i): _Holder(rho_values[i]) for i in range(len(rho_values))},
    )
    scen = types.SimpleNamespace(
        _pyomo_model=sm,  # keep Vars alive
        _mpisppy_data=data,
        _mpisppy_model=model,
    )
    return types.SimpleNamespace(
        local_scenarios={"Scen1": scen},
        cylinder_rank=cylinder_rank,
    )


class Test_check_rhos_positive(unittest.TestCase):
    """The single central rho > 0 guard."""

    def test_all_positive_passes(self):
        opt = _make_opt([1.0, 0.5, 1e-6])
        # should not raise
        check_rhos_positive(opt, source="unit test")

    def test_zero_raises(self):
        opt = _make_opt([1.0, 0.0, 2.0])
        with self.assertRaises(RuntimeError) as cm:
            check_rhos_positive(opt, source="unit test")
        msg = str(cm.exception)
        self.assertIn("Scen1", msg)
        self.assertIn("v[1]", msg)
        self.assertIn("unit test", msg)

    def test_negative_raises(self):
        opt = _make_opt([1.0, -3.0])
        with self.assertRaises(RuntimeError):
            check_rhos_positive(opt)

    def test_none_raises(self):
        opt = _make_opt([1.0, None])
        with self.assertRaises(RuntimeError):
            check_rhos_positive(opt)


class Test_report_zero_rho_fallback(unittest.TestCase):
    """The aggregated, report-only-on-change fallback reporter (no log spam)."""

    def setUp(self):
        self.opt = _make_opt([1.0])  # only cylinder_rank matters here
        self.state = {}

    def _emit(self, count):
        """Call the reporter, returning the captured stdout text."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_zero_rho_fallback(self.opt, "CoeffRho", count, 1.0, self.state)
        return buf.getvalue()

    def test_zero_count_is_silent(self):
        out = self._emit(0)
        self.assertNotIn("CoeffRho", out)
        self.assertEqual(self.state["CoeffRho"], 0)

    def test_reports_once_then_dedups(self):
        first = self._emit(3)
        self.assertIn("CoeffRho", first)
        self.assertIn("3 nonant", first)
        # identical situation: no second message
        second = self._emit(3)
        self.assertNotIn("CoeffRho", second)

    def test_reports_again_when_count_changes(self):
        self._emit(3)
        changed = self._emit(5)
        self.assertIn("5 nonant", changed)

    def test_non_rank0_is_silent(self):
        opt = _make_opt([1.0], cylinder_rank=1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_zero_rho_fallback(opt, "CoeffRho", 4, 1.0, {})
        self.assertNotIn("CoeffRho", buf.getvalue())


class Test_default_rho_validation(unittest.TestCase):
    """`--default-rho` must be strictly positive."""

    @staticmethod
    def _cfg(default_rho):
        return types.SimpleNamespace(
            default_rho=default_rho,
            sep_rho=False, coeff_rho=False, sensi_rho=False,
        )

    def test_zero_default_rho_raises(self):
        with self.assertRaises(RuntimeError):
            _get_rho_setter(types.SimpleNamespace(), self._cfg(0.0))

    def test_negative_default_rho_raises(self):
        with self.assertRaises(RuntimeError):
            _get_rho_setter(types.SimpleNamespace(), self._cfg(-1.0))

    def test_positive_default_rho_ok(self):
        # no rho_setter on the module, positive default -> returns None, no raise
        self.assertIsNone(_get_rho_setter(types.SimpleNamespace(), self._cfg(2.0)))


def _farmer_ph_options(default_rho=1.0):
    return {
        "asynchronousPH": False,
        "solver_name": solver_name,
        "PHIterLimit": 2,
        "defaultPHrho": default_rho,
        "convthresh": 1e-3,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
    }


@unittest.skipIf(not solver_available, "no solver available")
@unittest.skipIf(MPI.COMM_WORLD.Get_size() > 1, "serial test; run without mpiexec")
class Test_rho_enforcement_integration(unittest.TestCase):
    """End-to-end PH runs reject non-positive rho at its sources."""

    def setUp(self):
        self.names = ["scen0", "scen1", "scen2"]
        self.kwargs = {"num_scens": 3}

    def test_nonpositive_defaultPHrho_raises(self):
        ph = mpisppy.opt.ph.PH(
            _farmer_ph_options(default_rho=0.0),
            self.names, scenario_creator, scenario_denouement,
            scenario_creator_kwargs=self.kwargs,
        )
        with self.assertRaises(RuntimeError) as cm:
            ph.ph_main()
        self.assertIn("rho", str(cm.exception).lower())

    def test_nonpositive_rho_setter_raises(self):
        def bad_rho_setter(scenario):
            return [(id(v), 0.0)  # non-positive rho for every nonant
                    for v in scenario._mpisppy_data.nonant_indices.values()]

        ph = mpisppy.opt.ph.PH(
            _farmer_ph_options(default_rho=1.0),
            self.names, scenario_creator, scenario_denouement,
            scenario_creator_kwargs=self.kwargs,
            rho_setter=bad_rho_setter,
        )
        with self.assertRaises(RuntimeError):
            ph.ph_main()

    def test_positive_rho_runs(self):
        # control: a normal positive rho run completes without raising
        ph = mpisppy.opt.ph.PH(
            _farmer_ph_options(default_rho=1.0),
            self.names, scenario_creator, scenario_denouement,
            scenario_creator_kwargs=self.kwargs,
        )
        conv, obj, bound = ph.ph_main()
        self.assertIsNotNone(conv)


if __name__ == "__main__":
    unittest.main()
