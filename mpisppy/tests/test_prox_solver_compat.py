###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the quadratic-proximal-term / solver-capability checks.

Progressive Hedging's proximal term makes the subproblem objective quadratic
(unless it is linearized). A solver that cannot handle a quadratic objective
then fails -- cryptically, in the case of HiGHS on an MIQP (see issue #762).
mpi-sppy guards this in two complementary ways:

* a proactive check that uses the legacy ``has_capability`` API where it is
  available (catches e.g. cbc/glpk before any solve), and
* a reactive check after the first proximal solve, for solvers that do not
  expose capability information (e.g. HiGHS via the APPSI or
  ``pyomo.contrib.solver`` interfaces).

These tests exercise the decision logic directly and need neither MPI nor a
solver.
"""

import types
import unittest

import mpisppy.utils.sputils as sputils
from mpisppy.phbase import PHBase


class _FakePlugin:
    """Stand-in solver plugin exposing has_capability (legacy-style)."""

    def __init__(self, cap):
        self._cap = cap

    def has_capability(self, name):
        if isinstance(self._cap, Exception):
            raise self._cap
        return self._cap


class _NoCapPlugin:
    """Stand-in for APPSI / contrib.solver wrappers: no has_capability."""


def _scenario(plugin=None, solution_available=True):
    return types.SimpleNamespace(
        _solver_plugin=plugin,
        _mpisppy_data=types.SimpleNamespace(solution_available=solution_available),
    )


def _opt(prox_quadratic, scenarios, phiter=1, solver_name="testsolver"):
    """A minimal duck-typed stand-in for a PHBase instance.

    Only the attributes the checks under test actually touch are provided;
    ``allreduce_or`` defaults to the single-rank identity and can be
    overridden to simulate other ranks.
    """
    opt = types.SimpleNamespace()
    opt._prox_is_quadratic = lambda: prox_quadratic
    opt.local_scenarios = scenarios
    opt.options = {"solver_name": solver_name}
    opt._PHIter = phiter
    opt.allreduce_or = lambda local: local
    return opt


class TestSolverQuadraticCapabilityProbe(unittest.TestCase):
    def test_reports_supported(self):
        self.assertIs(
            sputils.solver_quadratic_objective_capability(_FakePlugin(True)), True
        )

    def test_reports_unsupported(self):
        self.assertIs(
            sputils.solver_quadratic_objective_capability(_FakePlugin(False)), False
        )

    def test_unknown_when_no_capability_method(self):
        self.assertIsNone(
            sputils.solver_quadratic_objective_capability(_NoCapPlugin())
        )

    def test_unknown_when_capability_raises(self):
        self.assertIsNone(
            sputils.solver_quadratic_objective_capability(
                _FakePlugin(RuntimeError("boom"))
            )
        )


class TestProxIsQuadratic(unittest.TestCase):
    @staticmethod
    def _stub(prox_approx, prox_disabled):
        return types.SimpleNamespace(
            _prox_approx=prox_approx, prox_disabled=prox_disabled
        )

    def test_quadratic_when_attached_and_not_linearized(self):
        self.assertTrue(PHBase._prox_is_quadratic(self._stub(False, False)))

    def test_not_quadratic_when_linearized(self):
        self.assertFalse(PHBase._prox_is_quadratic(self._stub(True, False)))

    def test_not_quadratic_when_prox_disabled(self):
        self.assertFalse(PHBase._prox_is_quadratic(self._stub(False, True)))


class TestCheckProxSolverCapability(unittest.TestCase):
    def test_raises_when_solver_reports_no_quadratic(self):
        opt = _opt(True, {"Scenario1": _scenario(_FakePlugin(False))})
        with self.assertRaisesRegex(RuntimeError, "linearize-proximal-terms"):
            PHBase._check_prox_solver_capability(opt)

    def test_no_raise_when_capability_unknown(self):
        # HiGHS-like: no has_capability -> deferred to the reactive check
        opt = _opt(True, {"Scenario1": _scenario(_NoCapPlugin())})
        PHBase._check_prox_solver_capability(opt)

    def test_no_raise_when_solver_supports_quadratic(self):
        opt = _opt(True, {"Scenario1": _scenario(_FakePlugin(True))})
        PHBase._check_prox_solver_capability(opt)

    def test_no_raise_when_prox_linearized(self):
        # quadratic check is skipped entirely when the prox term is linearized
        opt = _opt(False, {"Scenario1": _scenario(_FakePlugin(False))})
        PHBase._check_prox_solver_capability(opt)


class TestCheckProxSolveSucceeded(unittest.TestCase):
    def test_raises_when_no_solution_anywhere_at_iter1(self):
        opt = _opt(True, {"s1": _scenario(solution_available=False)}, phiter=1)
        with self.assertRaisesRegex(RuntimeError, "linearize-proximal-terms"):
            PHBase._check_prox_solve_succeeded(opt)

    def test_no_raise_when_a_subproblem_solved(self):
        opt = _opt(
            True,
            {
                "s1": _scenario(solution_available=False),
                "s2": _scenario(solution_available=True),
            },
            phiter=1,
        )
        PHBase._check_prox_solve_succeeded(opt)

    def test_no_raise_after_first_iteration(self):
        opt = _opt(True, {"s1": _scenario(solution_available=False)}, phiter=2)
        PHBase._check_prox_solve_succeeded(opt)

    def test_no_raise_when_prox_linearized(self):
        opt = _opt(False, {"s1": _scenario(solution_available=False)}, phiter=1)
        PHBase._check_prox_solve_succeeded(opt)

    def test_no_raise_when_another_rank_has_solution(self):
        # locally everything failed, but allreduce_or reports a solution on
        # some other rank -> the raise decision must be False on every rank
        opt = _opt(True, {"s1": _scenario(solution_available=False)}, phiter=1)
        opt.allreduce_or = lambda local: True
        PHBase._check_prox_solve_succeeded(opt)


class TestReraiseAsProxCapabilityError(unittest.TestCase):
    """The reactive check is unreachable when a solver signals 'no quadratic
    objective' by *raising* during the solve (e.g. cbc/glpk, whose LP writer
    raises before any solution is returned). This converts that raise into the
    same actionable message, preserving the original via exception chaining.
    """

    def test_reraises_actionable_message_at_iter1(self):
        opt = _opt(True, {"s1": _scenario()}, phiter=1)
        original = ValueError("contains nonlinear terms")
        with self.assertRaisesRegex(RuntimeError, "linearize-proximal-terms") as cm:
            PHBase._reraise_as_prox_capability_error(opt, original)
        # the original solver error is preserved for debugging
        self.assertIs(cm.exception.__cause__, original)

    def test_no_reraise_after_first_iteration(self):
        # a later raise is a genuine solve error -> caller re-raises it as-is
        opt = _opt(True, {"s1": _scenario()}, phiter=2)
        self.assertIsNone(
            PHBase._reraise_as_prox_capability_error(opt, ValueError("boom"))
        )

    def test_no_reraise_when_prox_linearized(self):
        # linearized prox is not quadratic, so a raise is not a capability issue
        opt = _opt(False, {"s1": _scenario()}, phiter=1)
        self.assertIsNone(
            PHBase._reraise_as_prox_capability_error(opt, ValueError("boom"))
        )


if __name__ == "__main__":
    unittest.main()
