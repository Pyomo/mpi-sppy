###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for mpisppy/utils/dual_certificate.py
#
# Most of these need no solver.  The certificate is a pure function of the point
# the variables hold and the values in the `dual` suffix, so setting both by
# hand gives exact arithmetic and covers the module without depending on ipopt
# being installed -- which it is not, in CI.  The solver-dependent tests are
# gathered in TestWithIpopt and skip cleanly.
#
# The running example throughout is
#
#     min (x-3)^2 + (y-2)^2   s.t.  x + y <= 1,   x, y in [-10, 10]
#
# whose optimum is x=1, y=0 with value 8 and multiplier 4 -- all analytic, so
# every expected number below is exact rather than a recorded observation.

import unittest

import pyomo.environ as pyo

from mpisppy.tests.utils import announce_hsl_if_used
from mpisppy.utils.dual_certificate import (
    CertificateError,
    certified_lower_bound,
    check_model_is_certifiable,
    unbounded_variables,
)

OPT = 8.0          # analytic optimum of the running example
MULTIPLIER = 4.0   # analytic multiplier of the active constraint

ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)

if ipopt_available:
    announce_hsl_if_used()


def _model(kind="le", bounds=(-10, 10), y_bounds=None):
    """The running example, with the single constraint written four ways.

    All four describe the same feasible set, so all four have the same optimum
    and the same |multiplier| -- only the sign Pyomo reports differs.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=bounds, initialize=0.0)
    m.y = pyo.Var(bounds=y_bounds if y_bounds is not None else bounds, initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x - 3) ** 2 + (m.y - 2) ** 2, sense=pyo.minimize)
    if kind == "le":
        m.c = pyo.Constraint(expr=m.x + m.y <= 1)
    elif kind == "ge":
        m.c = pyo.Constraint(expr=-m.x - m.y >= -1)
    elif kind == "eq":
        m.c = pyo.Constraint(expr=m.x + m.y == 1)
    elif kind == "range":
        m.c = pyo.Constraint(expr=pyo.inequality(-10, m.x + m.y, 1))
    else:
        raise ValueError(kind)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def _place(m, x, y, dual):
    """Put the model at a point with a given dual, as a solve would have."""
    m.x.value = x
    m.y.value = y
    m.dual[m.c] = dual
    return m


# The dual value ipopt reports for each orientation when the true multiplier is
# +4.  Verified against ipopt in TestWithIpopt.test_sign_table_matches_ipopt.
IPOPT_DUAL_AT_OPTIMUM = {"le": -4.0, "ge": 4.0, "eq": -4.0, "range": -4.0}


class TestCertificateMath(unittest.TestCase):
    """No solver required: the point and the duals are supplied directly."""

    def test_exact_at_kkt_point(self):
        # At the exact KKT point the correction term is identically zero and the
        # certificate collapses to the optimal value -- strong duality, exactly.
        m = _place(_model("le"), 1.0, 0.0, IPOPT_DUAL_AT_OPTIMUM["le"])
        self.assertEqual(certified_lower_bound(m, eps_rel=0.0), OPT)

    def test_sign_table_all_orientations(self):
        # Each orientation needs a different rule to recover the same canonical
        # multiplier.  A lost minus sign here shows up as a loose bound, so the
        # assertion is equality, not just validity.
        for kind, dual in IPOPT_DUAL_AT_OPTIMUM.items():
            with self.subTest(kind=kind):
                m = _place(_model(kind), 1.0, 0.0, dual)
                self.assertEqual(certified_lower_bound(m, eps_rel=0.0), OPT)

    def test_inactive_constraint_drops_out(self):
        # Complementarity: a zero dual must contribute nothing.  Unconstrained
        # optimum is x=3, y=2 with value 0.
        m = _model("le")
        m.c.deactivate()
        m.d = pyo.Constraint(expr=m.x + m.y <= 100)
        m.dual[m.d] = 0.0
        _place_x, _place_y = 3.0, 2.0
        m.x.value, m.y.value = _place_x, _place_y
        self.assertEqual(certified_lower_bound(m, eps_rel=0.0), 0.0)

    def test_valid_at_a_point_that_is_not_optimal(self):
        # phi(0,0) = 9 + 4 + 4*(0+0-1) = 9; grad = (-2, 0); the box term drives
        # x to its upper bound 10, contributing -2*(10-0) = -20.
        m = _place(_model("le"), 0.0, 0.0, IPOPT_DUAL_AT_OPTIMUM["le"])
        q = certified_lower_bound(m, eps_rel=0.0)
        self.assertEqual(q, -11.0)
        self.assertLessEqual(q, OPT)

    def test_infeasible_point_still_gives_a_valid_bound(self):
        # vhat need not be feasible: x+y = 4 violates x+y <= 1.  The certificate
        # does not care, which is why a truncated solve is safe.
        m = _place(_model("le"), 2.0, 2.0, IPOPT_DUAL_AT_OPTIMUM["le"])
        self.assertLessEqual(certified_lower_bound(m, eps_rel=0.0), OPT)

    def test_wrong_sign_is_loose_never_wrong(self):
        # Feed the `>=` dual to a `<=` constraint.  max(-(+4), 0) = 0 clips the
        # multiplier away, so phi degenerates to f -- loose, still valid.  This
        # is the robustness property that makes the whole approach safe.
        m = _place(_model("le"), 1.0, 0.0, -IPOPT_DUAL_AT_OPTIMUM["le"])
        q = certified_lower_bound(m, eps_rel=0.0)
        self.assertLess(q, OPT)
        self.assertEqual(q, -68.0)

    def test_looseness_scales_with_box_width(self):
        # The correction is |grad| times the distance to the far end of the box,
        # so wide bounds give a valid but weak bound.  Practical consequence:
        # tight variable bounds are what make this cylinder useful.
        narrow = certified_lower_bound(
            _place(_model("le", bounds=(-10, 10)), 0.0, 0.0, -4.0), eps_rel=0.0
        )
        wide = certified_lower_bound(
            _place(_model("le", bounds=(-1000, 1000)), 0.0, 0.0, -4.0), eps_rel=0.0
        )
        self.assertLess(wide, narrow)
        self.assertLessEqual(narrow, OPT)
        self.assertLessEqual(wide, OPT)

    def test_unbounded_direction_returns_none_not_minus_inf(self):
        # d(phi)/dy = 2*(y-2) + lam = 2y when lam = 4, so any y != 0 gives y a
        # nonzero gradient component.  With y unbounded the box minimization is
        # then -inf, and the contract is None ("no bound this time") rather than
        # a -inf that would poison Ebound's sum.
        m = _place(_model("le", y_bounds=(None, None)), 0.0, 1.0, -4.0)
        self.assertIsNone(certified_lower_bound(m))

    def test_unbounded_on_the_unused_side_only_still_bounds(self):
        # Half-open is enough when the gradient points the other way: here
        # d(phi)/dy = 2 > 0, so only the lower bound is consulted.
        m = _place(_model("le", y_bounds=(-10, None)), 0.0, 1.0, -4.0)
        self.assertIsNotNone(certified_lower_bound(m))
        m2 = _place(_model("le", y_bounds=(None, 10)), 0.0, 1.0, -4.0)
        self.assertIsNone(certified_lower_bound(m2))

    def test_unbounded_direction_with_zero_gradient_still_bounds(self):
        # An infinite bound only defeats the certificate if that variable's
        # gradient component is nonzero.  At the KKT point it is exactly zero
        # here, so a bound is still available.
        m = _place(_model("le", y_bounds=(None, None)), 1.0, 0.0, -4.0)
        self.assertEqual(certified_lower_bound(m, eps_rel=0.0), OPT)

    def test_cushion(self):
        m = _place(_model("le"), 1.0, 0.0, -4.0)
        exact = certified_lower_bound(m, eps_rel=0.0)
        cushioned = certified_lower_bound(m)          # default 1e-9
        self.assertEqual(exact, OPT)
        self.assertLess(cushioned, exact)
        self.assertAlmostEqual(exact - cushioned, 1e-9 * (1.0 + OPT), places=15)

    def test_fixed_variables_are_constants(self):
        # A fixed variable is not free in the box and must not contribute a
        # correction term.  Fix y at its optimal value; the bound is unchanged.
        m = _place(_model("le"), 1.0, 0.0, -4.0)
        m.y.fix(0.0)
        self.assertEqual(certified_lower_bound(m, eps_rel=0.0), OPT)


class TestUnboundedVariables(unittest.TestCase):
    def test_reports_variables_without_finite_bounds(self):
        m = _model("le", y_bounds=(None, None))
        self.assertEqual(unbounded_variables(m, do_fbbt=False), ["y"])

    def test_fbbt_rescues_bounds_implied_by_constraints(self):
        m = _model("le", y_bounds=(None, None))
        m.b1 = pyo.Constraint(expr=m.y >= 0)
        m.b2 = pyo.Constraint(expr=m.y <= 5)
        self.assertEqual(unbounded_variables(m, do_fbbt=False), ["y"])
        self.assertEqual(unbounded_variables(m, do_fbbt=True), [])
        self.assertEqual((m.y.lb, m.y.ub), (0, 5))

    def test_fully_bounded_model_reports_nothing(self):
        self.assertEqual(unbounded_variables(_model("le")), [])


class TestGuards(unittest.TestCase):
    def test_clean_model_passes(self):
        check_model_is_certifiable(_model("le"))   # must not raise

    def test_integer_variable(self):
        m = _model("le")
        m.z = pyo.Var(bounds=(0, 10), domain=pyo.Integers)
        with self.assertRaisesRegex(CertificateError, "discrete"):
            check_model_is_certifiable(m)

    def test_binary_variable(self):
        m = _model("le")
        m.z = pyo.Var(domain=pyo.Binary)
        with self.assertRaisesRegex(CertificateError, "discrete"):
            check_model_is_certifiable(m)

    def test_fixed_discrete_variable_is_allowed(self):
        # Fixed means constant, so it carries no convexity claim.
        m = _model("le")
        m.z = pyo.Var(bounds=(0, 10), domain=pyo.Integers)
        m.z.fix(3)
        check_model_is_certifiable(m)   # must not raise

    def test_nonlinear_equality(self):
        m = _model("le")
        m.bad = pyo.Constraint(expr=m.x * m.y == 1)
        with self.assertRaisesRegex(CertificateError, "nonlinear equality"):
            check_model_is_certifiable(m)

    def test_nonlinear_inequality_is_allowed(self):
        # Convex inequalities are the point of the exercise; only equalities
        # have to be affine.
        m = _model("le")
        m.ok = pyo.Constraint(expr=m.x**2 + m.y**2 <= 100)
        check_model_is_certifiable(m)   # must not raise

    def test_maximize(self):
        m = _model("le")
        m.obj.sense = pyo.maximize
        with self.assertRaisesRegex(CertificateError, "minimize-only"):
            check_model_is_certifiable(m)

    def test_no_objective(self):
        m = _model("le")
        m.obj.deactivate()
        with self.assertRaisesRegex(CertificateError, "exactly one active Objective"):
            check_model_is_certifiable(m)

    def test_missing_dual_suffix(self):
        m = _model("le")
        m.del_component(m.dual)
        with self.assertRaisesRegex(CertificateError, "no `dual` Suffix"):
            certified_lower_bound(m)

    def test_missing_dual_for_a_constraint(self):
        m = _model("le")          # suffix present but never populated
        m.x.value, m.y.value = 1.0, 0.0
        with self.assertRaisesRegex(CertificateError, "no dual available"):
            certified_lower_bound(m)

    def test_unknown_sign_convention(self):
        m = _place(_model("le"), 1.0, 0.0, -4.0)
        with self.assertRaisesRegex(CertificateError, "unknown sign convention"):
            certified_lower_bound(m, sign_convention="no-such-solver")


@unittest.skipUnless(ipopt_available, "ipopt is not available")
class TestWithIpopt(unittest.TestCase):
    """The parts that can only be checked against the real solver: that ipopt's
    reported dual signs are what the table in dual_certificate.py assumes."""

    @staticmethod
    def _solve(m, max_iter=None):
        opt = pyo.SolverFactory("ipopt")
        if max_iter is not None:
            opt.options["max_iter"] = max_iter
        opt.solve(m)
        return m

    def test_sign_table_matches_ipopt(self):
        for kind, expected in IPOPT_DUAL_AT_OPTIMUM.items():
            with self.subTest(kind=kind):
                m = self._solve(_model(kind))
                self.assertAlmostEqual(pyo.value(m.dual[m.c]), expected, places=5)

    def test_tight_at_convergence(self):
        for kind in IPOPT_DUAL_AT_OPTIMUM:
            with self.subTest(kind=kind):
                q = certified_lower_bound(self._solve(_model(kind)), eps_rel=0.0)
                self.assertLessEqual(q, OPT + 1e-9)
                self.assertLess(OPT - q, 1e-6)

    def test_valid_under_truncated_solves(self):
        # Validity must hold at every truncation level; this is the property a
        # lost minus sign would break by producing a bound *above* the optimum.
        for max_iter in (1, 2, 3, 5, 8):
            with self.subTest(max_iter=max_iter):
                q = certified_lower_bound(
                    self._solve(_model("le"), max_iter=max_iter), eps_rel=0.0
                )
                if q is not None:
                    self.assertLessEqual(q, OPT + 1e-9)

    def test_converged_bound_is_at_least_as_tight_as_truncated(self):
        # Deliberately not asserting pairwise monotonicity across max_iter: the
        # iterate path is a solver detail and may differ by linear-solver build.
        converged = certified_lower_bound(self._solve(_model("le")), eps_rel=0.0)
        for max_iter in (1, 2, 3, 5):
            with self.subTest(max_iter=max_iter):
                q = certified_lower_bound(
                    self._solve(_model("le"), max_iter=max_iter), eps_rel=0.0
                )
                if q is not None:
                    self.assertLessEqual(q, converged + 1e-9)

    def test_solver_objective_value_can_be_an_invalid_bound(self):
        # The motivating failure: at a badly truncated solve the returned
        # objective value sits *above* the optimum, so "just use f(vhat)" is not
        # an outer bound at all, while the certificate stays valid.
        m = self._solve(_model("le"), max_iter=1)
        self.assertGreater(pyo.value(m.obj), OPT)
        q = certified_lower_bound(m, eps_rel=0.0)
        self.assertTrue(q is None or q <= OPT + 1e-9)


if __name__ == "__main__":
    unittest.main()
