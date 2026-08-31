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

import math
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

    def test_nan_dual_yields_no_bound(self):
        # A diverged solve can leave NaN in the `dual` suffix, and NaN
        # propagates silently through phi and the correction.  The contract is
        # None, the same word this module already uses for "no bound this time".
        m = _place(_model("le"), 1.0, 0.0, float("nan"))
        self.assertIsNone(certified_lower_bound(m, eps_rel=0.0))

    def test_nan_in_the_point_yields_no_bound(self):
        m = _place(_model("le"), 1.0, 0.0, -4.0)
        m.x.set_value(float("nan"), skip_validation=True)
        self.assertIsNone(certified_lower_bound(m, eps_rel=0.0))

    def test_infinite_duals_yield_no_bound_or_a_valid_one(self):
        # -inf clips to lam = +inf, and inf*0 is NaN; +inf clips to lam = 0,
        # which just drops the constraint and leaves a loose but valid bound.
        # Neither may return a number above the optimum.
        for d in (float("inf"), float("-inf")):
            with self.subTest(dual=d):
                q = certified_lower_bound(_place(_model("le"), 1.0, 0.0, d),
                                          eps_rel=0.0)
                self.assertTrue(q is None or q <= OPT)

    def test_a_non_finite_bound_never_reaches_the_caller(self):
        # The guard is on the result, so it catches every route to a non-finite
        # qhat, not just the ones enumerated above.  +inf matters most: unlike
        # NaN it would compare as an *improvement* to a hub tracking the best
        # outer bound seen.
        for x, y, d in ((float("nan"), 0.0, -4.0),
                        (1.0, float("nan"), -4.0),
                        (1.0, 0.0, float("-inf")),
                        (float("inf"), 0.0, -4.0)):
            with self.subTest(x=x, y=y, dual=d):
                m = _model("le")
                m.x.set_value(x, skip_validation=True)
                m.y.set_value(y, skip_validation=True)
                m.dual[m.c] = d
                q = certified_lower_bound(m, eps_rel=0.0)
                self.assertTrue(q is None or math.isfinite(q))

    def test_fixed_variables_are_constants(self):
        # A fixed variable is not free in the box and must not contribute a
        # correction term.  Fix y at its optimal value; the bound is unchanged.
        m = _place(_model("le"), 1.0, 0.0, -4.0)
        m.y.fix(0.0)
        self.assertEqual(certified_lower_bound(m, eps_rel=0.0), OPT)


class TestCushion(unittest.TestCase):
    """eps_rel is subtracted, so it is the one argument that can turn a valid
    bound into an invalid one."""

    def test_negative_cushion_is_rejected(self):
        m = _model()
        _place(m, x=0.5, y=0.5, dual=1.0)
        with self.assertRaisesRegex(CertificateError, "non-negative"):
            certified_lower_bound(m, eps_rel=-1.0)

    def test_nan_cushion_is_rejected(self):
        m = _model()
        _place(m, x=0.5, y=0.5, dual=1.0)
        with self.assertRaisesRegex(CertificateError, "non-negative"):
            certified_lower_bound(m, eps_rel=float("nan"))

    def test_infinite_cushion_is_rejected(self):
        # inf passes `>= 0`, and subtracting it drives a finite qhat to -inf.
        # Unlike NaN, -inf is a number: it survives every downstream test and
        # is folded into Ebound's sum as though it were a bound.
        m = _model()
        _place(m, x=0.5, y=0.5, dual=1.0)
        with self.assertRaisesRegex(CertificateError, "finite"):
            certified_lower_bound(m, eps_rel=float("inf"))

    def test_a_huge_cushion_never_returns_minus_inf(self):
        # The finiteness screen must sit on the RETURNED value, after the
        # cushion is subtracted, not on the intermediate before it.
        m = _model()
        _place(m, x=0.5, y=0.5, dual=1.0)
        self.assertIsNone(certified_lower_bound(m, eps_rel=1e308))

    def test_zero_cushion_gives_the_theorem_quantity(self):
        m = _model()
        _place(m, x=0.5, y=0.5, dual=1.0)
        exact = certified_lower_bound(m, eps_rel=0.0)
        shaved = certified_lower_bound(m, eps_rel=1e-9)
        self.assertLess(shaved, exact)


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

    def test_nonlinear_le_inequality_is_allowed(self):
        # Convex inequalities are the point of the exercise. For a `<=` row the
        # canonical g is body - upper, so a convex body is what the theorem
        # wants -- and convexity is the caller's assertion, not checkable here.
        m = _model("le")
        m.ok = pyo.Constraint(expr=m.x**2 + m.y**2 <= 100)
        check_model_is_certifiable(m)   # must not raise

    def test_nonlinear_ge_inequality_is_allowed_but_needs_a_concave_body(self):
        # The sign trap. For a `>=` row the canonical g is lower - body, so the
        # BODY must be concave, not convex. That is still the caller's
        # assertion, so the guard lets it through -- but a convex body here
        # (x**2 >= 1, which looks perfectly ordinary) makes phi non-convex and
        # the certificate can then exceed the true optimum. The rule is stated
        # in the module docstring and in spokes.rst; this test exists to pin
        # down that the guard does NOT claim to catch it.
        m = _model("le")
        m.ok = pyo.Constraint(expr=m.x**2 >= 1)
        check_model_is_certifiable(m)   # must not raise

    def test_nonlinear_ranged_constraint_is_rejected(self):
        # A two-sided row splits into both g = body - upper and
        # g = lower - body, so the body would have to be convex AND concave --
        # affine. Unlike one-sided convexity, that IS decidable, so it is a
        # hard error rather than an assertion.
        m = _model("le")
        m.bad = pyo.Constraint(expr=pyo.inequality(1, m.x**2, 100))
        with self.assertRaisesRegex(CertificateError, "two-sided"):
            check_model_is_certifiable(m)

    def test_affine_ranged_constraint_is_allowed(self):
        m = _model("le")
        m.ok = pyo.Constraint(expr=pyo.inequality(1, 2 * m.x + m.y, 100))
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

    def test_missing_dual_for_a_constraint_is_reported_not_raised(self):
        """A dual-less row is lam = 0, which weak duality admits.

        Raising here killed the spoke for a whole run over one structurally
        dual-less constraint. Dropping the row costs tightness and nothing
        else, so the bound must still come back and must still be a bound.
        """
        m = _model("le")          # suffix present but never populated
        m.x.value, m.y.value = 1.0, 0.0
        skipped = []
        q = certified_lower_bound(m, eps_rel=0.0, missing_duals=skipped)
        self.assertEqual(skipped, ["c"])
        # A bound is a bound: it must not exceed the running example's
        # optimum, which the module declares as OPT (8.0, at x=1, y=0).
        self.assertLessEqual(q, OPT + 1e-9)
        # And dropping the row really is only a loss of tightness: taking the
        # same point WITH its dual gives a bound at least as good.
        tight = certified_lower_bound(
            _place(_model("le"), 1.0, 0.0, -4.0), eps_rel=0.0)
        self.assertLessEqual(q, tight + 1e-9)

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


# ---------------------------------------------------------------------------
# Ill-conditioning
#
# The question this answers is whether a solve that goes badly because the
# problem is badly conditioned can produce an *invalid* bound rather than
# merely a loose one.  It cannot, and the reason is structural: the certificate
# is an evaluation, not a solve.  A condition number measures how much a linear
# solve amplifies error, and `certified_lower_bound` never solves anything -- it
# evaluates phi and its gradient at whatever point ipopt stopped at.  The
# tangent inequality it rests on is pointwise and exact for any convex phi at
# any vhat, with no error constant, so kappa has nowhere to enter.  What
# conditioning does change is how far vhat lands from optimal and how large the
# gradient there is, and both of those show up in the correction term as
# looseness.
#
# The test problem is a Hilbert-matrix QP, the standard ill-conditioned test
# case:  min 1/2 x'Hx - 1'x  s.t.  sum(x) <= 5,  x in [-10,10]^n,  with
# H_ij = 1/(i+j+1).  It is convex (H is positive definite) so the hypotheses
# hold exactly, while cond(H) is about 1.6e13 at n=10 and 3.5e17 at n=16 --
# past the point where a double-precision solve can converge properly.
# ---------------------------------------------------------------------------

HILBERT_BOX = 10.0
HILBERT_RHS = 5.0


def _hilbert_qp(n=10, rowscale=1.0, offset=0.0):
    """min 1/2 x'Hx - 1'x + offset  s.t.  rowscale*sum(x) <= rowscale*5.

    `rowscale` leaves the feasible set alone and rescales the row, which
    rescales the multiplier ipopt reports by the same factor -- 1e-8 drives it
    to ~1e8 and makes the `lam^T g` term in phi dominate an objective of order
    one, which is the cancellation regime.  `offset` does the same to the `f`
    term.  Both are ways of making the arithmetic hard without touching the
    mathematics.
    """
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, bounds=(-HILBERT_BOX, HILBERT_BOX), initialize=0.0)
    m.obj = pyo.Objective(
        expr=0.5 * sum(m.x[i] * m.x[j] / (i + j + 1) for i in range(n) for j in range(n))
        - sum(m.x[i] for i in range(n))
        + offset
    )
    m.c = pyo.Constraint(
        expr=rowscale * sum(m.x[i] for i in range(n)) <= rowscale * HILBERT_RHS
    )
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def _hilbert_objective(xs, offset):
    n = len(xs)
    return (
        0.5 * sum(xs[i] * xs[j] / (i + j + 1) for i in range(n) for j in range(n))
        - sum(xs)
        + offset
    )


def _rigorous_upper_bound(xs, offset):
    """f at a point PROVABLY in the feasible set, hence >= the true optimum.

    Comparing the certificate against `f(vhat)` from a converged solve is the
    trap this exists to avoid: ipopt's vhat carries up to `constr_viol_tol` of
    infeasibility, so `f(vhat)` can sit *below* the optimum and manufacture an
    apparent violation where there is none.  Here a uniform downshift is
    bisected until the shifted, clipped point satisfies the constraint exactly
    as evaluated -- clipping keeps it in the box and the shifted sum is
    monotone in the shift, so the result is feasible by construction, whatever
    the solver did.
    """
    def clip(t):
        return [min(HILBERT_BOX, max(-HILBERT_BOX, v - t)) for v in xs]

    lo, hi = 0.0, 2.0 * HILBERT_BOX + abs(HILBERT_RHS) + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sum(clip(mid)) > HILBERT_RHS:
            lo = mid
        else:
            hi = mid
    feasible = clip(hi)
    assert sum(feasible) <= HILBERT_RHS
    assert max(abs(v) for v in feasible) <= HILBERT_BOX
    return _hilbert_objective(feasible, offset)


@unittest.skipUnless(ipopt_available, "ipopt is not available")
class TestIllConditioning(unittest.TestCase):
    """Validity must survive a solve that goes badly for numerical reasons."""

    # (label, n, rowscale, offset)
    VARIANTS = [
        ("cond 1.6e13", 10, 1.0, 0.0),
        ("tiny multiplier", 10, 1e8, 0.0),
        ("multiplier ~1e8", 10, 1e-8, 0.0),
        ("cond 3.5e17", 16, 1.0, 0.0),
        ("offset 1e12", 10, 1.0, 1e12),
    ]
    TRUNCATIONS = (1, 2, 3, 5, 10, None)

    @staticmethod
    def _solve(m, max_iter=None):
        opt = pyo.SolverFactory("ipopt")
        if max_iter is not None:
            opt.options["max_iter"] = max_iter
        opt.solve(m)
        return m

    def _reference(self, n, rowscale, offset):
        m = self._solve(_hilbert_qp(n, rowscale, offset))
        return _rigorous_upper_bound([pyo.value(m.x[i]) for i in range(n)], offset)

    def test_bound_never_exceeds_a_feasible_objective(self):
        for label, n, rowscale, offset in self.VARIANTS:
            upper = self._reference(n, rowscale, offset)
            # Both sides are computed in double precision, so the comparison
            # gets a few-thousand-ulp relative allowance.  That is the
            # arithmetic caveat dual_certificate.py documents, not slack in the
            # theorem: at an offset of 1e12 a double resolves about 1e-4
            # absolute, and an exact comparison there would be testing the
            # floating-point unit rather than the certificate.
            slop = 1e-12 * (1.0 + abs(upper))
            for max_iter in self.TRUNCATIONS:
                with self.subTest(variant=label, max_iter=max_iter):
                    m = self._solve(_hilbert_qp(n, rowscale, offset), max_iter)
                    q = certified_lower_bound(m, eps_rel=0.0)
                    if q is None:
                        continue
                    self.assertTrue(math.isfinite(q))
                    self.assertLessEqual(q, upper + slop)

    def test_severe_row_scaling_really_does_produce_a_huge_multiplier(self):
        # Without this the cancellation variant above could silently stop
        # exercising cancellation -- a passing test that tests nothing.
        m = self._solve(_hilbert_qp(10, rowscale=1e-8))
        self.assertGreater(abs(pyo.value(m.dual[m.c])), 1e6)

    def test_conditioning_costs_tightness_not_validity(self):
        # The well-scaled case closes to the last few digits; the badly scaled
        # one does not close at all.  Both stay valid, which is the whole
        # claim: conditioning moves the bound down, never up.
        good_n, good_scale = 10, 1.0
        good_upper = self._reference(good_n, good_scale, 0.0)
        good = certified_lower_bound(
            self._solve(_hilbert_qp(good_n, good_scale)), eps_rel=0.0)
        self.assertIsNotNone(good)
        self.assertLessEqual(good, good_upper + 1e-12 * (1.0 + abs(good_upper)))
        self.assertLess(good_upper - good, 1e-6 * (1.0 + abs(good_upper)))

        bad_upper = self._reference(10, 1e-8, 0.0)
        bad = certified_lower_bound(
            self._solve(_hilbert_qp(10, rowscale=1e-8)), eps_rel=0.0)
        self.assertIsNotNone(bad)
        self.assertLessEqual(bad, bad_upper + 1e-12 * (1.0 + abs(bad_upper)))


if __name__ == "__main__":
    unittest.main()
