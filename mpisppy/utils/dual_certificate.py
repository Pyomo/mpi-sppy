###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# A *certified* lower bound for a convex NLP, computed from an already-solved
# Pyomo model and its constraint duals.
#
# The problem is
#
#     L = min  f(v)   s.t.  g(v) <= 0,  h(v) == 0,  v in B = [lo, hi]
#
# and the point of this module is to return a number that is <= L by a theorem,
# not by an assertion that the solver converged.  A solver's returned objective
# value is the value at a point, hence >= L: an inner bound, the wrong
# direction.  Lagrangian weak duality gives the right direction and needs no
# convergence assumption at all:
#
#     phi(v)     = f(v) + lam^T g(v) + mu^T h(v)
#     q(lam, mu) = inf_{v in B} phi(v)   <=   L    for ANY lam >= 0, ANY mu
#
# The remaining trap is that q is an *infimum*.  Solving for it with an NLP
# solver returns a point, and the value there is >= q -- the wrong direction
# again.  What closes it is convexity: phi is convex on B, so its tangent at any
# vhat in B lies below it, and minimizing that tangent over a box is separable
# and closed-form:
#
#     q(lam, mu) >= phi(vhat) + sum_i  min_{v_i in [lo_i, hi_i]} d_i phi * (v_i - vhat_i)
#
# One gradient evaluation and a loop over variables.  No second solve, no
# tolerance argument, and vhat need not even be feasible.
#
# At an exact KKT point the correction term is identically zero (stationarity
# makes grad phi = z_L - z_U, which is zero on interior components and points
# the wrong way to help on active ones), so the bound collapses to f(vhat) and
# strong duality is recovered exactly.  The correction is precisely the price of
# inexactness, and it measures itself.
#
# CONVEXITY IS LOAD-BEARING.  If f or any component of g is non-convex over B,
# the tangent is not an underestimator and the returned number is simply wrong.
#
# Note carefully that the requirement is on the CANONICAL g, not on the
# constraint body as written, and the two differ by a sign on a `>=` row:
#
#     body <= upper   ->  g = body - upper    needs the body CONVEX
#     body >= lower   ->  g = lower - body    needs the body CONCAVE
#     lo <= body <= up->  both of the above   needs the body AFFINE
#     body == rhs     ->  h = body - rhs      needs the body AFFINE
#
# So `x**2 <= 4` is fine and `x**2 >= 1` is not, even though both are written
# with a convex body.  This is easy to get backwards: `x**2 >= 1` looks like an
# ordinary convex constraint and its feasible set is not convex at all.
#
# check_model_is_certifiable() rejects what is mechanically checkable -- the
# affine cases, since polynomial degree is decidable -- but convexity of a
# general nonlinear body is not, so on one-sided nonlinear rows it is the
# caller's assertion.
#
# By contrast a wrong multiplier -- bad sign convention, stale dual, clipped
# value -- can only make the bound loose, never wrong, because weak duality
# holds for any lam >= 0 and any mu.  In particular an inexact solve costs
# nothing but tightness: for a convex model there are no non-global local
# minima, so "the solver returned a sub-optimal answer" can only mean it stopped
# short of converging, and the theorem above never asked it to converge.
#
# THE ARITHMETIC IS A DIFFERENT MATTER, and it is the one place where an
# ill-conditioned model can hurt.  phi(vhat) is evaluated in double precision as
# f + sum lam_i g_i + sum mu_j h_j, whose rounding error is on the order of
# u * (|f| + sum |lam_i g_i| + sum |mu_j h_j|) -- driven by the size of the
# TERMS.  The eps_rel cushion below is proportional to |qhat| instead, i.e. to
# the size of the RESULT.  On a well-scaled model those track each other; on one
# where cancellation is severe -- multipliers of 1e8 against an objective of
# order one, say -- they do not, and the cushion can be the smaller of the two.
# Hence the honest description of eps_rel as hygiene rather than proof: a user
# who needs margin on a badly conditioned model should raise it.
#
# The correction term carries the same kind of error, and it is the half that
# can push the answer UP: the correction is -sum_i |d_i phi| * d_i, so a
# gradient component computed slightly small in magnitude makes it slightly less
# negative.  The clipping above gives no protection here -- that argument is
# about the multipliers, and a perturbed gradient is not one-directional the way
# a perturbed lam is.
#
# None of this is amplified by conditioning as such.  A condition number
# measures how much a LINEAR SOLVE magnifies error, and nothing here solves
# anything: phi and its gradient are evaluated at vhat, each component is
# compared with zero, an endpoint is chosen, and the results are summed.  The
# tangent inequality itself holds exactly at every vhat with no error constant.
# So an ill-conditioned subproblem gives a loose bound -- a worse vhat and a
# bigger gradient both enlarge the correction -- and not a wrong one.
#
# The caller is responsible for handing over a model that really is the
# relaxation it wants bounded.  In particular a Progressive Hedging proximal
# term, or nonanticipative variables fixed by an extension, make the model
# something other than the Lagrangian relaxation, and this module cannot detect
# either.

import math

import pyomo.environ as pyo
from pyomo.core.expr.calculus.derivatives import differentiate, Modes
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.fbbt.fbbt import fbbt

__all__ = [
    "CertificateError",
    "check_model_is_certifiable",
    "unbounded_variables",
    "certified_lower_bound",
]


class CertificateError(RuntimeError):
    """The model violates an assumption the certified bound depends on."""


# How a solver's Pyomo `dual` suffix maps onto canonical multipliers for
# g(v) <= 0 and h(v) == 0.  These were measured against analytically known
# multipliers, not assumed:  min (x-3)^2 with the constraint active has true
# multiplier 4, and ipopt reports d = -4 for `body <= upper`, d = +4 for
# `body >= lower`, and d = -4 for `body == rhs`.
_SIGN_CONVENTIONS = {
    # g = body - upper  (from `body <= upper`)
    "ipopt": {
        "lam_upper": lambda d: max(-d, 0.0),
        # g = lower - body  (from `body >= lower`)
        "lam_lower": lambda d: max(d, 0.0),
        # h = body - rhs  (from `body == rhs`)
        "mu": lambda d: -d,
    },
}


def _active_objective(model):
    objs = list(
        model.component_data_objects(pyo.Objective, active=True, descend_into=True)
    )
    if len(objs) != 1:
        raise CertificateError(
            f"expected exactly one active Objective, found {len(objs)}"
        )
    return objs[0]


def check_model_is_certifiable(model):
    """Raise CertificateError if `model` violates an assumption the
    certificate depends on.

    Checks only what is mechanically checkable.  Convexity of the objective and
    of one-sided nonlinear inequality bodies is the caller's assertion and is
    *not* checked -- including the direction of it: a `<=` row needs a convex
    body and a `>=` row needs a CONCAVE one, because the canonical g negates
    the body on a `>=`. See the module docstring.
    """
    obj = _active_objective(model)
    if obj.sense != pyo.minimize:
        raise CertificateError(
            "certified_lower_bound is minimize-only; maximization is handled by "
            "mirroring the model before calling here"
        )

    discrete = [
        v.name
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
        if not v.fixed and not v.is_continuous()
    ]
    if discrete:
        raise CertificateError(
            "a convexity claim is definitionally false with discrete variables; "
            f"these are not continuous: {', '.join(sorted(discrete)[:10])}"
            + (" ..." if len(discrete) > 10 else "")
        )

    # Both cases below need the body affine, and polynomial degree decides that,
    # so these are real checks rather than assertions. A one-sided nonlinear row
    # is left to the caller: whether its body is convex (for <=) or concave (for
    # >=) is not decidable here.
    nonlinear_eq = []
    nonlinear_ranged = []
    for con in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        two_sided = (not con.equality) and con.has_lb() and con.has_ub()
        if not (con.equality or two_sided):
            continue
        degree = con.body.polynomial_degree()
        if degree is None or degree > 1:
            (nonlinear_eq if con.equality else nonlinear_ranged).append(con.name)
    if nonlinear_eq:
        raise CertificateError(
            "a nonlinear equality makes mu^T h non-convex for one sign of mu, "
            "which breaks the underestimator; offending constraints: "
            f"{', '.join(sorted(nonlinear_eq)[:10])}"
            + (" ..." if len(nonlinear_eq) > 10 else "")
        )
    if nonlinear_ranged:
        raise CertificateError(
            "a two-sided constraint splits into g = body - upper AND "
            "g = lower - body, so its body would have to be both convex and "
            "concave -- i.e. affine -- for the underestimator to hold on both "
            "rows; offending constraints: "
            f"{', '.join(sorted(nonlinear_ranged)[:10])}"
            + (" ..." if len(nonlinear_ranged) > 10 else "")
        )


def unbounded_variables(model, do_fbbt=True):
    """Names of variables that still lack a finite bound, after optionally
    tightening with feasibility-based bounds tightening.

    `fbbt` mutates `model` in place, tightening variable bounds using the
    constraints.  That is sound here and makes the certificate tighter: it
    shrinks the box `B` without removing any feasible point.

    A non-empty return does not mean no bound is possible -- an unbounded
    variable only defeats the certificate if its gradient component in phi is
    nonzero -- but it does mean `certified_lower_bound` may return None.
    """
    if do_fbbt:
        fbbt(model)
    return sorted(
        v.name
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
        if not v.fixed and (v.lb is None or v.ub is None)
    )


def _lagrangian_expression(model, conv):
    """phi(v) = f(v) + lam^T g(v) + mu^T h(v), with the multipliers read off the
    model's `dual` suffix and canonicalized.

    Returns (expression, names_of_constraints_taken_with_multiplier_zero).  A
    constraint with no imported dual lands in the second element rather than
    raising: any lam >= 0 is admissible, so dropping one only costs tightness.
    """
    if not hasattr(model, "dual"):
        raise CertificateError(
            "model has no `dual` Suffix; attach "
            "pyo.Suffix(direction=pyo.Suffix.IMPORT) before solving"
        )
    dual = model.dual

    missing = []
    terms = [_active_objective(model).expr]
    for con in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        if con not in dual:
            # Weak duality holds for ANY lam >= 0 and any mu, so a constraint
            # whose dual the solve did not import is simply one taken with
            # multiplier zero: it drops out of phi and the bound gets looser,
            # never wrong. Raising here instead used to kill the bound for the
            # whole run over one structurally dual-less row.
            missing.append(con.name)
            continue
        d = float(pyo.value(dual[con]))
        body = con.body
        if con.equality:
            mu = conv["mu"](d)
            if mu != 0.0:
                terms.append(mu * (body - pyo.value(con.upper)))
            continue
        # A ranged constraint carries one dual for two rows; splitting it and
        # applying both rules lands the magnitude on the active side and zero
        # on the other.
        if con.has_ub():
            lam = conv["lam_upper"](d)
            if lam != 0.0:
                terms.append(lam * (body - pyo.value(con.upper)))
        if con.has_lb():
            lam = conv["lam_lower"](d)
            if lam != 0.0:
                terms.append(lam * (pyo.value(con.lower) - body))
    return sum(terms), missing


def certified_lower_bound(model, sign_convention="ipopt", eps_rel=1e-9,
                          missing_duals=None):
    """A number guaranteed <= the model's optimal value, or None.

    `model` must already be solved, with its `dual` Suffix populated and its
    variables holding the returned point.  Neither optimality nor feasibility of
    that point is required -- a truncated solve yields a valid but loose bound.

    Returns None when the box minimization is unbounded below, which happens
    when a variable with an infinite bound has a nonzero gradient component in
    phi, and also when the arithmetic produces a non-finite result -- a NaN or
    an infinity arriving from a diverged solve.  None means "no bound this
    time", never "-inf".

    `eps_rel` shaves a relative cushion off the result.  At the default 1e-9
    this is last-bit hygiene, not a proof-carrying margin; pass 0.0 to get the
    theorem's quantity exactly.  It must be non-negative -- a negative cushion
    would raise the result above the theorem's quantity, and CertificateError
    is raised rather than returning something that is not a bound.

    Pass a list as `missing_duals` to learn which constraints were taken with
    multiplier zero because the solve imported no dual for them.  That costs
    tightness and nothing else, so it is reported rather than raised.
    """
    if not (eps_rel >= 0.0):
        # The cushion is subtracted at the end.  A negative one would RAISE the
        # result above the theorem's quantity, so what came back would not be
        # an outer bound -- the single way this function can return a wrong
        # answer rather than a loose one.  NaN fails this test too, and must.
        raise CertificateError(
            f"eps_rel must be non-negative, got {eps_rel!r}; the cushion is "
            "subtracted from the bound, so a negative value would raise it "
            "above the certified quantity and the result would not be a bound"
        )

    try:
        conv = _SIGN_CONVENTIONS[sign_convention]
    except KeyError:
        raise CertificateError(
            f"unknown sign convention {sign_convention!r}; known: "
            f"{sorted(_SIGN_CONVENTIONS)}"
        )

    phi, skipped = _lagrangian_expression(model, conv)
    if missing_duals is not None:
        missing_duals.extend(skipped)
    vlist = list(identify_variables(phi, include_fixed=False))

    correction = 0.0
    if vlist:
        grad = differentiate(phi, wrt_list=vlist, mode=Modes.reverse_numeric)
        for v, g in zip(vlist, grad):
            g = float(g)
            if g == 0.0:
                continue
            vhat = pyo.value(v)
            # min over [lo, hi] of a linear term goes to whichever end the
            # gradient points away from.
            bound = v.lb if g > 0.0 else v.ub
            if bound is None:
                return None
            correction += g * (bound - vhat)

    qhat = pyo.value(phi) + correction

    # A solve that diverged or failed numerically can leave NaN in the point or
    # in the duals, and NaN propagates silently through everything above.  It
    # must not leave this function.  Returning it would be safe today only by
    # accident -- NaN loses every comparison, so the hub's `new > old` update
    # test happens to reject it -- and +inf, which an infinite multiplier can
    # also produce, would be an outright invalid bound if anything ever did
    # accept it.  Both are "no bound", which this module already has a word for.
    if not math.isfinite(qhat):
        return None

    return qhat - eps_rel * (1.0 + abs(qhat))
