###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""GradRho gradient extraction must aggregate across all per-sub-scenario
nonants in a proper bundle, not just the bundle's representative ref Var.
See Pyomo/mpi-sppy issue #673.

These are pure unit tests: they build tiny scenarios and a create_EF bundle
in-process, then drive ``GradRho._get_grad_exprs`` and ``_eval_grad_exprs``
directly with a stub PH-like object. No solver and no MPI required.
"""

import unittest

import numpy as np
import pyomo.environ as pyo

import mpisppy.MPI as MPI
import mpisppy.utils.sputils as sputils
from mpisppy.extensions.grad_rho import GradRho


def _build_linear_scen(name, c_coefs, prob):
    """Two-stage scenario with first-stage Var x[0..len-1] and linear
    objective sum_i c_coefs[i] * x[i]. No second stage (not needed)."""
    m = pyo.ConcreteModel(name=name)
    m.x = pyo.Var(range(len(c_coefs)), bounds=(0, None), initialize=0.0)
    m.obj = pyo.Objective(
        expr=sum(c_coefs[i] * m.x[i] for i in range(len(c_coefs)))
    )
    sputils.attach_root_node(m, 0, [m.x[i] for i in range(len(c_coefs))])
    m._mpisppy_probability = prob
    return m


def _build_quadratic_scen(name, q_coefs, prob):
    """Same shape as _build_linear_scen but with a quadratic objective so
    the gradient depends on the linearization point."""
    m = pyo.ConcreteModel(name=name)
    m.x = pyo.Var(range(len(q_coefs)), bounds=(None, None), initialize=0.0)
    m.obj = pyo.Objective(
        expr=sum(q_coefs[i] * m.x[i] ** 2 for i in range(len(q_coefs)))
    )
    sputils.attach_root_node(m, 0, [m.x[i] for i in range(len(q_coefs))])
    m._mpisppy_probability = prob
    return m


def _attach_spbase_metadata(scen):
    """Populate the ``_mpisppy_data`` Block with the fields SPBase normally
    sets (nlens, nonant_indices, varid_to_nonant_index). create_EF already
    creates the Block on the bundle; for bare scenarios we create it."""
    if not hasattr(scen, "_mpisppy_data"):
        scen._mpisppy_data = pyo.Block(
            name="For non-Pyomo mpi-sppy data"
        )
    scen._mpisppy_data.nlens = {
        node.name: len(node.nonant_vardata_list)
        for node in scen._mpisppy_node_list
    }
    nonant_indices = {}
    for node in scen._mpisppy_node_list:
        for i, v in enumerate(node.nonant_vardata_list):
            nonant_indices[(node.name, i)] = v
    scen._mpisppy_data.nonant_indices = nonant_indices
    scen._mpisppy_data.varid_to_nonant_index = {
        id(v): k for k, v in nonant_indices.items()
    }


def _build_bundle(scen_creator, scen_names):
    """Build a proper-bundle-shaped EF model: create_EF, then attach the
    ROOT node from non-surrogate ref_vars (mirroring proper_bundler), then
    attach SPBase-style metadata on the bundle's _mpisppy_data Block."""
    bundle = sputils.create_EF(scen_names, scen_creator, EF_name="bundle")
    nonantlist = [
        v for idx, v in bundle.ref_vars.items()
        if idx[0] == "ROOT" and idx not in bundle.ref_surrogate_vars
    ]
    surrogates = [
        v for idx, v in bundle.ref_surrogate_vars.items()
        if idx[0] == "ROOT"
    ]
    sputils.attach_root_node(bundle, 0, nonantlist, None, surrogates)
    _attach_spbase_metadata(bundle)
    return bundle


class _FakePH:
    """Minimum surface area GradRho._get_grad_exprs touches on self.opt."""

    def __init__(self, local_scenarios):
        self.local_scenarios = local_scenarios


class _FakeBuf:
    """Stand-in for the BEST_XHAT receive buffer; supplies value_array()
    so the NaN-gating in _eval_grad_exprs evaluates as 'best xhat known'."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def value_array(self):
        return self._arr


def _make_grad_rho(local_scenarios, *, eval_at_xhat=False, xhat_buf=None):
    """Construct a GradRho-like object without invoking GradRho.__init__
    (which requires PH options/cfg). _get_grad_exprs and _eval_grad_exprs
    only touch a handful of attributes; set them by hand."""
    g = GradRho.__new__(GradRho)
    g.opt = _FakePH(local_scenarios)
    g.eval_at_xhat = eval_at_xhat
    if xhat_buf is not None:
        g.best_xhat_buf = xhat_buf
    return g


class TestGradRhoBundles(unittest.TestCase):
    """Equivalence between bundled and non-bundled gradient extraction."""

    def setUp(self):
        # 3 scenarios x 3 first-stage vars, distinct linear cost coeffs.
        self.scen_names = ["scen0", "scen1", "scen2"]
        self.c = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        self.probs = [1.0 / 3, 1.0 / 3, 1.0 / 3]

    def _build_unbundled(self):
        scens = {}
        for name, c, p in zip(self.scen_names, self.c, self.probs):
            s = _build_linear_scen(name, c, p)
            _attach_spbase_metadata(s)
            scens[name] = s
        return scens

    def _build_bundled(self):
        c_by_name = dict(zip(self.scen_names, self.c))
        prob_by_name = dict(zip(self.scen_names, self.probs))

        def _scen_creator(sname, **kwargs):
            return _build_linear_scen(
                sname, c_by_name[sname], prob_by_name[sname]
            )

        return {"bundle": _build_bundle(_scen_creator, self.scen_names)}

    def test_unbundled_gradient_is_per_scenario_cost(self):
        # Sanity floor: with a linear objective the gradient at every point
        # equals the cost coefficient. Establishes the baseline that the
        # bundle test compares against.
        scens = self._build_unbundled()
        g = _make_grad_rho(scens)
        g._get_grad_exprs()
        for sname, s in scens.items():
            i = self.scen_names.index(sname)
            for k in range(len(self.c[i])):
                got = pyo.value(g.grad_exprs[s][("ROOT", k)])
                self.assertAlmostEqual(
                    got, self.c[i][k], places=10,
                    msg=f"unbundled {sname=} k={k}: {got=} != {self.c[i][k]=}",
                )

    def test_bundle_gradient_is_conditional_prob_weighted_sum(self):
        # The bundle objective from create_EF is
        #     EF_obj = (sum_s prob_s * obj_s) / EF_prob
        # so the bundled gradient at position (ROOT, k) must be
        #     sum_s (prob_s / EF_prob) * c[s][k]
        # i.e. the conditional-probability-weighted average of the
        # per-sub-scenario cost coefficients. Before the fix for #673,
        # differentiating wrt the ref Var only captured one sub-scenario
        # (the first one in scen_names), so the bundled gradient would
        # equal c[0][k] rather than this weighted sum.
        bundled = self._build_bundled()
        g = _make_grad_rho(bundled)
        g._get_grad_exprs()
        bundle = bundled["bundle"]
        bundle_prob = bundle._mpisppy_probability
        n_vars = len(self.c[0])
        for k in range(n_vars):
            expected = sum(
                (self.probs[s] / bundle_prob) * self.c[s][k]
                for s in range(len(self.c))
            )
            got = pyo.value(g.grad_exprs[bundle][("ROOT", k)])
            self.assertAlmostEqual(
                got, expected, places=10,
                msg=f"bundle k={k}: {got=} != weighted-sum {expected=}",
            )
            # Also confirm we are not accidentally returning the lone first
            # sub-scenario's coefficient (the pre-fix behavior).
            if self.c[0][k] != expected:
                self.assertNotAlmostEqual(
                    got, self.c[0][k], places=6,
                    msg=f"bundle k={k} matches representative-only "
                        f"value {self.c[0][k]}; aggregation regressed",
                )

    def test_bundle_eval_propagates_xhat_to_subscenario_vars(self):
        # With a quadratic objective the gradient depends on the
        # linearization point, so this test fails if _eval_grad_exprs sets
        # only the bundle ref Var and leaves the per-sub-scenario nonant
        # Vars (which the cached gradient expressions actually reference)
        # at their previous values.
        scen_names = ["A", "B"]
        q_by_name = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
        prob_by_name = {"A": 0.5, "B": 0.5}

        def _scen_creator(sname, **kwargs):
            return _build_quadratic_scen(
                sname, q_by_name[sname], prob_by_name[sname]
            )

        bundle = _build_bundle(_scen_creator, scen_names)
        xhat = np.array([5.0, 7.0])
        g = _make_grad_rho(
            {"bundle": bundle},
            eval_at_xhat=True,
            xhat_buf=_FakeBuf(xhat),
        )
        g._get_grad_exprs()

        # Force per-sub-scenario Vars to a stale value first; if the bundle
        # branch of _eval_grad_exprs forgot to override them, the gradient
        # would reflect the stale values rather than xhat.
        for scenario_name in bundle._ef_scenario_names:
            scen = bundle.component(scenario_name)
            for v in scen.x.values():
                v.value = -999.0

        grads = g._eval_grad_exprs(bundle, xhat)

        # bundle_obj = (0.5*sum_i qA[i]*x_Ai^2 + 0.5*sum_i qB[i]*x_Bi^2) / 1.0
        # d/dx_Ai (bundle_obj) = qA[i]*x_Ai
        # d/dx_Bi (bundle_obj) = qB[i]*x_Bi
        # aggregated grad at (ROOT, i) = qA[i]*x_Ai + qB[i]*x_Bi
        # at x_Ai = x_Bi = xhat[i]: = (qA[i] + qB[i]) * xhat[i]
        for i in range(2):
            expected = (q_by_name["A"][i] + q_by_name["B"][i]) * xhat[i]
            self.assertAlmostEqual(
                grads[("ROOT", i)], expected, places=10,
                msg=f"bundle eval at xhat[i={i}]: "
                    f"{grads[('ROOT', i)]=} != {expected=}",
            )


class _FakeParam:
    """Minimal stand-in for a Pyomo Param data object: just an ``_value``
    attribute that grad-rho reads (xbars) and assigns (rho)."""
    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = float(value)


class _FakeIndexedParam:
    """Dict-backed stand-in for an indexed Pyomo Param. Supports the two
    operations grad-rho uses: ``params[ndn_i]`` returning a _FakeParam, and
    ``params.items()`` for iterate-and-assign over the (ndn_i, param) pairs."""

    def __init__(self, values):
        self._data = {k: _FakeParam(v) for k, v in values.items()}

    def __getitem__(self, k):
        return self._data[k]

    def items(self):
        return self._data.items()


def _attach_phlike_state(scen, *, prob, root_xbar, root_var_value):
    """Layer on top of _attach_spbase_metadata: add the SPBase-side state
    grad-rho's ``_compute_and_update_rho`` reads (``prob_coeff``, ``xbars``,
    ``rho``) and prime every nonant Var to a deterministic value so the
    ``_scen_indep_denom`` computation is identical across the bundled and
    unbundled runs of the comparison test below.

    ``prob`` is the scenario probability (for a bundle, pass ``EF_prob``).
    """
    nlens = scen._mpisppy_data.nlens
    scen._mpisppy_data.prob_coeff = {
        ndn: np.full(n, prob) for ndn, n in nlens.items()
    }
    # Bundles arrive from create_EF with _mpisppy_model already set as a
    # Pyomo Block (can't be reassigned). Unbundled bare scenarios need one
    # created. In both cases, xbars / rho are then set as plain Python
    # attributes on the Block — non-Component RHS bypasses the Pyomo
    # add-as-component path.
    if not hasattr(scen, "_mpisppy_model"):
        scen._mpisppy_model = pyo.Block(
            name="For mpi-sppy Pyomo additions to the scenario model"
        )
    scen._mpisppy_model.xbars = _FakeIndexedParam(
        {(ndn, i): root_xbar
         for ndn, n in nlens.items() for i in range(n)}
    )
    scen._mpisppy_model.rho = _FakeIndexedParam(
        {(ndn, i): 0.0
         for ndn, n in nlens.items() for i in range(n)}
    )
    for node in scen._mpisppy_node_list:
        for v in node.nonant_vardata_list:
            v.value = root_var_value


class _FakePHwithComms:
    """``_compute_and_update_rho`` collectives sum across ranks; ``COMM_SELF``
    is the rank-1 communicator and gives global == local for a single-process
    test — the right semantics for a unit test of the formula."""

    def __init__(self, local_scenarios):
        self.local_scenarios = local_scenarios
        local_nodenames = set()
        for s in local_scenarios.values():
            for node in s._mpisppy_node_list:
                local_nodenames.add(node.name)
        self.comms = {ndn: MPI.COMM_SELF for ndn in local_nodenames}


def _make_full_grad_rho(local_scenarios, *, alpha=0.5, multiplier=1.0):
    """GradRho with the option surface ``_compute_and_update_rho`` reads.
    eval_at_xhat=False so the cached gradient expressions evaluate at the
    primed current Var values; indep_denom=True so the denominator is a
    deterministic shared-across-scenarios array (simplifies the assertion)."""
    g = GradRho.__new__(GradRho)
    g.opt = _FakePHwithComms(local_scenarios)
    g.alpha = alpha
    g.multiplier = multiplier
    g.denom_bound = 1.0 / 1e9  # very large bound -> never triggers
    g.eval_at_xhat = False
    g.indep_denom = True
    # value_array() is read to build the xhat arg even when eval_at_xhat is
    # False; an all-NaN buffer is fine because the xhat-priming branch is
    # gated and skipped here.
    nonants_total = sum(
        len(s._mpisppy_data.nonant_indices) for s in local_scenarios.values()
    )
    g.best_xhat_buf = _FakeBuf(np.full(nonants_total, np.nan))
    return g


class TestBundledRhoMatchesUnbundledMean(unittest.TestCase):
    """End-to-end grad-rho equivalence: same 3 scenarios, run unbundled and
    run as one bundle, compare the rho values written back to
    ``_mpisppy_model.rho``.

    With all-positive linear cost coefficients, ``EF_prob = 1`` (all 3
    scenarios in one bundle), ``alpha = 0.5`` (mean), ``indep_denom = True``,
    and every nonant Var primed to the same value across sub-scenarios:

        rho_unbundled[s][k] = |c_s[k]| / denom[k]  (= c_s[k] / denom[k])
        rho_bundle[k]       = |sum_s prob_s * c_s[k]| / denom[k]
                            = sum_s prob_s * c_s[k] / denom[k]  (signs match)
                            = sum_s prob_s * rho_unbundled[s][k]
                            = global_rho_mean[k]  (alpha=0.5 output)

    Pre-fix representative-only sourcing would have given
    ``rho_bundle[k] == |c_scen0[k]| / denom[k]`` instead of the mean — the
    discrimination guard at the end of the test rules out that regression.
    """

    def setUp(self):
        self.scen_names = ["scen0", "scen1", "scen2"]
        # Distinct, all-positive coefficients. Per-position the conditional-
        # prob-weighted mean (probs = 1/3 each) is [3.0, 4.0], well away
        # from scen0's [1.0, 2.0].
        self.c = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
        self.probs = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        # Var value = 2, xbar = 1 → unweighted_denom = 1 at every position;
        # weighted sum across scenarios with probs summing to 1 gives
        # denom = 1.0 in both runs. The expected mean rhos are therefore
        # exactly [3.0, 4.0].
        self.var_value = 2.0
        self.xbar_value = 1.0
        self.expected_denom = 1.0
        self.expected_mean_rhos = [3.0, 4.0]

    def _build_unbundled(self):
        scens = {}
        for name, c, p in zip(self.scen_names, self.c, self.probs):
            s = _build_linear_scen(name, c, p)
            _attach_spbase_metadata(s)
            _attach_phlike_state(
                s, prob=p,
                root_xbar=self.xbar_value,
                root_var_value=self.var_value,
            )
            scens[name] = s
        return scens

    def _build_bundled(self):
        c_by_name = dict(zip(self.scen_names, self.c))
        prob_by_name = dict(zip(self.scen_names, self.probs))

        def _scen_creator(sname, **kwargs):
            return _build_linear_scen(
                sname, c_by_name[sname], prob_by_name[sname]
            )

        bundle = _build_bundle(_scen_creator, self.scen_names)
        # The bundle is one "scenario" from PH's perspective; its probability
        # is EF_prob (here 1.0, the sum of sub-scenario probs).
        _attach_phlike_state(
            bundle, prob=bundle._mpisppy_probability,
            root_xbar=self.xbar_value,
            root_var_value=self.var_value,
        )
        return {"bundle": bundle}

    def test_bundle_rho_equals_unbundled_mean(self):
        # Run unbundled.
        unbundled = self._build_unbundled()
        g_u = _make_full_grad_rho(unbundled)
        g_u._get_grad_exprs()
        g_u._compute_and_update_rho()
        # _compute_and_update_rho writes the same global rho to every
        # scenario's rho param; pick any.
        scen0 = unbundled["scen0"]
        rho_u = {
            ndn_i: scen0._mpisppy_model.rho[ndn_i]._value
            for ndn_i in scen0._mpisppy_data.nonant_indices
        }

        # Run bundled.
        bundled = self._build_bundled()
        g_b = _make_full_grad_rho(bundled)
        g_b._get_grad_exprs()
        g_b._compute_and_update_rho()
        bundle = bundled["bundle"]
        rho_b = {
            ndn_i: bundle._mpisppy_model.rho[ndn_i]._value
            for ndn_i in bundle._mpisppy_data.nonant_indices
        }

        self.assertEqual(set(rho_u), set(rho_b))

        # Both runs match the analytical expected mean rhos.
        for k_idx, expected_rho in enumerate(self.expected_mean_rhos):
            self.assertAlmostEqual(
                rho_u[("ROOT", k_idx)], expected_rho, places=10,
                msg=f"unbundled rho at (ROOT, {k_idx}) = "
                    f"{rho_u[('ROOT', k_idx)]} != expected {expected_rho}",
            )
            self.assertAlmostEqual(
                rho_b[("ROOT", k_idx)], expected_rho, places=10,
                msg=f"bundle rho at (ROOT, {k_idx}) = "
                    f"{rho_b[('ROOT', k_idx)]} != expected {expected_rho}",
            )

        # And the user-visible property: bundle and unbundled produce the
        # same rho at each position.
        for ndn_i in rho_u:
            self.assertAlmostEqual(
                rho_b[ndn_i], rho_u[ndn_i], places=10,
                msg=f"bundle/unbundled rho mismatch at {ndn_i}: "
                    f"bundle={rho_b[ndn_i]} unbundled={rho_u[ndn_i]}",
            )

        # Discrimination guard against a regression to representative-only
        # sourcing: under that bug the bundle rho would equal scen0's
        # coefficient (|c[0][k]| / denom = c[0][k] / 1.0), not the mean.
        for k_idx in range(len(self.c[0])):
            scen0_only = abs(self.c[0][k_idx]) / self.expected_denom
            self.assertNotAlmostEqual(
                rho_b[("ROOT", k_idx)], scen0_only, places=3,
                msg=f"bundle rho at (ROOT, {k_idx}) matches scen0-only "
                    f"value {scen0_only}; aggregation regressed to "
                    f"representative-only sourcing",
            )


class TestBundleAndUnbundledDivergeUnderSignCancellation(unittest.TestCase):
    """Documents (and pins) that bundled and unbundled grad-rho do *not*
    generally agree — even with alpha=0.5 and identical primed state. The
    bundle objective is ``sum_s prob_s * obj_s`` so its gradient at a
    position aggregates per-sub-scenario contributions *before* the
    ``|.|`` in the rho formula:

        rho_bundle[k]   = |sum_s prob_s * c_s[k]| / denom[k]

    The unbundled mean does the absolute value *first*, then averages:

        rho_unbundled_mean[k] = sum_s prob_s * |c_s[k]| / denom[k]

    By the triangle inequality the bundle rho is bounded above by the
    unbundled mean; equality requires every per-sub-scenario contribution
    at that position to share a sign (the corner that
    ``TestBundledRhoMatchesUnbundledMean`` covers). When sub-scenarios
    pull in opposite directions, the bundle rho can be much smaller —
    arbitrarily so as the cancellation gets cleaner.

    This isn't a bug in the bundling — it's the actual algebraic
    consequence of bundling. The test exists so future readers don't
    expect the two paths to give similar rho numbers in general.
    """

    def test_opposing_signs_shrink_bundle_rho_relative_to_unbundled(self):
        scen_names = ["s0", "s1", "s2"]
        # Position 0: +5, -5, +1 — strong cancellation.
        # Position 1: +2, +2, +2 — no cancellation (sanity control).
        c = [
            [+5.0, +2.0],
            [-5.0, +2.0],
            [+1.0, +2.0],
        ]
        probs = [1.0 / 3] * 3
        var_value, xbar_value = 2.0, 1.0
        # abs(var_value - xbar_value) = 1 at every position; weighted prob
        # sum = 1 across both runs, so denom = 1.0 in both. The expected
        # rho numbers below assume that denom.

        unbundled = {}
        for n, ci, pi in zip(scen_names, c, probs):
            s = _build_linear_scen(n, ci, pi)
            _attach_spbase_metadata(s)
            _attach_phlike_state(
                s, prob=pi,
                root_xbar=xbar_value, root_var_value=var_value,
            )
            unbundled[n] = s
        g_u = _make_full_grad_rho(unbundled)
        g_u._get_grad_exprs()
        g_u._compute_and_update_rho()
        scen0 = unbundled["s0"]
        rho_u = {
            ndn_i: scen0._mpisppy_model.rho[ndn_i]._value
            for ndn_i in scen0._mpisppy_data.nonant_indices
        }

        c_by_name = dict(zip(scen_names, c))
        prob_by_name = dict(zip(scen_names, probs))

        def _scen_creator(sname, **kwargs):
            return _build_linear_scen(
                sname, c_by_name[sname], prob_by_name[sname]
            )

        bundle = _build_bundle(_scen_creator, scen_names)
        _attach_phlike_state(
            bundle, prob=bundle._mpisppy_probability,
            root_xbar=xbar_value, root_var_value=var_value,
        )
        bundled = {"bundle": bundle}
        g_b = _make_full_grad_rho(bundled)
        g_b._get_grad_exprs()
        g_b._compute_and_update_rho()
        rho_b = {
            ndn_i: bundle._mpisppy_model.rho[ndn_i]._value
            for ndn_i in bundle._mpisppy_data.nonant_indices
        }

        # Position 1 (no sign cancellation): bundle == unbundled mean = 2.0.
        # Pins the no-cancellation case still holds as a sanity check.
        self.assertAlmostEqual(rho_u[("ROOT", 1)], 2.0, places=10)
        self.assertAlmostEqual(rho_b[("ROOT", 1)], 2.0, places=10)

        # Position 0 (sign cancellation):
        #   unbundled mean = (|+5| + |-5| + |+1|) / 3 = 11/3 ≈ 3.667
        #   bundle         = |(+5 - 5 + 1) / 3|       =  1/3 ≈ 0.333
        # Pin both exactly; if either drifts the algebra has changed.
        self.assertAlmostEqual(rho_u[("ROOT", 0)], 11.0 / 3.0, places=10)
        self.assertAlmostEqual(rho_b[("ROOT", 0)], 1.0 / 3.0, places=10)

        # Triangle-inequality bound holds universally (sanity check that
        # would catch a coding error making bundle > unbundled mean).
        for ndn_i in rho_u:
            self.assertLessEqual(
                rho_b[ndn_i], rho_u[ndn_i] + 1e-10,
                msg=f"bundle rho exceeds unbundled mean at {ndn_i}: "
                    f"bundle={rho_b[ndn_i]} > unbundled-mean={rho_u[ndn_i]}",
            )
        # And the over-an-order-of-magnitude divergence is the property
        # being documented; if a future change brought the two paths into
        # closer agreement here without changing the algebra, that would
        # be a sign that the bundle aggregation got subtly broken (the
        # most likely regression is back to representative-only sourcing,
        # which on these coefficients gives |+5|/1 = 5.0 at position 0).
        ratio = rho_b[("ROOT", 0)] / rho_u[("ROOT", 0)]
        self.assertLess(
            ratio, 0.5,
            msg=f"sign-cancellation case expected to shrink bundle rho "
                f"well below unbundled mean; got ratio {ratio:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
