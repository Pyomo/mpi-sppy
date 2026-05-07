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


if __name__ == "__main__":
    unittest.main()
