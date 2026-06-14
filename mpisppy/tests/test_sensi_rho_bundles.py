###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""nonant_sensitivies must probe the consensus direction for proper bundles
(every per-sub-scenario nonant Var at each bundle position together) instead
of only the bundle's representative ref Var. See Pyomo/mpi-sppy issue #673.

The first test class exercises _bundle_consensus_groups directly and runs
without a solver; it is the primary correctness check for the fix and
catches a regression to representative-only behavior. The second class
runs the full nonant_sensitivies pipeline on a real Ipopt solve and is
skipped automatically if Ipopt isn't installed (so it is informational
on environments without Ipopt and a regression check on those that have
it).
"""

import unittest

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy.utils.nonant_sensitivities import (
    _bundle_consensus_groups,
    nonant_sensitivies,
)


def _ipopt_available():
    try:
        return bool(pyo.SolverFactory("ipopt").available(exception_flag=False))
    except Exception:
        return False


def _build_linear_scen(name, c_coefs, prob):
    m = pyo.ConcreteModel(name=name)
    m.x = pyo.Var(range(len(c_coefs)), bounds=(0, 10), initialize=0.0)
    m.obj = pyo.Objective(
        expr=sum(c_coefs[i] * m.x[i] for i in range(len(c_coefs)))
    )
    sputils.attach_root_node(m, 0, [m.x[i] for i in range(len(c_coefs))])
    m._mpisppy_probability = prob
    return m


def _attach_metadata(scen):
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


def _build_bundle(scen_creator, scen_names):
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
    _attach_metadata(bundle)
    return bundle


class TestBundleConsensusGroups(unittest.TestCase):
    """Solver-free unit tests of the bundle var-grouping helper.

    The helper is what guarantees the KKT sensitivity probe perturbs every
    per-sub-scenario nonant at a given bundle position together rather
    than only the representative ref Var (issue #673).
    """

    def _build(self, scen_names, c_by_name, prob_by_name):
        def _scen_creator(sname, **kwargs):
            return _build_linear_scen(
                sname, c_by_name[sname], prob_by_name[sname]
            )
        return _build_bundle(_scen_creator, scen_names)

    def test_groups_collect_one_var_per_sub_scenario(self):
        # 3 sub-scenarios x 2 first-stage vars: groups must have exactly
        # 2 keys and exactly 3 vars per key (one from each sub-scenario).
        # If the bug came back and the helper returned only the bundle
        # ref Var, each group would have 1 var; this test fails loudly.
        scen_names = ["scen0", "scen1", "scen2"]
        c_by_name = {
            "scen0": [1.0, 2.0],
            "scen1": [3.0, 4.0],
            "scen2": [5.0, 6.0],
        }
        prob_by_name = dict.fromkeys(scen_names, 1.0 / 3)

        bundle = self._build(scen_names, c_by_name, prob_by_name)
        groups = _bundle_consensus_groups(bundle)

        self.assertEqual(set(groups.keys()), {("ROOT", 0), ("ROOT", 1)})
        for k, sub_vars in groups.items():
            self.assertEqual(
                len(sub_vars), len(scen_names),
                msg=f"position {k}: expected {len(scen_names)} sub-scenario "
                    f"vars, got {len(sub_vars)}",
            )
            owners = {v.parent_component().parent_block() for v in sub_vars}
            self.assertEqual(
                len(owners), len(scen_names),
                msg=f"position {k}: expected {len(scen_names)} distinct "
                    f"sub-scenario blocks, got {len(owners)}",
            )

    def test_first_var_in_group_is_bundle_ref(self):
        # sputils.create_EF picks the first sub-scenario's nonant Var as
        # the bundle ref. The helper must include that ref *and* the other
        # sub-scenarios' nonants — not just the ref. This test pins both:
        # the ref is in the group (sanity), and there is at least one
        # other sub-scenario Var in the group that is not the ref
        # (the bug-catching half).
        scen_names = ["A", "B"]
        c_by_name = {"A": [10.0, 20.0], "B": [30.0, 40.0]}
        prob_by_name = {"A": 0.5, "B": 0.5}

        bundle = self._build(scen_names, c_by_name, prob_by_name)
        groups = _bundle_consensus_groups(bundle)

        for k_idx in range(2):
            key = ("ROOT", k_idx)
            ref = bundle.ref_vars[key]
            self.assertIn(
                ref, groups[key],
                msg=f"bundle ref Var missing from group {key}",
            )
            # At least one entry in the group must be a per-sub-scenario
            # Var that is NOT the ref. Without this the helper has
            # regressed to representative-only behavior.
            non_ref = [v for v in groups[key] if v is not ref]
            self.assertTrue(
                non_ref,
                msg=f"group {key} contains only the ref Var; per-sub-"
                    f"scenario nonants from other sub-scenarios are missing",
            )


@unittest.skipUnless(
    _ipopt_available(),
    "Ipopt not available; skipping nonant_sensitivies end-to-end test",
)
class TestNonantSensitiviesBundleEndToEnd(unittest.TestCase):
    """Smoke tests that go through the actual KKT sensitivity pipeline.

    Skipped automatically when Ipopt is not installed. The unit-test class
    above is what guards correctness in solver-less CI; this class runs in
    environments that have Ipopt and pins that the bundle path completes
    without raising and produces a sensitivity per bundle nonant position.
    """

    def _build(self, scen_names, c_by_name, prob_by_name):
        def _scen_creator(sname, **kwargs):
            return _build_linear_scen(
                sname, c_by_name[sname], prob_by_name[sname]
            )
        return _build_bundle(_scen_creator, scen_names)

    def test_bundle_returns_one_sensitivity_per_bundle_position(self):
        scen_names = ["A", "B", "C"]
        c_by_name = {
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
            "C": [5.0, 6.0],
        }
        prob_by_name = dict.fromkeys(scen_names, 1.0 / 3)
        bundle = self._build(scen_names, c_by_name, prob_by_name)
        sensis = nonant_sensitivies(bundle)
        # Keys match the bundle's nonant positions; values are scalars.
        self.assertEqual(set(sensis.keys()), {("ROOT", 0), ("ROOT", 1)})
        for k, v in sensis.items():
            # Either NumPy scalar or Python float — just check it is finite.
            self.assertTrue(
                float(v) == float(v),
                msg=f"sensitivity at {k} is NaN ({v=})",
            )

    def test_single_scenario_bundle_matches_unbundled(self):
        # A bundle of one sub-scenario must produce the same sensitivities
        # as the standalone scenario — proves the bundle branch degenerates
        # correctly and isn't introducing extra coupling.
        c = [1.0, 2.0]

        def _scen_creator(sname, **kwargs):
            return _build_linear_scen(sname, c, 1.0)

        # Standalone scenario.
        scen = _scen_creator("standalone")
        _attach_metadata(scen)
        scen_sensis = nonant_sensitivies(scen)

        # Bundle of one.
        bundle = _build_bundle(_scen_creator, ["only"])
        bundle_sensis = nonant_sensitivies(bundle)

        self.assertEqual(set(scen_sensis.keys()), set(bundle_sensis.keys()))
        for k in scen_sensis:
            self.assertAlmostEqual(
                float(scen_sensis[k]), float(bundle_sensis[k]), places=4,
                msg=f"single-scen bundle sensitivity at {k} diverges from "
                    f"standalone: bundle={bundle_sensis[k]} vs "
                    f"standalone={scen_sensis[k]}",
            )


if __name__ == "__main__":
    unittest.main()
