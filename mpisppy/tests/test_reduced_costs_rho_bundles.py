###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""ReducedCostsSpoke must aggregate per-sub-scenario reduced costs across
the consensus group at each bundle nonant position (issue #673). Pre-fix
behavior used ``sub.rc[ref_var]`` which only saw the bundle's representative
ref Var; in an EF bundle the bundle objective references each sub-scenario's
own nonant Vars (the rest are tied by NA equality constraints), so the rc
of the ref Var alone is not the correct rate-of-change input for rc-fixing.

Tests are solver-free: they synthesize a bundle, prime the bundle's ``rc``
Suffix with known values, and call the spoke's aggregation helper directly.
"""

import unittest

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy.cylinders.reduced_costs_spoke import _consensus_rc_sum
from mpisppy.utils.nonant_sensitivities import _bundle_consensus_groups


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


class TestConsensusRcSum(unittest.TestCase):
    """Unit tests for the spoke's _consensus_rc_sum helper."""

    def test_unbundled_returns_single_var_rc(self):
        # When consensus_groups is None (non-bundle path), the helper must
        # return the ref Var's rc unchanged so existing non-bundle behavior
        # is preserved bit-for-bit.
        scen = _build_linear_scen("solo", [1.0, 2.0], 1.0)
        _attach_metadata(scen)
        scen.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        scen.rc[scen.x[0]] = 7.5
        scen.rc[scen.x[1]] = 11.25
        for ndn_i, ref in scen._mpisppy_data.nonant_indices.items():
            self.assertEqual(
                _consensus_rc_sum(scen, ndn_i, ref, None),
                scen.rc[ref],
            )

    def test_bundle_returns_consensus_sum(self):
        # 3 sub-scenarios x 2 first-stage vars. Set bundle.rc[per_scen_var]
        # to distinct values per (sub, position) and verify the helper
        # returns the sum across the consensus group.
        scen_names = ["A", "B", "C"]
        c_by_name = {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]}
        prob_by_name = dict.fromkeys(scen_names, 1.0 / 3)

        def _scen_creator(sname, **kwargs):
            return _build_linear_scen(
                sname, c_by_name[sname], prob_by_name[sname]
            )

        bundle = _build_bundle(_scen_creator, scen_names)
        bundle.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # Encode rc[sub.x_k] = 100 * (sub_index + 1) + k so each sub-Var
        # carries a distinct, easy-to-reproduce rc value.
        rc_values = {}
        for sub_idx, sname in enumerate(scen_names):
            scen = bundle.component(sname)
            for k in range(2):
                v = scen.x[k]
                bundle.rc[v] = 100.0 * (sub_idx + 1) + k
                rc_values[(sname, k)] = bundle.rc[v]

        consensus_groups = _bundle_consensus_groups(bundle)
        for ndn_i, ref in bundle._mpisppy_data.nonant_indices.items():
            # Expected: sum across all 3 sub-scenarios at this bundle position.
            expected = sum(
                rc_values[(sname, ndn_i[1])] for sname in scen_names
            )
            got = _consensus_rc_sum(bundle, ndn_i, ref, consensus_groups)
            self.assertAlmostEqual(
                got, expected, places=10,
                msg=f"bundle {ndn_i=}: {got=} != consensus-sum {expected=}",
            )
            # Also confirm the helper does NOT silently return only the ref's
            # rc — the pre-fix behavior. (This guards against a regression
            # to representative-only sourcing.)
            ref_only = bundle.rc[ref]
            if ref_only != expected:
                self.assertNotAlmostEqual(
                    got, ref_only, places=6,
                    msg=f"bundle {ndn_i=} matches ref-only rc {ref_only}; "
                        f"aggregation regressed",
                )


if __name__ == "__main__":
    unittest.main()
