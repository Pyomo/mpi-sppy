###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Serial, exact-value tests for the Phase-4a xhat-field assembly math.

The np=6 integration test (test_flexible_rank_xhat) confirms the end-to-end
FWPH run converges at unequal ranks, but it cannot pin the exact byte layout of
the version-interleaved circular buffer. These tests do, directly and without
MPI, by exercising the assembly arithmetic against a hand-checked straddle
layout:

  * ``_expand_to_circular_versions`` -- replicating one version's overlap
    segments across the circular buffer, with a *different* version stride per
    source rank (the peer ranks here hold unequal scenario counts);
  * the per-scenario segment copy that assembles a local buffer from the source
    ranks' send buffers.

The xhat ring (``BEST_XHAT`` / ``RECENT_XHATS``) is read with strict coherence,
so an accepted read has every source rank at one write_id -- i.e. every rank
holds the *same* incumbent first stage. The assembler therefore reproduces a
NAC-consistent first stage across every scenario block with no post-assembly
fix-up, while routing each scenario's obj_val to its own slot. ``XFEAS`` carries
per-scenario *distinct* iterates (its consumer tries each scenario's x as its
own candidate), which the same assembler preserves per scenario.

The methods read only a handful of attributes, so a tiny stand-in object stands
in for a fully-constructed SPCommunicator.
"""

import types
import unittest

import numpy as np

from mpisppy.cylinders.spwindow import Field
from mpisppy.cylinders.spcommunicator import SPCommunicator
from mpisppy.cylinders.overlap_map import compute_overlap_segments


# --- fixed straddle layout shared by the tests ---------------------------
#
# 6 scenarios (global indices 0..5), 2 first-stage nonants each (K=2), so every
# per-scenario block is [n, n, obj_val] (length 3).
#
# Peer (source) cylinder: 2 ranks holding *unequal* counts --
#   rank 0 -> scenarios [0,1,2,3]   (4 scenarios; version stride 4*3 = 12)
#   rank 1 -> scenarios [4,5]       (2 scenarios; version stride 2*3 =  6)
# The unequal strides are the point: a single version stride would mis-assemble
# the later versions of one of the ranks.
#
# Receiver (local) rank holds scenarios [2,3,4,5] -- straddling both source
# ranks (2,3 from rank 0; 4,5 from rank 1).
K = 2
BLOCK = K + 1
PEER_SLICES = [[0, 1, 2, 3], [4, 5]]
LOCAL_SCEN_IDXS = [2, 3, 4, 5]
NSCEN_LOCAL = len(LOCAL_SCEN_IDXS)
BEST_XHAT_LEN = NSCEN_LOCAL * BLOCK          # one version's worth, this rank
NVERSIONS = 2
RECENT_LEN = NVERSIONS * BEST_XHAT_LEN


# First-stage marker for the xhat ring: under strict coherence every source rank
# holds the same incumbent first stage at a given version, so this depends only
# on the version (not the rank). The assembler must reproduce it across every
# scenario block.
def _ring_fs(version):
    return 0.5 + 10.0 * version

# First-stage marker for XFEAS: genuinely distinct per scenario (the consumer
# tries each scenario's x as its own candidate); keyed by global scenario index.
def _xfeas_fs(scen):
    return 100.0 + scen

# Per-scenario, per-version objective value (the block's second slot).
def _obj(scen, version):
    return 1000.0 * version + scen


def _make_stub():
    sp = types.SimpleNamespace()
    sp._field_lengths = {
        Field.BEST_XHAT: BEST_XHAT_LEN,
        Field.RECENT_XHATS: RECENT_LEN,
    }
    return sp


def _make_source_buffer(scens_on_rank, nversions, fs_of):
    """A peer rank's send buffer for a BEST_XHAT-style field: nversions copies of
    [block per scenario], each block = [first-stage(K), obj_val]. ``fs_of(scen,
    version)`` supplies the first-stage marker for each block."""
    per_version = len(scens_on_rank) * BLOCK
    buf = np.full(nversions * per_version, np.nan)
    for v in range(nversions):
        for j, s in enumerate(scens_on_rank):
            off = v * per_version + j * BLOCK
            buf[off:off + K] = fs_of(s, v)
            buf[off + K] = _obj(s, v)
    return buf


def _assemble(segments, sources, local_len):
    """Mirror the segment-copy loop in _flex_get_multi_source (no MPI)."""
    local_buf = np.full(local_len, np.nan)
    for seg in segments:
        src = sources[seg.remote_rank]
        local_buf[seg.local_offset:seg.local_offset + seg.count] = \
            src[seg.remote_offset:seg.remote_offset + seg.count]
    return local_buf


class TestFlexXhatAssembly(unittest.TestCase):

    def test_expand_circular_versions_offsets(self):
        """The expanded segments must carry a per-rank version stride."""
        sp = _make_stub()
        items_per_scen = [BLOCK] * 6
        base_segments = compute_overlap_segments(
            LOCAL_SCEN_IDXS, PEER_SLICES, items_per_scen
        )
        # base = 0 here, so remote_rank is already the (global) source rank.
        expanded = SPCommunicator._expand_to_circular_versions(
            sp, base_segments, PEER_SLICES, 0, items_per_scen
        )
        got = {(s.remote_rank, s.remote_offset, s.local_offset, s.count)
               for s in expanded}
        # rank 0 version stride = 4*3 = 12; rank 1 version stride = 2*3 = 6;
        # local version stride = 4*3 = 12.
        expected = {
            # version 0
            (0, 6, 0, 6),    # scen 2,3 from rank0 (offset 2*3=6), into local 0
            (1, 0, 6, 6),    # scen 4,5 from rank1 (offset 0),     into local 6
            # version 1: +12 remote on rank0, +6 remote on rank1, +12 local
            (0, 18, 12, 6),
            (1, 6, 18, 6),
        }
        self.assertEqual(got, expected)

    def test_recent_xhats_version_interleaved_assembly(self):
        """Version-interleaved assembly of the circular xhat buffer. Strict
        coherence guarantees all sources at one write_id, so both source ranks
        hold the same incumbent first stage per version; the assembly reproduces
        it across every scenario block (NAC-consistent, no fix-up) while routing
        each scenario's obj_val to its own slot."""
        sp = _make_stub()
        items_per_scen = [BLOCK] * 6
        base_segments = compute_overlap_segments(
            LOCAL_SCEN_IDXS, PEER_SLICES, items_per_scen
        )
        expanded = SPCommunicator._expand_to_circular_versions(
            sp, base_segments, PEER_SLICES, 0, items_per_scen
        )
        sources = {
            0: _make_source_buffer(PEER_SLICES[0], NVERSIONS, lambda s, v: _ring_fs(v)),
            1: _make_source_buffer(PEER_SLICES[1], NVERSIONS, lambda s, v: _ring_fs(v)),
        }
        local_buf = _assemble(expanded, sources, RECENT_LEN)

        expected = np.empty(RECENT_LEN)
        for v in range(NVERSIONS):
            ref = _ring_fs(v)
            for j, s in enumerate(LOCAL_SCEN_IDXS):
                off = v * BEST_XHAT_LEN + j * BLOCK
                expected[off:off + K] = ref
                expected[off + K] = _obj(s, v)
        np.testing.assert_allclose(local_buf, expected)
        # The two versions really carry distinct first stages (so the
        # version-stride assembly is doing visible work).
        self.assertNotEqual(_ring_fs(0), _ring_fs(1))

    def test_best_xhat_single_version_assembly(self):
        """BEST_XHAT is the one-version case: a single coherent first stage
        reproduced across scenarios, with per-scenario obj_val slots."""
        items_per_scen = [BLOCK] * 6
        segments = compute_overlap_segments(
            LOCAL_SCEN_IDXS, PEER_SLICES, items_per_scen
        )
        sources = {
            0: _make_source_buffer(PEER_SLICES[0], 1, lambda s, v: _ring_fs(v)),
            1: _make_source_buffer(PEER_SLICES[1], 1, lambda s, v: _ring_fs(v)),
        }
        local_buf = _assemble(segments, sources, BEST_XHAT_LEN)

        expected = np.empty(BEST_XHAT_LEN)
        ref = _ring_fs(0)
        for j, s in enumerate(LOCAL_SCEN_IDXS):
            off = j * BLOCK
            expected[off:off + K] = ref
            expected[off + K] = _obj(s, 0)
        np.testing.assert_allclose(local_buf, expected)

    def test_xfeas_per_scenario_iterates(self):
        """XFEAS carries per-scenario distinct iterates: assembled per scenario
        (no NAC fix-up), so each scenario keeps its own first stage."""
        items_per_scen = [BLOCK] * 6
        segments = compute_overlap_segments(
            LOCAL_SCEN_IDXS, PEER_SLICES, items_per_scen
        )
        sources = {
            0: _make_source_buffer(PEER_SLICES[0], 1, lambda s, v: _xfeas_fs(s)),
            1: _make_source_buffer(PEER_SLICES[1], 1, lambda s, v: _xfeas_fs(s)),
        }
        local_buf = _assemble(segments, sources, BEST_XHAT_LEN)

        expected = np.empty(BEST_XHAT_LEN)
        for j, s in enumerate(LOCAL_SCEN_IDXS):
            off = j * BLOCK
            expected[off:off + K] = _xfeas_fs(s)
            expected[off + K] = _obj(s, 0)
        np.testing.assert_allclose(local_buf, expected)
        # The per-scenario first stages really are distinct (so "preserved per
        # scenario" is a meaningful claim).
        self.assertEqual(len({_xfeas_fs(s) for s in LOCAL_SCEN_IDXS}), NSCEN_LOCAL)


# --- multistage assembly (Phase 4b) ---------------------------------------
#
# In multistage a scenario's block is its whole root->leaf nonant path plus
# obj_val, and non-anticipativity is a per-*node* property: the root segment is
# shared by every scenario, while a stage-2 node's segment is shared only by the
# scenarios in its subtree (different subtrees hold different stage-2 values).
#
# The assembler is index-based and stage-agnostic: it routes each scenario's
# whole block from the rank that holds it, exactly as in two-stage. Strict
# coherence guarantees all sources are at one write_id, so the source blocks are
# one coherent incumbent in which scenarios sharing a node already carry that
# node's values identically. Faithful routing therefore yields a per-node
# NAC-consistent assembly with NO post-assembly fix-up -- and does NOT collapse
# the distinct subtrees onto one another (which a whole-block "copy scenario 0"
# fix-up would have done).
#
# Layout: 3-stage tree, root with R=2 nonants shared by all; two stage-2 subtrees
# A and B with one nonant each. Per-scenario block = [root(2), stage2(1), obj].
K_M = 3                        # root(2) + stage2(1)
BLOCK_M = K_M + 1
ROOT_LEN = 2
# 6 scenarios in two subtrees of three: A = {0,1,2}, B = {3,4,5}.
SUBTREE = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 5: "B"}
# Source ranks: rank 0 straddles the subtree boundary (holds A and one B scen),
# rank 1 is wholly in B. Receiver holds one A scenario and three B scenarios,
# straddling both source ranks -- so the assembly must keep A's and B's stage-2
# values distinct while routing across ranks.
MS_PEER_SLICES = [[0, 1, 2, 3], [4, 5]]
MS_LOCAL_SCEN_IDXS = [2, 3, 4, 5]


def _root_val(version):
    return 7.0 + 100.0 * version              # shared by every scenario at root

def _s2_val(scen, version):
    # distinct per subtree, identical within a subtree
    return (10.0 if SUBTREE[scen] == "A" else 20.0) + 100.0 * version

def _ms_nonants(scen, version):
    return np.array([_root_val(version)] * ROOT_LEN + [_s2_val(scen, version)])


def _make_ms_source_buffer(scens_on_rank, nversions):
    """A peer rank's send buffer for a multistage BEST_XHAT-style field: each
    scenario block is [root(R), stage2(1), obj_val], the source data being one
    coherent (single-write_id) incumbent."""
    per_version = len(scens_on_rank) * BLOCK_M
    buf = np.full(nversions * per_version, np.nan)
    for v in range(nversions):
        for j, s in enumerate(scens_on_rank):
            off = v * per_version + j * BLOCK_M
            buf[off:off + K_M] = _ms_nonants(s, v)
            buf[off + K_M] = _obj(s, v)
    return buf


class TestFlexMultistageAssembly(unittest.TestCase):

    def test_assembly_preserves_per_node_nac(self):
        """Multi-node blocks routed wholesale across straddling ranks: the root
        segment ends up identical across every scenario (root NAC), each subtree
        keeps its own stage-2 value (per-node NAC, subtrees not collapsed), and
        obj_val stays per-scenario -- all with no fix-up."""
        items_per_scen = [BLOCK_M] * 6
        segments = compute_overlap_segments(
            MS_LOCAL_SCEN_IDXS, MS_PEER_SLICES, items_per_scen
        )
        sources = {
            0: _make_ms_source_buffer(MS_PEER_SLICES[0], 1),
            1: _make_ms_source_buffer(MS_PEER_SLICES[1], 1),
        }
        local_len = len(MS_LOCAL_SCEN_IDXS) * BLOCK_M
        local_buf = _assemble(segments, sources, local_len)

        expected = np.empty(local_len)
        for j, s in enumerate(MS_LOCAL_SCEN_IDXS):
            off = j * BLOCK_M
            expected[off:off + K_M] = _ms_nonants(s, 0)
            expected[off + K_M] = _obj(s, 0)
        np.testing.assert_allclose(local_buf, expected)

        # Root segment identical across every local scenario (root NAC)...
        roots = [local_buf[j * BLOCK_M:j * BLOCK_M + ROOT_LEN]
                 for j in range(len(MS_LOCAL_SCEN_IDXS))]
        for r in roots[1:]:
            np.testing.assert_allclose(r, roots[0])
        # ...while the two subtrees keep DISTINCT stage-2 values: a whole-block
        # collapse to scenario 0's would have erased subtree B's value.
        s2 = {s: local_buf[j * BLOCK_M + ROOT_LEN]
              for j, s in enumerate(MS_LOCAL_SCEN_IDXS)}
        self.assertEqual(s2[2], _s2_val(2, 0))          # subtree A
        self.assertEqual(s2[3], _s2_val(3, 0))          # subtree B
        self.assertNotEqual(s2[2], s2[3])               # subtrees not collapsed


if __name__ == "__main__":
    unittest.main()
