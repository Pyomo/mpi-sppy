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


if __name__ == "__main__":
    unittest.main()
