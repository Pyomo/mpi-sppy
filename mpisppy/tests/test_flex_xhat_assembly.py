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
FWPH run converges at unequal ranks, but a synchronous hub keeps every source at
the same write_id, so it cannot distinguish a correct first-stage NAC fix-up
from no fix-up at all, nor pin the exact byte layout of the version-interleaved
circular buffer. These tests do, directly and without MPI, by exercising the
two pieces of new arithmetic against a hand-checked straddle layout:

  * ``_expand_to_circular_versions`` -- replicating one version's overlap
    segments across the circular buffer, with a *different* version stride per
    source rank (the peer ranks here hold unequal scenario counts);
  * ``_enforce_first_stage_nac`` -- overwriting every scenario's first-stage
    portion with the first scenario's, per version, leaving the per-scenario
    cost slots alone.

To force the fix-up to do visible work, the two source ranks publish *different*
first-stage values (as if read at mixed write_ids); a correct fix-up collapses
them to a single coherent vector, an absent one would leave them split. XFEAS,
which carries per-scenario *distinct* iterates, must NOT be collapsed.

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
# per-scenario block is [n, n, cost] (length 3).
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

# Distinct first-stage markers per (source rank, version); the gap between the
# two ranks is what the NAC fix-up must erase.
def _fs(rank, version):
    return 0.5 + rank + 10.0 * version

# Per-scenario, per-version cost; never touched by the fix-up.
def _cost(scen, version):
    return 1000.0 * version + scen


def _make_stub():
    sp = types.SimpleNamespace()
    sp.opt = types.SimpleNamespace(nonant_length=K)
    sp._local_scen_idxs = list(LOCAL_SCEN_IDXS)
    sp._field_lengths = {
        Field.BEST_XHAT: BEST_XHAT_LEN,
        Field.RECENT_XHATS: RECENT_LEN,
    }
    return sp


def _make_source_buffer(scens_on_rank, rank, nversions):
    """A peer rank's send buffer for a BEST_XHAT-style field: nversions copies of
    [block per scenario], each block = [first-stage(K), cost]."""
    per_version = len(scens_on_rank) * BLOCK
    buf = np.full(nversions * per_version, np.nan)
    for v in range(nversions):
        for j, s in enumerate(scens_on_rank):
            off = v * per_version + j * BLOCK
            buf[off:off + K] = _fs(rank, v)
            buf[off + K] = _cost(s, v)
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

    def test_recent_xhats_assembly_and_nac_fixup(self):
        """Version-interleaved assembly + per-version first-stage NAC fix-up."""
        sp = _make_stub()
        items_per_scen = [BLOCK] * 6
        base_segments = compute_overlap_segments(
            LOCAL_SCEN_IDXS, PEER_SLICES, items_per_scen
        )
        expanded = SPCommunicator._expand_to_circular_versions(
            sp, base_segments, PEER_SLICES, 0, items_per_scen
        )
        sources = {
            0: _make_source_buffer(PEER_SLICES[0], 0, NVERSIONS),
            1: _make_source_buffer(PEER_SLICES[1], 1, NVERSIONS),
        }
        local_buf = _assemble(expanded, sources, RECENT_LEN)

        # Before the fix-up: scen 2,3 carry rank-0's first stage, scen 4,5
        # rank-1's -- inconsistent across scenarios (this is what the fix-up
        # must repair). Costs are already correctly routed per scenario.
        self.assertEqual(local_buf[0], _fs(0, 0))    # local scen 2, v0
        self.assertEqual(local_buf[6], _fs(1, 0))    # local scen 4, v0

        SPCommunicator._enforce_first_stage_nac(sp, local_buf, Field.RECENT_XHATS)

        # After: every scenario's first stage equals scenario-0's source
        # (rank 0), per version; costs untouched and per-scenario.
        expected = np.empty(RECENT_LEN)
        for v in range(NVERSIONS):
            ref = _fs(0, v)  # rank 0 holds local scenario 0 (global 2)
            for j, s in enumerate(LOCAL_SCEN_IDXS):
                off = v * BEST_XHAT_LEN + j * BLOCK
                expected[off:off + K] = ref
                expected[off + K] = _cost(s, v)
        np.testing.assert_allclose(local_buf, expected)

    def test_best_xhat_single_version_nac_fixup(self):
        """BEST_XHAT is the one-version case of the same fix-up."""
        sp = _make_stub()
        items_per_scen = [BLOCK] * 6
        segments = compute_overlap_segments(
            LOCAL_SCEN_IDXS, PEER_SLICES, items_per_scen
        )
        sources = {
            0: _make_source_buffer(PEER_SLICES[0], 0, 1),
            1: _make_source_buffer(PEER_SLICES[1], 1, 1),
        }
        local_buf = _assemble(segments, sources, BEST_XHAT_LEN)
        SPCommunicator._enforce_first_stage_nac(sp, local_buf, Field.BEST_XHAT)

        expected = np.empty(BEST_XHAT_LEN)
        ref = _fs(0, 0)
        for j, s in enumerate(LOCAL_SCEN_IDXS):
            off = j * BLOCK
            expected[off:off + K] = ref
            expected[off + K] = _cost(s, 0)
        np.testing.assert_allclose(local_buf, expected)

    def test_xfeas_is_not_collapsed(self):
        """XFEAS carries per-scenario distinct iterates: assembled per scenario,
        with NO NAC fix-up, so each scenario keeps its own first stage."""
        items_per_scen = [BLOCK] * 6
        segments = compute_overlap_segments(
            LOCAL_SCEN_IDXS, PEER_SLICES, items_per_scen
        )
        sources = {
            0: _make_source_buffer(PEER_SLICES[0], 0, 1),
            1: _make_source_buffer(PEER_SLICES[1], 1, 1),
        }
        local_buf = _assemble(segments, sources, BEST_XHAT_LEN)
        # No _enforce_first_stage_nac call: XFEAS is not in _FIRST_STAGE_NAC_FIELDS.

        expected = np.empty(BEST_XHAT_LEN)
        for j, s in enumerate(LOCAL_SCEN_IDXS):
            holding_rank = 0 if s in PEER_SLICES[0] else 1
            off = j * BLOCK
            expected[off:off + K] = _fs(holding_rank, 0)
            expected[off + K] = _cost(s, 0)
        np.testing.assert_allclose(local_buf, expected)
        # And the two source ranks really did differ (so the test is meaningful).
        self.assertNotEqual(_fs(0, 0), _fs(1, 0))


if __name__ == "__main__":
    unittest.main()
