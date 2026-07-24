###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unit tests for the per-field coherence policy used by the unequal-rank
multi-source reader (spcommunicator.reduce_source_write_ids) and for how the
reader (_flex_get_multi_source) acts on it.

The MPI integration tests run against a synchronous PH hub, where all sources
of a field always carry the same write_id -- so strict and relaxed behave
identically there and cannot distinguish the policies. An asynchronous sender
(e.g. an APH hub) is exactly where they differ, but timing makes it impossible
to deterministically force a *particular* mixed-write_id snapshot at a read in
a live MPI run. These tests pin the behavior directly, without MPI:
reduce_source_write_ids' policy logic (strict rejects a mixed-iteration read,
relaxed accepts the floor), and the reader's resulting retry semantics that an
asynchronous sender relies on -- a strict mixed-id read is rejected without
disturbing the receive buffer, and a later coherent read then succeeds."""

import unittest

import numpy as np

from mpisppy.cylinders.spcommunicator import (
    SPCommunicator,
    RecvArray,
    reduce_source_write_ids,
    _STRICT_COHERENCE_FIELDS,
)
from mpisppy.cylinders.overlap_map import OverlapSegment
from mpisppy.cylinders.spwindow import Field, padded_len_n_doubles


class TestReduceSourceWriteIds(unittest.TestCase):

    def test_relaxed_takes_floor(self):
        # Relaxed: the assembled value may mix iterations; report the floor so
        # "new" is detected only once every source has advanced past it.
        self.assertEqual(reduce_source_write_ids([5, 5, 5], strict=False), 5)
        self.assertEqual(reduce_source_write_ids([7, 5, 6], strict=False), 5)
        self.assertEqual(reduce_source_write_ids([3], strict=False), 3)

    def test_strict_accepts_only_when_all_agree(self):
        # Strict: a single shared id is accepted as-is.
        self.assertEqual(reduce_source_write_ids([5, 5, 5], strict=True), 5)
        self.assertEqual(reduce_source_write_ids([9], strict=True), 9)

    def test_strict_rejects_mixed_with_sentinel(self):
        # Strict: any disagreement -> sentinel -1 (reject). -1 is below any
        # real write_id, so `new_id > last_id` is false, and it breaks the
        # cross-reader sum-equality check so every reader rejects together.
        self.assertEqual(reduce_source_write_ids([5, 4], strict=True), -1)
        self.assertEqual(reduce_source_write_ids([7, 5, 6], strict=True), -1)
        self.assertEqual(reduce_source_write_ids([1, 1, 2], strict=True), -1)

    def test_empty_sources_is_zero(self):
        self.assertEqual(reduce_source_write_ids([], strict=True), 0)
        self.assertEqual(reduce_source_write_ids([], strict=False), 0)

    def test_sentinel_is_below_any_initial_id(self):
        # The reject sentinel must be < the initial buffer id (0) so a rejected
        # strict read is never reported new.
        self.assertLess(reduce_source_write_ids([2, 3], strict=True), 0)


class TestStrictCoherenceFields(unittest.TestCase):
    """Pin which per-scenario fields require strict coherence (all sources at
    one write_id) vs relaxed."""

    def test_xhat_objval_fields_are_strict(self):
        # BEST_XHAT / RECENT_XHATS carry per-scenario [first-stage nonants,
        # obj_val] blocks. Under a mixed-iteration assembly a block's obj_val
        # would be paired with another iteration's nonants -- and FWPH derives a
        # Frank-Wolfe column's recourse cost from that obj_val. Strict coherence
        # rejects the mixed read so each obj_val stays paired with its own
        # nonants (and the NAC-redundant first stage stays consistent across
        # scenarios).
        self.assertIn(Field.BEST_XHAT, _STRICT_COHERENCE_FIELDS)
        self.assertIn(Field.RECENT_XHATS, _STRICT_COHERENCE_FIELDS)

    def test_duals_is_strict(self):
        self.assertIn(Field.DUALS, _STRICT_COHERENCE_FIELDS)

    def test_xfeas_is_relaxed(self):
        # XFEAS carries genuinely distinct per-scenario iterates (no
        # cross-scenario NAC), so its obj_val is never desynced from its
        # nonants -- relaxed is correct.
        self.assertNotIn(Field.XFEAS, _STRICT_COHERENCE_FIELDS)

    def test_nonants_vals_is_relaxed(self):
        self.assertNotIn(Field.NONANTS_VALS, _STRICT_COHERENCE_FIELDS)


# --- reader retry semantics under mixed write_ids -------------------------
#
# An asynchronous sender (the APH hub) can leave the several remote ranks a
# spoke assembles a per-scenario field from at different write_ids. The
# coherence policy decides whether to accept such a read, and the reader
# (_flex_get_multi_source) must act on a rejection *without corrupting* the
# receive buffer, so a later coherent read still lands clean. These tests drive
# the reader against a tiny stand-in window holding source buffers whose
# write_ids we set by hand -- something the timing-dependent MPI test cannot do
# deterministically.
#
# Layout: a peer cylinder of two source ranks (global ranks 0 and 1), each
# publishing 2 data items; the receiver assembles a length-4 buffer, taking
# rank 0's two items into local [0:2] and rank 1's into local [2:4]. synchronize
# is False (the APH hub reads with synchronize=False), so no cylinder_comm is
# involved and the stand-in needs only a window.
_DATA_LEN = 2
_LOGICAL_LEN = _DATA_LEN + 1
_PADDED_LEN = padded_len_n_doubles(_LOGICAL_LEN)
_RANK0_DATA = [1.0, 2.0]
_RANK1_DATA = [3.0, 4.0]
_ASSEMBLED = _RANK0_DATA + _RANK1_DATA   # what a clean read must produce
_SEGMENTS = [
    OverlapSegment(remote_rank=0, remote_offset=0, local_offset=0, count=2),
    OverlapSegment(remote_rank=1, remote_offset=0, local_offset=2, count=2),
]


def _make_source(data, write_id):
    """A peer rank's padded send buffer: data, then the write_id at the logical
    end, then NaN padding -- the shape SPWindow.get returns."""
    arr = np.full(_PADDED_LEN, np.nan)
    arr[:_DATA_LEN] = data
    arr[_LOGICAL_LEN - 1] = float(write_id)
    return arr


class _StubWindow:
    """Stands in for SPWindow: hands back per-rank source buffers on get() and
    advertises their layout. The source write_ids are mutable so a test can
    flip a read from mixed to coherent between calls."""

    def __init__(self):
        self.sources = {
            0: _make_source(_RANK0_DATA, 0),
            1: _make_source(_RANK1_DATA, 0),
        }
        self.strata_buffer_layouts = {
            r: {Field.DUALS: (_DATA_LEN, _LOGICAL_LEN, _PADDED_LEN),
                Field.NONANTS_VALS: (_DATA_LEN, _LOGICAL_LEN, _PADDED_LEN)}
            for r in (0, 1)
        }

    def set_write_ids(self, id0, id1):
        self.sources[0][_LOGICAL_LEN - 1] = float(id0)
        self.sources[1][_LOGICAL_LEN - 1] = float(id1)

    def get(self, out_arr, rank, field):
        out_arr[:] = self.sources[rank]


def _make_reader():
    """A minimal SPCommunicator stand-in carrying just what
    _flex_get_multi_source touches (with synchronize=False)."""
    sp = SPCommunicator.__new__(SPCommunicator)
    sp.window = _StubWindow()
    peer = 1
    sp._overlap_source_ranks = {
        (Field.DUALS, peer): [0, 1],
        (Field.NONANTS_VALS, peer): [0, 1],
    }
    sp.overlap_maps = {
        (Field.DUALS, peer): _SEGMENTS,
        (Field.NONANTS_VALS, peer): _SEGMENTS,
    }
    # read-outcome diagnostic state (set by __init__ in the real class)
    sp.coherence_counters = {}
    sp._coherence_report_period = 0
    return sp, peer


class TestMultiSourceReaderRetry(unittest.TestCase):

    def _read(self, sp, buf, field, peer):
        return sp._flex_get_multi_source(buf, field, peer, synchronize=False)

    def test_strict_rejects_mixed_without_disturbing_buffer(self):
        # A strict (DUALS) read whose sources disagree must be rejected, leave
        # is_new False, leave the accepted id where it was, and -- the part an
        # async retry depends on -- not write any data into the buffer.
        sp, peer = _make_reader()
        buf = RecvArray(_DATA_LEN * 2)
        sp.window.set_write_ids(5, 4)  # mixed

        self.assertFalse(self._read(sp, buf, Field.DUALS, peer))
        self.assertFalse(buf.is_new())
        self.assertEqual(buf.id(), 0)
        # Untouched: the buffer still holds its freshly-allocated NaNs, not a
        # half-stitched value from the rejected read.
        self.assertTrue(np.all(np.isnan(buf.value_array())))

    def test_strict_accepts_after_sources_agree(self):
        # The same buffer that just rejected a mixed read accepts the next read
        # once the sources line up -- and lands the fully assembled value, with
        # no residue from the rejected attempt.
        sp, peer = _make_reader()
        buf = RecvArray(_DATA_LEN * 2)

        sp.window.set_write_ids(5, 4)
        self.assertFalse(self._read(sp, buf, Field.DUALS, peer))

        sp.window.set_write_ids(6, 6)  # now coherent
        self.assertTrue(self._read(sp, buf, Field.DUALS, peer))
        self.assertTrue(buf.is_new())
        self.assertEqual(buf.id(), 6)
        np.testing.assert_allclose(buf.value_array(), _ASSEMBLED)

    def test_strict_same_id_is_not_new(self):
        # After accepting id 6, re-reading the same coherent id reports not-new
        # (new_id > last_id is false), so a consumer is not handed stale data
        # twice.
        sp, peer = _make_reader()
        buf = RecvArray(_DATA_LEN * 2)
        sp.window.set_write_ids(6, 6)
        self.assertTrue(self._read(sp, buf, Field.DUALS, peer))
        self.assertFalse(self._read(sp, buf, Field.DUALS, peer))
        self.assertFalse(buf.is_new())

    def test_relaxed_accepts_mixed_at_the_floor(self):
        # A relaxed (NONANTS_VALS) read accepts mixed write_ids, assembling the
        # value as-is and recording the floor (5) as the accepted id -- so it is
        # reported new again only once every source has advanced past 5.
        sp, peer = _make_reader()
        buf = RecvArray(_DATA_LEN * 2)
        sp.window.set_write_ids(7, 5)  # mixed; floor is 5

        self.assertTrue(self._read(sp, buf, Field.NONANTS_VALS, peer))
        self.assertTrue(buf.is_new())
        self.assertEqual(buf.id(), 5)
        np.testing.assert_allclose(buf.value_array(), _ASSEMBLED)


class TestCoherenceCounters(unittest.TestCase):
    """The read-outcome diagnostic, driven deterministically through the same
    stub: each read outcome must land in exactly one counter bucket, so the
    buckets partition the total and the miss rate is computable after the run.
    (synchronize is False throughout, so rejected_cross_reader stays 0; that
    bucket needs the collective check the MPI integration test exercises.)"""

    def _read(self, sp, buf, field, peer):
        return sp._flex_get_multi_source(buf, field, peer, synchronize=False)

    def _counters(self, sp, field):
        return sp.coherence_counters[field]

    def test_outcomes_bucketed_and_partition_total(self):
        sp, peer = _make_reader()
        buf = RecvArray(_DATA_LEN * 2)

        # strict + mixed -> rejected_incoherent (the fundamental miss)
        sp.window.set_write_ids(5, 4)
        self._read(sp, buf, Field.DUALS, peer)
        # coherent + advanced -> new_accepted
        sp.window.set_write_ids(6, 6)
        self._read(sp, buf, Field.DUALS, peer)
        # coherent, id unchanged -> not_new (a slow sender, not a miss)
        self._read(sp, buf, Field.DUALS, peer)

        counters = self._counters(sp, Field.DUALS)
        self.assertEqual(counters["rejected_incoherent"], 1)
        self.assertEqual(counters["new_accepted"], 1)
        self.assertEqual(counters["not_new"], 1)
        self.assertEqual(counters["accepted_mixed"], 0)
        self.assertEqual(counters["rejected_cross_reader"], 0)
        self.assertEqual(counters["total"], 3)

    def test_relaxed_mixed_counts_as_accepted_mixed(self):
        sp, peer = _make_reader()
        buf = RecvArray(_DATA_LEN * 2)

        sp.window.set_write_ids(7, 5)  # mixed; relaxed accepts at the floor
        self._read(sp, buf, Field.NONANTS_VALS, peer)
        # mixed again but the floor has not advanced -> not_new, not a miss
        self._read(sp, buf, Field.NONANTS_VALS, peer)

        counters = self._counters(sp, Field.NONANTS_VALS)
        self.assertEqual(counters["accepted_mixed"], 1)
        self.assertEqual(counters["not_new"], 1)
        self.assertEqual(counters["rejected_incoherent"], 0)
        self.assertEqual(counters["total"], 2)


if __name__ == "__main__":
    unittest.main()
