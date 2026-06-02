###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unit tests for the per-field coherence policy used by the unequal-rank
multi-source reader (spcommunicator.reduce_source_write_ids).

The MPI integration tests run against a synchronous PH hub, where all sources
of a field always carry the same write_id -- so strict and relaxed behave
identically there and cannot distinguish the policies. These tests pin the
policy logic directly: strict rejects a mixed-iteration read, relaxed accepts
the floor."""

import unittest

from mpisppy.cylinders.spcommunicator import (
    reduce_source_write_ids,
    _STRICT_COHERENCE_FIELDS,
)
from mpisppy.cylinders.spwindow import Field


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


if __name__ == "__main__":
    unittest.main()
