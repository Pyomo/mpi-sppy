###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Partial-read support in SPWindow.get(), exercised on a single-process
RMA window (MPI.COMM_SELF). Verifies the default whole-field read is
unchanged and that item_offset/item_count read the right sub-range."""

import struct
import unittest
from unittest import mock

import numpy as np

from mpisppy import MPI
from mpisppy.cylinders.spwindow import (
    Field,
    PUBLICATION_MAX_GENERATION,
    SPWindow,
    padded_len_n_doubles,
    transmitted_canary,
)


class _RecordingWindow:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls = []

    def Put(self, *args):
        self.calls.append(("Put",))
        return self.wrapped.Put(*args)

    def Get(self, *args):
        self.calls.append(("Get",))
        return self.wrapped.Get(*args)

    def Flush(self, rank):
        self.calls.append(("Flush", rank))
        return self.wrapped.Flush(rank)

    def Sync(self):
        self.calls.append(("Sync",))
        return self.wrapped.Sync()

    def Unlock_all(self):
        self.calls.append(("Unlock_all",))
        return self.wrapped.Unlock_all()

    def Free(self):
        self.calls.append(("Free",))
        return self.wrapped.Free()

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class TestPartialGet(unittest.TestCase):

    def setUp(self):
        self.logical = 10
        self.padded = padded_len_n_doubles(self.logical)
        my_fields = {Field.NONANTS_VALS: (self.logical, self.padded)}
        self.win = SPWindow(my_fields, MPI.COMM_SELF)
        # Known, distinct values across the whole padded field.
        self.data = np.arange(100, 100 + self.padded, dtype="d")
        self.win.put(self.data, Field.NONANTS_VALS)

    def tearDown(self):
        self.win.free()

    def test_full_read_unchanged(self):
        # Default args: whole padded field, as before.
        dest = np.empty(self.padded, dtype="d")
        self.win.get(dest, 0, Field.NONANTS_VALS)
        np.testing.assert_array_equal(dest, self.data)

    def test_get_retries_when_publication_overlaps_snapshot(self):
        # The first payload is rejected because its bracketing metadata
        # differs. The second attempt observes one clean generation.
        metadata = [
            (0, 1),
            (1, 1),
            (0, 2),
            (0, 2),
        ]
        dest = np.empty(self.padded, dtype="d")
        with mock.patch.object(
                self.win, "_get_publication_metadata",
                side_effect=metadata) as get_metadata:
            self.win.get(dest, 0, Field.NONANTS_VALS)

        self.assertEqual(get_metadata.call_count, 4)
        np.testing.assert_array_equal(dest, self.data)

    def test_partial_prefix(self):
        dest = np.empty(3, dtype="d")
        self.win.get(dest, 0, Field.NONANTS_VALS, item_offset=0, item_count=3)
        np.testing.assert_array_equal(dest, self.data[0:3])

    def test_partial_middle(self):
        dest = np.empty(4, dtype="d")
        self.win.get(dest, 0, Field.NONANTS_VALS, item_offset=2, item_count=4)
        np.testing.assert_array_equal(dest, self.data[2:6])

    def test_partial_suffix(self):
        dest = np.empty(5, dtype="d")
        self.win.get(dest, 0, Field.NONANTS_VALS,
                     item_offset=self.padded - 5, item_count=5)
        np.testing.assert_array_equal(dest, self.data[self.padded - 5:])

    def test_partial_full_via_count(self):
        # item_count equal to padded length reproduces the full read.
        dest = np.empty(self.padded, dtype="d")
        self.win.get(dest, 0, Field.NONANTS_VALS,
                     item_offset=0, item_count=self.padded)
        np.testing.assert_array_equal(dest, self.data)

    def test_single_item(self):
        dest = np.empty(1, dtype="d")
        self.win.get(dest, 0, Field.NONANTS_VALS, item_offset=7, item_count=1)
        self.assertEqual(dest[0], self.data[7])

    def test_out_of_range_raises(self):
        dest = np.empty(2, dtype="d")
        with self.assertRaises(AssertionError):
            self.win.get(dest, 0, Field.NONANTS_VALS,
                         item_offset=self.padded - 1, item_count=2)

    def test_dest_size_mismatch_raises(self):
        dest = np.empty(5, dtype="d")  # wrong size for count=3
        with self.assertRaises(AssertionError):
            self.win.get(dest, 0, Field.NONANTS_VALS, item_offset=0, item_count=3)

    def test_get_detects_corrupt_transmitted_canary(self):
        layout = self.win.buffer_layout[Field.NONANTS_VALS]
        offset = layout.left_canary_offset_bytes
        self.win.buff[offset] ^= 0xff
        try:
            with self.assertRaisesRegex(
                    RuntimeError, "left-transmitted-canary"):
                self.win.get(
                    np.empty(self.padded, dtype="d"),
                    0,
                    Field.NONANTS_VALS,
                )
        finally:
            self.win._set_guard_bytes(Field.NONANTS_VALS, layout)

    def test_put_detects_corrupt_red_zone(self):
        layout = self.win.buffer_layout[Field.NONANTS_VALS]
        offset = layout.right_red_zone_offset_bytes
        self.win.buff[offset] ^= 0xff
        try:
            with self.assertRaisesRegex(RuntimeError, "right-red-zone"):
                self.win.put(self.data, Field.NONANTS_VALS)
        finally:
            self.win._set_guard_bytes(Field.NONANTS_VALS, layout)

    def test_transmitted_canaries_identify_record(self):
        layout = self.win.buffer_layout[Field.NONANTS_VALS]
        baseline = transmitted_canary(
            0, Field.NONANTS_VALS, "left", layout.padded_len)

        self.assertNotEqual(
            baseline,
            transmitted_canary(
                1, Field.NONANTS_VALS, "left", layout.padded_len),
        )
        self.assertNotEqual(
            baseline,
            transmitted_canary(
                0, Field.DUALS, "left", layout.padded_len),
        )
        self.assertNotEqual(
            baseline,
            transmitted_canary(
                0, Field.NONANTS_VALS, "right", layout.padded_len),
        )
        self.assertNotEqual(
            baseline,
            transmitted_canary(
                0, Field.NONANTS_VALS, "left", layout.padded_len + 1),
        )

    def test_put_and_get_use_publication_protocol(self):
        recording = _RecordingWindow(self.win.window)
        self.win.window = recording

        self.win.put(self.data, Field.NONANTS_VALS)
        self.win.get(
            np.empty(self.padded, dtype="d"), 0, Field.NONANTS_VALS)

        self.assertEqual(recording.calls, [
            # Publication: local busy, payload, generation, and idle stores;
            # each Sync publishes one stage before the next starts.
            ("Sync",),
            ("Sync",),
            ("Sync",),
            ("Sync",),
            # Get: metadata, payload, metadata.
            ("Get",),
            ("Flush", 0),
            ("Get",),
            ("Flush", 0),
            ("Get",),
            ("Flush", 0),
        ])

    def test_put_advances_generation_and_leaves_record_idle(self):
        layout = self.win.buffer_layout[Field.NONANTS_VALS]
        metadata = self.win.buff[
            layout.metadata_offset_bytes:
            layout.metadata_offset_bytes + 16
        ]
        busy_before, generation_before = struct.unpack("<B7xQ", metadata)

        self.win.put(self.data, Field.NONANTS_VALS)

        busy_after, generation_after = struct.unpack("<B7xQ", metadata)
        self.assertEqual(busy_before, 0)
        self.assertEqual(busy_after, 0)
        self.assertEqual(generation_after, generation_before + 1)

    def test_generation_overflow_fails_before_opening_publication(self):
        field = Field.NONANTS_VALS
        layout = self.win.buffer_layout[field]
        metadata = self.win.buff[
            layout.metadata_offset_bytes:
            layout.metadata_offset_bytes + 16
        ]
        metadata_before = bytes(metadata)
        self.win._publication_generations[field] = \
            PUBLICATION_MAX_GENERATION

        with self.assertRaisesRegex(
                OverflowError, "publication generation exhausted"):
            self.win.put(self.data, field)

        self.assertEqual(bytes(metadata), metadata_before)

    def test_free_closes_epoch_before_freeing_window(self):
        recording = _RecordingWindow(self.win.window)
        self.win.window = recording

        self.win.free()

        self.assertEqual(recording.calls, [
            ("Sync",),
            ("Unlock_all",),
            ("Free",),
        ])


if __name__ == "__main__":
    unittest.main()
