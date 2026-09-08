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

import unittest

import numpy as np

from mpisppy import MPI
from mpisppy.cylinders.spwindow import Field, SPWindow, padded_len_n_doubles


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


class _RecordingWindow:
    def __init__(self):
        self.calls = []

    def Lock(self, rank, lock_type):
        self.calls.append(("Lock", rank, lock_type))

    def Sync(self):
        self.calls.append(("Sync",))

    def Unlock(self, rank):
        self.calls.append(("Unlock", rank))

    def Lock_all(self):
        self.calls.append(("Lock_all",))

    def Get(self, origin, rank, disp):
        self.calls.append(("Get", rank, disp, origin[1], origin[2]))

    def Flush(self, rank):
        self.calls.append(("Flush", rank))

    def Unlock_all(self):
        self.calls.append(("Unlock_all",))

    def Put(self, *args):
        self.calls.append(("Put",))


class TestRMAProtocol(unittest.TestCase):
    def setUp(self):
        self.logical = 3
        self.padded = padded_len_n_doubles(self.logical)
        self.layout = {
            Field.NONANTS_VALS: (0, self.logical, self.padded),
            Field.WHOLE: (0, self.logical, self.padded),
        }
        self.win = SPWindow.__new__(SPWindow)
        self.win.strata_rank = 2
        self.win.buffer_layout = self.layout
        self.win.strata_buffer_layouts = [self.layout] * 4
        self.win.buff = np.full(self.padded, np.nan, dtype="d")
        self.win.window = _RecordingWindow()

    def test_put_uses_local_store_in_exclusive_self_epoch(self):
        values = np.arange(self.padded, dtype="d")

        self.win.put(values, Field.NONANTS_VALS)

        np.testing.assert_array_equal(self.win.buff, values)
        self.assertEqual(
            self.win.window.calls,
            [
                ("Lock", 2, MPI.LOCK_EXCLUSIVE),
                ("Sync",),
                ("Unlock", 2),
            ],
        )
        self.assertNotIn(("Put",), self.win.window.calls)

    def test_get_completes_short_lock_all_epoch(self):
        dest = np.empty(self.padded, dtype="d")

        self.win.get(dest, 3, Field.NONANTS_VALS)

        self.assertEqual(
            self.win.window.calls,
            [
                ("Lock_all",),
                ("Get", 3, 0, self.padded, MPI.DOUBLE),
                ("Flush", 3),
                ("Unlock_all",),
            ],
        )


if __name__ == "__main__":
    unittest.main()
