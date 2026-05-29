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


if __name__ == "__main__":
    unittest.main()
