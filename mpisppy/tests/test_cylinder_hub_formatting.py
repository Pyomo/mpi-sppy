###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for adaptive trace formatting in cylinders.hub."""

import unittest

from mpisppy.cylinders.hub import Hub


class TestTraceFormatting(unittest.TestCase):
    def test_fixed_format_is_preserved(self):
        self.assertEqual(Hub._format_trace_value(123.4567, 14, 4), "      123.4567")

    def test_small_values_switch_to_scientific(self):
        self.assertIn("e", Hub._format_trace_value(1.2e-8, 14, 4))
        self.assertIn("e", Hub._format_trace_value(-4.9e-5, 14, 4))

    def test_large_values_switch_to_scientific(self):
        self.assertIn("e", Hub._format_trace_value(1234567890.0, 14, 4))


if __name__ == "__main__":
    unittest.main()
