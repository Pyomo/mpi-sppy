###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import unittest

from mpisppy.spin_the_wheel import (
    _format_equal_window_distribution,
    _format_flexible_window_distribution,
)


class TestWindowDistributionMessages(unittest.TestCase):
    def test_equal_cylinders_summarize_distinct_strata_windows(self):
        records = (
            [(0, ("node-a",))] * 4
            + [(1, ("node-a", "node-b"))] * 4
            + [(2, ("node-b",))] * 4
        )

        message = _format_equal_window_distribution(records, n_cylinders=4)

        self.assertIn("3 strata windows (4 ranks each", message)
        self.assertIn("2 node-local and 1 cross-node", message)
        self.assertIn("1 node: 2, 2 nodes: 1", message)

    def test_flexible_cylinders_report_full_world_window(self):
        message = _format_flexible_window_distribution(
            ["node-a", "node-a", "node-b", "node-c"], [2, 1, 1])

        self.assertIn("one full-world window with 4 ranks", message)
        self.assertIn("spanning 3 nodes", message)
        self.assertIn("per-cylinder rank counts [2, 1, 1]", message)


if __name__ == "__main__":
    unittest.main()
