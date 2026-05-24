###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unit tests for mpisppy.utils.rank_apportionment.apportion_ranks."""

import unittest

from mpisppy.utils.rank_apportionment import apportion_ranks


class TestApportionRanks(unittest.TestCase):

    def test_equal_ratios_divisible(self):
        # Clean split when ranks divide evenly among equal cylinders.
        self.assertEqual(apportion_ranks([1.0, 1.0, 1.0], 9), [3, 3, 3])

    def test_equal_ratios_indivisible_breaks_ties_by_order(self):
        # 10 ranks, 3 equal cylinders: each floors to 3 (sum 9), and the
        # single leftover goes to the lowest-index cylinder on the tie.
        self.assertEqual(apportion_ranks([1.0, 1.0, 1.0], 10), [4, 3, 3])

    def test_two_equal_cylinders_tie(self):
        # 3 ranks, 2 equal cylinders: shares 1.5 / 1.5, leftover -> idx 0.
        self.assertEqual(apportion_ranks([1.0, 1.0], 3), [2, 1])

    def test_design_doc_example(self):
        # The worked example from the design doc: hub 1.0, lagrangian 0.5,
        # xhat 0.25, np 14  ->  8 / 4 / 2 (exact, no remainder).
        self.assertEqual(apportion_ranks([1.0, 0.5, 0.25], 14), [8, 4, 2])

    def test_ratio_scale_invariance(self):
        # Only relative magnitudes matter.
        self.assertEqual(
            apportion_ranks([4.0, 2.0, 1.0], 14),
            apportion_ranks([1.0, 0.5, 0.25], 14),
        )

    def test_floor_of_one_enforced(self):
        # A tiny ratio would round to zero; the floor-of-one pass rescues it
        # by taking a rank from the largest cylinder.
        counts = apportion_ranks([10.0, 0.01], 3)
        self.assertEqual(counts, [2, 1])

    def test_floor_of_one_many_small(self):
        # Several near-zero cylinders all get rescued to 1.
        counts = apportion_ranks([100.0, 0.001, 0.001, 0.001], 7)
        self.assertEqual(sum(counts), 7)
        self.assertTrue(all(c >= 1 for c in counts))
        self.assertEqual(counts[0], 4)  # big cylinder keeps the rest

    def test_cylinders_equal_ranks(self):
        # C == n_proc: everyone gets exactly one.
        self.assertEqual(apportion_ranks([1.0, 2.0, 3.0], 3), [1, 1, 1])

    def test_single_cylinder_gets_all(self):
        self.assertEqual(apportion_ranks([1.0], 5), [5])

    def test_more_cylinders_than_ranks_raises(self):
        with self.assertRaises(ValueError):
            apportion_ranks([1.0, 1.0, 1.0, 1.0], 3)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            apportion_ranks([], 4)
        with self.assertRaises(ValueError):
            apportion_ranks([1.0, 1.0], 0)
        with self.assertRaises(ValueError):
            apportion_ranks([1.0, -1.0], 4)
        with self.assertRaises(ValueError):
            apportion_ranks([1.0, 0.0], 4)

    def test_invariants_over_many_cases(self):
        # Property sweep: counts always sum to n_proc, are all >= 1, and
        # match the cylinder count.
        ratio_pools = [
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 0.25],
            [3.0, 1.0],
            [5.0, 2.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        for ratios in ratio_pools:
            for n_proc in range(len(ratios), len(ratios) + 25):
                counts = apportion_ranks(ratios, n_proc)
                self.assertEqual(len(counts), len(ratios))
                self.assertEqual(sum(counts), n_proc)
                self.assertTrue(all(c >= 1 for c in counts),
                                msg=f"{ratios=} {n_proc=} -> {counts=}")

    def test_monotone_in_ratio(self):
        # With enough ranks, a larger ratio never gets fewer ranks than a
        # smaller one (sanity on the apportionment ordering).
        counts = apportion_ranks([3.0, 2.0, 1.0], 30)
        self.assertGreaterEqual(counts[0], counts[1])
        self.assertGreaterEqual(counts[1], counts[2])

    def test_order_preserved(self):
        # Output position i corresponds to input ratio i.
        a = apportion_ranks([1.0, 0.5, 0.25], 14)
        b = apportion_ranks([0.25, 0.5, 1.0], 14)
        self.assertEqual(a, list(reversed(b)))


if __name__ == "__main__":
    unittest.main()
