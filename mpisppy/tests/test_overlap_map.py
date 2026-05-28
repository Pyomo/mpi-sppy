###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unit tests for mpisppy.cylinders.overlap_map.compute_overlap_segments."""

import unittest

from mpisppy.cylinders.overlap_map import OverlapSegment, compute_overlap_segments


def _seg(remote_rank, remote_offset, local_offset, count):
    return OverlapSegment(remote_rank, remote_offset, local_offset, count)


class TestOverlapMap(unittest.TestCase):

    def _assert_tiles_local(self, segments, local_scen_idxs, items_per_scen):
        # Segments must tile the local buffer contiguously from offset 0,
        # with total length equal to the local rank's item count.
        expected_local = sum(items_per_scen[s] for s in local_scen_idxs)
        running = 0
        for seg in segments:
            self.assertEqual(seg.local_offset, running)
            running += seg.count
        self.assertEqual(running, expected_local)

    def test_equal_ranks_is_identity_single_segment(self):
        # Same rank count on both cylinders -> one identity segment per
        # local rank (the no-op-at-equal-ranks backbone).
        remote_slices = [[0, 1], [2, 3], [4, 5], [6, 7]]
        items = [1] * 8
        for rank, local in enumerate(remote_slices):
            segs = compute_overlap_segments(local, remote_slices, items)
            self.assertEqual(segs, [_seg(rank, 0, 0, len(local))])

    def test_equal_ranks_identity_multi_item(self):
        # Identity holds with >1 item per scenario.
        remote_slices = [[0, 1], [2, 3]]
        items = [2] * 4
        segs = compute_overlap_segments([2, 3], remote_slices, items)
        self.assertEqual(segs, [_seg(1, 0, 0, 4)])

    def test_fewer_remote_ranks_doc_example(self):
        # 4-rank hub reading a 2-rank spoke, 10 scenarios, 1 item each.
        # Spoke split is contiguous: rank0 = scen0-4, rank1 = scen5-9.
        remote_slices = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        items = [1] * 10

        # Hub rank holding scen0,1: wholly inside spoke rank 0.
        self.assertEqual(
            compute_overlap_segments([0, 1], remote_slices, items),
            [_seg(0, 0, 0, 2)],
        )
        # Hub rank holding scen4,5: straddles the spoke split -> two
        # segments (scen4 from spoke rank0 @offset4, scen5 from spoke
        # rank1 @offset0).
        self.assertEqual(
            compute_overlap_segments([4, 5], remote_slices, items),
            [_seg(0, 4, 0, 1), _seg(1, 0, 1, 1)],
        )
        # Hub rank holding scen7,8,9: wholly inside spoke rank1, coalesced.
        self.assertEqual(
            compute_overlap_segments([7, 8, 9], remote_slices, items),
            [_seg(1, 2, 0, 3)],
        )

    def test_more_remote_ranks(self):
        # 2-rank local cylinder reading a 4-rank remote cylinder.
        remote_slices = [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]
        items = [1] * 10
        segs = compute_overlap_segments([0, 1, 2, 3, 4], remote_slices, items)
        self.assertEqual(segs, [_seg(0, 0, 0, 2), _seg(1, 0, 2, 3)])
        self._assert_tiles_local(segs, [0, 1, 2, 3, 4], items)

    def test_coalesces_contiguous_run_within_remote_rank(self):
        remote_slices = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        items = [1] * 10
        segs = compute_overlap_segments([2, 3, 4], remote_slices, items)
        self.assertEqual(segs, [_seg(0, 2, 0, 3)])

    def test_variable_items_per_scenario(self):
        # Multistage-style: scenarios contribute different item counts.
        items = [1, 2, 3, 1, 2]
        remote_slices = [[0, 1, 2], [3, 4]]
        # Local rank holds scen1,2 (both in remote rank0). scen1 starts at
        # remote offset items[0]=1; coalesced count = 2+3 = 5.
        segs = compute_overlap_segments([1, 2], remote_slices, items)
        self.assertEqual(segs, [_seg(0, 1, 0, 5)])
        self._assert_tiles_local(segs, [1, 2], items)

    def test_variable_items_across_remote_ranks(self):
        items = [1, 2, 3, 1, 2]
        remote_slices = [[0, 1], [2, 3, 4]]
        # Local holds scen1 (remote rank0 @offset1, count2) then scen2
        # (remote rank1 @offset0, count3).
        segs = compute_overlap_segments([1, 2], remote_slices, items)
        self.assertEqual(segs, [_seg(0, 1, 0, 2), _seg(1, 0, 2, 3)])
        self._assert_tiles_local(segs, [1, 2], items)

    def test_single_scenario(self):
        remote_slices = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        items = [1] * 10
        self.assertEqual(
            compute_overlap_segments([6], remote_slices, items),
            [_seg(1, 1, 0, 1)],
        )


if __name__ == "__main__":
    unittest.main()
