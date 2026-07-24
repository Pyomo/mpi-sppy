###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Direct test of the unequal-rank two-level buffer-layout exchange.

``two_level_layout_exchange`` replaces the flat ``fullcomm.allgather``
layout exchange on the unequal-rank path (an O(N) startup collective that
must not be the exchange at total rank counts in the thousands) with an
allgather within each cylinder, an allgather across one anchor rank per
cylinder, and a within-cylinder broadcast.  Its contract is that the
result is *identical* to what ``fullcomm.allgather`` would have produced
-- a list of every rank's layout indexed by global rank -- so this test
pins the two against each other, with rank-unique payloads (misrouting
any rank's layout breaks equality) across several cylinder partitions,
including single-rank cylinders at either end and a single-cylinder run.

mpiexec -np 6 python -m mpi4py -m pytest mpisppy/tests/test_flex_layout_exchange.py
"""

import unittest

from mpi4py import MPI

from mpisppy.cylinders.spcommunicator import two_level_layout_exchange
from mpisppy.utils.rank_apportionment import cylinder_bases, rank_to_cylinder

comm = MPI.COMM_WORLD


@unittest.skipUnless(comm.size == 6, "needs exactly 6 MPI ranks")
class TestTwoLevelLayoutExchange(unittest.TestCase):

    PARTITIONS = (
        [6],
        [1, 5],
        [5, 1],
        [4, 2],
        [2, 4],
        [3, 3],
        [2, 2, 2],
        [1, 2, 3],
        [3, 2, 1],
        [1, 1, 1, 1, 1, 1],
    )

    def _exchange_matches_allgather(self, rank_counts):
        global_rank = comm.Get_rank()
        cylinder_index, _ = rank_to_cylinder(global_rank, rank_counts)
        # same split spin_the_wheel uses on the unequal-rank path
        cylinder_comm = comm.Split(color=cylinder_index, key=global_rank)
        try:
            # a rank-unique picklable payload standing in for the layout dict
            my_layout = {
                "global_rank": global_rank,
                "cylinder": cylinder_index,
                "data": list(range(global_rank + 1)),
            }
            result = two_level_layout_exchange(
                my_layout, comm, cylinder_comm, cylinder_bases(rank_counts)
            )
            self.assertEqual(result, comm.allgather(my_layout))
        finally:
            cylinder_comm.Free()

    def test_matches_allgather_across_partitions(self):
        for rank_counts in self.PARTITIONS:
            with self.subTest(rank_counts=rank_counts):
                self._exchange_matches_allgather(rank_counts)


if __name__ == "__main__":
    unittest.main()
