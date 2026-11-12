###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import contextlib
import io
import unittest
from unittest import mock

from mpisppy import MPI
from mpisppy.spin_the_wheel import (
    _format_equal_window_distribution,
    _format_flexible_window_distribution,
    _require_multinode_progress,
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


class _FakeComm:
    def __init__(self, hosts, *, rank=0, all_ready=None):
        self.hosts = hosts
        self.rank = rank
        self.all_ready = all_ready
        self.aborted = None
        self.barrier_called = False

    def Get_size(self):
        return len(self.hosts)

    def Get_rank(self):
        return self.rank

    def allgather(self, value):
        return self.hosts

    def allreduce(self, value, op=None):
        return value if self.all_ready is None else self.all_ready

    def Barrier(self):
        self.barrier_called = True

    def Abort(self, code):
        self.aborted = code


class TestMultinodeProgressGuard(unittest.TestCase):
    def _check(self, comm, available):
        with (
            mock.patch.object(MPI, "Get_processor_name", return_value="node-a", create=True),
            mock.patch.object(MPI, "progress_available", available),
        ):
            _require_multinode_progress(comm)

    def test_single_node_does_not_require_progress(self):
        for hosts in (("node-a",), ("node-a", "node-a")):
            with self.subTest(hosts=hosts):
                comm = _FakeComm(hosts)
                self._check(comm, False)
                self.assertIsNone(comm.aborted)

    def test_multinode_with_progress_on_every_rank_continues(self):
        comm = _FakeComm(("node-a", "node-b"))
        self._check(comm, True)
        self.assertIsNone(comm.aborted)

    def test_multinode_without_progress_prints_and_aborts(self):
        comm = _FakeComm(("node-a", "node-b"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(
            RuntimeError, "MPI ranks span multiple nodes"
        ):
            self._check(comm, False)
        self.assertIn("MPISPPY_MPI_PROGRESS_THREAD=1", output.getvalue())
        self.assertIn("MVAPICH_ASYNC_PROGRESS=1", output.getvalue())
        self.assertTrue(comm.barrier_called)
        self.assertEqual(comm.aborted, 1)

    def test_one_rank_without_progress_aborts_everyone(self):
        comm = _FakeComm(("node-a", "node-b"), rank=1, all_ready=0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(RuntimeError):
            self._check(comm, True)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(comm.aborted, 1)


if __name__ == "__main__":
    unittest.main()
