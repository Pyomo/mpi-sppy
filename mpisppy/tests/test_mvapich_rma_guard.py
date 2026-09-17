###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import os
import unittest
import warnings
from unittest import mock

from mpisppy.cylinders import spcommunicator


class _FakeComm:
    def __init__(self, hosts=(), reduced_result=None, broadcast_result=None):
        self.hosts = list(hosts)
        self.reduced_result = reduced_result
        self.broadcast_result = broadcast_result
        self.allgather_calls = 0

    def allgather(self, value):
        self.allgather_calls += 1
        return self.hosts

    def allreduce(self, value, op=None):
        return value if self.reduced_result is None else self.reduced_result

    def bcast(self, value, root=0):
        return value if self.broadcast_result is None else self.broadcast_result


class TestMVAPICHRMAGuard(unittest.TestCase):
    def _vendor(self, name="MVAPICH", version=(2, 3, 7)):
        return mock.patch.object(
            spcommunicator.MPI,
            "get_vendor",
            return_value=(name, version),
        )

    def _processor_name(self, name="node-a"):
        return mock.patch.object(
            spcommunicator.MPI,
            "Get_processor_name",
            return_value=name,
        )

    def test_other_mpi_vendor_is_not_checked(self):
        window_comm = _FakeComm()
        with self._vendor("Open MPI", (5, 0, 0)), self._processor_name():
            spcommunicator._guard_mvapich_cross_node_rma(
                window_comm, _FakeComm(), 0)
        self.assertEqual(window_comm.allgather_calls, 0)

    def test_newer_mvapich_version_is_not_checked(self):
        window_comm = _FakeComm()
        with self._vendor(version=(2, 3, 8)), self._processor_name():
            spcommunicator._guard_mvapich_cross_node_rma(
                window_comm, _FakeComm(), 0)
        self.assertEqual(window_comm.allgather_calls, 0)

    def test_older_mvapich_cross_node_window_is_rejected(self):
        window_comm = _FakeComm(("node-a", "node-b"))
        with self._vendor(version=(2, 3, 6)), self._processor_name(), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(
                spcommunicator._ALLOW_UNSAFE_MVAPICH_RMA_ENV, None)
            with self.assertRaisesRegex(
                    RuntimeError, "earlier release has not been tested"):
                spcommunicator._guard_mvapich_cross_node_rma(
                    window_comm, _FakeComm(), 0)

    def test_older_mvapich_can_be_explicitly_overridden(self):
        window_comm = _FakeComm(("node-a", "node-b"))
        with self._vendor(version=(2, 3, 6)), self._processor_name(), \
                mock.patch.dict(os.environ, {
                    spcommunicator._ALLOW_UNSAFE_MVAPICH_RMA_ENV: "1",
                }, clear=False), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spcommunicator._guard_mvapich_cross_node_rma(
                window_comm, _FakeComm(), 0)

        self.assertEqual(len(caught), 1)
        self.assertIn("MVAPICH 2.3.6", str(caught[0].message))

    def test_node_local_window_is_allowed(self):
        window_comm = _FakeComm(("node-a", "node-a"))
        with self._vendor(), self._processor_name():
            spcommunicator._guard_mvapich_cross_node_rma(
                window_comm, _FakeComm(), 0)

    def test_cross_node_window_is_rejected(self):
        window_comm = _FakeComm(("node-a", "node-b"))
        with self._vendor(), self._processor_name(), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(
                spcommunicator._ALLOW_UNSAFE_MVAPICH_RMA_ENV, None)
            with self.assertRaisesRegex(RuntimeError, "cross-node MPI_Get"):
                spcommunicator._guard_mvapich_cross_node_rma(
                    window_comm, _FakeComm(), 0)

    def test_mvapich2_vendor_alias_is_rejected(self):
        window_comm = _FakeComm(("node-a", "node-b"))
        with self._vendor(name="MVAPICH2"), self._processor_name(), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(
                spcommunicator._ALLOW_UNSAFE_MVAPICH_RMA_ENV, None)
            with self.assertRaisesRegex(RuntimeError, "cross-node MPI_Get"):
                spcommunicator._guard_mvapich_cross_node_rma(
                    window_comm, _FakeComm(), 0)

    def test_cross_node_result_is_shared_across_fullcomm(self):
        window_comm = _FakeComm(("node-a", "node-a"))
        fullcomm = _FakeComm(reduced_result=True)
        with self._vendor(), self._processor_name(), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(
                spcommunicator._ALLOW_UNSAFE_MVAPICH_RMA_ENV, None)
            with self.assertRaises(RuntimeError):
                spcommunicator._guard_mvapich_cross_node_rma(
                    window_comm, fullcomm, 0)

    def test_explicit_override_warns_and_continues(self):
        window_comm = _FakeComm(("node-a", "node-b"))
        with self._vendor(), self._processor_name(), mock.patch.dict(
                os.environ,
                {spcommunicator._ALLOW_UNSAFE_MVAPICH_RMA_ENV: "1"},
                clear=False,
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spcommunicator._guard_mvapich_cross_node_rma(
                window_comm, _FakeComm(), 0, flexible_ranks=True)

        self.assertEqual(len(caught), 1)
        self.assertIn("Unequal-rank cylinders", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
