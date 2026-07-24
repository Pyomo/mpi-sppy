###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Serial tests for the console-script wrappers in mpisppy.entry_points.

import io
import contextlib
import unittest

import mpisppy.MPI as MPI
import mpisppy.entry_points as entry_points


class _FakeComm:
    """Stands in for MPI.COMM_WORLD so no test ever really aborts."""
    def __init__(self, size):
        self._size = size
        self.abort_code = None

    def Get_size(self):
        return self._size

    def Abort(self, errorcode=0):
        self.abort_code = errorcode


class _FakeCommNoAbort:
    """Mimics the no-mpi4py mock comm, which has no Abort method."""
    def __init__(self, size):
        self._size = size

    def Get_size(self):
        return self._size


class TestEntryPoints(unittest.TestCase):

    def setUp(self):
        self._saved_comm = MPI.COMM_WORLD

    def tearDown(self):
        MPI.COMM_WORLD = self._saved_comm

    def _run_failing_main(self, comm, exc=ValueError):
        MPI.COMM_WORLD = comm
        def failing_main():
            raise exc("boom")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(exc):
                entry_points._run_with_mpi_abort(failing_main)

    def test_serial_reraises_without_abort(self):
        comm = _FakeComm(1)
        self._run_failing_main(comm)
        self.assertIsNone(comm.abort_code)

    def test_multirank_aborts(self):
        comm = _FakeComm(3)
        self._run_failing_main(comm)
        self.assertEqual(comm.abort_code, 1)

    def test_mock_comm_without_abort_reraises(self):
        self._run_failing_main(_FakeCommNoAbort(3))

    def test_system_exit_passes_through(self):
        comm = _FakeComm(3)
        self._run_failing_main(comm, exc=SystemExit)
        self.assertIsNone(comm.abort_code)

    def test_successful_main_runs_once(self):
        MPI.COMM_WORLD = _FakeComm(3)
        calls = []
        entry_points._run_with_mpi_abort(lambda: calls.append(1))
        self.assertEqual(calls, [1])

    def test_console_script_targets_exist(self):
        # the callables named in pyproject.toml [project.scripts]
        for name in ("generic_cylinders_main",
                     "mrp_generic_main",
                     "one_sided_test_main"):
            self.assertTrue(callable(getattr(entry_points, name)))


if __name__ == "__main__":
    unittest.main()
