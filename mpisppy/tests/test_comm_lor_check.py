###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy.debug_utils.comm_lor_check.

The all-zero path is exercised with the real (mock) MPI.COMM_WORLD. The
non-zero path is exercised with a fake comm whose Allreduce writes 1 into
the recv buffer, so the failure-print code path runs without changing the
function under test.
"""

import contextlib
import io
import unittest

from mpisppy import MPI
from mpisppy.debug_utils import comm_lor_check


class _FakeNonzeroComm:
    """Stand-in for an MPI comm that forces the LOR result to 1.

    Get_rank() returns 0 so START/STOP also print, letting the test confirm
    both the announcement and the failure-print run in the same call.
    """

    def Get_rank(self):
        return 0

    def Allreduce(self, sendbuf, recvbuf, op=None):
        recvbuf[0] = 1


class TestCommLorCheck(unittest.TestCase):

    def test_prints_start_and_stop_with_comm_name(self):
        # Single-rank MPI.COMM_WORLD works for both mpi4py and the mock shim.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            comm_lor_check(MPI.COMM_WORLD, "world")
        out = buf.getvalue()
        self.assertIn("START", out)
        self.assertIn("STOP", out)
        self.assertIn("'world'", out)

    def test_no_nonzero_announcement_and_no_stack_dump(self):
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), \
             contextlib.redirect_stderr(err_buf):
            comm_lor_check(MPI.COMM_WORLD, "world")
        # NONZERO header goes to stdout; stack trace goes to stderr.
        self.assertNotIn("NONZERO", out_buf.getvalue())
        self.assertNotIn('File "', err_buf.getvalue())

    def test_nonzero_path_prints_value_name_and_stack_dump(self):
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), \
             contextlib.redirect_stderr(err_buf):
            comm_lor_check(_FakeNonzeroComm(), "broken")
        out = out_buf.getvalue()
        err = err_buf.getvalue()
        self.assertIn("NONZERO", out)
        self.assertIn("value=1", out)
        self.assertIn("'broken'", out)
        self.assertIn("rank=0", out)
        # traceback.print_stack writes 'File "..."' frames to stderr.
        self.assertIn('File "', err)
        # START/STOP still announced on this rank.
        self.assertIn("START", out)
        self.assertIn("STOP", out)


if __name__ == "__main__":
    unittest.main()
