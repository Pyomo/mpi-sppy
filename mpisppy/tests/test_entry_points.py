###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Serial tests for the console-script wrappers in mpisppy.entry_points.

import contextlib
import io
import sys
import unittest
from unittest import mock

import mpisppy.MPI as MPI
import mpisppy.entry_points as entry_points
import mpisppy.utils.mpi_abort as mpi_abort

try:
    import mpi4py  # noqa: F401
    have_mpi4py = True
except ImportError:
    have_mpi4py = False


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
    """What the console scripts install, and when it declines to.

    The mechanism itself is mpi4py's: an excepthook that hands the exception
    to ``mpi4py.run.set_abort_status``, which lets interpreter exit call
    ``COMM_WORLD.Abort``. That deferral cannot be observed in-process, so
    what a real job does is pinned by ``test_mpi_abort.py``, which spawns
    one. These cover the decisions made *before* handing over: whether to
    install at all, and doing it once.
    """

    def setUp(self):
        self._saved_comm = MPI.COMM_WORLD
        self._saved_hook = sys.excepthook
        self._saved_installed = mpi_abort._installed
        mpi_abort._installed = False

    def tearDown(self):
        MPI.COMM_WORLD = self._saved_comm
        sys.excepthook = self._saved_hook
        mpi_abort._installed = self._saved_installed

    def test_a_multirank_job_gets_the_hook(self):
        MPI.COMM_WORLD = _FakeComm(3)
        self.assertTrue(mpi_abort.abort_on_uncaught_exception())
        self.assertIsNot(sys.excepthook, self._saved_hook)

    def test_a_serial_job_is_left_alone(self):
        """A traceback and an exit code already say everything an abort
        would, and say it more clearly."""
        MPI.COMM_WORLD = _FakeComm(1)
        self.assertFalse(mpi_abort.abort_on_uncaught_exception())
        self.assertIs(sys.excepthook, self._saved_hook)

    def test_without_mpi4py_nothing_is_installed(self):
        MPI.COMM_WORLD = _FakeCommNoAbort(3)
        with mock.patch.object(mpi_abort, "haveMPI", False):
            self.assertFalse(mpi_abort.abort_on_uncaught_exception())
        self.assertIs(sys.excepthook, self._saved_hook)

    def test_a_comm_that_cannot_be_asked_is_not_fatal(self):
        """The caller's own exception is what should be reported, not one
        raised while deciding whether to report it."""
        class _Unusable:
            def Get_size(self):
                raise RuntimeError("MPI_COMM_NULL")
        MPI.COMM_WORLD = _Unusable()
        self.assertFalse(mpi_abort.abort_on_uncaught_exception())
        self.assertIs(sys.excepthook, self._saved_hook)

    def test_installing_twice_does_not_stack_hooks(self):
        MPI.COMM_WORLD = _FakeComm(3)
        mpi_abort.abort_on_uncaught_exception()
        once = sys.excepthook
        mpi_abort.abort_on_uncaught_exception()
        self.assertIs(sys.excepthook, once)

    def test_the_hook_records_the_status_and_still_reports(self):
        """Both halves matter: without the status the job hangs, without
        the report nobody learns why it died."""
        MPI.COMM_WORLD = _FakeComm(3)
        reported = []
        sys.excepthook = lambda t, e, tb: reported.append(e)
        recorded = []
        # Patched around the install, not just the call: the hook binds
        # set_abort_status when it is installed. And the real one would set
        # a status that aborts *this* process at interpreter exit.
        with mock.patch("mpi4py.run.set_abort_status", recorded.append):
            mpi_abort.abort_on_uncaught_exception()
            boom = ValueError("boom")
            sys.excepthook(ValueError, boom, None)
        self.assertEqual(recorded, [boom])
        self.assertEqual(reported, [boom])

    def test_the_status_is_recorded_even_if_reporting_fails(self):
        MPI.COMM_WORLD = _FakeComm(3)
        def _broken_hook(t, e, tb):
            raise RuntimeError("the reporter itself failed")
        sys.excepthook = _broken_hook
        recorded = []
        with mock.patch("mpi4py.run.set_abort_status", recorded.append):
            mpi_abort.abort_on_uncaught_exception()
            with self.assertRaises(RuntimeError):
                sys.excepthook(ValueError, ValueError("boom"), None)
        self.assertEqual(len(recorded), 1,
                         msg="the job would hang: no abort status was set")

    def test_the_wrappers_install_it_before_importing_the_target(self):
        """A failure during the target's import need not strike every rank
        -- a flaky shared filesystem -- so the hook has to be in place
        before that import runs."""
        MPI.COMM_WORLD = _FakeComm(3)
        seen = []
        entry_points._run_with_mpi_abort(
            lambda: seen.append(sys.excepthook is not self._saved_hook))
        self.assertEqual(seen, [True])

    def test_console_script_targets_exist(self):
        # the callables named in pyproject.toml [project.scripts]
        for name in ("generic_cylinders_main",
                     "mrp_generic_main",
                     "one_sided_test_main"):
            self.assertTrue(callable(getattr(entry_points, name)))


class TestConsoleScriptWrappers(unittest.TestCase):
    """Run each real console-script wrapper end-to-end on its cheap serial
    path: with no arguments (and one rank) every wrapped main prints a usage
    message and raises SystemExit, which the wrapper must pass through. This
    executes the wrappers' real imports, so a broken import target (e.g. the
    repo-root ``mpi_one_sided_test`` py-module going missing from the install)
    fails here instead of on a user's command line."""

    def _run_wrapper(self, wrapper, prog):
        saved_argv = sys.argv
        sys.argv = [prog]
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    wrapper()
            return caught.exception.code
        finally:
            sys.argv = saved_argv

    def test_generic_cylinders_usage_exit(self):
        # no args -> generic_cylinders.main prints usage and quit()s
        code = self._run_wrapper(entry_points.generic_cylinders_main,
                                 "mpi-sppy-generic-cylinders")
        self.assertIn(code, (None, 0))

    def test_mrp_generic_usage_exit(self):
        # no args -> mrp_generic.main prints usage and quit()s
        code = self._run_wrapper(entry_points.mrp_generic_main,
                                 "mpi-sppy-mrp-generic")
        self.assertIn(code, (None, 0))

    @unittest.skipUnless(have_mpi4py, "mpi_one_sided_test imports mpi4py directly")
    @unittest.skipUnless(MPI.COMM_WORLD.Get_size() == 1,
                         "at more than one rank the one-sided test really runs")
    def test_one_sided_test_single_rank_exit(self):
        # at one rank the script demands an mpiexec launch and exits 2
        code = self._run_wrapper(entry_points.one_sided_test_main,
                                 "mpi-sppy-one-sided-test")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
