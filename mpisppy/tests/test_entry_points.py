###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Serial tests for the console scripts in mpisppy.entry_points and for the
# abort hook in mpisppy.utils.mpi_abort that importing mpi-sppy installs.

import contextlib
import io
import os
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


class _HostileComm:
    """Any access is a bug: where the installer declines on grounds it can
    check without a communicator, it must not reach for one."""
    def __getattr__(self, name):
        raise AssertionError(f"the installer consulted the comm ({name})")


class _RestoresHookState(unittest.TestCase):
    """Both suites below install hooks and move COMM_WORLD aside."""

    def setUp(self):
        self._saved_comm = MPI.COMM_WORLD
        self._saved_hook = sys.excepthook
        self._saved_installed = mpi_abort._installed_excepthook
        self._saved_announced = mpi_abort._announced_opt_out
        mpi_abort._installed_excepthook = None
        mpi_abort._announced_opt_out = False
        # The docs tell people to export this for sweep-shaped jobs, so a
        # developer or runner that has it set must not see these fail.
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        os.environ.pop(mpi_abort.OPT_OUT_ENVVAR, None)

    def tearDown(self):
        self._env.stop()
        MPI.COMM_WORLD = self._saved_comm
        sys.excepthook = self._saved_hook
        mpi_abort._installed_excepthook = self._saved_installed
        mpi_abort._announced_opt_out = self._saved_announced


class TestWithoutMpi4pyNothingIsInstalled(_RestoresHookState):
    """What must hold in an install with no mpi4py, and so cannot be gated
    on having it. Importing mpi-sppy calls the installer unconditionally, so
    a regression that made it reach for mpi4py or for a communicator would
    break every import of the library in that configuration."""

    def test_nothing_is_installed_and_the_comm_is_not_consulted(self):
        MPI.COMM_WORLD = _HostileComm()
        with mock.patch.object(mpi_abort, "haveMPI", False):
            self.assertFalse(mpi_abort.abort_on_uncaught_exception())
        self.assertIs(sys.excepthook, self._saved_hook)

    def test_the_opt_out_declines_and_announces(self):
        """A job whose ranks are independent turns the abort off, and gets
        no hook however many ranks it has. The announcement is the point:
        the cost of setting this by mistake is a hang."""
        MPI.COMM_WORLD = _FakeComm(3)
        said = []
        with mock.patch.dict(os.environ, {mpi_abort.OPT_OUT_ENVVAR: "1"}), \
                mock.patch("mpisppy.global_toc", said.append):
            self.assertFalse(mpi_abort.abort_on_uncaught_exception())
            self.assertFalse(mpi_abort.abort_on_uncaught_exception())
        self.assertIs(sys.excepthook, self._saved_hook)
        self.assertEqual(len(said), 1, msg="said nothing, or said it twice")
        self.assertIn(mpi_abort.OPT_OUT_ENVVAR, said[0])

    def test_a_value_that_reads_as_off_is_off(self):
        """Exporting the variable empty, or set to something that plainly
        means no, must not disarm the job."""
        for value in ("", "0", "false", "FALSE", "no", "off", "  0  "):
            MPI.COMM_WORLD = _HostileComm()
            said = []
            with mock.patch.dict(os.environ,
                                 {mpi_abort.OPT_OUT_ENVVAR: value}), \
                    mock.patch("mpisppy.global_toc", said.append), \
                    mock.patch.object(mpi_abort, "haveMPI", False):
                # got past the opt-out and declined for the other reason,
                # without reaching for a communicator on the way
                self.assertFalse(mpi_abort.abort_on_uncaught_exception())
            self.assertEqual(said, [], msg=f"{value!r} disarmed the job")


@unittest.skipUnless(have_mpi4py, "the abort hook is mpi4py's mechanism")
class TestAbortHookInstallation(_RestoresHookState):
    """What importing mpi-sppy installs, and when it declines to.

    The mechanism itself is mpi4py's: an excepthook that hands the exception
    to ``mpi4py.run.set_abort_status``, which lets interpreter exit call
    ``COMM_WORLD.Abort``. That deferral cannot be observed in-process, so
    what a real job does is pinned by ``test_mpi_abort.py``, which spawns
    one. These cover the decisions made *before* handing over: whether to
    install at all, and whether the hook is still the one we installed.
    """

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

    def test_a_replaced_hook_is_wrapped_again(self):
        """The reason the module holds the hook object and not a bool.

        Something that replaces sys.excepthook after import -- a driver
        installing a crash reporter -- would otherwise leave a later call
        believing the abort was still in front, and the job would hang.
        """
        MPI.COMM_WORLD = _FakeComm(3)
        mpi_abort.abort_on_uncaught_exception()
        ours = sys.excepthook
        recorded, reported = [], []

        def usurper(t, e, tb):
            reported.append(e)
        sys.excepthook = usurper
        with mock.patch("mpi4py.run.set_abort_status", recorded.append):
            self.assertTrue(mpi_abort.abort_on_uncaught_exception())
            self.assertIsNot(sys.excepthook, usurper)
            self.assertIsNot(sys.excepthook, ours)
            boom = ValueError("boom")
            sys.excepthook(ValueError, boom, None)
        # both halves, or the re-install is worse than none: the job would
        # hang without the status, and lose the driver's report without it
        # chaining to the hook it displaced.
        self.assertEqual(recorded, [boom])
        self.assertEqual(reported, [boom])


class TestConsoleScripts(unittest.TestCase):
    """Run each real console script end-to-end on its cheap serial path:
    with no arguments (and one rank) every main prints a usage message and
    raises SystemExit. This executes the scripts' real imports, so a broken
    import target (e.g. the repo-root ``mpi_one_sided_test`` py-module going
    missing from the install) fails here instead of on a user's command
    line."""

    def test_console_script_targets_exist(self):
        # the callables named in pyproject.toml [project.scripts]
        for name in ("generic_cylinders_main",
                     "mrp_generic_main",
                     "one_sided_test_main"):
            self.assertTrue(callable(getattr(entry_points, name)))

    def _run_console_script(self, script, prog):
        saved_argv = sys.argv
        sys.argv = [prog]
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    script()
            return caught.exception.code
        finally:
            sys.argv = saved_argv

    def test_generic_cylinders_usage_exit(self):
        # no args -> generic_cylinders.main prints usage and quit()s
        code = self._run_console_script(entry_points.generic_cylinders_main,
                                        "mpi-sppy-generic-cylinders")
        self.assertIn(code, (None, 0))

    def test_mrp_generic_usage_exit(self):
        # no args -> mrp_generic.main prints usage and quit()s
        code = self._run_console_script(entry_points.mrp_generic_main,
                                        "mpi-sppy-mrp-generic")
        self.assertIn(code, (None, 0))

    @unittest.skipUnless(have_mpi4py, "mpi_one_sided_test imports mpi4py directly")
    @unittest.skipUnless(MPI.COMM_WORLD.Get_size() == 1,
                         "at more than one rank the one-sided test really runs")
    def test_one_sided_test_single_rank_exit(self):
        # at one rank the script demands an mpiexec launch and exits 2
        code = self._run_console_script(entry_points.one_sided_test_main,
                                        "mpi-sppy-one-sided-test")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
