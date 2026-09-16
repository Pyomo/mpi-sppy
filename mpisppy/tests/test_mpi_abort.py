###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""A rank that raises must kill the job, not hang the other ranks.

These spawn real two-rank mpiexec jobs under *plain* ``python``: the point
is the launcher that does not abort by itself, so a test running in-process
could not tell the fix from its absence. Each asserts the job ends, which
means the wrong answer here is a timeout rather than a failed assertion --
hence the short timeout and the message that says so.

None of the scripts below install the hook: importing mpi-sppy is what does
that, so leaving the call out is how these pin it. ``test_entry_points.py``
covers the install decisions (serial, no mpi4py, an unusable comm, a hook
someone replaced) in-process with a fake comm; what only a real job can show
is that the survivor does not sit in its collective.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

mpiexec_available = shutil.which("mpiexec") is not None

#: Set where mpiexec and mpi4py are known to exist. Without this a
#: launcher-less environment reports skips and the job goes green with
#: nothing having asserted the guard, which is the one outcome these tests
#: must not have.
_require = os.environ.get("MPISPPY_REQUIRE_MPIEXEC", "") not in ("", "0")

#: Extra arguments for the mpiexec jobs these tests spawn -- the workflow
#: computes "-oversubscribe" for OpenMPI, where a runner has fewer usable
#: slots than the ranks asked for. Empty everywhere else; MPICH rejects it.
_MPIEXEC_ARGS = os.environ.get("OVERSUBSCRIBE", "").split()

try:
    import mpi4py  # noqa: F401
    have_mpi4py = True
except ImportError:
    have_mpi4py = False

#: Both halves, or the cell goes green on skips with nothing asserted --
#: the class below skips on either, so checking only the launcher would
#: leave a broken mpi4py looking like a pass.
if _require and not (mpiexec_available and have_mpi4py):
    missing = ", ".join(
        name for name, ok in (("mpiexec", mpiexec_available),
                              ("mpi4py", have_mpi4py)) if not ok)
    raise RuntimeError(
        f"MPISPPY_REQUIRE_MPIEXEC is set but {missing} is not available, so "
        "the rank-failure abort tests cannot run.")

#: Long enough for interpreter start-up and MPI_Init on a loaded CI runner,
#: short enough that a hang is reported rather than waited out.
TIMEOUT = 120

#: How long to wait for the killed job to release the pipes. Only reached
#: when the job has already hung, so it just bounds the report.
_REAP_TIMEOUT = 30

#: How long the launcher gets to take its own ranks down after SIGTERM
#: before the kills in ``_reap`` stop asking.
_TERM_TIMEOUT = 10

#: The checkout under test. The children are plain ``python`` on a script in
#: a temp directory, so without this they would import whatever mpi-sppy is
#: installed -- which on a developer machine need not be this one.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

#: Rank 1 fails; rank 0 walks into a collective rank 1 will never reach.
#: Without the abort this is a job that hangs forever holding a traceback
#: nobody sees. The finally and atexit lines pin the *deferred* abort: the
#: failing rank finishes unwinding, and its buffers reach the log, before
#: the job ends.
_UNCAUGHT = """
import atexit
from mpisppy import MPI

rank = MPI.COMM_WORLD.Get_rank()
atexit.register(lambda: print(f"rank {rank} ATEXIT RAN", flush=True))

try:
    if rank == 1:
        raise RuntimeError("boom on rank 1")
    MPI.COMM_WORLD.Barrier()
finally:
    print(f"rank {rank} FINALLY RAN", flush=True)
"""

#: The property that protects every caller with a try/except of its own --
#: a driver retrying with another solver, a test asserting that a call
#: raises. A wrapper around the driver would fire on these; a hook does not.
_CAUGHT = """
from mpisppy import MPI

rank = MPI.COMM_WORLD.Get_rank()
try:
    if rank == 1:
        raise RuntimeError("caught on rank 1, and handled")
except RuntimeError:
    print(f"rank {rank} HANDLED ITS OWN FAILURE", flush=True)
MPI.COMM_WORLD.Barrier()
print(f"rank {rank} PAST THE COLLECTIVE", flush=True)
"""

#: Ctrl-C. A rank sitting in a collective cannot raise KeyboardInterrupt --
#: mpi4py releases the GIL inside the C call and the signal stays pending --
#: so letting it propagate strands exactly the ranks it was supposed to
#: spare, and the job needs kill -9.
_INTERRUPT = """
from mpisppy import MPI

rank = MPI.COMM_WORLD.Get_rank()
if rank == 1:
    raise KeyboardInterrupt
MPI.COMM_WORLD.Barrier()
print(f"rank {rank} PAST THE COLLECTIVE", flush=True)
"""

#: sys.exit never reaches an excepthook, so it keeps its own status. That is
#: what leaves argparse alone: --help and a usage error are uniform across
#: ranks and end the job by themselves. Every rank exits, so this runs at the
#: default two: at one rank no hook is installed and the test would pass
#: whatever an installed hook did to SystemExit.
_SYSTEM_EXIT = """
import sys
import mpisppy  # the import is what installs the hook

print("about to exit", flush=True)
sys.exit(2)
"""

#: The same failure reached through WheelSpinner.run, which is where a real
#: run loses a rank (opt construction, make_windows).
_THROUGH_THE_WHEEL = """
from mpisppy import MPI
from mpisppy.spin_the_wheel import WheelSpinner

class StubOpt:
    def __init__(self, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 1:
            raise RuntimeError("boom on rank 1")
        MPI.COMM_WORLD.Barrier()

class StubSPComm:
    BestInnerBound = None
    BestOuterBound = None
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return lambda *a, **k: None

_cylinder = {
    "opt_class": StubOpt,
    "opt_kwargs": {"all_scenario_names": ["Scenario1"]},
}
WheelSpinner(dict(hub_class=StubSPComm, **_cylinder),
             [dict(spoke_class=StubSPComm, **_cylinder)]).run()
"""


#: A driver that installs its own excepthook after importing mpi-sppy and
#: chains to sys.__excepthook__ rather than to the hook it displaced -- a
#: crash reporter is usually written exactly this way. The import-time
#: install is gone by the time the wheel runs, so only run()'s re-assert
#: keeps rank 0 from waiting in its collective forever.
_HOSTILE_HOOK = """
import sys
from mpisppy import MPI
from mpisppy.spin_the_wheel import WheelSpinner

def crash_reporter(exc_type, exc, tb):
    print("REPORTER RAN", flush=True)
    sys.__excepthook__(exc_type, exc, tb)

sys.excepthook = crash_reporter

class StubOpt:
    def __init__(self, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 1:
            raise RuntimeError("boom on rank 1")
        MPI.COMM_WORLD.Barrier()

class StubSPComm:
    BestInnerBound = None
    BestOuterBound = None
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return lambda *a, **k: None

_cylinder = {
    "opt_class": StubOpt,
    "opt_kwargs": {"all_scenario_names": ["Scenario1"]},
}
WheelSpinner(dict(hub_class=StubSPComm, **_cylinder),
             [dict(spoke_class=StubSPComm, **_cylinder)]).run()
"""


#: Turning the abort off is announced. A job that hangs later should not
#: leave anyone hunting for why the guard did not fire.
_OPT_OUT_SAYS_SO = """
import mpisppy
from mpisppy.utils.mpi_abort import abort_on_uncaught_exception

print("installed:", abort_on_uncaught_exception(), flush=True)
"""


#: A job that hangs whatever the code under test does, for the test that
#: the harness takes the ranks down and not just the launcher. Rank 0 waits
#: in a collective rank 1 never reaches, because *that* is the rank that
#: survives: one merely asleep notices its launcher die and goes with it,
#: while one inside a collective busy-waits and stays. It is also the shape
#: of every job the tests above leave behind when they report a hang.
_HANGS = """
import os
import time
from mpisppy import MPI

rank = MPI.COMM_WORLD.Get_rank()
here = os.path.dirname(os.path.abspath(__file__))
# Announced on disk rather than on stdout: the pipes cannot be read until
# the job is over, and the test has to know both ranks are up before it
# kills them or it would be asserting on a job that never started.
open(os.path.join(here, "ready.%d" % rank), "w").close()
if rank == 1:
    time.sleep(9999)
MPI.COMM_WORLD.Barrier()
"""


def _launch(path, np, env_extra=None):
    """Start the `np`-rank job for the script at `path`. Returns argv, proc.

    The session is our own so that `_reap` can take the whole job down and
    not just the process it started.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_ROOT] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    # The docs tell sweep users to export this; inheriting it here would
    # disarm the very guard these jobs exist to exercise, and every _died
    # test would wait out its timeout before failing.
    env.pop("MPISPPY_NO_ABORT_HOOK", None)
    env.update(env_extra or {})
    # Plain python, not "python -m mpi4py": mpi4py's runner would end the
    # job on its own and the tests would pass without the code under test.
    argv = ["mpiexec", *_MPIEXEC_ARGS, "-np", str(np), sys.executable, path]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env,
                            start_new_session=True)
    return argv, proc


def _session_members(sid):
    """The pids still in session `sid`, the session leader included.

    Empty where pgrep is missing, which costs the confirmation rather than
    the kill: the signals in `_reap` do the work, and this only reports
    whether they landed.
    """
    try:
        found = subprocess.run(["pgrep", "-s", str(sid)], capture_output=True,
                               text=True, timeout=_REAP_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(pid) for pid in found.stdout.split()]


def _reap(proc):
    """End the whole job behind `proc`, ranks included. Returns survivors.

    ``start_new_session=True`` makes mpiexec a session leader, but prterun
    gives each rank a process group of its own -- the ranks share only the
    session -- so killing the launcher's group reaches the launcher and
    stops there. What is left is orphaned to init, and a rank abandoned in
    a collective busy-waits, so each survivor holds a core at 100% with no
    output and no log until someone goes looking.

    SIGTERM to the launcher first: a launcher that is still healthy takes
    its own ranks down, and that is the only route that also reaches ranks
    on another node. The group kill is the backstop for a launcher that is
    itself wedged, and the sweep by session covers the ranks no signal to
    the launcher's group can name.

    `proc` itself is not a survivor: it is still to be waited on by the
    caller, which is what reads the pipes.
    """
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=_TERM_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        # It finished as the deadline lapsed, or the group is not ours to
        # signal. The sweep below is what says whether anything is left.
        pass
    deadline = time.monotonic() + _REAP_TIMEOUT
    while True:
        survivors = [pid for pid in _session_members(proc.pid)
                     if pid != proc.pid]
        if not survivors or time.monotonic() >= deadline:
            return survivors
        for pid in survivors:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.1)


def _run(script, np=2, env_extra=None):
    """Run `script` as an `np`-rank plain-python job. Returns the result.

    Not subprocess.run: on the timeout path it kills only mpiexec and then
    reads the pipes with no deadline of its own, so an orphaned rank still
    holding the write end blocks the report forever. Since a timeout is the
    failure these tests exist to report, the reporting path is the one that
    must not hang -- hence a session of our own, torn down whole by `_reap`
    before the pipes are read. That teardown is on every way out and not
    just the timeout, since a Ctrl-C orphans exactly the same job.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "leg.py")
        with open(path, "w") as f:
            f.write(script)
        argv, proc = _launch(path, np, env_extra)
        try:
            out, err = proc.communicate(timeout=TIMEOUT)
        except BaseException:
            # A timeout is the way in that these tests are written around,
            # but Ctrl-C and anything else that takes the run down arrive
            # here too and leave the same job behind, which is where the
            # survivors found spinning days later have come from.
            leaked = _reap(proc)
            if leaked:
                # Nothing may be raised over the exception the caller has
                # to see, so a leak is reported rather than raised.
                print(f"WARNING: {len(leaked)} process(es) from the abandoned "
                      f"job outlived the kill and are still running: {leaked}",
                      file=sys.stderr, flush=True)
            try:
                proc.communicate(timeout=_REAP_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass  # the caller is about to fail the test regardless
            raise
        return subprocess.CompletedProcess(argv, proc.returncode, out, err)


@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
@unittest.skipIf(not have_mpi4py, "mpi4py is not available")
class TestAbortInsteadOfHang(unittest.TestCase):

    def _died(self, script):
        try:
            result = _run(script)
        except subprocess.TimeoutExpired:
            self.fail(f"the job hung for {TIMEOUT}s: a rank is still waiting "
                      "in its collective for a rank that failed")
        self.assertNotEqual(result.returncode, 0,
                            msg="the job reported success although a rank "
                                "failed")
        return result.stdout + result.stderr

    def test_an_uncaught_exception_ends_the_job(self):
        out = self._died(_UNCAUGHT)
        self.assertIn("boom on rank 1", out,
                      msg="the job died without printing what killed it")

    def test_the_failing_rank_finishes_unwinding_first(self):
        """The abort is deferred to interpreter exit, so cleanup runs.

        An immediate Abort takes the process out mid-unwind: no finally, no
        atexit, and whatever is still buffered is lost with it.
        """
        out = self._died(_UNCAUGHT)
        self.assertIn("rank 1 FINALLY RAN", out,
                      msg="the failing rank was killed before its finally")
        self.assertIn("rank 1 ATEXIT RAN", out,
                      msg="the failing rank was killed before its atexit")

    def test_a_caught_exception_leaves_the_job_alone(self):
        """The property that protects a caller with its own try/except."""
        try:
            result = _run(_CAUGHT)
        except subprocess.TimeoutExpired:
            self.fail(f"the job hung for {TIMEOUT}s")
        out = result.stdout + result.stderr
        self.assertIn("rank 1 HANDLED ITS OWN FAILURE", out)
        for rank in (0, 1):
            self.assertIn(f"rank {rank} PAST THE COLLECTIVE", out,
                          msg=f"rank {rank} was taken down by a failure "
                              "another rank had already handled")
        self.assertEqual(result.returncode, 0)

    def test_an_uncaught_keyboard_interrupt_ends_the_job(self):
        out = self._died(_INTERRUPT)
        self.assertNotIn("PAST THE COLLECTIVE", out,
                         msg="a rank got past a collective the interrupted "
                             "rank never reached")

    def test_sys_exit_keeps_its_own_status(self):
        """argparse and every other uniform exit are left alone."""
        try:
            result = _run(_SYSTEM_EXIT)
        except subprocess.TimeoutExpired:
            self.fail(f"the job hung for {TIMEOUT}s")
        out = result.stdout + result.stderr
        self.assertIn("about to exit", out,
                      msg="the job did not get as far as the exit")
        self.assertEqual(result.returncode, 2,
                         msg="sys.exit(2) did not exit 2")
        # Lowered because the implementations disagree on the spelling:
        # OpenMPI prints "MPI_ABORT was invoked", MPICH "application called
        # MPI_Abort(MPI_COMM_WORLD, 2)". Matching only one of them would let
        # the abort matrix's MPICH cells pass an intercepted SystemExit --
        # the returncode cannot tell them apart, since an abort on a status
        # of 2 also exits 2.
        self.assertNotIn("mpi_abort", out.lower(),
                         msg="SystemExit was turned into an abort")

    def test_the_opt_out_declines_and_says_so(self):
        """MPISPPY_NO_ABORT_HOOK is for ranks doing independent work, where
        one rank's failure should not end the others. It is loud because
        the cost of setting it by mistake is a hang."""
        try:
            result = _run(_OPT_OUT_SAYS_SO,
                          env_extra={"MPISPPY_NO_ABORT_HOOK": "1"})
        except subprocess.TimeoutExpired:
            self.fail(f"the job hung for {TIMEOUT}s")
        out = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=out)
        self.assertIn("installed: False", out,
                      msg="the hook was installed despite the opt-out")
        self.assertIn("MPISPPY_NO_ABORT_HOOK is set", out,
                      msg="the abort was turned off without saying so")

    def test_the_wheel_ends_the_job(self):
        out = self._died(_THROUGH_THE_WHEEL)
        self.assertIn("boom on rank 1", out)

    def test_a_driver_that_took_the_excepthook_still_ends_the_job(self):
        """Why run() re-asserts the hook the import already installed.

        A driver that replaces sys.excepthook and chains to
        sys.__excepthook__ instead of to the hook it displaced drops the
        abort. Without the call in run() this hangs to the timeout.
        """
        out = self._died(_HOSTILE_HOOK)
        self.assertIn("boom on rank 1", out)
        self.assertIn("REPORTER RAN", out,
                      msg="the driver's own excepthook was bypassed rather "
                          "than wrapped")


@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
@unittest.skipIf(not have_mpi4py, "mpi4py is not available")
@unittest.skipIf(shutil.which("pgrep") is None, "pgrep is not available")
class TestTheTimedOutJobIsGone(unittest.TestCase):
    """Nothing from a timed-out job outlives the test that reported it.

    Every test above reports a hang by timing out, so the kill on that path
    runs exactly when these tests are earning their keep. A rank it misses
    is orphaned in a collective and spins on a core for as long as the
    machine is up, saying nothing -- which is why this asserts on the pids
    rather than trusting the signal.
    """

    def test_the_kill_takes_the_ranks_and_not_just_the_launcher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hang.py")
            with open(path, "w") as f:
                f.write(_HANGS)
            _, proc = _launch(path, 2)
            try:
                # Killing before both ranks are past MPI_Init would pass
                # without either having been at risk, and rank 0 has to be
                # in its collective for this to be the job the timeout path
                # actually leaves behind.
                markers = [os.path.join(tmpdir, f"ready.{r}") for r in (0, 1)]
                deadline = time.monotonic() + TIMEOUT
                while (not all(os.path.exists(m) for m in markers)
                       and time.monotonic() < deadline):
                    time.sleep(0.1)
                self.assertTrue(
                    all(os.path.exists(m) for m in markers),
                    msg=f"the job did not reach two running ranks in "
                        f"{TIMEOUT}s, so the kill below had nothing to miss")
                time.sleep(1)  # rank 0 is a few instructions from Barrier
                leaked = _reap(proc)
            finally:
                try:
                    proc.communicate(timeout=_REAP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self.assertEqual(
                leaked, [],
                msg="the kill reached mpiexec and stopped there: these pids "
                    "are ranks left running, each one holding a core")


if __name__ == "__main__":
    unittest.main()
