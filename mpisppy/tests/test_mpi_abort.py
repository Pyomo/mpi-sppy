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

``mpisppy/tests/test_entry_points.py`` covers the wrapper's branches (serial,
a clean exit, a comm with no Abort, Ctrl-C) in-process with a fake comm; what
only a real job can show is that the survivor does not sit in its collective.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

mpiexec_available = shutil.which("mpiexec") is not None

#: Set where mpiexec is known to exist. Without it a launcher-less
#: environment reports skips and the job goes green with nothing having
#: asserted the guard, which is the one outcome these tests must not have.
if (os.environ.get("MPISPPY_REQUIRE_MPIEXEC", "") not in ("", "0")
        and not mpiexec_available):
    raise RuntimeError(
        "MPISPPY_REQUIRE_MPIEXEC is set but mpiexec is not on the PATH, so "
        "the rank-failure abort tests cannot run.")

#: Extra arguments for the mpiexec jobs these tests spawn -- the workflow
#: computes "-oversubscribe" for OpenMPI, where a runner has fewer usable
#: slots than the ranks asked for. Empty everywhere else; MPICH rejects it.
_MPIEXEC_ARGS = os.environ.get("OVERSUBSCRIBE", "").split()

try:
    import mpi4py  # noqa: F401
    have_mpi4py = True
except ImportError:
    have_mpi4py = False

#: Long enough for interpreter start-up and MPI_Init on a loaded CI runner,
#: short enough that a hang is reported rather than waited out.
TIMEOUT = 120

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
import atexit, sys
from mpisppy import MPI
from mpisppy.utils.mpi_abort import abort_on_uncaught_exception

rank = MPI.COMM_WORLD.Get_rank()
abort_on_uncaught_exception()
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
#: raises. A wrapper around run() fires on these; an excepthook does not.
_CAUGHT = """
from mpisppy import MPI
from mpisppy.utils.mpi_abort import abort_on_uncaught_exception

rank = MPI.COMM_WORLD.Get_rank()
abort_on_uncaught_exception()
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
from mpisppy.utils.mpi_abort import abort_on_uncaught_exception

rank = MPI.COMM_WORLD.Get_rank()
abort_on_uncaught_exception()
if rank == 1:
    raise KeyboardInterrupt
MPI.COMM_WORLD.Barrier()
print(f"rank {rank} PAST THE COLLECTIVE", flush=True)
"""

#: sys.exit never reaches an excepthook, so it keeps its own status. That is
#: what leaves argparse alone: --help and a usage error are uniform across
#: ranks and end the job by themselves.
_SYSTEM_EXIT = """
import sys
from mpisppy import MPI
from mpisppy.utils.mpi_abort import abort_on_uncaught_exception

abort_on_uncaught_exception()
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


def _run(script, np=2):
    """Run `script` as an `np`-rank plain-python job. Returns the result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "leg.py")
        with open(path, "w") as f:
            f.write(script)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [_ROOT] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        # Plain python, not "python -m mpi4py": mpi4py's runner would end the
        # job on its own and the tests would pass without the code under test.
        return subprocess.run(
            ["mpiexec", *_MPIEXEC_ARGS, "-np", str(np), sys.executable, path],
            capture_output=True, text=True, timeout=TIMEOUT, check=False,
            env=env)


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
            result = _run(_SYSTEM_EXIT, np=1)
        except subprocess.TimeoutExpired:
            self.fail(f"the job hung for {TIMEOUT}s")
        self.assertEqual(result.returncode, 2,
                         msg="sys.exit(2) did not exit 2")
        self.assertNotIn("MPI_ABORT", result.stdout + result.stderr)

    def test_the_wheel_ends_the_job(self):
        out = self._died(_THROUGH_THE_WHEEL)
        self.assertIn("boom on rank 1", out)


if __name__ == "__main__":
    unittest.main()
