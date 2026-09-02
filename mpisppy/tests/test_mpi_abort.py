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

#: Rank 1 raises; rank 0 walks into a collective that rank 1 will never
#: reach. Without the abort this is a job that hangs forever holding a
#: traceback nobody sees.
_DIRECT = """
from mpisppy import MPI
from mpisppy.utils.mpi_abort import run_with_mpi_abort

def main():
    if MPI.COMM_WORLD.Get_rank() == 1:
        raise RuntimeError("boom on rank 1")
    MPI.COMM_WORLD.Barrier()

run_with_mpi_abort(main)
"""

#: Same, with sys.exit in place of the raise. A scenario_creator that exits
#: on the one rank owning a bad data file strands the others exactly as a
#: raise does, so a nonzero exit has to abort too.
_SYS_EXIT = """
from mpisppy import MPI
from mpisppy.utils.mpi_abort import run_with_mpi_abort

def main():
    if MPI.COMM_WORLD.Get_rank() == 1:
        raise SystemExit("boom on rank 1")
    MPI.COMM_WORLD.Barrier()

run_with_mpi_abort(main)
"""

#: The same shape, but reached through WheelSpinner.run: the stub opt class
#: raises on rank 1 while rank 0 blocks in a collective inside the wheel,
#: which is where a real run loses a rank (opt construction, make_windows).
_THROUGH_THE_WHEEL = """
from mpisppy import MPI
from mpisppy.spin_the_wheel import WheelSpinner

class StubOpt:
    def __init__(self, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 1:
            raise RuntimeError("boom on rank 1")
        MPI.COMM_WORLD.Barrier()

class StubSPComm:
    def __init__(self, *args, **kwargs):
        pass

_cylinder = {
    "opt_class": StubOpt,
    "opt_kwargs": {"all_scenario_names": ["Scenario1"]},
}
WheelSpinner(dict(hub_class=StubSPComm, **_cylinder),
             [dict(spoke_class=StubSPComm, **_cylinder)]).run()
"""

#: One wheel per rank, each on its own single-rank comm -- the shape
#: boot-sp's batch executor runs. Rank 1's wheel fails; rank 0's batch has
#: nothing to do with it and must still finish, and rank 1 must be left to
#: handle its own failure. Aborting COMM_WORLD here destroys both.
_PER_RANK_WHEELS = """
from mpisppy import MPI
from mpisppy.spin_the_wheel import WheelSpinner

rank = MPI.COMM_WORLD.Get_rank()
mine = MPI.COMM_WORLD.Split(color=rank, key=0)

class StubOpt:
    def __init__(self, **kwargs):
        if rank == 1:
            raise RuntimeError("boom in rank 1's own wheel")

class StubSPComm:
    # Rank 0's wheel has to run all the way through, so this answers every
    # hook the wheel calls; the two bounds are read as values, not called.
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
wheel = WheelSpinner(dict(hub_class=StubSPComm, **_cylinder), [])
try:
    wheel.run(comm_world=mine)
except RuntimeError:
    print(f"RANK {rank} RECORDED ITS OWN FAILURE")
except BaseException as e:
    print(f"RANK {rank} UNEXPECTED {type(e).__name__}: {e}")
else:
    print(f"RANK {rank} FINISHED")
MPI.COMM_WORLD.Barrier()
print(f"RANK {rank} REACHED THE GATHER")
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
        # Plain python, not "python -m mpi4py": mpi4py's runner would abort
        # on its own and the tests would pass without the code under test.
        return subprocess.run(
            ["mpiexec", *_MPIEXEC_ARGS, "-np", str(np), sys.executable, path],
            capture_output=True, text=True, timeout=TIMEOUT, check=False,
            env=env)


@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
@unittest.skipIf(not have_mpi4py, "mpi4py is not available")
class TestAbortInsteadOfHang(unittest.TestCase):

    def _assert_died_reporting(self, script):
        try:
            result = _run(script)
        except subprocess.TimeoutExpired:
            self.fail(f"the job hung for {TIMEOUT}s: rank 0 is still waiting "
                      "in its collective for a rank that failed")
        self.assertNotEqual(result.returncode, 0,
                            msg="the job reported success although a rank "
                                "failed")
        self.assertIn("boom on rank 1", result.stdout + result.stderr,
                      msg="the job died without printing what killed it")
        return result

    def test_the_wrapper_aborts_the_job(self):
        self._assert_died_reporting(_DIRECT)

    def test_a_nonzero_system_exit_aborts_the_job(self):
        self._assert_died_reporting(_SYS_EXIT)

    def test_the_wheel_aborts_the_job(self):
        self._assert_died_reporting(_THROUGH_THE_WHEEL)

    def test_a_wheel_on_its_own_comm_leaves_the_other_ranks_alone(self):
        """One group's failure is that group's, not the whole job's."""
        try:
            result = _run(_PER_RANK_WHEELS)
        except subprocess.TimeoutExpired:
            self.fail(f"the job hung for {TIMEOUT}s")
        out = result.stdout + result.stderr
        self.assertIn("RANK 0 FINISHED", out,
                      msg="rank 0's own wheel was destroyed by a failure in "
                          "rank 1's")
        self.assertIn("RANK 1 RECORDED ITS OWN FAILURE", out,
                      msg="the caller's except never ran: the process was "
                          "killed inside it")
        for rank in (0, 1):
            self.assertIn(f"RANK {rank} REACHED THE GATHER", out,
                          msg=f"rank {rank} never reached the collective "
                              "after the failed batch")
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
