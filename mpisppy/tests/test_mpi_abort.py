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
SystemExit, a comm with no Abort) in-process with a fake comm; what only a
real job can show is that the survivor does not sit in its collective.
"""

import os
import shutil
import subprocess
import sys
import unittest

mpiexec_available = shutil.which("mpiexec") is not None

try:
    import mpi4py  # noqa: F401
    have_mpi4py = True
except ImportError:
    have_mpi4py = False

#: Long enough for interpreter start-up and MPI_Init on a loaded CI runner,
#: short enough that a hang is reported rather than waited out.
TIMEOUT = 120

_HERE = os.path.dirname(os.path.abspath(__file__))

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


def _run(script, tmp_name):
    path = os.path.join(_HERE, tmp_name)
    with open(path, "w") as f:
        f.write(script)
    try:
        # Plain python, not "python -m mpi4py": mpi4py's runner would abort
        # on its own and the test would pass without the code under test.
        return subprocess.run(
            ["mpiexec", "-np", "2", sys.executable, path],
            capture_output=True, text=True, timeout=TIMEOUT, check=False)
    finally:
        os.remove(path)


@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
@unittest.skipIf(not have_mpi4py, "mpi4py is not available")
class TestAbortInsteadOfHang(unittest.TestCase):

    def _assert_died_reporting(self, script, tmp_name):
        try:
            result = _run(script, tmp_name)
        except subprocess.TimeoutExpired:
            self.fail(f"the job hung for {TIMEOUT}s: rank 0 is still waiting "
                      "in its collective for a rank that raised")
        self.assertNotEqual(result.returncode, 0,
                            msg="the job reported success although a rank "
                                "raised")
        self.assertIn("boom on rank 1", result.stdout + result.stderr,
                      msg="the job died without printing what killed it")

    def test_the_wrapper_aborts_the_job(self):
        self._assert_died_reporting(_DIRECT, "_abort_direct_tmp.py")

    def test_the_wheel_aborts_the_job(self):
        self._assert_died_reporting(_THROUGH_THE_WHEEL, "_abort_wheel_tmp.py")


if __name__ == "__main__":
    unittest.main()
