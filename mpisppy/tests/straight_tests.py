###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# straight smoke tests (with no unittest, which is a bummer but we need mpiexec)

import os
import shlex
import subprocess
import sys

from mpisppy.tests.utils import get_solver

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

badguys = []

_tests_dir = os.path.dirname(os.path.abspath(__file__))


def _doone(cmdstr: str) -> bool:
    """Run a shell command. Return True on success, False on failure.
    Write stdout/stderr to mpisppy/tests/last_mpiexec.log for debugging.
    """
    print("testing:", cmdstr)

    log = os.path.join(_tests_dir, "last_mpiexec.log")

    # Use bash so the redirection and quoting behave predictably.
    script = (
        "set +e\n"
        "echo '=== preflight ==='\n"
        "echo 'pwd:' $(pwd)\n"
        "echo 'python:' $(command -v python)\n"
        "echo 'python -V:' $(python -V 2>&1)\n"
        "echo 'sys.executable:' " + shlex.quote(sys.executable) + "\n"
        "echo 'mpiexec:' $(command -v mpiexec || echo MISSING)\n"
        "echo 'mpiexec --version:'\n"
        "mpiexec --version 2>&1 || true\n"
        "echo '=== command ==='\n"
        f"echo {shlex.quote(cmdstr)}\n"
        "echo '=== output ==='\n"
        f"{cmdstr}\n"
        "rc=$?\n"
        "echo '=== done; rc='${rc}\n"
        "exit ${rc}\n"
    )

    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(["bash", "-lc", script], stdout=f, stderr=subprocess.STDOUT)

    if p.returncode != 0:
        # include tail of log for convenience
        try:
            with open(log, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail = "".join(lines[-200:])
        except Exception as e:
            tail = f"(Could not read log {log}: {e})\n"

        badguys.append(
            f"Test failed with code {p.returncode}:\n{cmdstr}\n"
            f"--- log (tail) {log} ---\n{tail}\n"
        )
        return False

    return True


#####################################################
# aircond
example_dir = os.path.join(_tests_dir, "..", "..", "examples", "aircond")
fpath = os.path.join(example_dir, "aircond_cylinders.py")
jpath = os.path.join(example_dir, "lagranger_factors.json")

# PH and lagranger rescale factors w/ FWPH
fwphSaveFile = os.path.join(_tests_dir, "fwph_trace.txt")

# Use the exact interpreter running this test (important for CI/conda/venv)
pyexe = shlex.quote(sys.executable)

cmdstr = (
    f"mpiexec -np 4 {pyexe} -m mpi4py {shlex.quote(fpath)} "
    f"--bundles-per-rank=0 --max-iterations=5 --default-rho=1 "
    f"--solver-name={shlex.quote(solver_name)} "
    f'--branching-factors "4 3 2" '
    f"--Capacity 200 --QuadShortCoeff 0.3 --BeginInventory 50 "
    f"--rel-gap 0.01 --mu-dev 0 --sigma-dev 40 "
    f"--max-solver-threads 2 --start-seed 0 "
    f"--lagranger --lagranger-rho-rescale-factors-json {shlex.quote(jpath)} "
    f"--fwph --fwph-save-file {shlex.quote(fwphSaveFile)} --xhatshuffle"
)

ok = _doone(cmdstr)

# If the run succeeded, verify that FWPH wrote the file; then delete it.
if ok:
    if os.path.exists(fwphSaveFile):
        try:
            os.remove(fwphSaveFile)
        except OSError as e:
            badguys.append(f"Test wrote {fwphSaveFile} but could not delete it: {e}\n{cmdstr}")
    else:
        badguys.append(f"Test failed to write {fwphSaveFile}:\n{cmdstr}")


#####################################################
# generic_cylinders with MMW CI (farmer, wheel-based xhat)
gc_path = os.path.abspath(os.path.join(_tests_dir, "..", "..", "mpisppy", "generic_cylinders.py"))
farmer_module = os.path.abspath(os.path.join(_tests_dir, "..", "..", "examples", "farmer", "farmer"))

cmdstr = (
    f"mpiexec -np 2 {pyexe} -m mpi4py {shlex.quote(gc_path)} "
    f"--module-name {shlex.quote(farmer_module)} "
    f"--num-scens 3 "
    f"--solver-name {shlex.quote(solver_name)} "
    f"--default-rho 1 "
    f"--max-iterations 3 "
    f"--xhatshuffle "
    f"--mmw-num-batches 2 "
    f"--mmw-batch-size 3 "
    f"--mmw-start 4"
)

_doone(cmdstr)


#######################################################
if badguys:
    print("\nstraight_tests.py failed commands:")
    for msg in badguys:
        print(msg)
    raise RuntimeError("straight_tests.py had failed commands")
else:
    print("straight_test.py: all OK.")
