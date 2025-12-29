###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
###############################################################################
# straight smoke tests (with no unittest, which is a bummer but we need mpiexec)

import os
import subprocess

from mpisppy.tests.utils import get_solver

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

badguys = []


# --- Open MPI debugging / robustness knobs ---
# Make Open MPI print all error messages (don't aggregate/suppress)
os.environ["OMPI_MCA_orte_base_help_aggregate"] = "0"
# Reduce noise about unused components (optional)
os.environ["OMPI_MCA_btl_base_warn_component_unused"] = "0"

# Force Open MPI temp/session dirs to a known-writable location (often fixes -17 No permission)
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_ompi_tmp = os.path.join(_tests_dir, ".ompi_tmp")
os.makedirs(_ompi_tmp, exist_ok=True)
os.environ["TMPDIR"] = _ompi_tmp
os.environ["OMPI_MCA_tmpdir_base"] = _ompi_tmp


def _doone(cmdstr: str) -> bool:
    """Run a shell command. Return True on success, False on failure.
    Write stdout/stderr to a log file to help debug mpiexec failures.
    """
    print("testing:", cmdstr)

    log = os.path.join(_tests_dir, "last_mpiexec.log")
    cmd_with_log = f"{cmdstr} > {log} 2>&1"

    p = subprocess.run(cmd_with_log, shell=True)

    if p.returncode != 0:
        # include tail of log for convenience
        tail = ""
        try:
            with open(log, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail = "".join(lines[-120:])  # last 120 lines
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

cmdstr = (
    f"mpiexec --tag-output -np 4 python -m mpi4py {fpath} "
    f"--bundles-per-rank=0 --max-iterations=5 --default-rho=1 "
    f"--solver-name={solver_name} "
    f'--branching-factors "4 3 2" '
    f"--Capacity 200 --QuadShortCoeff 0.3 --BeginInventory 50 "
    f"--rel-gap 0.01 --mu-dev 0 --sigma-dev 40 "
    f"--max-solver-threads 2 --start-seed 0 "
    f"--lagranger --lagranger-rho-rescale-factors-json {jpath} "
    f"--fwph --fwph-save-file {fwphSaveFile} --xhatshuffle"
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


#######################################################
if badguys:
    print("\nstraight_tests.py failed commands:")
    for msg in badguys:
        print(msg)
    raise RuntimeError("straight_tests.py had failed commands")
else:
    print("straight_test.py: all OK.")
