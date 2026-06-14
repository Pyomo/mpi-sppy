###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# PR-time UC smoke suite. The full UC regression (run_uc.py) is too slow for
# every PR even on HiGHS, so this trimmed variant runs the PH hub with the
# IntegerRelaxThenEnforce extension at a high ratio — every subproblem solve
# is an LP for the duration of the run, which collapses UC iter-0 from
# minutes to seconds. Spokes that re-solve scenarios as MIPs (xhatshuffle,
# lagrangian) are deliberately omitted; the weekly run_uc.py covers them.
# Defaults to HiGHS (open-source, no size limit).
# Usage:
#   python run_uc_pr.py
#   python run_uc_pr.py appsi_highs
#   python run_uc_pr.py appsi_highs "" --python-args="-m coverage run --parallel-mode --source=mpisppy"
# Assumes you run from the examples directory.

import os
import sys

python_args = ""
_remaining = []
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i].startswith("--python-args="):
        python_args = sys.argv[_i].split("=", 1)[1]
    elif sys.argv[_i] == "--python-args" and _i + 1 < len(sys.argv):
        _i += 1
        python_args = sys.argv[_i]
    else:
        _remaining.append(sys.argv[_i])
    _i += 1
sys.argv = [sys.argv[0]] + _remaining

solver_name = "appsi_highs"
if len(sys.argv) > 1:
    solver_name = sys.argv[1]

mpiexec_arg = ""
if len(sys.argv) > 2:
    mpiexec_arg = sys.argv[2]

badguys = dict()


def egret_avail():
    try:
        import egret  # noqa: F401
    except Exception:
        return False
    return True


def do_one(dirname, progname, np, argstring):
    os.chdir(dirname)
    runstring = "mpiexec {} -np {} python -u {} -m mpi4py {} {}".format(
        mpiexec_arg, np, python_args, progname, argstring)
    code = os.system("echo {} && {}".format(runstring, runstring))
    if code != 0:
        badguys.setdefault(dirname, []).append(runstring)
    if '/' not in dirname:
        os.chdir("..")
    else:
        os.chdir("../..")
    return code


def do_one_serial(dirname, progname, argstring):
    os.chdir(dirname)
    runstring = "python -u {} {} {}".format(python_args, progname, argstring)
    code = os.system("echo {} && {}".format(runstring, runstring))
    if code != 0:
        badguys.setdefault(dirname, []).append(runstring)
    os.chdir("..")
    return code


if not egret_avail():
    print("run_uc_pr.py: egret is not importable; nothing to do.")
    sys.exit(0)

# 3-scenario EF (single rank). Cheap on HiGHS (~20 s).
do_one_serial("uc", "uc_ef.py", solver_name + " 3")

# Generic driver on uc_funcs: PH hub only, integers relaxed throughout.
# --integer-relax-then-enforce-ratio=10 with --max-iterations=1 means the
# unrelax trigger never fires, so every subproblem is an LP. No bound
# spokes (xhatshuffle / lagrangian) — those would still solve UC MIPs and
# blow the time budget. The weekly run_uc.py exercises them.
do_one("uc", "../../mpisppy/generic_cylinders.py", 3,
       "--module-name uc_funcs --max-iterations=1 --default-rho=1 "
       "--num-scens=3 "
       "--integer-relax-then-enforce --integer-relax-then-enforce-ratio=10 "
       "--linearize-proximal-terms "
       "--solver-name={}".format(solver_name))

if badguys:
    print("\nrun_uc_pr.py failed commands:")
    for dirname, cmds in badguys.items():
        print("Directory={}".format(dirname))
        for c in cmds:
            print("  " + c)
    raise RuntimeError("run_uc_pr.py had failed commands")
print("run_uc_pr.py: all OK.")
