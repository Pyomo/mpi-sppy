###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Small UC regression suite sized for CI.
# Same CLI as run_all.py but runs only a handful of short UC commands.
# Defaults to HiGHS (appsi_highs) because it is fully open-source and has no
# problem-size limit, unlike Xpress community edition which caps at 5000
# rows/cols and cannot solve this UC. The cs_uc and generic_cylinders runs
# pass --linearize-proximal-terms so HiGHS (a non-quadratic solver) can
# handle the PH proximal term; this is harmless with quadratic-capable
# solvers like gurobi/cplex/xpress-full.
# Usage:
#   python run_uc.py
#   python run_uc.py appsi_highs
#   python run_uc.py gurobi_persistent --oversubscribe
#   python run_uc.py appsi_highs "" --python-args="-m coverage run --parallel-mode --source=mpisppy"
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
    print("run_uc.py: egret is not importable; nothing to do.")
    sys.exit(0)

# 3-scenario EF (single rank)
do_one_serial("uc", "uc_ef.py", solver_name + " 3")

# Cross-scenario cuts demo: PH hub + xhatlooper + cross-scen cut spoke
do_one("uc", "cs_uc.py", 3,
       "--max-iterations=2 --default-rho=1 --num-scens=3 "
       "--linearize-proximal-terms "
       "--solver-name={}".format(solver_name))

# Generic driver on uc_funcs: PH hub + xhatshuffle + lagrangian
do_one("uc", "../../mpisppy/generic_cylinders.py", 3,
       "--module-name uc_funcs --max-iterations=3 --default-rho=1 "
       "--xhatshuffle --lagrangian --num-scens=3 "
       "--linearize-proximal-terms "
       "--rel-gap=0.01 --intra-hub-conv-thresh=-1 "
       "--solver-name={}".format(solver_name))

if badguys:
    print("\nrun_uc.py failed commands:")
    for dirname, cmds in badguys.items():
        print("Directory={}".format(dirname))
        for c in cmds:
            print("  " + c)
    raise RuntimeError("run_uc.py had failed commands")
print("run_uc.py: all OK.")
