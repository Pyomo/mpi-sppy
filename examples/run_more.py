###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Additional example-based regression tests, split from run_all.py to reduce
# CI time. These tests exercise code paths that are also covered by dedicated
# unit tests (test_aph.py, test_conf_int_*.py, etc.) but go through the
# example script entry points.
# Assumes you run from the examples directory.
# Optional command line arguments: solver_name mpiexec_arg
# E.g. python run_more.py
#      python run_more.py gurobi_persistent --oversubscribe
# For coverage: python run_more.py gurobi_persistent "" --python-args="-m coverage run --parallel-mode --source=mpisppy"

import os
import sys
import pandas as pd
from datetime import datetime as dt

# Parse --python-args (extra args inserted after "python" in subcommands, e.g. for coverage)
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

solver_name = "gurobi_persistent"
if len(sys.argv) > 1:
    solver_name = sys.argv[1]

# Use oversubscribe if your computer does not have enough cores.
# Don't use this unless you have to.
# (This may not be allowed on some versions of mpiexec)
mpiexec_arg = ""  # "--oversubscribe" or "-envall"
if len(sys.argv) > 2:
    mpiexec_arg = sys.argv[2]

badguys = dict()

def do_one(dirname, progname, np, argstring):
    """ return the code"""
    os.chdir(dirname)
    runstring = "mpiexec {} -np {} python -u {} -m mpi4py {} {}".\
                format(mpiexec_arg, np, python_args, progname, argstring)
    # The top process output seems to be cached by github actions
    # so we need oputput in the system call to help debug
    code = os.system("echo {} && {}".format(runstring, runstring))
    if code != 0:
        if dirname not in badguys:
            badguys[dirname] = [runstring]
        else:
            badguys[dirname].append(runstring)
    if '/' not in dirname:
        os.chdir("..")
    else:
        os.chdir("../..")   # hack for one level of subdirectories
    return code

def time_one(ID, dirname, progname, np, argstring):
    """ same as do_one, but also check the running time.
        ID must be unique and ID.perf.csv will be(come) a local file name
        and should be allowed to sit on your machine in your examples directory.
        Do not record a time for a bad guy."""

    if ID in time_one.ID_check:
        raise RuntimeError(f"Duplicate time_one ID={ID}")
    else:
        time_one.ID_check.append(ID)

    listfname = ID+".perf.csv"

    start = dt.now()
    code = do_one(dirname, progname, np, argstring)
    finish = dt.now()
    runsecs = (finish-start).total_seconds()
    if code != 0:
        return   # Nothing to see here, folks.

    # get a reference time
    start = dt.now()
    for i in range(int(1e7)):   # don't change this unless you *really* have to
        if (i % 2) == 0:
            foo = i * i
            bar = str(i)+"!"
    del foo
    del bar
    finish = dt.now()
    refsecs = (finish-start).total_seconds()

    if os.path.isfile(listfname):
        timelistdf = pd.read_csv(listfname)
        timelistdf.loc[len(timelistdf.index)] = [str(finish), refsecs, runsecs]
    else:
        print(f"{listfname} will be created.")
        timelistdf = pd.DataFrame([[finish, refsecs, runsecs]],
                                  columns=["datetime", "reftime", "time"])

    # Quick look for trouble
    if len(timelistdf) > 0:
        thisscaled = runsecs / refsecs
        lastrow = timelistdf.iloc[-1]
        lastrefsecs = lastrow["reftime"]
        lastrunsecs = lastrow["time"]
        lastscaled = lastrunsecs / lastrefsecs
        deltafrac = (thisscaled - lastscaled) / lastscaled
        if deltafrac > 0.1:
            print(f"**** WARNING: {100*deltafrac}% time increase for {ID}, see {listfname}")

    timelistdf.to_csv(listfname, index=False)
time_one.ID_check = list()


###################### tests moved from run_all.py ######################
# These are covered by dedicated unit tests but exercise the example
# script entry points.

# --- Timing tests ---

time_one("FarmerLinProx", "farmer/archive", "farmer_cylinders.py", 3,
       "--num-scens 3 --default-rho=1.0 --max-iterations=50 "
       "--display-progress --rel-gap=0.0 --abs-gap=0.0 "
       "--linearize-proximal-terms --proximal-linearization-tolerance=1.e-6 "
       "--solver-name={} --lagrangian --xhatshuffle".format(solver_name))

time_one("AircondAMA", "aircond", "aircond_ama.py", 3,
       "--branching-factors \'3 3\' --max-iterations=100 "
       "--default-rho=1 --lagrangian --xhatshuffle "
       "--solver-name={}".format(solver_name))

# --- APH farmer tests (also covered by test_aph.py) ---

do_one("farmer/archive",
       "farmer_cylinders.py",
       2,
       f"--num-scens 3 --max-iterations=10 --default-rho=1.0 --display-progress  --xhatshuffle --aph-gamma=1.0 --aph-nu=1.0 --aph-frac-needed=1.0 --aph-dispatch-frac=1.0 --abs-gap=1 --aph-sleep-seconds=0.01 --run-async --solver-name={solver_name}")
do_one("farmer/archive",
       "farmer_cylinders.py",
       2,
       f"--num-scens 3 --max-iterations=10 --default-rho=1.0 --display-progress --xhatlooper --aph-gamma=1.0 --aph-nu=1.0 --aph-frac-needed=1.0 --aph-dispatch-frac=0.25 --abs-gap=1 --display-convergence-detail --aph-sleep-seconds=0.01 --run-async --solver-name={solver_name}")

# --- Sequential sampling (also covered by test_conf_int_farmer.py) ---

do_one("farmer/CI",
       "farmer_seqsampling.py",
       1,
       f"--num-scens 3 --crops-multiplier=1  --EF-solver-name={solver_name} "
       "--BM-h 2 --BM-q 1.3 --confidence-level 0.95 --BM-vs-BPL BM")

do_one("farmer/CI",
       "farmer_seqsampling.py",
       1,
       f"--num-scens 3 --crops-multiplier=1  --EF-solver-name={solver_name} "
       "--BPL-c0 25 --BPL-eps 100 --confidence-level 0.95 --BM-vs-BPL BPL")

# --- Hydro without stage2_ef_solver_name (also covered by generic_tester) ---

do_one("hydro", "hydro_cylinders.py", 3,
       "--branching-factors \"3 3\" --max-iterations=100 "
       "--default-rho=1 --xhatshuffle --lagrangian "
       "--solver-name={}".format(solver_name))

# --- Aircond sequential sampling (also covered by test_conf_int_aircond.py) ---

do_one("aircond",
       "aircond_seqsampling.py",
       1,
       f"--branching-factors \'3 2\' --seed 1134 --solver-name={solver_name} "
       "--BM-h 2 --BM-q 1.3 --confidence-level 0.95 --BM-vs-BPL BM")
do_one("aircond",
       "aircond_seqsampling.py",
       1,
       f"--branching-factors \'3 2\' --seed 1134 --solver-name={solver_name} "
       "--BPL-c0 25 --BPL-eps 100 --confidence-level 0.95 --BM-vs-BPL BPL")


#### final processing ####
if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
    sys.exit(1)
else:
    print("\nAll OK.")
