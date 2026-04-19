###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Run a few examples; dlw June 2020
# See also runall.py
# Assumes you run from the examples directory.
# Optional command line arguments: solver_name mpiexec_arg
# E.g. python afew.py
#      python afew.py cplex
#      python afew.py gurobi_persistent --oversubscribe
# For coverage: python afew.py gurobi_persistent "" --python-args="-m coverage run --parallel-mode --source=mpisppy"

import os
import sys

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
# (This may not be allowed on versions of mpiexec)
mpiexec_arg = ""  # "--oversubscribe"
if len(sys.argv) > 2:
    mpiexec_arg = sys.argv[2]

badguys = dict()

def do_one(dirname, progname, np, argstring):
    os.chdir(dirname)
    runstring = "mpiexec {} -np {} python {} -m mpi4py {} {}".\
                format(mpiexec_arg, np, python_args, progname, argstring)
    print(runstring)
    code = os.system(runstring)
    if code != 0:
        if dirname not in badguys:
            badguys[dirname] = [runstring]
        else:
            badguys[dirname].append(runstring)
    if '/' not in dirname:
        os.chdir("..")
    else:
        os.chdir("../..")   # hack for one level of subdirectories


# for farmer, the first arg is num_scens and is required
do_one("farmer", "farmer_lshapedhub.py", 2,
       "--num-scens=3 --max-iterations=50 "
       "--solver-name={} --rel-gap=0.0 "
       " --xhatlshaped --max-solver-threads=1".format(solver_name))
do_one("hydro", "../../mpisppy/generic_cylinders.py", 3,
       "--module-name hydro --branching-factors \'3 3\' "
       "--max-iterations=10000 "
       "--default-rho=1 --xhatshuffle --lagrangian "
       "--abs-gap=0 --rel-gap=0 --time-limit=2 "
       "--stage2EFsolvern={} --solver-name={}".format(solver_name, solver_name))

if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
        sys.exit(1)
else:
    print("\nAll OK.")
