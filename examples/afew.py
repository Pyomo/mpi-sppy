# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Run a few examples; dlw June 2020
# See also runall.py
# Assumes you run from the examples directory.
# Optional command line arguments: solver_name mpiexec_arg
# E.g. python run_all.py
#      python run_all.py cplex
#      python run_all.py gurobi_persistent --oversubscribe

import os
import sys

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
    runstring = "mpiexec {} -np {} python -m mpi4py {} {}".\
                format(mpiexec_arg, np, progname, argstring)
    print(runstring)
    code = os.system(runstring)
    if code != 0:
        if dirname not in badguys:
            badguys[dirname] = [runstring]
        else:
            badguys[dirname].append(runstring)
    os.chdir("..")


# for farmer, the first arg is num_scens and is required
do_one("farmer", "farmer_cylinders.py", 3,
       "3 --bundles-per-rank=0 --max-iterations=50 "
       "--default-rho=1 "
       "--solver-name={} --no-fwph".format(solver_name))
do_one("sizes",
       "sizes_cylinders.py",
       4,
       "--num-scens=3 --bundles-per-rank=0 --max-iterations=5 "
       "--iter0-mipgap=0.01 --iterk-mipgap=0.001 "
       "--default-rho=1 --solver-name={} --with-display-progress".format(solver_name))
do_one("hydro", "hydro_cylinders.py", 3,
       "--BFs=3,3 --bundles-per-rank=0 --max-iterations=100 "
       "--default-rho=1 --with-xhatspecific --with-lagrangian "
       "--solver-name={}".format(solver_name))

if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
        sys.exit(1)
else:
    print("\nAll OK.")
