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

solver_name = "gurobi_persistent"


def do_one_mmw(dirname, progname, npyfile, argstring):
    os.chdir(dirname)

    runstring = "python -m mpisppy.confidence_intervals.mmw_conf {} {} {}".\
                format(progname, npyfile , argstring)
    code = os.system("echo {} && {}".format(runstring, runstring))
    if code != 0:
        if dirname not in badguys:
            badguys[dirname] = [runstring]
        else:
            badguys[dirname].append(runstring)
    os.remove(npyfile)

    os.chdir("..")

# for farmer, the first arg is num_scens and is required
do_one("farmer", "farmer_cylinders.py", 3,
       "3 --bundles-per-rank=0 --max-iterations=50 "
       "--default-rho=1 --with-display-convergence-detail "
       "--solver-name={} --no-fwph --use-norm-rho-updater".format(solver_name))
do_one("farmer", "farmer_lshapedhub.py", 2,
       "3 --bundles-per-rank=0 --max-iterations=50 "
       "--solver-name={} --rel-gap=0.0 "
       "--no-fwph --max-solver-threads=1".format(solver_name))
do_one("sizes",
       "sizes_cylinders.py",
       4,
       "--num-scens=3 --bundles-per-rank=0 --max-iterations=5 "
       "--iter0-mipgap=0.01 --iterk-mipgap=0.001 "
       "--default-rho=1 --solver-name={} --with-display-progress".format(solver_name))
do_one("hydro", "hydro_cylinders.py", 3,
       "--BFs 3 3 --bundles-per-rank=0 --max-iterations=100 "
       "--default-rho=1 --with-xhatspecific --with-lagrangian "
       "--solver-name={}".format(solver_name))

#mmw tests
#write .npy file for farmer
os.chdir("farmer")
os.system("echo python afarmer.py --num-scens=3 && python afarmer.py --num-scens=3")
os.chdir("..")
#run mmw, remove .npy file
do_one_mmw("farmer", "afarmer.py", "farmer_root_nonants_temp.npy", "--alpha 0.95 --num-scens=3 --solver-name gurobi")

if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
        sys.exit(1)
else:
    print("\nAll OK.")
