# This software is distributed under the 3-clause BSD License.
# Run a lot of examples; dlw May 2020
# Not intended to be user-friendly.
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

def egret_avail():
    try:
        import egret
    except:
        return False

    p = str(egret.__path__)
    l = p.find("'")
    r = p.find("'", l+1)
    egretrootpath = p[l+1:r]

    egret_thirdparty_path = os.path.join(egretrootpath, "thirdparty")
    if os.path.exists(os.path.join(egret_thirdparty_path, "pglib-opf-master")):
        return True
    
    from egret.thirdparty.get_pglib_opf import get_pglib_opf
    get_pglib_opf(egret_thirdparty_path)
    return True

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
do_one("farmer", "farmer_cylinders.py", 3,
       "6 --bundles-per-rank=2 --max-iterations=50 "
       "--default-rho=1 "
       "--solver-name={} --no-fwph".format(solver_name))
do_one("farmer", "farmer_cylinders.py", 4,
       "6 --bundles-per-rank=2 --max-iterations=50 "
       "--fwph-stop-check-tol 0.1 "
       "--default-rho=1 --solver-name={} ".format(solver_name))
do_one("farmer", "farmer_cylinders.py", 2,
       "6 --bundles-per-rank=2 --max-iterations=50 "
       "--default-rho=1 "
       "--solver-name={} --no-fwph --no-lagrangian".format(solver_name))
do_one("sizes",
       "sizes_cylinders.py",
       3,
       "--num-scens=10 --bundles-per-rank=0 --max-iterations=5 "
       "--default-rho=1 "
       "--solver-name={} --no-fwph".format(solver_name))
do_one("sizes",
       "sizes_cylinders.py",
       4,
       "--num-scens=3 --bundles-per-rank=0 --max-iterations=5 "
       "--default-rho=1 --solver-name={} --with-display-progress".format(solver_name))
do_one("sslp",
       "sslp_cylinders.py",
       4,
       "--instance-name=sslp_15_45_10 --bundles-per-rank=2 "
       "--max-iterations=5 --default-rho=1 "
       "--solver-name={} --fwph-stop-check-tol 0.01".format(solver_name))
do_one("hydro", "hydro_cylinders.py", 3,
       "--BFs=3,3 --bundles-per-rank=0 --max-iterations=100 "
       "--default-rho=1 --with-xhatspecific --with-lagrangian "
       "--solver-name={}".format(solver_name))

if egret_avail():
    do_one("acopf3", "ccopf2wood.py", 2, "2 3 2 0")

print("\nSlow runs ahead...\n")
do_one("uc", "uc3wood.py", 3, "10 0 2 fixer")
do_one("uc", "uc4wood.py", 4, "10 0 2 fixer")
do_one("uc", "uc_lshaped.py", 2,
       "--bundles-per-rank=0 --max-iterations=2 "
       "--default-rho=1 --num-scens=10 "
       "--solver-name={} --threads=1".format(solver_name))
do_one("farmer", "farmer_lshapedhub.py", 2,
       "3 --bundles-per-rank=0 --max-iterations=50 "
       "--default-rho=1 "
       "--solver-name={} --no-fwph --threads=1".format(solver_name))


if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
else:
    print("\nAll OK.")
