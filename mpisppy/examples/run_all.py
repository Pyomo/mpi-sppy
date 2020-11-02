# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Run a lot of examples for regression testing; dlw May 2020
# Not intended to be user-friendly.
# Assumes you run from the examples directory.
# Optional command line arguments: solver_name mpiexec_arg nouc
# E.g. python run_all.py
#      python run_all.py cplex
#      python run_all.py gurobi_persistent --oversubscribe
#      python run_all.py gurobi_persistent -envall nouc
#      (envall does nothing; it is just a place-holder)

import os
import sys

solver_name = "gurobi_persistent"
if len(sys.argv) > 1:
    solver_name = sys.argv[1]

# Use oversubscribe if your computer does not have enough cores.
# Don't use this unless you have to.
# (This may not be allowed on some versions of mpiexec)
mpiexec_arg = ""  # "--oversubscribe" or "-envall"
if len(sys.argv) > 2:
    mpiexec_arg = sys.argv[2]

# set nouc for testing with community solvers
nouc = False
if len(sys.argv) > 3:
    nouc = True
    if sys.argv[3] != "nouc":
        raise RuntimeError("Third arg can only be nouc (you have {})".\
                           format(sys.argv[3]))
    
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

do_one("farmer", "farmer_ef.py", 1,
       "1 3 {}".format(solver_name))
do_one("farmer", "farmer_lshapedhub.py", 2,
       "3 --bundles-per-rank=0 --max-iterations=50 "
       "--solver-name={} --rel-gap=0.0 "
       "--no-fwph --max-solver-threads=1".format(solver_name))
# for farmer_cylinders, the first arg is num_scens and is required
do_one("farmer", "farmer_cylinders.py", 3,
       "3 --bundles-per-rank=0 --max-iterations=50 "
       "--default-rho=1 "
       "--solver-name={} --no-fwph".format(solver_name))
do_one("farmer", "farmer_lagranger.py", 3,
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
do_one("farmer", "farmer_cylinders.py", 3,
       "3 --bundles-per-rank=0 --max-iterations=1 "
       "--default-rho=1 --with-tee-rank0-solves "
       "--solver-name={} --no-fwph".format(solver_name))
do_one("farmer/from_pysp", "concrete_ampl.py", 1, solver_name)
do_one("farmer/from_pysp", "abstract.py", 1, solver_name)
do_one("netdes", "netdes_cylinders.py", 5,
       "--max-iterations=3 --instance-name=network-10-20-L-01 "
       "--solver-name={} --rel-gap=0.0 --default-rho=1 "
       "--no-fwph --max-solver-threads=2".format(solver_name))
do_one("sizes",
       "sizes_cylinders.py",
       3,
       "--num-scens=10 --bundles-per-rank=0 --max-iterations=5 "
       "--default-rho=1 "
       "--iter0-mipgap=0.01 --iterk-mipgap=0.001 "
       "--solver-name={} --no-fwph".format(solver_name))
do_one("sizes",
       "sizes_cylinders.py",
       4,
       "--num-scens=3 --bundles-per-rank=0 --max-iterations=5 "
       "--iter0-mipgap=0.01 --iterk-mipgap=0.001 "
       "--default-rho=1 --solver-name={} --with-display-progress".format(solver_name))
do_one("sizes", "sizes_pysp.py", 1, "3 {}".format(solver_name))
do_one("sizes", "sizes_demo.py", 1, " {}".format(solver_name))
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
    do_one("acopf3", "ccopf2wood.py", 2, f"2 3 2 0 {solver_name}")
    do_one("acopf3", "fourstage.py", 4, f"2 2 2 1 0 {solver_name}")        


if not nouc and egret_avail():
    print("\nSlow runs ahead...\n")
    # 3-scenario UC
    do_one("uc", "uc_ef.py", 1, solver_name+" 3")
    do_one("uc", "uc_lshaped.py", 2,
           "--bundles-per-rank=0 --max-iterations=5 "
           "--default-rho=1 --num-scens=3 "
           "--solver-name={} --max-solver-threads=1 --no-fwph".format(solver_name))
    do_one("uc", "uc_cylinders.py", 4,
           "--bundles-per-rank=0 --max-iterations=2 "
           "--default-rho=1 --num-scens=3 --max-solver-threads=2 "
           "--lagrangian-iter0-mipgap=1e-7 --no-cross-scenario-cuts "
           "--ph-mipgaps-json=phmipgaps.json "
           "--solver-name={}".format(solver_name))
    # 10-scenario UC
    do_one("uc", "uc_cylinders.py", 3,
           "--bundles-per-rank=5 --max-iterations=2 "
           "--default-rho=1 --num-scens=10 --max-solver-threads=2 "
           "--lagrangian-iter0-mipgap=1e-7 --no-cross-scenario-cuts "
           "--ph-mipgaps-json=phmipgaps.json "
           "--no-fwph "
           "--solver-name={}".format(solver_name))
    do_one("uc", "uc_cylinders.py", 4,
           "--bundles-per-rank=5 --max-iterations=2 "
           "--default-rho=1 --num-scens=10 --max-solver-threads=2 "
           "--lagrangian-iter0-mipgap=1e-7 --no-cross-scenario-cuts "
           "--ph-mipgaps-json=phmipgaps.json "
           "--solver-name={}".format(solver_name))
    do_one("uc", "uc_cylinders.py", 5,
           "--bundles-per-rank=5 --max-iterations=2 "
           "--default-rho=1 --num-scens=10 --max-solver-threads=2 "
           "--lagrangian-iter0-mipgap=1e-7 --with-cross-scenario-cuts "
           "--ph-mipgaps-json=phmipgaps.json --cross-scenario-iter-cnt=4 "
           "--solver-name={}".format(solver_name))


if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
else:
    print("\nAll OK.")
