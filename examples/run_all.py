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
import pandas as pd
from datetime import datetime as dt

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
    """ return the code"""
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
    
def do_one_mmw(dirname, runefstring, npyfile, mmwargstring):
    # assume that the dirname matches the module name
    
    os.chdir(dirname)
    # solve ef, save .npy file (file name hardcoded in progname at the moment)
    code = os.system("echo {} && {}".format(runefstring, runefstring))

    if code!=0:
        if dirname not in badguys:
            badguys[dirname] = [runefstring]
        else:
            badguys[dirname].append(runefstring)
    # run mmw, remove .npy file
    else:
        runstring = "python -m mpisppy.confidence_intervals.mmw_conf {} --xhatpath {} {}".\
                    format(dirname, npyfile, mmwargstring)
        code = os.system("echo {} && {}".format(runstring, runstring))
        if code != 0:
            if dirname not in badguys:
                badguys[dirname] = [runstring]
            else:
                badguys[dirname].append(runstring)
        
        os.remove(npyfile)
    os.chdir("..")

do_one("farmer", "farmer_ef.py", 1,
       "1 3 {}".format(solver_name))
# for farmer_cylinders, the first arg is num_scens and is required
do_one("farmer", "farmer_cylinders.py",  4,
       "--num-scens 3 --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name={} "
       "--use-norm-rho-converger --use-norm-rho-updater --lagrangian --xhatshuffle --fwph "
       "--display-convergence-detail".format(solver_name))
do_one("farmer", "farmer_lshapedhub.py", 2,
       "--num-scens 3 --bundles-per-rank=0 --max-iterations=50 "
       "--solver-name={} --rel-gap=0.0 "
       "--xhatlshaped --max-solver-threads=1".format(solver_name))
do_one("farmer", "farmer_cylinders.py", 3,
       "--num-scens 3 --bundles-per-rank=0 --max-iterations=50 "
       "--default-rho=1 "
       "--solver-name={} --lagranger --xhatlooper".format(solver_name))
do_one("farmer", "farmer_cylinders.py", 3,
       "--num-scens 6 --bundles-per-rank=2 --max-iterations=50 "
       "--default-rho=1 --lagrangian --xhatshuffle "
       "--solver-name={}".format(solver_name))
do_one("farmer", "farmer_cylinders.py", 4,
       "--num-scens 6 --bundles-per-rank=2 --max-iterations=50 "
       "--fwph-stop-check-tol 0.1 "
       "--default-rho=1 --solver-name={} --lagrangian --xhatshuffle --fwph".format(solver_name))
do_one("farmer", "farmer_cylinders.py", 2,
       "--num-scens 6 --bundles-per-rank=2 --max-iterations=50 "
       "--default-rho=1 "
       "--solver-name={} --xhatshuffle".format(solver_name))
do_one("farmer", "farmer_cylinders.py", 3,
       "--num-scens 3 --bundles-per-rank=0 --max-iterations=1 "
       "--default-rho=1 --tee-rank0-solves "
       "--solver-name={} --lagrangian --xhatshuffle".format(solver_name))
time_one("FarmerLinProx", "farmer", "farmer_cylinders.py", 3,
       "--num-scens 3 --default-rho=1.0 --max-iterations=50 "
       "--display-progress --rel-gap=0.0 --abs-gap=0.0 "
       "--linearize-proximal-terms --proximal-linearization-tolerance=1.e-6 "
       "--solver-name={} --lagrangian --xhatshuffle".format(solver_name))

do_one("farmer/from_pysp", "concrete_ampl.py", 1, solver_name)
do_one("farmer/from_pysp", "abstract.py", 1, solver_name)

do_one("farmer",
       "farmer_cylinders.py",
       2,
       f"--num-scens 3 --max-iterations=10 --default-rho=1.0 --display-progress  --bundles-per-rank=0 --xhatshuffle --aph-gamma=1.0 --aph-nu=1.0 --aph-frac-needed=1.0 --aph-dispatch-frac=1.0 --abs-gap=1 --aph-sleep-seconds=0.01 --run-async --solver-name={solver_name}")
do_one("farmer",
       "farmer_cylinders.py",
       2,
       f"--num-scens 3 --max-iterations=10 --default-rho=1.0 --display-progress  --bundles-per-rank=0 --xhatlooper --aph-gamma=1.0 --aph-nu=1.0 --aph-frac-needed=1.0 --aph-dispatch-frac=0.25 --abs-gap=1 --aph-sleep-seconds=0.01 --run-async --solver-name={solver_name}")
do_one("farmer",
       "farmer_cylinders.py",
       2,
       f"--num-scens 30 --max-iterations=10 --default-rho=1.0 --display-progress  --bundles-per-rank=0 --xhatlooper --aph-gamma=1.0 --aph-nu=1.0 --aph-frac-needed=1.0 --aph-dispatch-frac=1 --abs-gap=1 --aph-sleep-seconds=0.01 --run-async --bundles-per-rank=5 --solver-name={solver_name}")

do_one("farmer",
       "farmer_cylinders.py", 4,
       f"--num-scens 3 --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name={solver_name}  --lagrangian --xhatshuffle --fwph --max-stalled-iters 1")

do_one("farmer",
       "farmer_cylinders.py",
       2,
       f"--num-scens 30 --max-iterations=10 --default-rho=1.0 --display-progress  --bundles-per-rank=0 --xhatshuffle --aph-gamma=1.0 --aph-nu=1.0 --aph-frac-needed=1.0 --aph-dispatch-frac=0.5 --abs-gap=1 --aph-sleep-seconds=0.01 --run-async --bundles-per-rank=5 --solver-name={solver_name}")

do_one("farmer",
       "farmer_ama.py",
       3,
       f"--num-scens=10 --crops-multiplier=3 --farmer-with-integer --EF-solver-name={solver_name}")

do_one("farmer",
       "farmer_seqsampling.py",
       1,
       f"--num-scens 3 --crops-multiplier=1  --EF-solver-name={solver_name} "
       "--BM-h 2 --BM-q 1.3 --confidence-level 0.95 --BM-vs-BPL BM")

do_one("farmer",
       "farmer_seqsampling.py",
       1,
       f"--num-scens 3 --crops-multiplier=1  --EF-solver-name={solver_name} "
       "--BPL-c0 25 --BPL-eps 100 --confidence-level 0.95 --BM-vs-BPL BPL")

do_one("netdes", "netdes_cylinders.py", 5,
       "--max-iterations=3 --instance-name=network-10-20-L-01 "
       "--solver-name={} --rel-gap=0.0 --default-rho=1 "
       "--slammax --lagrangian --xhatshuffle --cross-scenario-cuts --max-solver-threads=2".format(solver_name))

# sizes is slow for xpress so try linearizing the proximal term.
do_one("sizes",
       "sizes_cylinders.py",
       3,
       "--linearize-proximal-terms "
       "--num-scens=10 --bundles-per-rank=0 --max-iterations=5 "
       "--default-rho=1 --lagrangian --xhatshuffle "
       "--iter0-mipgap=0.01 --iterk-mipgap=0.001 "
       "--solver-name={}".format(solver_name))

do_one("sizes",
       "sizes_cylinders.py",
       3,
       "--linearize-proximal-terms "
       "--num-scens=10 --bundles-per-rank=0 --max-iterations=5 "
       "--default-rho=1 --lagrangian --xhatxbar "
       "--iter0-mipgap=0.01 --iterk-mipgap=0.001 "
       "--solver-name={}".format(solver_name))

do_one("sizes",
       "sizes_cylinders.py",
       4,
       "--num-scens=3 --bundles-per-rank=0 --max-iterations=5 "
       "--iter0-mipgap=0.01 --iterk-mipgap=0.005 "
       "--default-rho=1 --lagrangian --xhatshuffle --fwph "
       "--solver-name={} --display-progress".format(solver_name))
do_one("sizes", "sizes_pysp.py", 1, "3 {}".format(solver_name))

do_one("sslp",
       "sslp_cylinders.py",
       4,
       "--instance-name=sslp_15_45_10 --bundles-per-rank=2 "
       "--max-iterations=5 --default-rho=1 "
       "--lagrangian --xhatshuffle --fwph "
       "--linearize-proximal-terms "
       "--solver-name={} --fwph-stop-check-tol 0.01".format(solver_name))

do_one("hydro", "hydro_cylinders.py", 3,
       "--branching-factors \"3 3\" --bundles-per-rank=0 --max-iterations=100 "
       "--default-rho=1 --xhatshuffle --lagrangian "
       "--solver-name={}".format(solver_name))
do_one("hydro", "hydro_cylinders.py", 3,
       "--branching-factors \'3 3\' --bundles-per-rank=0 --max-iterations=100 "
       "--default-rho=1 --xhatshuffle --lagrangian "
       "--solver-name={} --stage2EFsolvern={}".format(solver_name, solver_name))

do_one("hydro", "hydro_cylinders_pysp.py", 3,
       "--bundles-per-rank=0 --max-iterations=100 "
       "--default-rho=1 --xhatshuffle --lagrangian "
       "--solver-name={}".format(solver_name))

do_one("hydro", "hydro_ef.py", 1, solver_name)

# the next might hang with 6 ranks
do_one("aircond", "aircond_cylinders.py", 3,
       "--branching-factors \'4 3 2\' --bundles-per-rank=0 --max-iterations=100 "
       "--default-rho=1 --lagrangian --xhatshuffle "
       "--solver-name={}".format(solver_name))
do_one("aircond", "aircond_ama.py", 3,
       "--branching-factors \'3 3\' --bundles-per-rank=0 --max-iterations=100 "
       "--default-rho=1 --lagrangian --xhatshuffle "
       "--solver-name={}".format(solver_name))
time_one("AircondAMA", "aircond", "aircond_ama.py", 3,
       "--branching-factors \'3 3\' --bundles-per-rank=0 --max-iterations=100 "
       "--default-rho=1 --lagrangian --xhatshuffle "
       "--solver-name={}".format(solver_name))

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

#=========MMW TESTS==========
# do_one_mmw is special
do_one_mmw("farmer", f"python farmer_ef.py 3 3 {solver_name}", "farmer_cyl_nonants.npy", f"--MMW-num-batches=5 --confidence-level 0.95 --MMW-batch-size=10 --start-scen 4 --EF-solver-name={solver_name}")


#============================

if egret_avail():
    do_one("acopf3", "ccopf2wood.py", 2, f"2 3 2 0 {solver_name}")
    do_one("acopf3", "fourstage.py", 4, f"2 2 2 1 0 {solver_name}")        

#  sizes kills the github tests using xpress
#  so we use linearized proximal terms
do_one("sizes", "sizes_demo.py", 1, " {}".format(solver_name))

do_one("sizes",
       "special_cylinders.py",
       3,
       "--lagrangian --xhatshuffle "
       "--num-scens=3 --bundles-per-rank=0 --max-iterations=5 "
       "--iter0-mipgap=0.01 --iterk-mipgap=0.001 --linearize-proximal-terms "
       "--default-rho=1 --solver-name={} --display-progress".format(solver_name))

if not nouc and egret_avail():
    print("\nSlow runs ahead...\n")
    # 3-scenario UC
    do_one("uc", "uc_ef.py", 1, solver_name+" 3")

    do_one("uc", "uc_cylinders.py", 4,
           "--bundles-per-rank=0 --max-iterations=2 "
           "--default-rho=1 --num-scens=3 --max-solver-threads=2 "
           "--lagrangian-iter0-mipgap=1e-7 --fwph "
           " --lagrangian --xhatshuffle "
           "--ph-mipgaps-json=phmipgaps.json "
           "--solver-name={}".format(solver_name))
    do_one("uc", "uc_lshaped.py", 2,
           "--bundles-per-rank=0 --max-iterations=5 "
           "--default-rho=1 --num-scens=3 --xhatlshaped "
           "--solver-name={} --max-solver-threads=1".format(solver_name))
    do_one("uc", "uc_cylinders.py", 3,
           "--run-aph --bundles-per-rank=0 --max-iterations=2 "
           "--default-rho=1 --num-scens=3 --max-solver-threads=2 "
           "--lagrangian-iter0-mipgap=1e-7 --lagrangian --xhatshuffle "
           "--ph-mipgaps-json=phmipgaps.json "
           "--solver-name={}".format(solver_name))    
    # as of May 2022, this one works well, but outputs some crazy messages
    do_one("uc", "uc_ama.py", 3,
           "--bundles-per-rank=0 --max-iterations=2 "
           "--default-rho=1 --num-scens=3 "
           "--fixer-tol=1e-2 --lagranger --xhatshuffle "
           "--solver-name={}".format(solver_name))
    
    # 10-scenario UC
    time_one("UC_cylinder10scen", "uc", "uc_cylinders.py", 3,
             "--bundles-per-rank=5 --max-iterations=2 "
             "--default-rho=1 --num-scens=10 --max-solver-threads=2 "
             "--lagrangian-iter0-mipgap=1e-7 "
             "--ph-mipgaps-json=phmipgaps.json "
             "--lagrangian --xhatshuffle "
             "--solver-name={}".format(solver_name))
    # note that fwph takes a long time to do one iteration
    do_one("uc", "uc_cylinders.py", 4,
           "--bundles-per-rank=5 --max-iterations=2 "
           "--default-rho=1 --num-scens=10 --max-solver-threads=2 "
           " --lagrangian --xhatshuffle --fwph "
           "--lagrangian-iter0-mipgap=1e-7 "
           "--ph-mipgaps-json=phmipgaps.json "
           "--solver-name={}".format(solver_name))
    do_one("uc", "uc_cylinders.py", 5,
           "--bundles-per-rank=5 --max-iterations=2 "
           "--default-rho=1 --num-scens=10 --max-solver-threads=2 "
           " --lagrangian --xhatshuffle --fwph "
           "--lagrangian-iter0-mipgap=1e-7 --cross-scenario-cuts "
           "--ph-mipgaps-json=phmipgaps.json --cross-scenario-iter-cnt=4 "
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
