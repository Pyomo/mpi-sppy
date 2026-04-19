###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Run a lot of examples for regression testing; dlw May 2020
# Not intended to be user-friendly.
# Assumes you run from the examples directory.
# Optional command line arguments: solver_name mpiexec_arg nouc
# E.g. python run_all.py
#      python run_all.py cplex
#      python run_all.py gurobi_persistent --oversubscribe
#      python run_all.py gurobi_persistent -envall nouc
#      (envall does nothing; it is just a place-holder; might not work with your mpiexec)
# For coverage: python run_all.py gurobi_persistent "" "" --python-args="-m coverage run --parallel-mode --source=mpisppy"

import os
import sys

# Parse --python-args (extra args inserted after "python" in subcommands, e.g. for coverage)
# and --first-part-only / --second-part-only (split pre-nouc work in half so CI can parallelize).
python_args = ""
first_part_only = False
second_part_only = False
_remaining = []
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i].startswith("--python-args="):
        python_args = sys.argv[_i].split("=", 1)[1]
    elif sys.argv[_i] == "--python-args" and _i + 1 < len(sys.argv):
        _i += 1
        python_args = sys.argv[_i]
    elif sys.argv[_i] == "--first-part-only":
        first_part_only = True
    elif sys.argv[_i] == "--second-part-only":
        second_part_only = True
    else:
        _remaining.append(sys.argv[_i])
    _i += 1
sys.argv = [sys.argv[0]] + _remaining
if first_part_only and second_part_only:
    raise RuntimeError("--first-part-only and --second-part-only are mutually exclusive")
run_first_part = not second_part_only
run_second_part = not first_part_only

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
    except Exception:
        return False

    path = str(egret.__path__)
    left = path.find("'")
    right = path.find("'", left+1)
    egretrootpath = path[left+1:right]

    egret_thirdparty_path = os.path.join(egretrootpath, "thirdparty")
    if os.path.exists(os.path.join(egret_thirdparty_path, "pglib-opf-master")):
        return True

    from egret.thirdparty.get_pglib_opf import get_pglib_opf
    get_pglib_opf(egret_thirdparty_path)
    return True

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

def do_one_mmw(dirname, modname, runefstring, npyfile, mmwargstring):
    # assume that the dirname matches the module name

    os.chdir(dirname)
    # solve ef, save .npy file (file name hardcoded in progname at the moment)
    if python_args:
        runefstring = runefstring.replace("python ", f"python {python_args} ", 1)
    code = os.system("echo {} && {}".format(runefstring, runefstring))

    if code!=0:
        if dirname not in badguys:
            badguys[dirname] = [runefstring]
        else:
            badguys[dirname].append(runefstring)
    # run mmw, remove .npy file
    else:
        runstring = "python {} -m mpisppy.confidence_intervals.mmw_conf {} --xhatpath {} {}".\
                    format(python_args, modname, npyfile, mmwargstring)
        code = os.system("echo {} && {}".format(runstring, runstring))
        if code != 0:
            if dirname not in badguys:
                badguys[dirname] = [runstring]
            else:
                badguys[dirname].append(runstring)

        os.remove(npyfile)
    os.chdir("..")
    os.chdir("..")  # moved to CI directory

# -------- First part: farmer family --------
if run_first_part:
    do_one("farmer/CI", "farmer_ef.py", 1,
           "1 3 {}".format(solver_name))
    # for farmer_cylinders, the first arg is num_scens and is required
    do_one("farmer/archive", "farmer_cylinders.py",  3,
           "--num-scens 3 --max-iterations=50 --default-rho=1 --solver-name={} "
           "--primal-dual-converger --primal-dual-converger-tol=0.5 --lagrangian --xhatshuffle "
           "--intra-hub-conv-thresh -0.1 --rel-gap=1e-6".format(solver_name))
    do_one("farmer/archive", "farmer_cylinders.py",  5,
           "--num-scens 3 --max-iterations=20 --default-rho=1 --solver-name={} "
           "--use-norm-rho-converger --use-norm-rho-updater --rel-gap=1e-6 --lagrangian --lagranger "
           "--xhatshuffle --fwph --W-fname=out_ws.txt --Xbar-fname=out_xbars.txt "
           "--ph-track-progress --track-convergence=4 --track-xbar=4 --track-nonants=4 "
           "--track-duals=4".format(solver_name))
    do_one("farmer/archive", "farmer_cylinders.py",  5,
           "--num-scens 3 --max-iterations=20 --default-rho=1 --solver-name={} "
           "--use-norm-rho-converger --use-norm-rho-updater --lagrangian --lagranger --xhatshuffle --fwph "
           "--init-W-fname=out_ws.txt --init-Xbar-fname=out_xbars.txt --ph-track-progress --track-convergence=4 "  "--track-xbar=4 --track-nonants=4 --track-duals=4 ".format(solver_name))
    do_one("farmer", "farmer_lshapedhub.py", 2,
           "--num-scens 3 --max-iterations=50 "
           "--solver-name={} --rel-gap=0.0 "
           "--xhatlshaped --max-solver-threads=1".format(solver_name))
    do_one("farmer/archive", "farmer_cylinders.py", 3,
           "--num-scens 3 --max-iterations=50 "
           "--default-rho=1 "
           "--solver-name={} --lagranger --xhatlooper".format(solver_name))
    do_one("farmer", "../../mpisppy/generic_cylinders.py", 3,
           "--module-name farmer --num-scens 6 "
           "--rel-gap 0.001 --max-iterations=50 "
           "--grad-rho --grad-order-stat 0.5 "
           "--default-rho=2 --solver-name={} --lagrangian --xhatshuffle".format(solver_name))
    do_one("farmer", "../../mpisppy/generic_cylinders.py", 3,
           "--module-name farmer --num-scens 6 "
           "--rel-gap 0.001 --max-iterations=50 "
           "--grad-rho --indep-denom "
           "--default-rho=2 --solver-name={} --lagrangian --xhatshuffle".format(solver_name))
    do_one("farmer", "../../mpisppy/generic_cylinders.py", 4,
           "--module-name farmer --num-scens 6 "
           "--rel-gap 0.001 --max-iterations=50 "
           "--ph-primal-hub --ph-dual --ph-dual-rescale-rho-factor=0.1 "
           "--default-rho=2 --solver-name={} --lagrangian --xhatshuffle".format(solver_name))
    do_one("farmer", "../../mpisppy/generic_cylinders.py", 4,
           "--module-name farmer "
           "--num-scens 6 --max-iterations=50 --grad-rho --grad-order-stat 0.5 "
           "--ph-dual-grad-order-stat 0.3 "
           "--ph-primal-hub --ph-dual --ph-dual-rescale-rho-factor=0.1 --ph-dual-rho-multiplier 0.2 "
           "--default-rho=1 --solver-name={} --lagrangian --xhatshuffle".format(solver_name))

    do_one("farmer/from_pysp", "concrete_ampl.py", 1, solver_name)
    do_one("farmer/from_pysp", "abstract.py", 1, solver_name)

    do_one("farmer/archive",
           "farmer_cylinders.py", 4,
           f"--num-scens 3 --max-iterations=20 --default-rho=1 --solver-name={solver_name}  --lagrangian --xhatshuffle --fwph --max-stalled-iters 1")

    do_one("farmer/archive",
           "../../../mpisppy/generic_cylinders.py",
           4,
           "--module-name farmer --farmer-with-integer "
           "--num-scens=3 "
           "--lagrangian --ph-primal-hub "
           "--max-iterations=10 --default-rho=0.1 "
           "--relaxed-ph-rescale-rho-factor=10 "
           "--relaxed-ph --relaxed-ph-fixer --xhatshuffle "
           "--linearize-proximal-terms "
           "--rel-gap=0.0 "
           "--solver-name={}".format(solver_name))

# -------- Second part: netdes, sizes, sslp, hydro, aircond, MMW --------
if run_second_part:
    # NOTE: Pyomo OBBT does not support persistent solvers as of Aug 2025
    direct_solver_name = solver_name.replace("_persistent", "_direct") if "_persistent" in solver_name else solver_name
    do_one("netdes", "netdes_cylinders.py", 4,
           "--max-iterations=3 --instance-name=network-10-20-L-01 "
           "--solver-name={} --rel-gap=0.0 --default-rho=10000 --presolve --obbt --obbt-solver={} "
           "--slammax --subgradient-hub --xhatshuffle --cross-scenario-cuts --max-solver-threads=2".format(solver_name, direct_solver_name))

    # sizes is slow for xpress so try linearizing the proximal term.
    do_one("sizes",
           "sizes_cylinders.py",
           3,
           "--config-file=sizes_config.txt "
           "--num-scens=10 "
           "--solver-name={}".format(solver_name))

    do_one("sizes",
           "sizes_cylinders.py",
           3,
           "--linearize-proximal-terms "
           "--num-scens=10 --max-iterations=5 "
           "--default-rho=1 --lagrangian --xhatxbar "
           "--iter0-mipgap=0.01 --iterk-mipgap=0.001 "
           "--solver-name={}".format(solver_name))

    do_one("sizes", "sizes_pysp.py", 1, "3 {}".format(solver_name))
    do_one("sslp",
           "sslp_cylinders.py",
           4,
           "--instance-name=sslp_15_45_10 "
           "--integer-relax-then-enforce "
           "--integer-relax-then-enforce-ratio=0.8 "
           "--lagrangian "
           "--reduced-costs-rho "
           "--max-iterations=20 --default-rho=1e-6 "
           "--reduced-costs --rc-fixer --xhatshuffle "
           "--linearize-proximal-terms "
           "--rel-gap=0.0 --surrogate-nonant "
           "--use-primal-dual-rho-updater --primal-dual-rho-update-threshold=10 "
           "--solver-name={}".format(solver_name))
    do_one("hydro", "hydro_cylinders.py", 3,
           "--branching-factors \'3 3\' --max-iterations=100 "
           "--default-rho=1 --xhatshuffle --lagrangian "
           "--solver-name={} --stage2EFsolvern={}".format(solver_name, solver_name))

    do_one("hydro", "hydro_cylinders_pysp.py", 3,
           "--max-iterations=100 "
           "--default-rho=1 --xhatshuffle --lagrangian "
           "--solver-name={}".format(solver_name))

    # the next might hang with 6 ranks
    do_one("aircond", "aircond_cylinders.py", 3,
           "--branching-factors \'4 3 2\' --max-iterations=100 "
           "--default-rho=1 --lagrangian --xhatshuffle "
           "--solver-name={}".format(solver_name))
    do_one("aircond", "aircond_ama.py", 3,
           "--branching-factors \'3 3\' --max-iterations=100 "
           "--default-rho=1 --lagrangian --xhatshuffle "
           "--solver-name={}".format(solver_name))

    # aircondMulti: multi-product aircond, model module in
    # mpisppy/tests/examples/aircondMulti.py. generic_cylinders needs
    # --stage2EFsolvern for multistage --xhatshuffle and this module
    # doesn't register it, so use --xhatxbar for the inner bound.
    do_one("aircondMulti", "../../mpisppy/generic_cylinders.py", 3,
           "--module-name ../../mpisppy/tests/examples/aircondMulti "
           "--branching-factors \'3 3\' --max-iterations=5 "
           "--default-rho=1 --lagrangian --xhatxbar "
           "--solver-name={}".format(solver_name))

    #=========MMW TESTS==========
    # do_one_mmw is special
    do_one_mmw("farmer/CI", "farmer", f"python farmer_ef.py 1 3 0 {solver_name}", "farmer_root_nonants.npy", f"--MMW-num-batches=5 --confidence-level 0.95 --MMW-batch-size=10 --start-scen 4 --EF-solver-name={solver_name}")


#============================

#  sizes kills the github tests using xpress
#  so we use linearized proximal terms

if not nouc:
    # put a few slow runs and/or runs that are trouble on github in the uc group

    do_one("sizes",
           "special_cylinders.py",
           3,
           "--lagrangian --xhatshuffle "
           "--num-scens=3 --max-iterations=5 "
           "--iter0-mipgap=0.01 --iterk-mipgap=0.001 --linearize-proximal-terms "
           "--default-rho=1 --solver-name={} --display-progress".format(solver_name))

    do_one("sizes",
           "sizes_cylinders.py",
           4,
           "--num-scens=3 --max-iterations=5 "
           "--iter0-mipgap=0.01 --iterk-mipgap=0.005 "
           "--default-rho=1 --lagrangian --xhatshuffle --fwph "
           "--solver-name={} --display-progress".format(solver_name))

    if egret_avail():
        print("\nSlow runs ahead...\n")
        do_one("acopf3", "ccopf2wood.py", 2, f"2 3 2 0 {solver_name}")
        do_one("acopf3", "fourstage.py", 4, f"2 2 2 1 0 {solver_name}")

        # 3-scenario UC
        do_one("uc", "uc_ef.py", 1, solver_name+" 3")

        do_one("uc", "gradient_uc_cylinders.py", 15,
               "--max-iterations=100 --default-rho=1 "
               "--xhatshuffle --lagrangian --num-scens=5 --max-solver-threads=2 "
               "--lagrangian-iter0-mipgap=1e-7 --ph-mipgaps-json=phmipgaps.json "
               f"--solver-name={solver_name} --xhatpath uc_cyl_nonants.npy "
               "--rel-gap 0.00001 --abs-gap=1 --intra-hub-conv-thresh=-1 "
               "--grad-rho-setter --grad-order-stat 0.5 "
               "--grad-dynamic-primal-crit")

        do_one("uc", "uc_cylinders.py", 4,
               "--max-iterations=2 "
               "--default-rho=1 --num-scens=3 --max-solver-threads=2 "
               "--lagrangian-iter0-mipgap=1e-7 --fwph "
               " --lagrangian --xhatshuffle "
               "--ph-mipgaps-json=phmipgaps.json "
               "--solver-name={}".format(solver_name))
        do_one("uc", "uc_lshaped.py", 2,
               "--max-iterations=5 "
               "--default-rho=1 --num-scens=3 --xhatlshaped "
               "--solver-name={} --max-solver-threads=1".format(solver_name))
        do_one("uc", "uc_cylinders.py", 3,
               "--run-aph --max-iterations=2 "
               "--default-rho=1 --num-scens=3 --max-solver-threads=2 "
               "--lagrangian-iter0-mipgap=1e-7 --lagrangian --xhatshuffle "
               "--ph-mipgaps-json=phmipgaps.json "
               "--solver-name={}".format(solver_name))
        # as of May 2022, this one works well, but outputs some crazy messages
        do_one("uc", "uc_ama.py", 3,
              "--max-iterations=2 "
               "--default-rho=1 --num-scens=3 "
               "--fixer-tol=1e-2 --lagranger --xhatshuffle "
               "--solver-name={}".format(solver_name))

        do_one("sizes", "sizes_demo.py", 1, " {}".format(solver_name))

if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
    sys.exit(1)
else:
    print("\nAll OK.")
