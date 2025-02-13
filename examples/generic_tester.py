###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# This might make a mess in terms of output files....
# (re)baseline by uncommenting rebaseline_xhat lines
# The baselines are in the subdirectories of the examples/test_data 
# NOTE: the asynchronous nature of mip-sppy makes it hard to hit baselines.
# Run a lot of generic_cylinders examples for regression testing; dlw Aug 2024
# Not intended to be user-friendly.
# Assumes you run from the examples directory (that is what .. often refers to)
# Uses baselines from subdirectories of that directory.
# The basline directory for each command line uses mod name within...
# Optional command line arguments: solver_name mpiexec_arg nouc
# E.g. python generic_tester.py
#      python generic_tester.py cplex
#      python generic_tester.py gurobi_persistent --oversubscribe
#      python generic_tester.py gurobi_persistent -envall nouc
#      (envall does nothing; it is just a place-holder)

import os
import sys
import shutil
import numpy as np
from numpy.linalg import norm

# Write solutions here, then delete. (.. makes it local to examples)
XHAT_TEMP = os.path.join("..","AAA_delete_this_from_generic_tester")

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
        raise RuntimeError(f"Third arg can only be nouc; you have {sys.argv[3]}")

badguys = dict()  # bad return code
xhat_losers = dict()  # does not match baseline well enough

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


def _append_soln_output_dir(argstring, outbase):
    outarg = "--solution-base-name"
    assert outarg not in argstring, "The tester controls solution writing"
    retval = argstring + f" {outarg} {outbase}"
    return retval


def rebaseline_xhat(dirname, modname, np, argstring, baseline_dir):
    # Add the write output to the command line, then do one
    # note: baseline_dir must exist (and probably needs to be pushed to github)
    fullarg = _append_soln_output_dir(argstring,
                                      os.path.join("..", baseline_dir, modname))
    do_one(dirname, modname, np, fullarg)  # do not check a baseline_dir, write one


def _check_baseline_xhat(modname, argstring, baseline_dir, tol=1e-6):
    # return true if OK, False otherwise
    # compare the baseline_dir to XHAT_TEMP
    
    xhat = np.load(os.path.join(XHAT_TEMP, f"{modname}.npy"))
    base_xhat = np.load(os.path.join(baseline_dir, f"{modname}.npy"))
    d2 = norm(xhat - base_xhat, 2)
    if d2 > tol:
        print(f"{d2=}, {tol=}, {argstring=}")
        return False
    else:
        return True


def _xhat_dir_setup(modname):
    if os.path.exists(XHAT_TEMP):
        shutil.rmtree(XHAT_TEMP)    
    os.makedirs(XHAT_TEMP)
    # .. is prepended elsewhere
    return os.path.join(XHAT_TEMP, modname)


def _xhat_dir_teardown():
    if os.path.exists(XHAT_TEMP):
        shutil.rmtree(XHAT_TEMP)    


def do_one(dirname, modname, np, argstring, xhat_baseline_dir=None, tol=1e-6):
    """ return the code"""
    os.chdir(dirname)  # taking us down a level from examples
    if xhat_baseline_dir is not None:
        fullarg = _append_soln_output_dir(argstring, _xhat_dir_setup(modname))         
    else:
        # we might be making a baseline, or just not using one
        fullarg = argstring

    runstring = "mpiexec {} -np {} python -u -m mpi4py -m mpisppy.generic_cylinders --module-name {} {}".\
                format(mpiexec_arg, np, modname, fullarg)
    # The top process output seems to be cached by github actions
    # so we need oputput in the system call to help debug
    code = os.system("echo {} && {}".format(runstring, runstring))
    if code != 0:
        if dirname not in badguys:
            badguys[dirname] = [runstring]
        else:
            badguys[dirname].append(runstring)
    if xhat_baseline_dir is not None:
        bd = os.path.join("..", xhat_baseline_dir)
        if not _check_baseline_xhat(modname, argstring, bd, tol=tol):
            if dirname not in xhat_losers:
                xhat_losers[dirname] = [runstring]
            else:
                xhat_losers[dirname].append(runstring)
        _xhat_dir_teardown()
    os.chdir("..")
    return code

###################### main code ######################
# directory names are given relative to the examples directory whence this runs

farmeref = (f"--EF --num-scens 3 --EF-solver-name={solver_name}")
#rebaseline_xhat("farmer", "farmer", 1, farmeref, "test_data/farmeref_baseline")
do_one("farmer", "farmer", 1, farmeref, xhat_baseline_dir = "test_data/farmeref_baseline")


hydroef = (f"--EF --branching-factors '3 3' --EF-solver-name={solver_name}")
#rebaseline_xhat("hydro", "hydro", 1, hydroef, "test_data/hydroef_baseline")
do_one("hydro", "hydro", 1, hydroef, xhat_baseline_dir="test_data/hydroef_baseline")


hydroa = ("--max-iterations 100 --bundles-per-rank=0 --default-rho 1 "
          "--lagrangian --xhatshuffle --rel-gap 0.001 --branching-factors '3 3' "
          f"--stage2EFsolvern {solver_name} --solver-name={solver_name}")
#rebaseline_xhat("hydro", "hydro", 3, hydroa, "test_data/hydroa_baseline")
do_one("hydro", "hydro", 3, hydroa, xhat_baseline_dir="test_data/hydroa_baseline")

# write hydro bundles for at least some testing of multi-stage proper bundles
# (just looking for smoke)
hydro_wr = ("--pickle-bundles-dir hydro_pickles --scenarios-per-bundle 3"
            "--branching-factors '3 3' ")
do_one("hydro", "hydro", 3, hydro_wr, xhat_baseline_dir=None)

# write, then read, pickled scenarios
print("starting write/read pickled scenarios")
farmer_wr = "--pickle-scenarios-dir farmer_pickles --crops-mult 2 --num-scens 10"
do_one("farmer", "farmer", 2, farmer_wr, xhat_baseline_dir=None)
farmer_rd = f"--num-scens 10 --solver-name {solver_name} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01 --unpickle-scenarios-dir farmer_pickles"
#rebaseline_xhat("farmer", "farmer", 3, farmer_rd, "test_data/farmer_rd_baseline")
do_one("farmer", "farmer", 3, farmer_rd, xhat_baseline_dir="test_data/farmer_rd_baseline")

# Just a smoke test to make sure sizes_expression still exists and
# that lpfiles still executes.
sizese = ("--module-name sizes_expression --num-scens 3 --default-rho 1"
          f" --solver-name {solver_name} --max-iterations 0"
          " --scenario-lpfiles")
do_one("sizes", "sizes", 3, sizese, xhat_baseline_dir=None)   

quit()

# proper bundles
sslp_pb = ("--sslp-data-path ./data --instance-name sslp_15_45_10 "
           "--scenarios-per-bundle 1 --default-rho 1 "
           f"--solver-name {solver_name} --max-iterations 5 --lagrangian "
           "--xhatshuffle --rel-gap 0.001")
#rebaseline_xhat("sslp", "sslp", 3, sslp_pb, "test_data/sslp_pb_baseline")
do_one("sslp", "sslp", 3, sslp_pb, xhat_baseline_dir="test_data/sslp_pb_baseline")

# write, then read, pickled bundles
sslp_wr = "--module-name sslp --sslp-data-path ./data --instance-name sslp_15_45_10 --pickle-bundles-dir sslp_pickles --scenarios-per-bundle 1 --default-rho 1"
do_one("sslp", "sslp", 2, sslp_wr, xhat_baseline_dir=None)
sslp_rd = ("--sslp-data-path ./data --instance-name sslp_15_45_10 "
           "--unpickle-bundles-dir sslp_pickles --scenarios-per-bundle 1 "
           f"--default-rho 1 --solver-name={solver_name} "
           "--max-iterations 5 --lagrangian --xhatshuffle --rel-gap 0.001")
#rebaseline_xhat("sslp", "sslp", 3, sslp_rd, "test_data/sslp_rd_baseline")
do_one("sslp", "sslp", 3, sslp_rd, xhat_baseline_dir="test_data/sslp_rd_baseline")

hydroa_rc = ("--max-iterations 100 --bundles-per-rank=0 --default-rho 1 "
          "--reduced-costs --xhatshuffle --rel-gap 0.001 --branching-factors '3 3' "
          "--rc-fixer --reduced-costs-rho --reduced-costs-rho-multiplier=1.0 "
          f"--stage2EFsolvern {solver_name} --solver-name={solver_name}")
#rebaseline_xhat("hydro", "hydro", 3, hydroa, "test_data/hydroa_baseline")
do_one("hydro", "hydro", 3, hydroa_rc, xhat_baseline_dir = "test_data/hydroa_baseline")


if not nouc:
    uca = ("--num-scens 5 --max-iterations 3 --max-solver-threads 4 "
           "--default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01 "
           f" --solver-name={solver_name}")
    #rebaseline_xhat("uc", "uc_funcs", 3, uca, "test_data/uca_baseline")    
    do_one("uc", "uc_funcs", 3, uca, xhat_baseline_dir="test_data/uca_baseline")

    # This particular sizes command line is not very deterministic, also
    # linearize prox to help xpress.
    sizesa = ("--linearize-proximal-terms "
              " --num-scens=10 --bundles-per-rank=0 --max-iterations=5"
              " --default-rho=5 --lagrangian --xhatshuffle"
              " --iter0-mipgap=0.01 --iterk-mipgap=0.001 --rel-gap 0.001"
              f" --solver-name={solver_name}")
    #rebaseline_xhat("sizes", "sizes", 3, sizesa, "test_data/sizesa_baseline")
    do_one("sizes", "sizes", 3, sizesa, xhat_baseline_dir = "test_data/sizesa_baseline")

#### final processing ####
if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
    sys.exit(1)
elif len(xhat_losers) > 0:
    print("\nXhat Losers:")
    for i,v in xhat_losers.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
    sys.exit(1)
else:
    print("\nAll OK.")
