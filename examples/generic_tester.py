# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# This might make a mess in terms of output files....
# (re)baseline by uncommenting rebaseline_xhat lines
# Run a lot of generic_cylinders examples for regression testing; dlw Aug 2024
# Not intended to be user-friendly.
# Assumes you run from the examples directory.
# Optional command line arguments: solver_name mpiexec_arg nouc
# E.g. python generic_tester.py
#      python generic_tester.py cplex
#      python generic_tester.py gurobi_persistent --oversubscribe
#      python generic_tester.py gurobi_persistent -envall nouc
#      (envall does nothing; it is just a place-holder)

import os
import sys
import pandas as pd
from datetime import datetime as dt

XHAT_TEMP = "AAA_delete_this_from_generic_tester"  # write solutions here, then delete.

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

badguys = dict()  # bad return code
xhat_losers = dict()  # does not match baseline well enough

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


def _append_soln_output_dir(argstring, outdir):
    outarg = "--solution-base-name"
    assert outarg not in argstring, "The tester controls solution writing"
    assert os.path.isdir(outdir), f"The solution dir given ({outdir}) is not an existing directory"
    retval = argstring + f" {outarg} outdir"
    return retval


def rebaseline_xhat(dirname, modname, np, argstring, baseline_dir):
    # Add the write output to the command line, do_one,
    fullarg = _append_soln_output_dir(argstring, baseline_dir)
    do_one(dirname, modname, np, fullarg)  # do not check a baseline_dir, write one


def _check_baseline_xhat(modname, argstring, baseline_dir):
    # return true if OK, False otherwise
    # compare the baseline_dir to XHAT_TEMP
    return True

def _xhat_dir_setup(modname):
    if os.path.exists(XHAT_TEMP):
        shutil.rmtree(XHAT_TEMP)    
    os.makedirs(XHAT_TEMP)
    return os.path.join(XHAT_TEMP, modname)

def _xhat_dir_setup(modname):
    if os.path.exists(XHAT_TEMP):
        shutil.rmtree(XHAT_TEMP)    
    os.makedirs(XHAT_TEMP)
    return os.path.join(XHAT_TEMP, modname)


def do_one(dirname, modname, np, argstring, xhat_baseline_dir=None):
    """ return the code"""
    os.chdir(dirname)
    if xhat_baseline_dir is not None: 
        fullarg = _append_soln_output_dir(argstring, _xhat_dir_setup(modname))         
    else:
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
        if not _check_baseline_xhat(modname, argstring, baseline_dir):
            xhat_losers.append(runstring)
        _xhat_dir_teardown()        
    if '/' not in dirname:
        os.chdir("..")
    else:
        os.chdir("../..")   # hack for one level of subdirectories
    return code

###################### main code ######################

do_one("sizes", "sizes", 1, "--help")

sizesa = ("--linearize-proximal-terms "
          " --num-scens=10 --bundles-per-rank=0 --max-iterations=5"
          " --default-rho=1 --lagrangian --xhatshuffle"
          " --iter0-mipgap=0.01 --iterk-mipgap=0.001"
          f" --solver-name={solver_name}")
#rebaseline_xhat("sizes", "sizes", 3, sizesa, "test_data/sizesa_baseline")
do_one("sizes", "sizes", 3, sizesa, xhat_baseline_dir = "test_data/sizesa_baseline")

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
