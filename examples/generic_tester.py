# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
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
losers = dict()  # does not match baseline well enough

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
    outarg = "--solution_base_name"
    assert outarg not in argstring, "The tester controls solution writing"
    assert os.path.isdir(outdir), f"The solution dir given ({outdir}) is not an existing directory"
    retval = argstring + f" {outarg} outdir"
    return retval


def rebaseline(dirname, modname, np, argstring, baseline_dir):
    # Add the write output to the command line, do_one,
    fullarg = _append_soln_output_dir(argstring, baseline_dir)
    do_one(dirname, modname, np, fullarg)  # no baseline_dir!


def _check_baseline(modname, argstring, baseline_dir):
    # return true if OK, False otherwise
    return True


def do_one(dirname, modname, np, argstring, baseline_dir=None):
    """ return the code"""
    os.chdir(dirname)
    fullarg = argstring if baseline_dir is None else _append_soln_output_dir(argstring, baseline_dir)
    runstring = "mpiexec {} -np {} python -u -m mpi4py {} {}".\
                format(mpiexec_arg, np, progname, fullarg)
    # The top process output seems to be cached by github actions
    # so we need oputput in the system call to help debug
    code = os.system("echo {} && {}".format(runstring, runstring))
    if code != 0:
        if dirname not in badguys:
            badguys[dirname] = [runstring]
        else:
            badguys[dirname].append(runstring)
    if baseline_dir is not None:
        if not check_baseline(modname, argstring, baseline_dir):
    if '/' not in dirname:
        os.chdir("..")
    else:
        os.chdir("../..")   # hack for one level of subdirectories
    return code

do_one("sizes", "sizes", 1, "--help")

if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
    sys.exit(1)
elif len(losers) > 0:
    print("\nLosers:")
    for i,v in losers.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
    sys.exit(1)
else:
    print("\nAll OK.")
