import os
import sys

solver_name = "gurobi_persistent"

def do_one_mmw(dirname, progname, npyfile, solver_name, argstring):
    os.chdir(dirname)
    runstring = "python -m mpisppy.confidence_intervals.mmw_conf {} {} {} {}".\
                format(progname, npyfile, solver_name, argstring)
    code = os.system("echo {} && {}".format(runstring, runstring))

    os.chdir("..")

do_one_mmw("examples", "farmer.py", "farmer_xhat.npy", solver_name, "--alpha 0.95")
do_one_mmw("examples", "apl1p.py", "apl1p_xhat.npy", solver_name, "--alpha 0.95 --objective-gap 0")
#do_one_mmw("examples", "aircond_submodels.py", "aircond_xhat.npy", solver_name, "--alpha 0.95")