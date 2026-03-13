###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Run a few examples; dlw Nov 2024; user-unfriendly
# python afew_agnostic.py
# python afew_agnostic.py --oversubscribe
# For coverage: python afew_agnostic.py --python-args="-m coverage run --parallel-mode --source=mpisppy"

import os
import sys

# Parse --python-args (extra args inserted after "python" in subcommands, e.g. for coverage)
python_args = ""
_remaining = []
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i].startswith("--python-args="):
        python_args = sys.argv[_i].split("=", 1)[1]
    elif sys.argv[_i] == "--python-args" and _i + 1 < len(sys.argv):
        _i += 1
        python_args = sys.argv[_i]
    else:
        _remaining.append(sys.argv[_i])
    _i += 1
sys.argv = [sys.argv[0]] + _remaining

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGNOSTIC_CYLINDERS = os.path.join(_THIS_DIR, "..", "agnostic_cylinders.py")

pyomo_solver_name = "cplex_direct"
ampl_solver_name = "gurobi"
gams_solver_name = "cplex"

# Use oversubscribe if your computer does not have enough cores.
# Don't use this unless you have to.
# (This may not be allowed on versions of mpiexec)
mpiexec_arg = ""  # "--oversubscribe"
if len(sys.argv) == 2:
    mpiexec_arg = sys.argv[1]

badguys = list()

def do_one(np, argstring):
    runstring = f"mpiexec -np {np} {mpiexec_arg} python {python_args} -m mpi4py {_AGNOSTIC_CYLINDERS} {argstring}"
    print(runstring)
    code = os.system(runstring)
    if code != 0:
        badguys.append(runstring)



print("skipping Pyomo because it is not working on github due to cplex_direct versus cplexdirect")
#do_one(3, f"--module-name farmer4agnostic --default-rho 1 --num-scens 6 --solver-name {pyomo_solver_name} --guest-language Pyomo  --max-iterations 5")

do_one(3, f"--module-name mpisppy.agnostic.examples.farmer_ampl_model --default-rho 1 --num-scens 3 --solver-name {ampl_solver_name} --guest-language AMPL --ampl-model-file farmer.mod --lagrangian --xhatshuffle --max-iterations 5")

do_one(3, f"--module-name mpisppy.agnostic.examples.steel_ampl_model --default-rho 1 --num-scens 3 --seed 17 --solver-name {ampl_solver_name} --guest-language AMPL --ampl-model-file steel.mod --ampl-data-file steel.dat --max-iterations 10 --rel-gap 0.01 --xhatshuffle --lagrangian --solution-base-name steel")

do_one(3, f"--module-name mpisppy.agnostic.examples.farmer_gams_model --num-scens 3 --default-rho 1 --solver-name {gams_solver_name} --max-iterations=5 --rel-gap 0.01 --display-progress --guest-language GAMS --gams-model-file farmer_average.gms")


if len(badguys) > 0:
    print("\nBad Guys:")
    for i in badguys:
        print(i)
        sys.exit(1)
else:
    print("\nAll OK.")
