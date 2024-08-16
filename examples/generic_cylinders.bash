#!/bin/bash
# This runs a few command lines to illustrate the use of generic_cylinders.py

SOLVER="cplex"

# You still need to name a module to get help
python -m mpi4py ../mpisppy/generic_cylinders.py --module-name farmer/farmer --help

# A not-so-useful run that does not use MPI, so only runs a hub
python -m mpi4py ../mpisppy/generic_cylinders.py --module-name farmer/farmer --num-scens 3 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 
