#!/bin/bash
# Illustrate the use of generic cylinders for the farmer example.
# We assumng the current directory is examples.farmer

SOLVER=gurobi

# Note that to get help, you still need to specify a module-name
python ../../mpisppy/generic_cylinders.py --module-name farmer --help

# Here is a simple ef command, that writes solution data to two files with the base name farmersol
#    and to a directory named farmersol_soldir
echo "Starting EF"
python ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --EF --EF-solver-name gurobi --solution-base-name farmersol

# Here is a very simple PH command that also computes bounds
echo "Starting PH"
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01 

