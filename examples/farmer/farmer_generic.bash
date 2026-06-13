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

# Risk-averse (CVaR) variants: minimize cvar-mean-weight*E[Cost] + cvar-weight*CVaR_alpha(Cost).
# Adding --cvar is all it takes; the VaR variable eta is just another first-stage
# variable, so the EF solve and every cylinder inherit risk aversion unchanged.
echo "Starting risk-averse (CVaR) EF"
python ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --EF --EF-solver-name gurobi --cvar --cvar-weight 2.0 --cvar-alpha 0.8

# For PH, eta has a much larger cost scale than the acreage variables, so a
# uniform rho stalls; use a cost-aware rho (--grad-rho here). See the docs.
echo "Starting risk-averse (CVaR) PH"
mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 --solver-name ${SOLVER} --max-iterations 100 --max-solver-threads 4 --default-rho 1 --grad-rho --grad-order-stat 0.5 --lagrangian --xhatshuffle --rel-gap 1e-6 --cvar --cvar-weight 2.0 --cvar-alpha 0.8

