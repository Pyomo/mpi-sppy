#!/bin/bash

SOLVERNAME=gurobi

# mpiexec -np 1 python -m mpi4py agnostic_gurobipy_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=10 --rel-gap 0.01 --display-progress

# mpiexec -np 2 python -m mpi4py agnostic_gurobipy_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=10 --xhatshuffle --rel-gap 0.01 --display-progress

mpiexec -np 3 python -m mpi4py agnostic_gurobipy_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=10 --xhatshuffle --lagrangian --rel-gap 0.01 --display-progress
