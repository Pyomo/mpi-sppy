#!/bin/bash

SOLVERNAME="cplex"

# get an xhat 
# xhat output file name is hardwired
mpiexec --oversubscribe -np 3 python3 -m mpi4py aircond_cylinders.py --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors 4 3 2 --Capacity 200 --QuadShortCoeff 0.3 --start-ups --BeginInventory 50

