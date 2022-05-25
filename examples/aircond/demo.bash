#!/bin/bash

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignored).

SOLVERNAME="gurobi_persistent"
SOLVERNAME="cplex"

# get an xhat 
# xhat output file name is hardwired
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --bundles-per-rank=0 --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors "4 3 2" --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory 50 --rel-gap 0.01 --mu-dev 0 --sigma-dev 40 --max-solver-threads 2 --start-seed 0 --xhatshuffle --lagrangian
#--no-fwph --no-lagranger
