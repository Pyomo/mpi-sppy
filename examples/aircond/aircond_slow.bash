#!/bin/bash

SOLVERNAME="cplex"

# get an xhat 
# xhat output file name is hardwired
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors "4 3 2" --Capacity 200 --QuadShortCoeff 0.3 --start-ups --BeginInventory 50 --rel-gap 0.01 --xhatshuffle --lagrangian --max-solver-threads 2

# on many shared-memory machines, this will be slow without max-solver-threads. With it, you can
increase the -np argument to go faster.

