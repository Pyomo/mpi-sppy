#!/bin/bash

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignore).

#SOLVERNAME="xpress_persistent"
SOLVERNAME="cplex"

# get an xhat 
# xhat output file name is hardwired
# if you want to see numerical tolerance warnings emitted by FWPH, change branching factors to "20 5 4"
mpiexec --oversubscribe -np 4 python -m mpi4py aircond_cylinders.py --bundles-per-rank=0 --max-iterations=100 --default-rho=0.1 --solver-name=${SOLVERNAME} --branching-factors "4 3 2" --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory 50 --rel-gap -0.01 --mu-dev 0 --sigma-dev 40 --max-solver-threads 1 --start-seed 0 --fwph --start-ups --iter0-mipgap=0.0 --intra-hub-conv-thresh=-0.01 --max-stalled-iters=500 --xhatshuffle --lagrangian
