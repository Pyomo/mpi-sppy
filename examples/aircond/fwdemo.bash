#!/bin/bash

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignore).

SOLVERNAME="xpress_persistent"

# get an xhat 
# xhat output file name is hardwired
mpiexec --oversubscribe -np 5 python -m mpi4py aircond_cylinders.py --bundles-per-rank=7 --max-iterations=1000 --default-rho=0.1 --solver-name=${SOLVERNAME} --branching-factors 4 3 2 --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory 50 --rel-gap -0.01 --mu-dev 0 --sigma-dev 40 --max-solver-threads 1 --start-seed 0 --with-fwph --start-ups --iter0-mipgap=0.0 --intra-hub-conv-thresh=-0.01
