#!/bin/bash

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignored).
# as of March 2, 2022 the cost and demand parameters in this example make it too easy for bundling

SOLVERNAME="cplex"

BF1=16
BF2=10
BF3=10
let SPB=BF2*BF3
let SC=BF1*BF2*BF3

BI=50

# this command can run under mpi for speed-up (see try_picles.bash)
python bundle_pickler.py --branching-factors "$BF1 $BF2 $BF3" --pickle-bundles-dir="." --scenarios-per-bundle=$SPB --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory $BI --mu-dev 0 --sigma-dev 40 --start-seed 0 

# It is entirely up to the user to make sure that the scenario count and scenarios per bundle match between creating the pickles and using them.
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors $SC --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory $BI --rel-gap 0.01 --mu-dev 0 --sigma-dev 40 --max-solver-threads 2 --start-seed 0 --xhatshuffle --lagrangian --start-ups --bundles-per-rank=0 --unpickle-bundles-dir="." --scenarios-per-bundle=$SPB
