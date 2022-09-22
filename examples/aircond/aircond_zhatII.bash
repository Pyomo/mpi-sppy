#!/bin/bash

SOLVERNAME="cplex"
SAMPLES=3

# get an xhat 
# xhat output file name is hardwired
mpiexec --oversubscribe -np 3 python3 -m mpi4py aircond_cylinders.py --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors "4 3 2" --Capacity 200 --QuadShortCoeff 0.3 --start-ups --BeginInventory 50 --xhatshuffle --lagrangian --max-solver-threads 2

echo "starting zhat4xhat with ${SAMPLES} samples"
python -m mpisppy.confidence_intervals.zhat4xhat mpisppy.tests.examples.aircond --xhatpath aircond_cyl_nonants.npy --solver-name ${SOLVERNAME} --branching-factors "4 3 2" --num-samples ${SAMPLES} --confidence-level 0.95 --Capacity 200 --QuadShortCoeff 0.3 --start-ups --BeginInventory 50
