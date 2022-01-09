#!/bin/bash

SOLVERNAME="gurobi_persistent"

# get an xhat
# xhat output file name is hardwired
mpiexec --oversubscribe -np 9 python -m mpi4py aircond_cylinders.py  --branching-factors 5 4 3 --bundles-per-rank=0 --max-iterations=50 --default-rho=1.0 --solver-name=${SOLVERNAME} --rel-gap=0.001 --abs-gap=2 --start-ups --max-solver-threads 2

echo ""
echo "starting zhat4xhat"
python -m mpisppy.confidence_intervals.zhat4xhat mpisppy.tests.examples.aircond aircond_cyl_nonants.npy --solver-name ${SOLVERNAME} --branching-factors 4 3 2 --num-samples 5 --confidence-level 0.95
