#!/bin/bash

SOLVERNAME="cplex"

# get an xhat
# xhat output file name is hardwired
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py  3 --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name=${SOLVERNAME}

echo "starting zhat4xhat"
# NOTE: num_scens is restricted by the availability of data directories
python -m mpisppy.confidence_intervals.zhat4xhat aaircond aircond_cyl_nonants.npy --solver-name ${SOLVERNAME} --branching-factors 4 3 2 --num-samples 5 --confidence-level 0.95
