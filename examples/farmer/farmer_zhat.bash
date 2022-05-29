#!/bin/bash

SOLVERNAME="cplex"

# get an xhat
# xhat output file name is hardwired to 'farmer_cyl_nonants.npy'
mpiexec -np 3 python -m mpi4py farmer_cylinders.py  --num-scens 3 --lagrangian --xhatshuffle --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name=${SOLVERNAME}

echo "starting zhat4xhat"
python -m mpisppy.confidence_intervals.zhat4xhat farmer --xhatpath farmer_cyl_nonants.npy --solver-name ${SOLVERNAME} --branching-factors 10 --num-samples 5 --confidence-level 0.95
