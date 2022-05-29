#!/bin/bash

SOLVERNAME="cplex"

# taking defaults for most farmer args

# get an xhat
# xhat output file name is hardwired to 'farmer_cyl_nonants.npy'
mpiexec -np 3 python -m mpi4py farmer_cylinders.py  --num-scens 3 --lagrangian --xhatshuffle --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name=${SOLVERNAME}

echo "starting mmw"

python -m mpisppy.confidence_intervals.mmw_conf afarmer --xhat-path farmer_cyl_nonants.npy --solver-name ${SOLVERNAME} --MMW-num-batches 5 --MMW-batch-size 10 --confidence-level 0.9 --start-scen 10
