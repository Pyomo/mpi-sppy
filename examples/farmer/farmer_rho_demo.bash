#!/bin/bash

SOLVERNAME="cplex"

# get an xhat
# xhat output file name is hardwired to 'farmer_cyl_nonants.npy'
mpiexec -np 3 python -m mpi4py farmer_cylinders.py  --num-scens 3 --lagrangian --xhatshuffle --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name=${SOLVERNAME}

mpiexec -np 3 python -m mpi4py farmer_rho_demo.py --num-scens 3 --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVERNAME} --xhatpath=./farmer_cyl_nonants.npy --grad-order-stat 0.5 --xhatshuffle --lagrangian --max-stalled-iters 5000 --grad-rho-setter --rel-gap 0.001

#--rho-relative-bound
#--grad-rho-file=./grad_rhos_demo.csv --grad-cost-file=./grad_cost_demo.csv --whatpath=./grad_cost_demo.csv --order-stat=0.5

#--rho-setter --rho-path=./rhos_demo.csv




