#!/bin/bash

SOLVERNAME="cplex"

mpiexec -np 3 python -m mpi4py farmer_rho_demo.py --num-scens 3 --max-iterations=100 --default-rho=1 --solver-name=${SOLVERNAME} --grad-order-stat 0.5 --xhatshuffle --lagrangian --max-stalled-iters 5000 --grad-rho --rel-gap 0.001

#--rho-relative-bound
#--grad-rho-file=./grad_rhos_demo.csv --grad-cost-file=./grad_cost_demo.csv --whatpath=./grad_cost_demo.csv --order-stat=0.5

#--rho-setter --rho-path=./rhos_demo.csv




