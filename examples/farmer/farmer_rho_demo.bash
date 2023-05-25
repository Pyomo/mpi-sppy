#!/bin/bash

SOLVERNAME="cplex"

mpiexec -np 2 python -m mpi4py farmer_rho_demo.py --num-scens 3 --bundles-per-rank=0 --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --xhatpath=./farmer_cyl_nonants.npy --rho-file=./rhos_test.csv --grad-cost-file=./grad_cost_test.csv --whatpath=./grad_cost_test.csv --order-stat=0.5


#--rho-setter --rho-path=/./rhos_test.csv




