#!/bin/bash

SOLVERNAME="cplex"

mpiexec -np 2 python -m mpi4py farmer_rho_demo.py --num-scens 3 --bundles-per-rank=0 --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --xhatpath=/export/home/uinaepels/subgradient/mpi-sppy23/examples/farmer/farmer_cyl_nonants.npy --rho-file=/export/home/uinaepels/subgradient/mpi-sppy23/examples/farmer/rhos_test.csv --grad-cost-file=/export/home/uinaepels/subgradient/mpi-sppy23/examples/farmer/grad_cost_test.csv --whatpath=/export/home/uinaepels/subgradient/mpi-sppy23/examples/farmer/grad_cost_test.csv


#--rho-setter --rho-path=/export/home/uinaepels/subgradient/mpi-sppy23/examples/farmer/rhos_test.csv




