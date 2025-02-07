#!/bin/bash

SOLVER=xpress_persistent
#SOLVER=cplex

mpiexec -n 6 python -u -m mpi4py sslp_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=1.0 --instance-name=sslp_15_45_10 --max-iterations=50 --rel-gap=0.0 --subgradient --xhatshuffle --presolve --intra-hub-conv-thresh=-0.1 --subgradient-rho-multiplier=1.0 --surrogate-nonant
