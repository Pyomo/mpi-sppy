#!/bin/bash

#SOLVER=xpress_persistent
SOLVER=cplex

mpiexec --oversubscribe -n 6 python netdes_cylinders.py --solver-name=${SOLVER} --max-solver-threads=2 --default-rho=10000.0 --instance-name=network-10-20-L-01 --max-iterations=100 --rel-gap=0.01 --lagrangian --xhatshuffle
