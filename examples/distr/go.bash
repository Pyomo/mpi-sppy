#!/bin/bash

SOLVERNAME=xpress

# Runs a hard-wired example
mpiexec -np 3 python -u -m mpi4py distr_admm_cylinders.py --num-scens 3 --default-rho 10 --solver-name $SOLVERNAME --max-iterations 50 --xhatxbar --lagrangian --rel-gap 0.05

# Runs a scalable example with xhatxbar as inner bounder and lagrangian as outer bounder
#mpiexec -np 3 python -u -m mpi4py distr_admm_cylinders.py --num-scens 3 --default-rho 10 --solver-name $SOLVERNAME --max-iterations 50 --xhatxbar --lagrangian --mnpr 4 --rel-gap 0.05 --scalable
