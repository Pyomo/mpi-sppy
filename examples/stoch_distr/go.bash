#!/bin/bash

# How to run this bash script:

# This runs a hard wired example on 3 spokes
#mpiexec -np 3 python -u -m mpi4py stoch_distr_admm_cylinders.py --default-rho 10 --solver-name xpress --max-iterations 10 --xhatxbar --lagrangian

# This runs a scalable example on only one spoke
#mpiexec -np 1 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 2 --num-admm-subproblems 3 --default-rho 10 --solver-name xpress --max-iterations 10 --scalable --mnpr 6

# This runs a scalable example on 3 spokes
mpiexec -np 3 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 2 --num-admm-subproblems 3 --default-rho 10 --solver-name xpress --max-iterations 10 --scalable --xhatxbar --lagrangian --mnpr 6

