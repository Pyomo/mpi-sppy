#!/bin/bash

# How to run this bash script:

# This runs a hard wired example on 3 spokes
#mpiexec -np 3 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 10 --num-admm-subproblems 4 --default-rho 10 --solver-name xpress --max-iterations 50 --xhatxbar --lagrangian

# This runs a scalable example on only one spoke
#mpiexec -np 1 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 3 --num-admm-subproblems 3 --default-rho 10 --solver-name xpress --max-iterations 200 --scalable --mnpr 6

# This runs a scalable example on 3 spokes
mpiexec -np 3 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 3 --num-admm-subproblems 2 --default-rho 10 --solver-name cplex_direct --max-iterations 50 --scalable --xhatxbar --lagrangian --mnpr 9 --ensure-xhat-feas
#mpiexec -np 6 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 20 --num-admm-subproblems 5 --default-rho 10 --solver-name cplex_direct --max-iterations 100 --scalable --xhatxbar --lagrangian --mnpr 8 --ensure-xhat-feas --rel-gap 0.01

#running async
#mpiexec -np 1 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 3 --num-admm-subproblems 3 --default-rho 10 --solver-name cplex_direct --max-iterations 200 --scalable --mnpr 6 --run-async
