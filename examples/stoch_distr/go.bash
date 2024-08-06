#!/bin/bash

SOLVERNAME=xpress

# This runs a hard wired example on 3 spokes
#mpiexec -np 3 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 10 --num-admm-subproblems 4 --default-rho 10 --solver-name $SOLVERNAME --max-iterations 100 --xhatxbar --lagrangian --ensure-xhat-feas --rel-gap 0.001

# This runs a scalable example on 3 spokes
mpiexec -np 3 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 3 --num-admm-subproblems 2 --default-rho 10 --solver-name $SOLVERNAME --max-iterations 50 --scalable --xhatxbar --lagrangian --mnpr 9 --ensure-xhat-feas

# This runs a scalable example on 6 spokes
#mpiexec -np 6 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 20 --num-admm-subproblems 5 --default-rho 10 --solver-name $SOLVERNAME --max-iterations 100 --scalable --xhatxbar --lagrangian --mnpr 8 --ensure-xhat-feas --rel-gap 0.01

# Running async on 1 spoke, doesn't work
#mpiexec -np 1 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 3 --num-admm-subproblems 3 --default-rho 10 --solver-name $SOLVERNAME --max-iterations 200 --scalable --mnpr 6 --run-async
