#!/bin/bash

# How to run this bash script:
# Execute with "bash go.bash scalable" for the scalable example
# Execute with "bash go.bash anything" otherwise

# This file runs either a scalable example or a non scalable example
if [ ${1} = "scalable" ]
then
    mpiexec -np 3 python -u -m mpi4py distr_admm_cylinders.py --num-scens 3 --default-rho 10 --solver-name xpress --max-iterations 50 --xhatxbar --lagrangian --mnpr 40 --rel-gap 0.05 --scalable
else
    mpiexec -np 3 python -u -m mpi4py distr_admm_cylinders.py --num-scens 3 --default-rho 10 --solver-name xpress --max-iterations 10 --xhatxbar --lagrangian --rel-gap 0.05
fi

#mpiexec -np 3 python -u -m mpi4py distr_admm_cylinders.py --num-scens 3 --default-rho 10 --solver-name xpress --max-iterations 50 --xhatxbar --lagrangian --mnpr 40 --rel-gap 0.05 --scalable
#mpiexec -np 3 python -u -m mpi4py distr_admm_cylinders.py --num-scens 3 --default-rho 10 --solver-name xpress --max-iterations 10 --xhatxbar --lagrangian --rel-gap 0.05