#!/bin/bash

SOLVERNAME=cplex

#python agnostic_cylinders.py --help

#mpiexec -np 3 python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=5 --xhatshuffle --lagrangian --display-progress --rel-gap 0.01

#python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=5 --rel-gap 0.01

#mpiexec -np 2 python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=5 --lagrangian --rel-gap 0.01

mpiexec -np 2 python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=5 --xhatshuffle --display-progress --rel-gap 0.01
