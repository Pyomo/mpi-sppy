#!/bin/bash

SOLVERNAME=gurobi

#python agnostic_cylinders.py --help

mpiexec -np 3 python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 0.5 --solver-name $SOLVERNAME --max-iterations=50 --xhatshuffle --lagrangian --rel-gap 0.01 #--display-progress 

#python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=5 --rel-gap 0.01 --display-progress

#mpiexec -np 2 python -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 1 --solver-name $SOLVERNAME --max-iterations=40 --lagrangian --rel-gap 0.01

#mpiexec -np 2 python -u -m mpi4py agnostic_gams_cylinders.py --num-scens 3 --default-rho 0.5 --solver-name $SOLVERNAME --max-iterations=50 --xhatshuffle --rel-gap 0.01 --display-progress
