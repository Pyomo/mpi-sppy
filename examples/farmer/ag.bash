#!/bin/bash


python agnostic_cylinders.py --help

python agnostic_cylinders.py --num-scens 3 --default-rho 1 --solver-name cplex

mpiexec -np 2 python -m mpi4py agnostic_cylinders.py --num-scens 3 --default-rho 1 --solver-name cplex --max-iterations=4 --xhatshuffle
