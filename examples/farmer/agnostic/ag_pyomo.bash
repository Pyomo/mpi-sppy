#!/bin/bash



#python agnostic_cylinders.py --help

#mpiexec -np 3 python -m mpi4py farmer_pyomo_cylinders.py --num-scens 3 --default-rho 1 --solver-name cplex --max-iterations=5 --xhatshuffle --lagrangian --rel-gap 0.01

#mpiexec -np 3 python -m mpi4py agnostic_pyomo_cylinders.py --num-scens 3 --default-rho 1 --solver-name cplex --max-iterations=5 --xhatshuffle --lagrangian --rel-gap 0.01

#python agnostic_pyomo_cylinders.py --num-scens 3 --default-rho 1 --solver-name cplex --max-iterations=2 

#mpiexec -np 2 python -m mpi4py agnostic_pyomo_cylinders.py --num-scens 3 --default-rho 1 --solver-name cplex --max-iterations=10 --lagrangian --rel-gap 0.01

mpiexec -np 2 python -m mpi4py agnostic_pyomo_cylinders.py --num-scens 3 --default-rho 1 --solver-name cplex --max-iterations=10 --xhatshuffle --rel-gap 0.01
