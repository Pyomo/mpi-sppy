#!/bin/bash
#SOLVERNAME="gurobi_persistent"
SOLVERNAME="cplex"
EMLOC="/export/home/dlwoodruff/Documents/DLWFORKS/mpi-sppy-1/examples/EMPRISE"

#mpiexec -n 24 python -m mpi4py ${EMLOC}/emprise_cylinders.py --max-solver-threads=8 --bundles-per-rank=1 --max-iterations=50 --default-rho=75 --with-xhatshuffle --with-lagrangian --solver-name=${SOLVERNAME}
# mpi dies on webster with np > 3 ???
# max-solver-threads=2 might be better?
mpiexec -n 6 python -m mpi4py ${EMLOC}/emprise_cylinders.py --max-solver-threads=16 --bundles-per-rank=1 --max-iterations=50 --default-rho=750 --with-xhatshuffle --with-lagrangian --solver-name=${SOLVERNAME} --with-display-convergence-detail --use-norm-rho-updater --use-norm-rho-converger
