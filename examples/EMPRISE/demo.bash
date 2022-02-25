#!/bin/bash
SOLVERNAME="gurobi_persistent"
EMLOC="/export/home/dlwoodruff/Documents/DLWFORKS/mpi-sppy-1/examples/EMPRISE"

#mpiexec -n 24 python -m mpi4py ${EMLOC}/emprise_cylinders.py --max-solver-threads=8 --bundles-per-rank=1 --max-iterations=50 --default-rho=75 --with-xhatshuffle --with-lagrangian --solver-name=${SOLVERNAME}
mpiexec -n 12 python -m mpi4py ${EMLOC}/emprise_cylinders.py --max-solver-threads=2 --bundles-per-rank=2 --max-iterations=10 --default-rho=75 --with-xhatshuffle --with-lagrangian --solver-name=${SOLVERNAME}
