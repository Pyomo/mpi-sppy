#!/bin/bash

SOLVERNAME="cplex"

# uc_cylinders has the uc_cyl_nonants.npy name hard-wired
mpiexec --oversubscribe -np 3 python -m mpi4py gradient_uc_cylinders.py --xhatshuffle --lagrangian --bundles-per-rank=0 --max-iterations=5 --default-rho=1 --num-scens=5 --max-solver-threads=2 --lagrangian-iter0-mipgap=1e-7 --ph-mipgaps-json=phmipgaps.json --solver-name=${SOLVERNAME} --xhatpath uc_cyl_nonants.npy --rel-gap 0.000001 --display-progress --grad-rho-setter --grad-order-stat 0.5

####mpiexec --oversubscribe -np 1 python -m mpi4py gradient_uc_cylinders.py --bundles-per-rank=0 --max-iterations=20 --default-rho=1 --num-scens=5 --max-solver-threads=2 --lagrangian-iter0-mipgap=1e-7 --ph-mipgaps-json=phmipgaps.json --solver-name=${SOLVERNAME} --xhatpath uc_cyl_nonants.npy --rel-gap 0.000001 --display-progress --grad-rho-setter --grad-order-stat 0.5

# --fwph
echo

# --rho-setter --order-stat 0.5
# --display-progress
