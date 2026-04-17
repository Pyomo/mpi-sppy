#!/bin/bash

SOLVERNAME="xpress_direct"

mpiexec -np 3 python -m mpi4py gradient_uc_cylinders.py --xhatshuffle --lagrangian --max-iterations=10 --default-rho=1 --num-scens=5 --max-solver-threads=2 --lagrangian-iter0-mipgap=1e-7 --ph-mipgaps-json=phmipgaps.json --solver-name=${SOLVERNAME} --rel-gap 0.000001 --display-progress --grad-rho --grad-order-stat 0.5 --integer-relax-then-enforce --integer-relax-then-enforce-ratio=0.9
