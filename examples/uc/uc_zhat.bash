#!/bin/bash

SOLVERNAME="cplex"

# uc_cylinders has the uc_cyl_nonants.spy.npy name hard-wired
echo "Skipping xhat and using an old one..."
#mpiexec -np 4 python -m mpi4py uc_cylinders.py --bundles-per-rank=0 --max-iterations=2 --default-rho=1 --num-scens=3 --max-solver-threads=2 --lagrangian-iter0-mipgap=1e-7 --no-cross-scenario-cuts --ph-mipgaps-json=phmipgaps.json --solver-name=${SOLVERNAME}


# NOTE: num_scens is restricted by the availability of data directories
python -m mpisppy.confidence_intervals.zhat4xhat uc_funcs uc_cyl_nonants.spy.npy --solver-name ${SOLVERNAME} --branching-factors 3 --UC-count-for-path 100 --num-samples 5
