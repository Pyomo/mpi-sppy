#!/bin/bash

SOLVERNAME="cplex"

# --solution-base-name produces uc_cyl_nonants.npy (and .csv and _soldir/)
##mpiexec --oversubscribe -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py --module-name uc_funcs --xhatshuffle --lagrangian --max-iterations=2 --default-rho=1 --num-scens=3 --max-solver-threads=2 --lagrangian-iter0-mipgap=1e-7 --ph-mipgaps-json=phmipgaps.json --solver-name=${SOLVERNAME} --solution-base-name uc_cyl_nonants
# --fwph
echo
echo "done finding xhat, now evaluating zhat"


# NOTE: num_scens is restricted by the availability of data directories
python -m mpisppy.confidence_intervals.mmw_conf uc_funcs --xhatpath uc_cyl_nonants.npy --solver-name ${SOLVERNAME} --MMW-num-batches 5 --MMW-batch-size 10 --UC-count-for-path 100 --confidence-level 0.9 --start-scen 10
