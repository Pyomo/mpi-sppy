#!/bin/bash

SOLVERNAME="cplex"

# taking defaults for most farmer args

# get an xhat
# xhat output file name is hardwired to 'farmer_cyl_nonants.npy'
mpiexec -np 4 python -m mpi4py farmer_cylinders.py  3 --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name=${SOLVERNAME}


# evaluate the zhat for the xhat computed above
python -m mpisppy.confidence_intervals.mmw_conf farmer farmer_cyl_nonants.npy ${SOLVERNAME} --MMW-num-batches 5 --MMW-batch-size 10 --confidence-level 0.9 --start-scen 10
