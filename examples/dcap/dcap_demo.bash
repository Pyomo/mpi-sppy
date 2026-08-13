#!/bin/bash
# Demonstrate the DCAP (SIPLIB) example using the generic cylinders driver.
# DCAP instances are supplied in SMPS format (.cor/.tim/.sto), so there is no
# model module: the generic driver selects mpisppy.problem_io.smps_module
# automatically when you pass --smps-dir.
# We assume the current directory is examples/dcap.

SOLVER=gurobi
#SOLVER=cplex
#SOLVER=xpress_persistent

INSTANCE=dcap233_200

# Progressive hedging hub with a Lagrangian outer bound and an xhat inner bound.
mpiexec -np 3 python -u -m mpi4py ../../mpisppy/generic_cylinders.py \
    --smps-dir ${INSTANCE} --solver-name=${SOLVER} \
    --max-iterations=20 --default-rho=1 \
    --lagrangian --xhatshuffle --rel-gap=1e-4 --intra-hub-conv-thresh=-0.1
