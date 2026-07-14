#!/bin/bash
# Demonstrate the sslp example using the generic cylinders driver.
# sslp.py provides the model-module hooks (scenario_creator, inparser_adder,
# kw_creator, ...), so runs go through mpisppy/generic_cylinders.py.
# We assume the current directory is examples/sslp.

SOLVER=xpress_persistent
#SOLVER=cplex

mpiexec -n 6 python -u -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --instance-name=sslp_15_45_10 --sslp-data-path=./data --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=1.0 --max-iterations=50 --rel-gap=0.0 --subgradient --xhatshuffle --presolve --intra-hub-conv-thresh=-0.1 --subgradient-rho-multiplier=1.0 --surrogate-nonant
