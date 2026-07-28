#!/bin/bash
# Demonstrate the sslp example using the generic cylinders driver.
# sslp.py provides the model-module hooks (scenario_creator, inparser_adder,
# kw_creator, ...), so runs go through mpisppy/generic_cylinders.py.
# We assume the current directory is examples/sslp.

SOLVER=xpress_persistent
#SOLVER=cplex

mpiexec -n 11 python -u -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path=./data/ --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=10.0 --instance-name=sslp_15_45_10 --max-iterations=100 --rel-gap=0.0 --xhatshuffle --presolve --intra-hub-conv-thresh=-0.1 --fwph-objgap-hub --xhatshuffle-rank-ratio=0.1 --sep-rho --surrogate-nonant

# --fwph-add-cylinder-columns
