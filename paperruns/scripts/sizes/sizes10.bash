#!/bin/bash

SLOC="../../../examples/sizes/"
SNAME=gurobi_persistent

###mpiexec -np 3 python ${SLOC}sizes_cylinders.py --num-scens=10 --bundles-per-rank=0 --max-iterations=500 --default-rho=1 --iter0-mipgap=0.01 --iterk-mipgap=0.001 --solver-name=${SNAME} --no-fwph --rel-gap=0.001

###mpiexec -np 4 python ${SLOC}sizes_cylinders.py --num-scens=10 --bundles-per-rank=0 --max-iterations=500 --default-rho=1 --iter0-mipgap=0.01 --iterk-mipgap=0.001 --solver-name=${SNAME} --with-fwph --rel-gap=0.001

##################################
# no mip gap
#mpiexec -np 3 python ${SLOC}sizes_cylinders.py --num-scens=10 --bundles-per-rank=0 --max-iterations=500 --default-rho=1 --solver-name=${SNAME} --no-fwph --rel-gap=0.001 --max-solver-threads=1

mpiexec -np 4 python ${SLOC}sizes_cylinders.py --num-scens=10 --bundles-per-rank=0 --max-iterations=500 --default-rho=1 --solver-name=${SNAME} --with-fwph --rel-gap=0.001 --max-solver-threads=1
