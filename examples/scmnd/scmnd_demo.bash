#!/bin/bash

SOLVER=xpress_persistent
num_procs=60
case_name=scmnd_20_120_40_20

# Solve instance with the active subspace progressive hedging algorithm
mpiexec -n ${num_procs} python ../../mpisppy/generic_cylinders.py --module-name scmnd --solver-name=${SOLVER} --max-iterations=1000 --max-solver-threads=1 --default-rho=1.0 --sep-rho --sep-rho-multiplier=1 --xhatshuffle --rel-gap=0.01 --intra-hub-conv-thresh=-0.0001 --presolve --reduced-costs --rc-fixer-converger --rc-verbose --rc-fix-fraction-iterk=1 --rc-fixer-require-improving-outer-bound --rc-debug --max-stalled-iters=100 --scmnd-data-path=./data/ --instance-name=${case_name} --time-limit=200 --rc-converger-tol=0.01

#  Solve instance with the progressive hedging algorithm
mpiexec -n ${num_procs} python ../../mpisppy/generic_cylinders.py --module-name scmnd --solver-name=${SOLVER} --max-iterations=1000 --max-solver-threads=1 --default-rho=1 --sep-rho --sep-rho-multiplier=1 --xhatshuffle --rel-gap=0.01  --intra-hub-conv-thresh=-0.0001 --presolve --max-stalled-iters=100 --scmnd-data-path=./data --instance-name=${case_name} --time-limit=200 --reduced-costs

#  Solve instance with the extensive form
mpiexec -n 1 python ../../mpisppy/generic_cylinders.py --module-name scmnd --solver-name=${SOLVER} --instance-name=${case_name} --scmnd-data-path=./data --EF --EF-solver-name=${SOLVER} --time-limit=200

