#!/bin/bash
rm *.csv
SEED=1134
#SNAME="gurobi_persistent"
SNAME="cplex"

LOC="../../../mpisppy/examples/farmer"

CROPSMULT=1000
SCENCNT=3
let NP=3*${SCENCNT}

mpiexec --oversubscribe -np ${NP} python -m mpi4py ${LOC}/farmer_cylinders.py ${SCENCNT} --max-iterations=100 --rel-gap=0.01 --with-xhatshuffle --default-rho=5 --max-solver-threads=2 --seed=${SEED} --solver-name=$SNAME --crops-mult=${CROPSMULT} --trace-prefix="f_${SCENCNT}_${SEED}_"  --no-fwph 






