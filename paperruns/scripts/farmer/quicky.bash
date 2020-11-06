#!/bin/bash
rm *.csv
SEED=1134
#SNAME="gurobi_persistent"
SNAME="cplex"

LOC="../../../examples/farmer"

CROPSMULT=500
SCENCNT=6
SCENPERBUN=2
# use 0 for no bundles
if [ ${SCENPERBUN} -gt 0 ]
   then
   let NP=3*${SCENCNT}/${SCENPERBUN}
   else
   let NP=3*${SCENCNT}
fi

mpiexec --oversubscribe -np ${NP} python -m mpi4py ${LOC}/farmer_cylinders.py ${SCENCNT} --max-iterations=100 --rel-gap=0.01 --with-xhatshuffle --default-rho=0.1 --max-solver-threads=2 --seed=${SEED} --solver-name=$SNAME --crops-mult=${CROPSMULT} --trace-prefix="f_${SCENCNT}_${SEED}_"  --no-fwph 






