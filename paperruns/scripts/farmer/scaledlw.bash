#!/bin/bash

SEED=1134
SNAME="gurobi_persistent"

LOC="../../../examples/farmer"

CROPSMULT=100
SCENCNT=2048
SCENPERBUN=32   # be sure this divides SCENCNT
PFACT=20        # be sure this devides (SCENCNT/SCENPERBUN)

rm *.csv

for PFACT in 32 16 8 4 2 1
  do
    echo "=========="
    let NP=3*${PFACT}
    let BPR=${SCENCNT}/${SCENPERBUN}/${PFACT}
    echo ${SCENCNT}
    echo ${SCENPERBUN}
    echo ${PFACT}
    echo ${BPR}
    #python ${LOC}/farmer_ef.py ${CROPSMULT} ${SCENCNT} ${SNAME}
    mpiexec --oversubscribe -np ${NP} python -m mpi4py ${LOC}/farmer_cylinders.py ${SCENCNT} --bundles-per-rank=${BPR} --max-iterations=100 --rel-gap=0.01 --with-xhatshuffle --default-rho=0.1 --seed=${SEED} --solver-name=$SNAME --crops-mult=${CROPSMULT} --trace-prefix="f_${NP}_${SCENCNT}_${SEED}_"  --no-fwph --max-solver-threads=1 >& PFACT_${PFACT}.out
    mv *.csv savecsv
  done
