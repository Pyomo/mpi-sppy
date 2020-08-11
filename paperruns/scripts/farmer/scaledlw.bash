#!/bin/bash
rm *.csv
SEED=1134
SNAME="gurobi_persistent"

LOC="../../../mpisppy/examples/farmer"

CROPSMULT=1000
SCENCNT=1000
SCENPERBUN=20   # be sure this divides SCENCNT
PFACT=10        # be sure this devides (SCENCNT/SCENPERBUN)


for PFACT in {30..1..-5}
  do
    let NP=3*${PFACT}
    let BPR=${SCENCNT}/${SCENPERBUN}/${PFACT}


    mpiexec --oversubscribe -np ${NP} python -m mpi4py ${LOC}/farmer_cylinders.py ${SCENCNT} --bundles-per-rank=${BPR} --max-iterations=100 --rel-gap=0.01 --with-xhatshuffle --default-rho=1 --max-solver-threads=2 --seed=${SEED} --solver-name=$SNAME --crops-mult=${CROPSMULT} --trace-prefix="f_${NP}_${SCENCNT}_${SEED}_"  --no-fwph --max-solver-threads=2 >& NP_${NP}.out
    mv *.csv savescv
  done
