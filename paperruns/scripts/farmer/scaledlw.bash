#!/bin/bash
rm *.csv
SEED=1134
SNAME="gurobi_persistent"

LOC="../../../mpisppy/examples/farmer"


CROPSMULT=500
SCENCNT=1000
SCENPERBUN=50   # be sure this divides SCENCNT
PFACT=20        # be sure this devides (SCENCNT/SCENPERBUN)


#for PFACT in {20..1..-5}
#  do
    let NP=3*${PFACT}
    let BPR=${SCENCNT}/${SCENPERBUN}/${PFACT}


#python ${LOC}/farmer_ef.py ${CROPSMULT} ${SCENCNT} ${SNAME}
    mpiexec --oversubscribe -np ${NP} python -m mpi4py ${LOC}/farmer_cylinders.py ${SCENCNT} --bundles-per-rank=${BPR} --max-iterations=100 --rel-gap=0.01 --with-xhatshuffle --default-rho=0.1 --max-solver-threads=2 --seed=${SEED} --solver-name=$SNAME --crops-mult=${CROPSMULT} --trace-prefix="f_${NP}_${SCENCNT}_${SEED}_"  --no-fwph --max-solver-threads=2 >& NP_${NP}.out
    mv *.csv savescv
#  done
