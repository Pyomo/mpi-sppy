#!/bin/bash
# four stage version

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignored).


SOLVERNAME="gurobi_persistent"

BF1=50
BF2=10
BF3=10
let SPB=BF2*BF3
let SC=BF1*BF2*BF3
let PBF=BF1  # by design, this is the number of bundles

BI=50
NC=1
QSC=0.3
SD=40
OTC=5

EC="--Capacity 200 --QuadShortCoeff $QSC  --BeginInventory $BI --mu-dev 0 --sigma-dev $SD --start-seed 0 --NegInventoryCost=$NC --OvertimeProdCost=$OTC"
# --start-ups

echo "^^^^^^^^^ Make pickle bundles"
python bundle_pickler.py --branching-factors $BF1 $BF2 $BF3 --pickle-bundles-dir="." --scenarios-per-bundle=$SPB $EC


echo "***** Use pickle bundles"
# It is entirely up to the user to make sure that the scenario count and scenarios per bundle match between creating the pickles and using them (the costs probably don't matter, since the pickle has it all)
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors $PBF --rel-gap 0.001 --max-solver-threads 2 --start-seed 0 --no-fwph --no-lagranger --bundles-per-rank=0  --scenarios-per-bundle=$SPB --write-solution --intra-hub-conv-thresh 0 --unpickle-bundles-dir="." $EC --with-display-progress --solver-options="method=0"
#--with-display-convergence-detail

###python aircond_cylinders.py --help

