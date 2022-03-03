#!/bin/bash
# three stage version

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignored).
# as of March 2, 2022 the cost and demand parameters in this example make it too easy for bundling

SOLVERNAME="cplex"

BF1=2
BF2=2
let SPB=BF2
let SC=BF1*BF2

BI=150
NC=1
QSC=0.3
SD=80
OTC=1.5

EC="--Capacity 200 --QuadShortCoeff $QSC  --BeginInventory $BI --mu-dev 0 --sigma-dev $SD --start-seed 0 --NegInventoryCost=$NC --OvertimeProdCost=$OTC"
# --start-ups

echo "-------- EF Directly"
python aircond_cylinders.py --EF-directly --solver-name=${SOLVERNAME} --branching-factors $BF1 $BF2 $EC

echo "***** Cylinders with no bundles"
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME}  --rel-gap 0.01  --max-solver-threads 2 --start-seed 0 --no-fwph --no-lagranger --bundles-per-rank=0  --write-solution --intra-hub-conv-thresh 0 --branching-factors $BF1 $BF2 $EC

echo "======== Cylinders with ${BF1} bundles per rank (not proper bundles)"
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME}  --rel-gap 0.01  --max-solver-threads 2 --start-seed 0 --no-fwph --no-lagranger --bundles-per-rank=$BF1  --write-solution --intra-hub-conv-thresh 0 --branching-factors $BF1 $BF2 $EC


echo "^^^^^^^^^ Make pickle bundles"
python bundle_pickler.py --branching-factors $BF1 $BF2 --pickle-bundles-dir="." --scenarios-per-bundle=$SPB $EC


echo "***** Use pickle bundles"
# It is entirely up to the user to make sure that the scenario count and scenarios per bundle match between creating the pickles and using them (the costs probably don't matter, since the pickle has it all)
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors $SC --rel-gap 0.01 --max-solver-threads 2 --start-seed 0 --no-fwph --no-lagranger --bundles-per-rank=0  --scenarios-per-bundle=$SPB --write-solution --intra-hub-conv-thresh 0 --unpickle-bundles-dir="." $EC

