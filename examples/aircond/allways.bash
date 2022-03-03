#!/bin/bash

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignored).
# as of March 2, 2022 the cost and demand parameters in this example make it too easy for bundling

SOLVERNAME="cplex"

BF1=2
BF2=2
BF3=2
let SPB=BF2*BF3
let SC=BF1*BF2*BF3

BI=150
NC=1
QSC=0.3
SD=80
OTC=1.5

echo "-------- EF Directly"
python aircond_cylinders.py --EF-directly --branching-factors $BF1 $BF2 $BF3 --scenarios-per-bundle=$SPB --Capacity 200 --QuadShortCoeff $QSC  --BeginInventory $BI --mu-dev 0 --sigma-dev $SD --start-seed 0 --NegInventoryCost=$NC --OvertimeProdCost=$OTC --solver-name=${SOLVERNAME}

echo "***** Cylinders with no bundles"
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors $SC --Capacity 200 --QuadShortCoeff $QSC  --BeginInventory $BI --rel-gap 0.01 --mu-dev 0 --sigma-dev $SD --max-solver-threads 2 --start-seed 0 --no-fwph --no-lagranger --bundles-per-rank=0  --write-solution --intra-hub-conv-thresh 0 --NegInventoryCost=$NC --OvertimeProdCost=$OTC

echo "======== Cylinders with ${BF1} bundles per rank (not proper bundles)"
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors $SC --Capacity 200 --QuadShortCoeff $QSC  --BeginInventory $BI --rel-gap 0.01 --mu-dev 0 --sigma-dev $SD --max-solver-threads 2 --start-seed 0 --no-fwph --no-lagranger --bundles-per-rank=$BF1  --scenarios-per-bundle=$SPB --write-solution --intra-hub-conv-thresh 0 --NegInventoryCost=$NC --OvertimeProdCost=$OTC

echo "^^^^^^^^^ Make pickle bundles"
python bundle_pickler.py --branching-factors $BF1 $BF2 $BF3 --pickle-bundles-dir="." --scenarios-per-bundle=$SPB --Capacity 200 --QuadShortCoeff $QSC  --BeginInventory $BI --mu-dev 0 --sigma-dev $SD --start-seed 0 --NegInventoryCost=$NC --OvertimeProdCost=$OTC


echo "***** Use pickle bundles"
# It is entirely up to the user to make sure that the scenario count and scenarios per bundle match between creating the pickles and using them (the costs probably don't matter, since the pickle has it all)
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors $SC --Capacity 200 --QuadShortCoeff $QSC  --BeginInventory $BI --rel-gap 0.01 --mu-dev 0 --sigma-dev $SD --max-solver-threads 2 --start-seed 0 --no-fwph --no-lagranger --bundles-per-rank=0  --scenarios-per-bundle=$SPB --write-solution --intra-hub-conv-thresh 0 --NegInventoryCost=$NC --OvertimeProdCost=$OTC --unpickle-bundles-dir="."

# --start-ups
# --unpickle-bundles-dir="."
