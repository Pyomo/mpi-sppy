#!/bin/bash
# three stage version

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignored).
# as of March 2, 2022 the cost and demand parameters in this example make it too easy for bundling

SOLVERNAME="cplex"

SS=1134
BF1=2
BF2=2
let SPB=BF2
let SC=BF1*BF2
let PBF=BF1

BI=150
NC=1
QSC=0.3
SD=80
OTC=1.5

EC="--Capacity 200 --QuadShortCoeff $QSC  --BeginInventory $BI --mu-dev 0 --sigma-dev $SD --start-seed $SS --NegInventoryCost=$NC --OvertimeProdCost=$OTC"
# --start-ups

echo "-------- EF Directly"
python aircond_cylinders.py --EF-directly --solver-name=${SOLVERNAME} --branching-factors "$BF1 $BF2" $EC

echo "***** Cylinders with no bundles"
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME}  --rel-gap 0.01  --max-solver-threads 2 --start-seed $SS --lagrangian --xhatshuffle --bundles-per-rank=0  --write-solution --intra-hub-conv-thresh 0 --branching-factors "$BF1 $BF2" $EC

echo "======== Cylinders with ${BF1} bundles per rank (not proper bundles)"
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME}  --rel-gap 0.01  --max-solver-threads 2 --start-seed $SS  --lagrangian --xhatshuffle --bundles-per-rank=$BF1  --write-solution --intra-hub-conv-thresh 0 --branching-factors "$BF1 $BF2" $EC

# NOTE: As of 3 March 2022, you can't compare pickle bundle problems with non-pickled. See _demands_creator in aircondB.py for more discusion.

echo "^^^^^^^^^ Make pickle bundles"
mpiexec --oversubscribe -np 2 python bundle_pickler.py --branching-factors "$BF1 $BF2" --pickle-bundles-dir="." --scenarios-per-bundle=$SPB $EC --start-seed=$SS

echo ""
echo "***** Use pickle bundles"
echo "This will NOT match the others"
echo ""
# It is entirely up to the user to make sure that the scenario count and scenarios per bundle match between creating the pickles and using them
# (the costs and the start-seed probably don't matter, since the pickle has it all)
mpiexec --oversubscribe -np 3 python -m mpi4py aircond_cylinders.py --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors $PBF --rel-gap 0.01 --max-solver-threads 2 --start-seed $SS  --lagrangian --xhatshuffle --bundles-per-rank=0  --scenarios-per-bundle=$SPB --write-solution --intra-hub-conv-thresh 0 --unpickle-bundles-dir="." $EC

echo ""
echo "The pickle bundle objective  should NOT have matched the others"
echo ""
