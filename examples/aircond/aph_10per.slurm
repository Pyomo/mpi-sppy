#!/bin/bash -l
#SBATCH --job-name=aph_reallystarved
#SBATCH --output=try_big_pickles_aph_reallystarved_REVISION-Feb28.out
#SBATCH --ntasks=300
#SBATCH --nodes=34
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

# Slurm script for the one million scenarios, 10 bundles per rank example in
# The Online Companion for Projective Hedging Algorithms for Multi-Stage Stochastic Programming,
#       Supporting Distributed and Asynchronous Implementation

# This script will probably need to be modified to work your paricular computer.

# The next line is needed to make mpi work correctly on some super computers
export MPICH_ASYNC_PROGRESS=1
# The next line should be changed to setup your Python environment for mpi-sppy
source ${HOME}/venvs/mpisppy-jan2023/bin/activate

date
NODENAME="$(srun hostname)"
echo ${NODENAME[0]}

SOLVERNAME="gurobi_persistent"

REPLICANT=1
# Change the next line to name a directory on your computer
PICKLE_DIR="/p/lustre1/watson61/aircond/$REPLICANT"

BF1=1000
BF2=25
BF3=10
BF4=4
let SPB=BF2*BF3*BF4
let SC=BF1*BF2*BF3*BF4
let PBF=BF1  # by design, this is the number of bundles

BI=50
NC=1
QSC=0.3
SD=40
OTC=5

let SEED=(REPLICANT-1)*1000000

EC="--Capacity 200 --QuadShortCoeff $QSC  --BeginInventory $BI --mu-dev 0 --sigma-dev $SD --start-seed $SEED --NegInventoryCost=$NC --OvertimeProdCost=$OTC"
# --start-ups

# There are no restrictions on the number of processors for the pickler and it could be given as many as the maxiumum number of cores (or threads?) that will be used in the next step.
echo "^^^^^^^^^ Make pickle bundles"
srun -n $SLURM_NTASKS unbuffer python -m mpi4py bundle_pickler.py --branching-factors "$BF1 $BF2 $BF3 $BF4" --pickle-bundles-dir=$PICKLE_DIR --scenarios-per-bundle=$SPB $EC

#echo "***** Use pickle bundles"
# It is entirely up to the user to make sure that the scenario count and scenarios per bundle match between creating the pickles and using them (the costs probably don't matter, since the pickle has it all)
srun -n $SLURM_NTASKS unbuffer python -m mpi4py aircond_cylinders.py --run-async --aph-frac-needed=1.0 --aph-dispatch-frac=0.20 --max-iterations=100 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors $PBF --rel-gap 0.0001 --abs-gap 0.5 --max-solver-threads 2 --start-seed $SEED --bundles-per-rank=0  --scenarios-per-bundle=$SPB --write-solution --intra-hub-conv-thresh 0 --unpickle-bundles-dir=$PICKLE_DIR $EC --display-progress --solver-options="method=0" --lagrangian --xhatshuffle --trace-prefix "${SLURM_JOB_NAME}_"