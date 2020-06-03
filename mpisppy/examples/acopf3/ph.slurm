#!/bin/bash -l
#SBATCH --job-name=cc
#SBATCH --output=phaph.out
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=2
#SBATCH --time=0-1:00:00
#SBATCH --nodelist=c[4]

MAXIT=15
BF1=5
BF2=10
SCENPERBUN=0

srun date
NODENAME="$(srun hostname)"
srun echo $NODENAME
for BF in {3..6..2}
do
#   for SCENPERBUN in {0..4..2}
#   do
   mpiexec -n $SLURM_NTASKS  python -m mpi4py dlw2wood.py ${BF1} ${BF2} ${MAXIT} ${SCENPERBUN} >& phaph_${BF1}_${BF2}_${MAXIT}_${SCENPERBUN}.out
#   done
done
srun uptime

# it is going to be tough to deal with licenses for PH... maybe use cplex
# --mpi=pmi2
