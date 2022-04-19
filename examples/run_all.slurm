#!/bin/bash -l

# This can also be run as a straight bash script

#!/bin/bash -l
#SBATCH --job-name=run_all
#SBATCH --output=run_all_slurm.out
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --time=0-2:00:00
#SBATCH --nodelist=c[4]

conda activate mpisppy
SOLVERNAME="gurobi"  # "gurobi_persistent"
export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c4/gurobi.lic

python run_all.py $SOLVERNAME

