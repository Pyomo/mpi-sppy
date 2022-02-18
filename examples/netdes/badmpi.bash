#!/bin/bash -l

# This can also be run as a straight bash script

#!/bin/bash -l
#SBATCH --job-name=netdes_demo_slurm
#SBATCH --output=netdes_demo_slurm.out
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=2
#SBATCH --time=0-0:10:00
#SBATCH --nodelist=c[4]

SOLVER=gurobi_persistent
#export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c4/gurobi.lic

SLURM_NTASKS=10
mpiexec -np $SLURM_NTASKS python netdes_cylinders.py --solver-name=${SOLVER} --max-solver-threads=2 --default-rho=5000.0 --instance-name=network-10-20-L-02 --max-iterations=100 --rel-gap=0.01 --no-cross-scenario-cuts

