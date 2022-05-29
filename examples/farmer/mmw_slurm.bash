#!/bin/bash -l

#!/bin/bash -l
#SBATCH --job-name=mmw_slurm
#SBATCH --output=mmw_slurm.out
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --time=0-0:30:00
#SBATCH --nodelist=c[3]


SOLVERNAME="gurobi"
export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c3/gurobi.lic

# taking defaults for most farmer args

# get an xhat
# xhat output file name is hardwired to 'farmer_cyl_nonants.npy'
mpiexec -np 3 python -m mpi4py farmer_cylinders.py  --num-scens 3 --lagrangian --xhatshuffle --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name=${SOLVERNAME}

echo "starting zhat4xhat"
python -m mpisppy.confidence_intervals.zhat4xhat farmer --xhatpath farmer_cyl_nonants.npy --solver-name ${SOLVERNAME} --branching-factors 10 --num-samples 5 --confidence-level 0.95
