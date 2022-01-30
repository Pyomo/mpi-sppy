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
mpiexec -np 4 python -m mpi4py farmer_cylinders.py  3 --bundles-per-rank=0 --max-iterations=50 --default-rho=1 --solver-name=${SOLVERNAME}


# evaluate the zhat for the xhat computed above
mpiexec -np $SLURM_NTASKS python -m mpisppy.confidence_intervals.mmw_conf afarmer farmer_cyl_nonants.npy ${SOLVERNAME} --MMW-num-batches 12 --MMW-batch-size 10 --confidence-level 0.9 --start-scen 10
