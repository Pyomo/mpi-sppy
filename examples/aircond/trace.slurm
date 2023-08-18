#!/bin/bash -l

# Outputs a trace file (but throws an error if the file exists)

#!/bin/bash -l
#SBATCH --job-name=aircond_demo_slurm
#SBATCH --output=demo_slurm.out
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=2
#SBATCH --time=0-0:30:00
#SBATCH --nodelist=c[3]

conda activate mpisppy
SOLVERNAME="gurobi_persistent"
export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c3/gurobi.lic

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignored).

# xhat output file name is hardwired
mpiexec -np $SLURM_NTASKS python -m mpi4py aircond_cylinders.py --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors "20 5 4" --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory 50 --rel-gap 0.001 --mu-dev 0 --sigma-dev 40 --max-solver-threads 2 --start-seed 0 --start-ups --lagrangian --xhatshuffle --trace-prefix ${SLURM_JOB_NAME}_

# --no-lagrangian --with-lagranger --lagranger-rho-rescale-factors-json lagranger_factors.json
# --with-lagrangian --no-lagranger
