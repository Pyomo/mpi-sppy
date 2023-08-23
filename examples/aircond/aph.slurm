#!/bin/bash -l

# starved

#!/bin/bash -l
#SBATCH --job-name=aircond_aph_slurm
#SBATCH --output=aph_slurm.out
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=2
#SBATCH --time=0-0:30:00
#SBATCH --nodelist=c[3]

conda activate mpisppy

SOLVERNAME="gurobi_persistent"
export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c3/gurobi.lic

# TBD: aircond uses start-seed (but seed is allowed as an arg that is ignored).

# xhat output file name is hardwired
mpiexec -np $SLURM_NTASKS python -m mpi4py aircond_cylinders.py --bundles-per-rank=0 --max-iterations=10 --default-rho=1 --solver-name=${SOLVERNAME} --branching-factors "20 5 4" --Capacity 200 --QuadShortCoeff 0.3  --BeginInventory 50 --rel-gap 0.001 --mu-dev 0 --sigma-dev 40 --max-solver-threads 2 --start-seed 0 --start-ups --lagrangian --xhatshuffle   --aph-dispatch-frac 0.2 --run-async 


# --no-lagrangian --with-lagranger --lagranger-rho-rescale-factors-json lagranger_factors.json
# --with-lagrangian --no-lagranger
