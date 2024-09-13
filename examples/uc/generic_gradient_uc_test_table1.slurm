#!/bin/bash

#SBATCH -N 6
#SBATCH -J 50scen_gradient
#SBATCH -t 00:20:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -A mpisppy

export MPICH_ASYNC_PROGRESS=1
source ${HOME}/venvs/mpisppy/bin/activate
cd ${HOME}/mpi-sppy/examples/uc

SOLVERNAME="gurobi_persistent"

# IMPT: run limited number of iterations (2) to get the uc_cyl_nonants.npy xhat file"

# the command line below executes UC with cost-based, gradient rho, and dynamic primal/dual-based updates
srun -n 150 python -u -m mpi4py ../../mpisppy/generic_cylinders.py --module-name uc_funcs --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --xhatshuffle --ph-ob --num-scens=50 --max-solver-threads=2 --mipgaps-json=phmipgaps.json --solver-name=${SOLVERNAME} --xhatpath uc_cyl_nonants_table1.npy --rel-gap 0.00001 --abs-gap=1 --intra-hub-conv-thresh=-1 --grad-rho-setter --grad-order-stat 0.5 --grad-dynamic-primal-crit --grad-dynamic-dual-crit

#--grad-rho-setter --grad-order-stat 0.5 --grad-dynamic-primal-crit

# other ...

# uses uc_funcs rho-setter
# --use-cost-based-rho

# --grad-dynamic-primal-crit
# --grad-dynamic-dual-crit
# --grad-dynamic-primal-thresh
# --grad-dynamic-dual-thresh

####mpiexec --oversubscribe -np 1 python -m mpi4py gradient_uc_cylinders.py --bundles-per-rank=0 --max-iterations=20 --default-rho=1 --num-scens=5 --max-solver-threads=2 --lagrangian-iter0-mipgap=1e-7 --ph-mipgaps-json=phmipgaps.json --solver-name=${SOLVERNAME} --xhatpath uc_cyl_nonants.npy --rel-gap 0.000001 --display-progress --grad-rho-setter --grad-order-stat 0.5 --grad-display-rho

# ? do we need ph_ob_rho_rescale_factors_json", ph_ob_gradient_rho", ??

# --fwph
echo

# --rho-setter --order-stat 0.5
# --display-progress
