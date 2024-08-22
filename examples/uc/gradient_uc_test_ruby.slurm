#!/bin/bash

#SBATCH -N 2
#SBATCH -J 10scen_gradient
#SBATCH -t 00:05:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -A mpisppy

export MPICH_ASYNC_PROGRESS=1
source ${HOME}/venvs/mpisppy/bin/activate
cd ${HOME}/mpi-sppy/examples/uc

SOLVERNAME="gurobi_persistent"

#echo "First solve the EF to get a npy file for xhat ; the uc_cyl_nonants.npy name is hard-wired (you don't really such a good xhat, but this one is just easy to get)"
#python simple_ef.py $SOLVERNAME

echo "Now run the cylinders"
srun -n 15 python -u -m mpi4py gradient_uc_cylinders.py --bundles-per-rank=0 --max-iterations=100 --default-rho=1 --xhatshuffle --ph-ob --num-scens=5 --max-solver-threads=2 --lagrangian-iter0-mipgap=1e-7 --ph-mipgaps-json=phmipgaps.json --solver-name=${SOLVERNAME} --xhatpath uc_cyl_nonants.npy --rel-gap 0.00001 --abs-gap=1 --intra-hub-conv-thresh=-1 --grad-rho-setter --grad-order-stat 0.5 --grad-dynamic-primal-crit

#--grad-rho-setter --grad-order-stat 0.5 --grad-dynamic-primal-crit

## --grad-dynamic-primal-crit
## --use-cost-based-rho

####mpiexec --oversubscribe -np 1 python -m mpi4py gradient_uc_cylinders.py --bundles-per-rank=0 --max-iterations=20 --default-rho=1 --num-scens=5 --max-solver-threads=2 --lagrangian-iter0-mipgap=1e-7 --ph-mipgaps-json=phmipgaps.json --solver-name=${SOLVERNAME} --xhatpath uc_cyl_nonants.npy --rel-gap 0.000001 --display-progress --grad-rho-setter --grad-order-stat 0.5 --grad-display-rho

# ? do we need ph_ob_rho_rescale_factors_json", ph_ob_gradient_rho", ??

# --fwph
echo

# --rho-setter --order-stat 0.5
# --display-progress
