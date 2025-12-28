#!/bin/bash

# wrap the mpiexec run in python so the rank can be determined
# run with $ SOLVER=gurobi mpiexec -np 3 ./tests.sh
# (do chmod once)

set -euo pipefail

SOLVER="${SOLVER:-gurobi}"

# Determine MPI rank from common env vars (OpenMPI / MPICH / Slurm)
RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-}}}"
if [[ -z "${RANK}" ]]; then
  echo "Could not determine MPI rank from environment (OMPI_COMM_WORLD_RANK / PMI_RANK / SLURM_PROCID)." >&2
  exit 1
fi

echo "^^^ sslp_15_45_10 ^^^ rank=${RANK}"

python -m scalene run \
  --outfile "scalene_rank_${RANK}.txt" \
  ../../mpisppy/generic_cylinders.py \
  --module-name ../sslp/sslp --sslp-data-path ../sslp/data --instance-name sslp_15_45_10 --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 4 --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01

