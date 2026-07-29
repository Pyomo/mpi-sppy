#!/bin/bash
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# Rank-aware scalene launcher, meant to be the program that mpiexec runs.
# Each rank profiles itself and writes $OUTDIR/scalene_rank_<rank>.json
#
# Required environment:
#   OUTDIR  directory to write the per-rank profile into (must already exist)
# Arguments:
#   the script and arguments to profile (use absolute paths; cwd is $OUTDIR)
#
# Not normally run by hand; see run_experiments.bash.

set -euo pipefail

if [[ -z "${OUTDIR:-}" ]]; then
  echo "scalene_wrapper.bash: OUTDIR must be set" >&2
  exit 1
fi

# Determine MPI rank from common env vars (OpenMPI / MPICH / Slurm)
RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-}}}"
if [[ -z "${RANK}" ]]; then
  echo "Could not determine MPI rank from environment" \
       "(OMPI_COMM_WORLD_RANK / PMI_RANK / SLURM_PROCID)." >&2
  exit 1
fi

cd "${OUTDIR}"

exec python -m scalene run \
  --outfile "scalene_rank_${RANK}.json" \
  "$@"
