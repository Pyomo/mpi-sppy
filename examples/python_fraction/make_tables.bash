#!/bin/bash
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# Regenerate the LaTeX tables in python_fraction.tex from the profiles that
# run_experiments.bash wrote. Cheap to re-run; does not re-run any experiment.
#
# Usage:
#   ./make_tables.bash [results_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${1:-${HERE}/results}"

# Rank order comes from generic_cylinders: the hub is rank 0 and the spokes
# follow in the order build_spoke_list appends them, which for --lagrangian
# --xhatshuffle is lagrangian then xhatshuffle.
LABELS="PH hub,lagrangian,xhatshuffle"

CASES=(farmer3 farmer60 farmer240 farmer240_bun10
       sslp_15_45_10 sslp_15_45_15 sslp_15_45_15_bun3 sslp_5_25_50)

# Primary results: the persistent interface.
python "${HERE}/summarize_reps.py" \
  --results "${RESULTS}" \
  --solvers gurobi_persistent \
  --cases "${CASES[@]}" \
  --rank-labels "${LABELS}" \
  --out "${HERE}/scalene_summary_persistent.tex"

# Secondary: the file-based interface, for the contrast noted in the writeup.
if [[ -d "${RESULTS}/gurobi" ]]; then
  python "${HERE}/summarize_reps.py" \
    --results "${RESULTS}" \
    --solvers gurobi \
    --cases "${CASES[@]}" \
    --rank-labels "${LABELS}" \
    --out "${HERE}/scalene_summary_file_interface.tex"
fi

echo "Tables written to ${HERE}"
