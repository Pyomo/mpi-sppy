#!/bin/bash
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# Run the "fraction of time in Python" experiments under scalene.
#
# Every case runs the same three cylinders (PH hub, lagrangian, xhatshuffle) so
# that only the model and the instance size change. Each case is repeated REPS
# times so that the report can show run-to-run spread rather than a single
# sample; scalene works by sampling, so a single run says very little.
#
# Usage:
#   ./run_experiments.bash                 # all cases, REPS reps each
#   REPS=1 ./run_experiments.bash farmer3  # just one case, one rep
#
# Environment:
#   SOLVER   solver name (default gurobi)
#   REPS     repetitions per case (default 3)
#   NP       number of MPI ranks, i.e. cylinders (default 3)
#   THREADS  --max-solver-threads (default 2)
#   RESULTS  output directory (default ./results)
#   TRIES    attempts per rep before giving up (default 3); see the retry note below
#   PROFILE  1 (default) to run under scalene; 0 to run the identical cases with
#            no profiler and record only wall time, in <outdir>/wall.txt. The
#            unprofiled times are what make it possible to state how much of the
#            measured Python time is scalene's own instrumentation overhead.
#
# The solver name is part of the output path, because which Pyomo solver
# interface is used turns out to dominate the answer: "gurobi" is the
# file-based interface, which writes an LP file and parses a solution file in
# Python, while "gurobi_persistent" keeps the model in the solver through its C
# API. Run the sweep once per interface and compare.
#
# Profiles land in $RESULTS/<solver>/<case>/rep<n>/scalene_rank_<rank>.json
# Then run ./make_tables.bash to regenerate the LaTeX.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
EXAMPLES="${REPO}/examples"
DRIVER="${REPO}/mpisppy/generic_cylinders.py"

SOLVER="${SOLVER:-gurobi}"
REPS="${REPS:-3}"
NP="${NP:-3}"
THREADS="${THREADS:-2}"
RESULTS="${RESULTS:-${HERE}/results}"
TRIES="${TRIES:-3}"
PROFILE="${PROFILE:-1}"

# Cylinders and PH settings shared by every case. rel-gap is 0 so that runs end
# on max-iterations (or exact convergence) instead of on a gap tolerance, which
# keeps the amount of work per case predictable.
COMMON=(--solver-name "${SOLVER}"
        --max-solver-threads "${THREADS}"
        --default-rho 1
        --lagrangian
        --xhatshuffle
        --rel-gap 0.0)

# Case definitions. Iteration counts were calibrated so that the "long" cases
# each take roughly a minute of wall time; farmer3 is deliberately left short
# because the short-run numbers are themselves a finding.
#
# The bundled cases exist because subproblem size turns out to drive the answer.
# farmer240 and farmer240_bun10 are a controlled pair: same instance, same
# iteration count, differing only in whether ten scenarios are gathered into one
# subproblem. sslp_15_45_15_bun3 is the same comparison on a MIP.
case_args() {
  case "$1" in
    farmer3)
      echo "--module-name ${EXAMPLES}/farmer/farmer --num-scens 3 --max-iterations 100"
      ;;
    farmer60)
      echo "--module-name ${EXAMPLES}/farmer/farmer --num-scens 60 --max-iterations 400"
      ;;
    farmer240)
      echo "--module-name ${EXAMPLES}/farmer/farmer --num-scens 240 --max-iterations 200"
      ;;
    farmer240_bun10)
      echo "--module-name ${EXAMPLES}/farmer/farmer --num-scens 240 --scenarios-per-bundle 10 --max-iterations 200"
      ;;
    sslp_15_45_10)
      echo "--module-name ${EXAMPLES}/sslp/sslp --sslp-data-path ${EXAMPLES}/sslp/data --instance-name sslp_15_45_10 --max-iterations 50"
      ;;
    sslp_15_45_15)
      echo "--module-name ${EXAMPLES}/sslp/sslp --sslp-data-path ${EXAMPLES}/sslp/data --instance-name sslp_15_45_15 --max-iterations 50"
      ;;
    sslp_15_45_15_bun3)
      echo "--module-name ${EXAMPLES}/sslp/sslp --sslp-data-path ${EXAMPLES}/sslp/data --instance-name sslp_15_45_15 --scenarios-per-bundle 3 --max-iterations 50"
      ;;
    sslp_5_25_50)
      echo "--module-name ${EXAMPLES}/sslp/sslp --sslp-data-path ${EXAMPLES}/sslp/data --instance-name sslp_5_25_50 --max-iterations 150"
      ;;
    *)
      echo "unknown case: $1" >&2
      return 1
      ;;
  esac
}

ALL_CASES=(farmer3 farmer60 farmer240 farmer240_bun10
           sslp_15_45_10 sslp_15_45_15 sslp_15_45_15_bun3 sslp_5_25_50)
CASES=("${@:-}")
if [[ -z "${CASES[0]}" ]]; then
  CASES=("${ALL_CASES[@]}")
fi

echo "solver=${SOLVER} ranks=${NP} threads=${THREADS} reps=${REPS}"
echo "results -> ${RESULTS}"

# Scalene instruments the import machinery, and occasionally a rank dies during
# startup with a KeyError out of importlib's lock handling before any mpi-sppy
# code runs. That is a profiler startup race, not a property of the case, so a
# rep that comes back with fewer than NP profiles is simply run again. Failures
# are counted and reported at the end so a retried-away problem is still
# visible; a rep that never succeeds is left out of the results rather than
# silently reported as if it had worked.
failures=0
retries=0

# Unprofiled runs go somewhere else so that they cannot overwrite the profiles.
SUBDIR="${SOLVER}"
if [[ "${PROFILE}" == "0" ]]; then
  SUBDIR="${SOLVER}-unprofiled"
fi

for c in "${CASES[@]}"; do
  read -r -a cargs <<< "$(case_args "$c")"
  for rep in $(seq 1 "${REPS}"); do
    outdir="${RESULTS}/${SUBDIR}/${c}/rep${rep}"
    echo "^^^ ${c} rep ${rep} ^^^"
    for try in $(seq 1 "${TRIES}"); do
      rm -rf "${outdir}"
      mkdir -p "${outdir}"
      if [[ "${PROFILE}" == "0" ]]; then
        # Same case, no profiler: record wall time only.
        /usr/bin/time -f "%e" -o "${outdir}/wall.txt" \
          mpiexec --oversubscribe -np "${NP}" \
          python -m mpi4py "${DRIVER}" "${cargs[@]}" "${COMMON[@]}" \
          > "${outdir}/run.log" 2>&1 || true
        if grep -q 'Cylinder finalization complete' "${outdir}/run.log"; then
          n_found="${NP}"
          echo "  wall $(cat "${outdir}/wall.txt")s (unprofiled)"
          break
        fi
        n_found=0
      else
        OUTDIR="${outdir}" mpiexec --oversubscribe -np "${NP}" \
          "${HERE}/scalene_wrapper.bash" \
          "${DRIVER}" "${cargs[@]}" "${COMMON[@]}" \
          > "${outdir}/run.log" 2>&1 || true
        n_found=$(find "${outdir}" -name 'scalene_rank_*.json' | wc -l)
        if [[ "${n_found}" -eq "${NP}" ]]; then
          break
        fi
      fi
      if [[ "${PROFILE}" == "0" ]]; then
        echo "  attempt ${try}/${TRIES}: run did not complete" \
             "(see ${outdir}/run.log)" >&2
      else
        echo "  attempt ${try}/${TRIES}: expected ${NP} profiles, found ${n_found}" \
             "(see ${outdir}/run.log)" >&2
      fi
      retries=$((retries + 1))
    done
    if [[ "${n_found}" -ne "${NP}" ]]; then
      echo "  GIVING UP on ${c} rep ${rep} after ${TRIES} attempts" >&2
      failures=$((failures + 1))
      continue
    fi
    if [[ "${PROFILE}" != "0" ]]; then
      tail -1 "${outdir}/run.log"
    fi
  done
done

echo "done: ${retries} retried attempt(s), ${failures} rep(s) abandoned"
if [[ "${failures}" -ne 0 ]]; then
  exit 1
fi
