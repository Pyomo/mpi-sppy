#!/bin/bash
# Example / regression check: sequential sampling (MRP) via mrp_generic with the
# farmer model, plus a multistage aircond run.  Generic-driver equivalent of
# farmer_sequential.bash.
#
# Solver can be overridden via the first CLI arg or the SOLVERNAME env var.
# Defaults to xpress because that is what the CI images have a usable
# license for; a developer can pass "cplex" or "gurobi" locally.
#
# This is more than a smoke test: run_and_check aborts (nonzero exit) if any
# invocation fails OR does not print a completed run with a finite
# CI=[0, <number>] result, so a crash, non-convergence, or an inf/nan CI is a
# CI failure rather than being silently ignored.

set -eo pipefail

SOLVERNAME="${1:-${SOLVERNAME:-xpress}}"
CL="0.95"
# Path to the farmer module (relative to this directory)
FARMER="farmer"

# run_and_check <label> <command...>
# Run mrp_generic and confirm it completed with a finite CI=[0, <number>].
run_and_check () {
    local label="$1"; shift
    echo ""
    echo "===== ${label} ====="
    local out rc
    set +e
    out=$("$@" 2>&1); rc=$?
    set -e
    echo "$out"
    [ "$rc" -eq 0 ] \
        || { echo "FAIL (${label}): command exited ${rc}"; exit 1; }
    # A completed run prints "MRP complete: T=<int>, CI=[0, <number>]". The
    # upper bound may be a bare float (100.0) or a numpy repr
    # (np.float64(2136.78), from numpy>=2); requiring a digit right after either
    # form still rejects inf/nan/None and no-output.
    echo "$out" | grep -qE "MRP complete: T=[0-9]+, CI=\[0, (np\.float64\()?[0-9]" \
        || { echo "FAIL (${label}): no completed MRP result with a finite CI"; exit 1; }
}

run_and_check "BM sequential sampling (EF)" \
    python -m mpisppy.mrp_generic \
    --module-name ${FARMER} \
    --num-scens 3 \
    --solver-name ${SOLVERNAME} \
    --stopping-criterion BM \
    --BM-h 2.0 \
    --BM-q 1.3 \
    --confidence-level ${CL} \
    --solution-base-name farmer_mrp_xhat

run_and_check "BPL sequential sampling (EF)" \
    python -m mpisppy.mrp_generic \
    --module-name ${FARMER} \
    --num-scens 3 \
    --solver-name ${SOLVERNAME} \
    --stopping-criterion BPL \
    --BPL-eps 100.0 \
    --BPL-c0 25 \
    --confidence-level ${CL}

run_and_check "BM sequential sampling (cylinders)" \
    mpiexec -np 3 python -m mpi4py -m mpisppy.mrp_generic \
    --module-name ${FARMER} \
    --num-scens 3 \
    --solver-name ${SOLVERNAME} \
    --xhat-method cylinders \
    --stopping-criterion BM \
    --BM-h 2.0 \
    --BM-q 1.3 \
    --confidence-level ${CL} \
    --default-rho 1 \
    --max-iterations 10 \
    --lagrangian --xhatshuffle

# Multistage e2e: exercises the do_mrp multistage path (IndepScens_SeqSampling)
# with real multistage-EF solves.  The pytest multistage test only mocks the
# sampler, and the aircond seqsampling tests only construct it, so this is the
# only place a full multistage run() is executed.  aircond is not in this
# directory, so use its importable module name (works regardless of cwd).
run_and_check "BM sequential sampling, multistage (aircond, EF)" \
    python -m mpisppy.mrp_generic \
    --module-name mpisppy.tests.examples.aircond \
    --branching-factors "3 2 2" \
    --solver-name ${SOLVERNAME} \
    --stopping-criterion BM \
    --BM-h 1.75 \
    --BM-hprime 0.5 \
    --BM-eps 0.2 \
    --BM-eps-prime 0.1 \
    --BM-p 0.1 \
    --BM-q 1.2 \
    --confidence-level ${CL}
