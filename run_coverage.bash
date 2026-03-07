#!/bin/bash
###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Run mpi-sppy tests with code coverage collection.
#
# This mirrors all the test jobs from .github/workflows/test_pr_and_main.yml.
#
# Usage:  bash run_coverage.bash [SOLVER]
#
#   SOLVER defaults to "cplex" (matches CI). Use e.g. "xpress" or "gurobi".
#
# Tests that require unavailable optional dependencies (amplpy, parapint, mip)
# are either skipped automatically via has_module checks or fail gracefully
# through run_phase (which continues to the next phase on error).
#
# Results are written to htmlcov/index.html

set -e

SOLVER="${1:-cplex}"
SOLVER_PERSISTENT="${SOLVER}_persistent"
SOLVER_DIRECT="${SOLVER}_direct"

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# Clean previous coverage data
rm -f .coverage .coverage.*

# --- Environment for automatic subprocess coverage ---
export COVERAGE_PROCESS_START="$PROJ_DIR/.coveragerc"
# Put our sitecustomize on the path so every python process auto-starts coverage
export PYTHONPATH="$PROJ_DIR:${PYTHONPATH:-}"

# Helper: run a phase, continue on failure
phase=0
run_phase() {
    phase=$((phase + 1))
    echo ""
    echo "=== Phase $phase: $1 ==="
    shift
    "$@" 2>&1 || echo "  (phase $phase finished with non-zero exit)"
}

# Helper: check if a python module is importable
has_module() {
    python -c "import $1" 2>/dev/null
}

# ---------- Serial pytest / unittest tests ----------

run_phase "test_ef_ph (serial)" \
    coverage run --rcfile=.coveragerc -m pytest mpisppy/tests/test_ef_ph.py -v

run_phase "test_component_map_usage (serial)" \
    coverage run --rcfile=.coveragerc -m pytest mpisppy/tests/test_component_map_usage.py -v

run_phase "test_admmWrapper (serial, spawns mpiexec)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_admmWrapper.py

run_phase "test_stoch_admmWrapper (serial, spawns mpiexec)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_stoch_admmWrapper.py

run_phase "test_aph (spawns mpiexec)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_aph.py

run_phase "test_pickle_bundle (spawns mpiexec)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_pickle_bundle.py

run_phase "test_mps (serial)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_mps.py

run_phase "test_conf_int_farmer (spawns mpiexec)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_conf_int_farmer.py

run_phase "test_conf_int_aircond (spawns mpiexec)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_conf_int_aircond.py

run_phase "test_gradient_rho (spawns mpiexec)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_gradient_rho.py

run_phase "test_xbar_w_reader_writer (spawns mpiexec)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_xbar_w_reader_writer.py

run_phase "test_pysp_model (serial)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/test_pysp_model.py

run_phase "pysp_model pytest (serial)" \
    coverage run --rcfile=.coveragerc -m pytest mpisppy/utils/pysp_model/ -v

# ---------- MPI tests (direct launch) ----------

run_phase "test_with_cylinders (mpiexec -np 2)" \
    mpiexec -np 2 coverage run --rcfile="$PROJ_DIR/.coveragerc" -m mpi4py mpisppy/tests/test_with_cylinders.py

# ---------- Tests that spawn mpiexec internally ----------

run_phase "straight_tests.py (spawns mpiexec)" \
    coverage run --rcfile=.coveragerc mpisppy/tests/straight_tests.py

# ---------- Example-based tests ----------

run_phase "examples/afew.py $SOLVER_PERSISTENT" \
    bash -c "cd '$PROJ_DIR/examples' && coverage run --rcfile='$PROJ_DIR/.coveragerc' afew.py '$SOLVER_PERSISTENT'"

run_phase "examples/run_all.py $SOLVER_PERSISTENT nouc" \
    bash -c "cd '$PROJ_DIR/examples' && coverage run --rcfile='$PROJ_DIR/.coveragerc' run_all.py '$SOLVER_PERSISTENT' '' nouc"

run_phase "examples/run_all.py $SOLVER_DIRECT nouc" \
    bash -c "cd '$PROJ_DIR/examples' && coverage run --rcfile='$PROJ_DIR/.coveragerc' run_all.py '$SOLVER_DIRECT' '' nouc"

run_phase "examples/generic_tester.py ${SOLVER}_direct nouc" \
    bash -c "cd '$PROJ_DIR/examples' && coverage run --rcfile='$PROJ_DIR/.coveragerc' generic_tester.py '${SOLVER}_direct' '' nouc"

# ---------- Optional tests (skip if deps missing) ----------

if has_module amplpy; then
    run_phase "test_agnostic (serial)" \
        coverage run --rcfile=.coveragerc mpisppy/tests/test_agnostic.py

    run_phase "afew_agnostic.py" \
        bash -c "cd '$PROJ_DIR/mpisppy/agnostic/examples' && coverage run --rcfile='$PROJ_DIR/.coveragerc' afew_agnostic.py"
else
    echo ""
    echo "=== SKIPPED: agnostic tests (amplpy not installed) ==="
fi

if has_module parapint; then
    run_phase "test_sc (serial)" \
        coverage run --rcfile=.coveragerc -m pytest mpisppy/tests/test_sc.py -v

    run_phase "test_sc (mpiexec -np 3)" \
        mpiexec -np 3 coverage run --rcfile="$PROJ_DIR/.coveragerc" -m pytest mpisppy/tests/test_sc.py -v
else
    echo ""
    echo "=== SKIPPED: schur-complement tests (parapint not installed) ==="
fi

# ---------- Combine and report ----------

echo ""
echo "=== Combining and reporting ==="
coverage combine --rcfile=.coveragerc
coverage report --rcfile=.coveragerc --show-missing | head -120
coverage html --rcfile=.coveragerc

echo ""
echo "HTML report: file://$PROJ_DIR/htmlcov/index.html"
