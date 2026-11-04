#!/usr/bin/env bash
# Exercise an installed worker from outside the source checkout.
set -euo pipefail

worker="$1"
export MPISPPY_EXPECT_PROGRESS_WORKER="$worker"
cd "$RUNNER_TEMP"

python "$GITHUB_WORKSPACE/mpisppy/tests/test_mpi_progress.py" -v

timeout 90s mpiexec -np 2 python -m mpi4py -m mpi_one_sided_test \
  2>&1 | tee "$RUNNER_TEMP/progress-rma-$worker.log"
rg -q 'Test passed' "$RUNNER_TEMP/progress-rma-$worker.log"
rg -q "worker=$worker" "$RUNNER_TEMP/progress-rma-$worker.log"
rg -q 'compute_regions=[1-9]' "$RUNNER_TEMP/progress-rma-$worker.log"
rg -q 'probes=[1-9]' "$RUNNER_TEMP/progress-rma-$worker.log"
if [ "$worker" = native ]; then
  rg -q 'mpi_error=0, wait_error=0' "$RUNNER_TEMP/progress-rma-$worker.log"
fi

timeout 300s mpiexec -np 2 python -m mpi4py -m unittest \
  mpisppy.tests.test_with_cylinders.Test_farmer_with_cylinders.test_xhatshuffle_extended \
  2>&1 | tee "$RUNNER_TEMP/progress-cylinders-$worker.log"
rg -q "worker=$worker" "$RUNNER_TEMP/progress-cylinders-$worker.log"
rg -q 'compute_regions=[1-9]' "$RUNNER_TEMP/progress-cylinders-$worker.log"
rg -q 'probes=[1-9]' "$RUNNER_TEMP/progress-cylinders-$worker.log"
