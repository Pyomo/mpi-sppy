#!/usr/bin/env bash
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.

# Exercise an installed worker from outside the source checkout.
set -euo pipefail

worker="$1"
export MPISPPY_EXPECT_PROGRESS_WORKER="$worker"
cd "$RUNNER_TEMP"

python "$GITHUB_WORKSPACE/mpisppy/tests/test_mpi_progress.py" -v

timeout 90s mpiexec -np 2 python -m mpi4py -m mpi_one_sided_test \
  2>&1 | tee "$RUNNER_TEMP/progress-rma-$worker.log"
grep -q 'Test passed' "$RUNNER_TEMP/progress-rma-$worker.log"
grep -q "worker=$worker" "$RUNNER_TEMP/progress-rma-$worker.log"

timeout 300s mpiexec -np 2 python -m mpi4py -m unittest \
  mpisppy.tests.test_with_cylinders.Test_farmer_with_cylinders.test_xhatshuffle_extended \
  2>&1 | tee "$RUNNER_TEMP/progress-cylinders-$worker.log"
grep -q "worker=$worker" "$RUNNER_TEMP/progress-cylinders-$worker.log"
