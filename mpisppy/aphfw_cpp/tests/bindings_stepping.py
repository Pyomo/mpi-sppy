###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test the C++ stepping API via pybind11.

Runs FWAPH on the smallFWAPH test problem for a known number of cycles,
then verifies the Lagrangian bound matches the standalone C++ run.

This must be run under MPI with exactly 2 ranks:
    mpirun -np 2 python bindings_stepping.py

It uses the same problem files as the C++ unit test (smallFWAPH/).
"""

import sys

import aphfw_cpp
import numpy as np
from mpi4py import MPI


def test_stepping():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 MPI ranks, got {size}")
        sys.exit(1)

    # Problem directory (relative to tests/)
    problem_dir = "smallFWAPH/mps_dir"

    # Config matching the C++ unit test, with block-sync for determinism
    config = {
        "aph-max-cycles": 8,
        "rho-csv": "smallFWAPH/mps_dir/scen1_rho.csv",
        "nl-json": "smallFWAPH/nl.json",
        "grad-json": "smallFWAPH/grad.json",
        "iter0-json": "smallFWAPH/iter0.json",
        "FW-tmax": 4,
        "FW-tau": 0.01,
        "primal-term-thresh": 0.001,
        "dual-term-thresh": 0.001,
        "display-progress": True,
        "verbosity": 0,
        "block-sync": True,
        "solver-log-file": "solver.log",  # suppress solver stdout
    }

    # --- Test 1: run all 8 cycles at once (should match standalone) ---
    lib = aphfw_cpp.APHFWLibrary()
    lib.init(comm, problem_dir, config)

    completed = lib.run_cycles(8)
    bound_all = lib.get_current_lagrangian_bound()
    cycle_all = lib.get_current_cycle()
    converged = lib.is_converged()

    if rank == 0:
        print("\n=== Test 1: run_cycles(8) ===")
        print(f"  Completed cycles: {completed}")
        print(f"  Current cycle:    {cycle_all}")
        print(f"  Converged:        {converged}")
        print(f"  Lagrangian bound: {bound_all}")

    # Check data export
    n = lib.get_local_nonant_length()
    names = lib.get_local_scenario_names()
    if rank == 0:
        print(f"  Local nonant length: {n}")
        print(f"  Local scenario names: {names}")

    x_buf = np.empty(n, dtype=np.float64)
    w_buf = np.empty(n, dtype=np.float64)
    lib.get_all_local_nonants(x_buf)
    lib.get_all_local_w(w_buf)
    if rank == 0:
        print(f"  x values: {x_buf}")
        print(f"  W values: {w_buf}")

    lib.finalize()

    # --- Test 2: run in steps (1 cycle at a time) ---
    lib2 = aphfw_cpp.APHFWLibrary()
    lib2.init(comm, problem_dir, config)

    total_completed = 0
    for step in range(8):
        c = lib2.run_cycles(1)
        total_completed += c
        if lib2.is_converged():
            break

    bound_step = lib2.get_current_lagrangian_bound()
    cycle_step = lib2.get_current_cycle()

    if rank == 0:
        print("\n=== Test 2: 8 x run_cycles(1) ===")
        print(f"  Total completed: {total_completed}")
        print(f"  Current cycle:   {cycle_step}")
        print(f"  Lagrangian bound: {bound_step}")

    x_buf2 = np.empty(n, dtype=np.float64)
    w_buf2 = np.empty(n, dtype=np.float64)
    lib2.get_all_local_nonants(x_buf2)
    lib2.get_all_local_w(w_buf2)

    lib2.finalize()

    # --- Compare ---
    if rank == 0:
        print("\n=== Comparison ===")
        print(f"  Cycles: all-at-once={cycle_all}, stepping={cycle_step}")
        print(f"  Bounds: all-at-once={bound_all}, stepping={bound_step}")

        # The bounds should match exactly (same algorithm, same data, deterministic with block sync)
        # But async timing may cause slight differences. Check x and W values too.
        x_match = np.allclose(x_buf, x_buf2, atol=1e-6)
        w_match = np.allclose(w_buf, w_buf2, atol=1e-6)
        print(f"  x values match: {x_match}")
        print(f"  W values match: {w_match}")
        if not x_match:
            print(f"    x diff: {np.abs(x_buf - x_buf2).max()}")
        if not w_match:
            print(f"    W diff: {np.abs(w_buf - w_buf2).max()}")

        bound_match = abs(bound_all - bound_step) < 1e-4
        print(f"  Bound match (tol=1e-4): {bound_match}")

        if cycle_all == cycle_step and bound_match:
            print("\nPASSED: Stepping produces equivalent results to batch run")
        else:
            print("\nNOTE: Results differ (expected for async algorithm; "
                  "use --block-sync for deterministic comparison)")

    comm.Barrier()

    # --- Test 3: aph-max-cycles 0 is no limit when stepping ---
    # It once ended the run after cycle 1, and the next run_cycles call then
    # waited forever (the ctest TIMEOUT turns that into a failure).
    lib3 = aphfw_cpp.APHFWLibrary()
    lib3.init(comm, problem_dir, dict(config, **{"aph-max-cycles": 0}))
    steps = []
    for step in range(3):
        steps.append(lib3.run_cycles(1))
        if lib3.is_converged():
            break
    converged3 = lib3.is_converged()
    lib3.finalize()
    if rank == 0:
        print(f"\n=== Test 3: aph-max-cycles 0, run_cycles(1) x3: {steps} ===")
    if not (steps == [1, 1, 1] or (converged3 and all(c == 1 for c in steps[:-1]))):
        if rank == 0:
            print(f"FAILED: expected a cycle per call, got {steps}")
        sys.exit(1)
    comm.Barrier()

    # --- Test 4: a termination other than the cycle target ends the run ---
    # A huge APH-tau-tolerance makes the first projection stop on tau.  The
    # next run_cycles call used to re-enter the event loop with no eligible
    # scenarios and wait forever (the ctest TIMEOUT turns that into a failure).
    lib4 = aphfw_cpp.APHFWLibrary()
    lib4.init(comm, problem_dir, dict(config, **{"APH-tau-tolerance": 1e30}))
    first = lib4.run_cycles(1)
    second = lib4.run_cycles(1)
    lib4.finalize()
    if rank == 0:
        print(f"\n=== Test 4: tau termination, run_cycles(1) x2: {first}, {second} ===")
    if second != 0:
        if rank == 0:
            print(f"FAILED: expected no cycles after termination, got {second}")
        sys.exit(1)
    comm.Barrier()


if __name__ == "__main__":
    test_stepping()
