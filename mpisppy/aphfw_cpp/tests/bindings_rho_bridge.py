###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test the C++ rho bridge API: get/set xbar and rho via pybind11.

Verifies:
  1. get_all_local_xbar returns values of the correct length
  2. get_all_local_rho returns values matching the initial rho-all setting
  3. set_all_local_rho + get_all_local_rho round-trips correctly
  4. Modified rho values persist across a cycle (rho is used, not reset)

Must be run under MPI with exactly 2 ranks:
    mpirun -np 2 python -m mpi4py bindings_rho_bridge.py
"""

import sys

import aphfw_cpp
import numpy as np
from mpi4py import MPI


def test_rho_bridge():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 MPI ranks, got {size}")
        sys.exit(1)

    problem_dir = "smallFWAPH/mps_dir"
    rho_val = 1.5

    config = {
        "aph-max-cycles": 4,
        "rho-all": rho_val,
        "nl-json": "smallFWAPH/nl.json",
        "grad-json": "smallFWAPH/grad.json",
        "iter0-json": "smallFWAPH/iter0.json",
        "FW-tmax": 2,
        "FW-tau": 0.01,
        "primal-term-thresh": 0.001,
        "dual-term-thresh": 0.001,
        "block-sync": True,
        "solver-log-file": "solver.log",
    }

    lib = aphfw_cpp.APHFWLibrary()
    lib.init(comm, problem_dir, config)

    n = lib.get_local_nonant_length()
    passed = True

    # --- Test 1: get_all_local_xbar returns correct length ---
    if rank == 0:
        print("\n=== Test 1: get_all_local_xbar length ===")
    xbar_buf = np.empty(n, dtype=np.float64)
    lib.get_all_local_xbar(xbar_buf)
    if rank == 0:
        print(f"  xbar length: {len(xbar_buf)}, expected: {n}")
        print(f"  xbar values: {xbar_buf}")
    assert len(xbar_buf) == n, f"xbar length mismatch: {len(xbar_buf)} != {n}"

    # --- Test 2: get_all_local_rho returns initial rho-all value ---
    if rank == 0:
        print("\n=== Test 2: get_all_local_rho initial values ===")
    rho_buf = np.empty(n, dtype=np.float64)
    lib.get_all_local_rho(rho_buf)
    if rank == 0:
        print(f"  rho values: {rho_buf}")
        print(f"  expected all: {rho_val}")
    if not np.allclose(rho_buf, rho_val):
        print(f"  [rank {rank}] FAIL: rho values don't match rho-all={rho_val}")
        passed = False
    else:
        if rank == 0:
            print("  PASS")

    # --- Test 3: set_all_local_rho round-trips ---
    if rank == 0:
        print("\n=== Test 3: set/get rho round-trip ===")
    new_rho = np.arange(1, n + 1, dtype=np.float64) * 0.5
    lib.set_all_local_rho(new_rho)

    readback = np.empty(n, dtype=np.float64)
    lib.get_all_local_rho(readback)
    if rank == 0:
        print(f"  set rho:  {new_rho}")
        print(f"  got rho:  {readback}")
    if not np.allclose(new_rho, readback):
        print(f"  [rank {rank}] FAIL: rho round-trip mismatch")
        print(f"    max diff: {np.abs(new_rho - readback).max()}")
        passed = False
    else:
        if rank == 0:
            print("  PASS")

    # --- Test 4: rho persists across a cycle ---
    if rank == 0:
        print("\n=== Test 4: rho persists across cycle ===")
    # Set a distinctive rho pattern
    distinctive_rho = np.full(n, 3.14, dtype=np.float64)
    lib.set_all_local_rho(distinctive_rho)

    # Run a cycle
    lib.run_cycles(1)

    # Read rho back — it should still be 3.14
    after_cycle = np.empty(n, dtype=np.float64)
    lib.get_all_local_rho(after_cycle)
    if rank == 0:
        print(f"  rho after cycle: {after_cycle}")
    if not np.allclose(after_cycle, 3.14):
        print(f"  [rank {rank}] FAIL: rho changed after cycle")
        print(f"    expected 3.14, got: {after_cycle}")
        passed = False
    else:
        if rank == 0:
            print("  PASS")

    lib.finalize()
    comm.Barrier()

    all_passed = comm.allreduce(int(passed), op=MPI.MIN)
    if rank == 0:
        if all_passed:
            print("\nAll rho bridge tests PASSED")
        else:
            print("\nSome rho bridge tests FAILED")

    return bool(all_passed)


if __name__ == "__main__":
    result = test_rho_bridge()
    if not result:
        sys.exit(1)
