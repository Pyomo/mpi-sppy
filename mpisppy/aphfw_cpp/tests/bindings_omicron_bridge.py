###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test the C++ omicron bridge API: get_all_local_omicron, freeze_omicron,
omicron_is_frozen. See design/omicron_implementation_plan.md.

Verifies:
  1. After init with defaults, omicron is NOT frozen.
  2. freeze_omicron(multiplier) sets omicron = multiplier * rho and flips
     the frozen flag.
  3. Second freeze_omicron() raises (one-shot).
  4. After freeze, changing rho does NOT change omicron (decoupling).
  5. --omicron-all at init freezes omicron at setup with a uniform scalar.
  6. --omicron-all above omicron-max raises during init (bounds check).

Must be run under MPI with exactly 2 ranks:
    mpirun -np 2 python -m mpi4py bindings_omicron_bridge.py
"""

import sys

import aphfw_cpp
import numpy as np
from mpi4py import MPI


def _base_config(extra=None):
    cfg = {
        "aph-max-cycles": 4,
        "rho-all": 1.0,
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
    if extra:
        cfg.update(extra)
    return cfg


def test_omicron_bridge():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print(f"ERROR: This test requires exactly 2 MPI ranks, got {size}")
        sys.exit(1)

    problem_dir = "smallFWAPH/mps_dir"
    passed = True

    # -------------------------------------------------------------------
    # Part 1: default config -> omicron not frozen after init
    # -------------------------------------------------------------------
    if rank == 0:
        print("\n=== Part 1: default init leaves omicron unfrozen ===")

    lib = aphfw_cpp.APHFWLibrary()
    lib.init(comm, problem_dir, _base_config())
    n = lib.get_local_nonant_length()

    if lib.omicron_is_frozen():
        print(f"  [rank {rank}] FAIL: omicron_is_frozen() True right after init with defaults")
        passed = False
    elif rank == 0:
        print("  PASS: omicron_is_frozen() is False")

    # omicron starts as all-ones
    o_buf = np.empty(n, dtype=np.float64)
    lib.get_all_local_omicron(o_buf)
    if not np.allclose(o_buf, 1.0):
        print(f"  [rank {rank}] FAIL: initial omicron not all-ones, got {o_buf}")
        passed = False
    elif rank == 0:
        print(f"  PASS: initial omicron is all-ones ({o_buf[0]})")

    # -------------------------------------------------------------------
    # Part 2: freeze_omicron(multiplier) -> omicron = multiplier * rho
    # -------------------------------------------------------------------
    if rank == 0:
        print("\n=== Part 2: freeze_omicron(2.5) captures 2.5*rho ===")

    rho_before = np.empty(n, dtype=np.float64)
    lib.get_all_local_rho(rho_before)

    multiplier = 2.5
    lib.freeze_omicron(multiplier)

    if not lib.omicron_is_frozen():
        print(f"  [rank {rank}] FAIL: omicron_is_frozen() still False after freeze")
        passed = False

    o_after = np.empty(n, dtype=np.float64)
    lib.get_all_local_omicron(o_after)
    expected = multiplier * rho_before
    if not np.allclose(o_after, expected):
        print(f"  [rank {rank}] FAIL: omicron != multiplier*rho")
        print(f"    expected {expected}")
        print(f"    got      {o_after}")
        passed = False
    elif rank == 0:
        print(f"  PASS: omicron == {multiplier} * rho (values: {o_after})")

    # -------------------------------------------------------------------
    # Part 3: second freeze_omicron raises
    # -------------------------------------------------------------------
    if rank == 0:
        print("\n=== Part 3: second freeze_omicron raises ===")

    try:
        lib.freeze_omicron(3.0)
        print(f"  [rank {rank}] FAIL: second freeze did not raise")
        passed = False
    except RuntimeError as e:
        if rank == 0:
            print(f"  PASS: second freeze raised: {e}")

    # -------------------------------------------------------------------
    # Part 4: after freeze, changing rho does NOT change omicron
    # -------------------------------------------------------------------
    if rank == 0:
        print("\n=== Part 4: rho drift does not perturb frozen omicron ===")

    omicron_snapshot = o_after.copy()
    new_rho = np.full(n, 7.0, dtype=np.float64)
    lib.set_all_local_rho(new_rho)
    lib.run_cycles(1)

    o_after_cycle = np.empty(n, dtype=np.float64)
    lib.get_all_local_omicron(o_after_cycle)
    rho_after_cycle = np.empty(n, dtype=np.float64)
    lib.get_all_local_rho(rho_after_cycle)

    if not np.allclose(o_after_cycle, omicron_snapshot):
        print(f"  [rank {rank}] FAIL: omicron drifted after rho change + cycle")
        print(f"    snapshot: {omicron_snapshot}")
        print(f"    now:      {o_after_cycle}")
        passed = False
    elif rank == 0:
        print(f"  PASS: omicron unchanged (value {o_after_cycle[0]}) while rho is {rho_after_cycle[0]}")

    lib.finalize()
    comm.Barrier()

    # -------------------------------------------------------------------
    # Part 5: --omicron-all freezes omicron at init
    # -------------------------------------------------------------------
    if rank == 0:
        print("\n=== Part 5: --omicron-all 3.0 freezes at init ===")

    lib2 = aphfw_cpp.APHFWLibrary()
    lib2.init(comm, problem_dir, _base_config({"omicron-all": 3.0}))
    n2 = lib2.get_local_nonant_length()

    if not lib2.omicron_is_frozen():
        print(f"  [rank {rank}] FAIL: omicron not frozen after init with --omicron-all 3.0")
        passed = False
    o2 = np.empty(n2, dtype=np.float64)
    lib2.get_all_local_omicron(o2)
    if not np.allclose(o2, 3.0):
        print(f"  [rank {rank}] FAIL: omicron-all values wrong, expected 3.0, got {o2}")
        passed = False
    elif rank == 0:
        print("  PASS: frozen at init with uniform value 3.0")
    lib2.finalize()
    comm.Barrier()

    # -------------------------------------------------------------------
    # Part 6: --omicron-all above omicron-max raises at init
    # -------------------------------------------------------------------
    if rank == 0:
        print("\n=== Part 6: --omicron-all 1e10 exceeds omicron-max, raises ===")

    lib3 = aphfw_cpp.APHFWLibrary()
    try:
        lib3.init(comm, problem_dir, _base_config({"omicron-all": 1e10}))
        print(f"  [rank {rank}] FAIL: init with --omicron-all 1e10 did not raise")
        passed = False
    except RuntimeError as e:
        if rank == 0:
            print(f"  PASS: init raised: {e}")
    # lib3 may not be fully initialized; don't finalize
    comm.Barrier()

    all_passed = comm.allreduce(int(passed), op=MPI.MIN)
    if rank == 0:
        if all_passed:
            print("\nAll omicron bridge tests PASSED")
        else:
            print("\nSome omicron bridge tests FAILED")
    return bool(all_passed)


if __name__ == "__main__":
    result = test_omicron_bridge()
    if not result:
        sys.exit(1)
