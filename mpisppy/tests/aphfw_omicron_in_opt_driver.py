###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test that the post-iter-0 omicron freeze hook fires when running via
APHFWOpt (the mpi-sppy hub integration path).

Verifies:
  1. After aphfw_main returns, omicron_is_frozen() is True.
  2. Frozen values equal multiplier * (rho-at-freeze-time).
  3. With multiplier != 1.0, the values reflect the scaling.

Runs the farmer example without mpi-sppy extensions. The decoupling claim
("omicron stays constant while rho drifts") is covered at a lower level
in test_omicron_bridge.py Part 4; this test only verifies that the
Python-side cycle loop in opt.py actually triggers freeze_omicron.

Must be run under MPI with exactly 3 ranks:
    mpirun -np 3 python -m mpi4py aphfw_omicron_in_opt_driver.py
"""

import sys

import numpy as np
from mpi4py import MPI

from mpisppy.opt.aphfw import APHFWOpt
from mpisppy.tests.examples.farmer import (
    scenario_creator,
    scenario_denouement,
)
from mpisppy.tests.utils import aphfw_test_solver


def _farmer_options(multiplier):
    num_scens = 3
    all_scenario_names = [f"Scenario{i}" for i in range(num_scens)]
    scenario_creator_kwargs = {"num_scens": num_scens}
    options = {
        "solver_name": aphfw_test_solver(),
        "defaultPHrho": 1.0,
        "convthresh": 0,
        "PHIterLimit": 0,
        "time_limit": 600,
        "verbose": False,
        "display_progress": False,
        "display_convergence_detail": False,
        "iter0_solver_options": {},
        "iterk_solver_options": {},
        "tee-rank0-solves": False,
        "trace_prefix": "",
        "presolve": False,
        "rounding_bias": 0.0,
        "warmstart_subproblems": False,
        "user_warmstart": False,
        # aphfw knobs
        "aphfw_fw_tmax": 2,
        "aphfw_fw_tau": 0.01,
        "aphfw_max_cycles": 4,
        "aphfw_primal_term_thresh": 0.001,
        "aphfw_dual_term_thresh": 0.001,
        "aphfw_display_progress": False,
        "aphfw_block_sync": True,
        "aphfw_verbosity": 0,
        "aphfw_omicron_rho_multiplier": multiplier,
    }
    kwargs = dict(
        all_scenario_names=all_scenario_names,
        scenario_creator=scenario_creator,
        scenario_denouement=scenario_denouement,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    return options, kwargs


def _run_check(multiplier, expected_scale):
    """Run APHFWOpt with given multiplier, return (passed_on_rank, message).

    expected_scale: the value all omicron entries should hold (given rho
    starts at defaultPHrho=1.0 and no drivers change it, the post-iter-0
    snapshot of rho is 1.0, so omicron = multiplier * 1.0).
    """
    options, kwargs = _farmer_options(multiplier)
    opt = APHFWOpt(options, **kwargs)
    opt.aphfw_main(finalize=False)

    n = opt._local_nonant_length
    omicron = np.empty(n, dtype=np.float64)
    rho = np.empty(n, dtype=np.float64)
    opt._cpp.get_all_local_omicron(omicron)
    opt._cpp.get_all_local_rho(rho)

    frozen = opt._cpp.omicron_is_frozen()
    opt.aphfw_finalize()

    return {
        "frozen": frozen,
        "omicron": omicron,
        "rho": rho,
        "expected": expected_scale,
    }


def test_omicron_in_opt():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 3:
        if rank == 0:
            print(f"ERROR: requires 3 MPI ranks, got {size}")
        sys.exit(1)

    passed = True

    # --- Run 1: default multiplier = 1.0 ---
    if rank == 0:
        print("\n=== Run 1: aphfw_omicron_rho_multiplier = 1.0 ===")
    r1 = _run_check(multiplier=1.0, expected_scale=1.0)

    if not r1["frozen"]:
        print(f"  [rank {rank}] FAIL: omicron not frozen after aphfw_main")
        passed = False
    elif rank == 0:
        print("  PASS: omicron_is_frozen() == True")
    if not np.allclose(r1["omicron"], r1["rho"], atol=1e-10):
        print(f"  [rank {rank}] FAIL: omicron != 1.0 * rho")
        print(f"    omicron: {r1['omicron']}")
        print(f"    rho:     {r1['rho']}")
        passed = False
    elif rank == 0:
        print(f"  PASS: omicron == rho (values: {r1['omicron']})")

    comm.Barrier()

    # --- Run 2: multiplier = 2.0, omicron should be 2x rho-at-freeze ---
    if rank == 0:
        print("\n=== Run 2: aphfw_omicron_rho_multiplier = 2.0 ===")
    r2 = _run_check(multiplier=2.0, expected_scale=2.0)

    if not r2["frozen"]:
        print(f"  [rank {rank}] FAIL: omicron not frozen with multiplier=2.0")
        passed = False
    # rho at freeze is the rho after post_iter0 drivers (none here) + any
    # initial tweaks. With default farmer setup rho stays at 1.0, so
    # omicron should be 2.0. But we verify against the live rho/2.0 to
    # account for any future driver behavior changes.
    ratio = r2["omicron"] / np.maximum(r2["rho"], 1e-30)
    if not np.allclose(ratio, 2.0, atol=1e-10):
        print(f"  [rank {rank}] FAIL: omicron / rho != 2.0")
        print(f"    omicron: {r2['omicron']}")
        print(f"    rho:     {r2['rho']}")
        print(f"    ratio:   {ratio}")
        passed = False
    elif rank == 0:
        print(f"  PASS: omicron == 2.0 * rho (omicron: {r2['omicron']})")

    all_passed = comm.allreduce(int(passed), op=MPI.MIN)
    if rank == 0:
        if all_passed:
            print("\nAll omicron-in-opt tests PASSED")
        else:
            print("\nSome omicron-in-opt tests FAILED")
    return bool(all_passed)


if __name__ == "__main__":
    result = test_omicron_in_opt()
    if not result:
        sys.exit(1)
