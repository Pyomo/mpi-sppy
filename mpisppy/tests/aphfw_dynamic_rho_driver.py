###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test dynamic rho via the mpi-sppy NormRhoUpdater extension.

Runs APHFWOpt on the farmer example with NormRhoUpdater enabled and verifies:
  1. The extension is called (rho values change from initial default)
  2. The algorithm still converges (or runs to max cycles without error)
  3. Comparison: same problem WITHOUT extensions produces different rho trajectory

Must be run under MPI with exactly 3 ranks:
    mpirun -np 3 python -m mpi4py aphfw_dynamic_rho_driver.py
"""

import sys

import numpy as np
from mpi4py import MPI

from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
from mpisppy.opt.aphfw import APHFWOpt
from mpisppy.tests.examples.farmer import (
    scenario_creator,
    scenario_denouement,
)
from mpisppy.tests.utils import aphfw_test_solver


def _make_options(extensions=None, extension_kwargs=None):
    """Build options dict for APHFWOpt on the farmer problem."""
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
        # aphfw-specific
        "aphfw_fw_tmax": 2,
        "aphfw_fw_tau": 0.01,
        "aphfw_max_cycles": 10,
        "aphfw_primal_term_thresh": 0.001,
        "aphfw_dual_term_thresh": 0.001,
        "aphfw_display_progress": True,
        "aphfw_block_sync": True,
        "aphfw_verbosity": 0,
    }

    if extensions is not None:
        options["norm_rho_options"] = {"verbose": True}

    kwargs = dict(
        all_scenario_names=all_scenario_names,
        scenario_creator=scenario_creator,
        scenario_denouement=scenario_denouement,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    if extensions is not None:
        kwargs["extensions"] = extensions
    if extension_kwargs is not None:
        kwargs["extension_kwargs"] = extension_kwargs

    return options, kwargs


def test_dynamic_rho():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 3:
        if rank == 0:
            print(f"ERROR: This test requires exactly 3 MPI ranks, got {size}")
        sys.exit(1)

    passed = True

    # --- Part 1: Run WITH NormRhoUpdater ---
    if rank == 0:
        print("\n=== Part 1: APHFWOpt with NormRhoUpdater ===")

    options, kwargs = _make_options(
        extensions=MultiExtension,
        extension_kwargs={"ext_classes": [NormRhoUpdater]},
    )

    opt = APHFWOpt(options, **kwargs)
    trivial_bound = opt.aphfw_main(finalize=False)

    cycle_count = opt._cpp.get_current_cycle()
    converged = opt._cpp.is_converged()
    n = opt._local_nonant_length

    # Read final rho from C++
    rho_after = np.empty(n, dtype=np.float64)
    opt._cpp.get_all_local_rho(rho_after)

    # Read final x
    x_with_ext = np.empty(n, dtype=np.float64)
    opt._cpp.get_all_local_nonants(x_with_ext)

    if rank == 0:
        print(f"  Cycles: {cycle_count}, Converged: {converged}")
        print(f"  Trivial bound: {trivial_bound}")
    print(f"  [rank {rank}] final rho: {rho_after}")
    print(f"  [rank {rank}] final x:   {x_with_ext}")

    # Check that rho changed from the initial value of 1.0
    initial_rho = 1.0
    rho_changed = not np.allclose(rho_after, initial_rho, atol=1e-8)
    if rank == 0:
        print(f"  Rho changed from initial: {rho_changed}")

    if not rho_changed:
        if rank == 0:
            print("  WARNING: Rho did not change. NormRhoUpdater may not have"
                  " triggered (could be OK if convergence was immediate).")
        # Only fail if we ran enough cycles for it to have had a chance
        if cycle_count >= 3:
            print(f"  [rank {rank}] FAIL: rho unchanged after {cycle_count} cycles")
            passed = False

    opt.aphfw_finalize()
    comm.Barrier()

    # --- Part 2: Run WITHOUT extensions (baseline) ---
    if rank == 0:
        print("\n=== Part 2: APHFWOpt without extensions (baseline) ===")

    options_base, kwargs_base = _make_options(extensions=None)

    opt2 = APHFWOpt(options_base, **kwargs_base)
    opt2.aphfw_main(finalize=False)

    cycle_count2 = opt2._cpp.get_current_cycle()
    n2 = opt2._local_nonant_length

    rho_base = np.empty(n2, dtype=np.float64)
    opt2._cpp.get_all_local_rho(rho_base)

    x_without_ext = np.empty(n2, dtype=np.float64)
    opt2._cpp.get_all_local_nonants(x_without_ext)

    if rank == 0:
        print(f"  Cycles: {cycle_count2}")
    print(f"  [rank {rank}] final rho (baseline): {rho_base}")
    print(f"  [rank {rank}] final x (baseline):   {x_without_ext}")

    # Baseline rho should still be 1.0
    if not np.allclose(rho_base, initial_rho):
        print(f"  [rank {rank}] FAIL: baseline rho changed unexpectedly")
        passed = False
    else:
        if rank == 0:
            print("  Baseline rho stayed at 1.0: PASS")

    opt2.aphfw_finalize()
    comm.Barrier()

    # --- Summary ---
    all_passed = comm.allreduce(int(passed), op=MPI.MIN)
    if rank == 0:
        if all_passed:
            print("\nAll dynamic rho tests PASSED")
        else:
            print("\nSome dynamic rho tests FAILED")

    return bool(all_passed)


if __name__ == "__main__":
    result = test_dynamic_rho()
    if not result:
        sys.exit(1)
