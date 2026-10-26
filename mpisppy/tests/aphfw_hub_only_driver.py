###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test the full aphfw hub pipeline using mpi-sppy's farmer example.

This test:
  1. Runs APHFWOpt directly (no WheelSpinner) on the farmer problem
  2. Verifies C++ produces reasonable results through the Python pipeline
  3. Regression: re-runs the same problem files via aphfw_cpp directly and
     compares x and W values

Must be run under MPI with exactly 3 ranks:
    mpirun -np 3 python -m mpi4py aphfw_hub_only_driver.py
"""

import os
import sys

import aphfw_cpp
import numpy as np

# Add aphfw Python package and build directory to path
from mpi4py import MPI


def test_hub_only():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 3:
        if rank == 0:
            print(f"ERROR: This test requires exactly 3 MPI ranks, got {size}")
        sys.exit(1)

    # --- Part 1: Run through APHFWOpt pipeline ---
    if rank == 0:
        print("\n=== Part 1: APHFWOpt pipeline (farmer, 3 scenarios) ===")

    # Import mpi-sppy farmer example
    from mpisppy.opt.aphfw import APHFWOpt
    from mpisppy.tests.examples.farmer import (
        scenario_creator,
        scenario_denouement,
    )

    num_scens = 3
    all_scenario_names = [f"Scenario{i}" for i in range(num_scens)]

    # Options matching what vanilla.shared_options would create
    options = {
        "solver_name": "gurobi",
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
        "aphfw_fw_tmax": 10,
        "aphfw_fw_tau": 0.01,
        "aphfw_max_cycles": 5,
        "aphfw_primal_term_thresh": 0.001,
        "aphfw_dual_term_thresh": 0.001,
        "aphfw_display_progress": True,
        "aphfw_block_sync": True,
        "aphfw_verbosity": 0,
        "aphfw_keep_scenario_files": True,  # keep for Part 2
    }

    scenario_creator_kwargs = {"num_scens": num_scens}

    opt = APHFWOpt(
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement=scenario_denouement,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    # Run without WheelSpinner (spcomm stays None)
    trivial_bound = opt.aphfw_main(finalize=False)

    cycle_count = opt._cpp.get_current_cycle()
    converged = opt._cpp.is_converged()
    n = opt._cpp.get_local_nonant_length()
    scen_names = opt._cpp.get_local_scenario_names()

    x1 = np.empty(n, dtype=np.float64)
    w1 = np.empty(n, dtype=np.float64)
    opt._cpp.get_all_local_nonants(x1)
    opt._cpp.get_all_local_w(w1)

    # Save the scenario dir for Part 2
    scenario_dir = opt._scenario_dir

    if rank == 0:
        print(f"  Trivial bound:  {trivial_bound}")
        print(f"  Cycles:         {cycle_count}")
        print(f"  Converged:      {converged}")
        print(f"  Scenario dir:   {scenario_dir}")
    print(f"  [rank {rank}] scenarios: {scen_names}, nonants: {n}")
    print(f"  [rank {rank}] x = {x1}")
    print(f"  [rank {rank}] W = {w1}")

    opt._cpp.finalize()
    opt._cpp = None

    comm.Barrier()

    # --- Part 2: Regression — re-run same files via aphfw_cpp directly ---
    if rank == 0:
        print("\n=== Part 2: Regression via aphfw_cpp on same files ===")

    config = {
        "aph-max-cycles": 5,
        "rho-all": 1.0,
        "nl-json": os.path.join(scenario_dir, "nl.json"),
        "grad-json": os.path.join(scenario_dir, "grad.json"),
        "iter0-json": os.path.join(scenario_dir, "iter0.json"),
        "FW-tmax": 10,
        "FW-tau": 0.01,
        "primal-term-thresh": 0.001,
        "dual-term-thresh": 0.001,
        "display-progress": True,
        "verbosity": 0,
        "block-sync": True,
        "solver-log-file": os.path.join(scenario_dir, "solver.log"),
    }

    lib2 = aphfw_cpp.APHFWLibrary()
    lib2.set_explicit_scenario_names(list(scen_names))
    lib2.init(comm, scenario_dir, config)

    lib2.run_cycles(5)
    cycle2 = lib2.get_current_cycle()
    n2 = lib2.get_local_nonant_length()

    x2 = np.empty(n2, dtype=np.float64)
    w2 = np.empty(n2, dtype=np.float64)
    lib2.get_all_local_nonants(x2)
    lib2.get_all_local_w(w2)

    if rank == 0:
        print(f"  Cycles:         {cycle2}")

    print(f"  [rank {rank}] x = {x2}")
    print(f"  [rank {rank}] W = {w2}")

    lib2.finalize()

    comm.Barrier()

    # --- Part 3: Compare ---
    if rank == 0:
        print("\n=== Comparison ===")

    x_match = np.allclose(x1, x2, atol=1e-6)
    w_match = np.allclose(w1, w2, atol=1e-6)

    print(f"  [rank {rank}] cycles: {cycle_count} vs {cycle2}, "
          f"x match: {x_match}, W match: {w_match}")

    if not x_match:
        print(f"  [rank {rank}] x diff: {np.abs(x1 - x2).max()}")
        print(f"  [rank {rank}] x1={x1}, x2={x2}")
    if not w_match:
        print(f"  [rank {rank}] W diff: {np.abs(w1 - w2).max()}")
        print(f"  [rank {rank}] W1={w1}, W2={w2}")

    all_match = comm.allreduce(int(x_match and w_match and cycle_count == cycle2),
                               op=MPI.MIN)

    if rank == 0:
        if all_match:
            print("\nPASSED: Hub pipeline produces same results as direct aphfw_cpp")
        else:
            print("\nFAILED: Results differ between hub pipeline and direct aphfw_cpp")

    # Clean up scenario files
    if rank == 0:
        import shutil
        parent = os.path.dirname(scenario_dir)
        try:
            shutil.rmtree(parent)
        except OSError:
            pass

    comm.Barrier()
    return all_match


if __name__ == "__main__":
    result = test_hub_only()
    if not result:
        sys.exit(1)
