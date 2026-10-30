###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test the aphfw WheelSpinner hub-only pipeline using mpi-sppy's farmer example.

This runs APHFWHub through WheelSpinner with no spokes, exercising the full
cylinder machinery (MPI RMA windows, spcomm.sync, _export_to_hub).

Must be run under MPI with exactly 3 ranks:
    mpirun -np 3 python -m mpi4py aphfw_wheelspinner_hub_only_driver.py
"""

import sys

from mpi4py import MPI

from mpisppy.tests.utils import aphfw_test_solver


def test_wheelspinner_hub_only():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 3:
        if rank == 0:
            print(f"ERROR: This test requires exactly 3 MPI ranks, got {size}")
        sys.exit(1)

    import mpisppy.utils.config as config
    from mpisppy.spin_the_wheel import WheelSpinner
    from mpisppy.tests.examples.farmer import (
        scenario_creator,
        scenario_denouement,
    )
    from mpisppy.utils.cfg_vanilla import aphfw_hub

    # Build a cfg object with all required attributes
    cfg = config.Config()
    cfg.popular_args()
    cfg.aphfw_args()

    # Hub convergence options (not in popular_args)
    cfg.add_to_config("rel_gap", description="relative gap", domain=float, default=0.0)
    cfg.add_to_config("abs_gap", description="absolute gap", domain=float, default=0.0)
    cfg.add_to_config("max_stalled_iters", description="max stalled iters", domain=int, default=100)
    cfg.add_to_config("num_scens", description="number of scenarios", domain=int, default=3)

    # Set values
    cfg.solver_name = aphfw_test_solver()
    cfg.default_rho = 1.0
    cfg.max_iterations = 5
    cfg.time_limit = 600
    cfg.verbose = False
    cfg.display_progress = False
    cfg.display_convergence_detail = False
    cfg.tee_rank0_solves = False
    cfg.trace_prefix = ""
    cfg.presolve = False
    cfg.rounding_bias = 0.0
    cfg.warmstart_subproblems = False
    cfg.user_warmstart = False
    cfg.num_scens = 3

    # aphfw options
    cfg.aphfw_fw_tmax = 10
    cfg.aphfw_fw_tau = 0.01
    cfg.aphfw_primal_term_thresh = 0.001
    cfg.aphfw_dual_term_thresh = 0.001
    cfg.aphfw_display_progress = True
    cfg.aphfw_block_sync = True
    cfg.aphfw_verbosity = 0

    # Hub convergence
    cfg.rel_gap = 0.0
    cfg.abs_gap = 0.0
    cfg.max_stalled_iters = 100

    num_scens = 3
    all_scenario_names = [f"Scenario{i}" for i in range(num_scens)]
    scenario_creator_kwargs = {"num_scens": num_scens}

    hub_dict = aphfw_hub(
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    if rank == 0:
        print("\n=== WheelSpinner hub-only test (farmer, 3 scenarios, no spokes) ===")

    wheel = WheelSpinner(hub_dict, [])
    wheel.spin()

    # Query results from the hub
    if wheel.strata_rank == 0:
        cycle_count = wheel.spcomm.current_iteration()

        print(f"\n  [rank {rank}] Cycles completed: {cycle_count}")
        print(f"  [rank {rank}] WheelSpinner completed successfully")

        # The hub should have run some cycles
        assert cycle_count > 0, f"Expected cycles > 0, got {cycle_count}"

    comm.Barrier()

    if rank == 0:
        print("\nPASSED: WheelSpinner hub-only test completed successfully")

    return True


if __name__ == "__main__":
    result = test_wheelspinner_hub_only()
    if not result:
        sys.exit(1)
