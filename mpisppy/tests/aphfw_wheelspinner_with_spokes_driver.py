###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test the aphfw WheelSpinner with hub + Lagrangian spoke using farmer example.

This exercises the full cylinder machinery with actual bound computation:
  - APHFWHub sends nonant values and W duals to the Lagrangian spoke
  - Lagrangian spoke computes outer bounds
  - Hub convergence checker monitors the gap

Must be run under MPI with exactly 6 ranks (3 hub + 3 spoke):
    mpirun -np 6 python -m mpi4py aphfw_wheelspinner_with_spokes_driver.py
"""

import os
import sys

from mpi4py import MPI

from mpisppy.tests.utils import aphfw_test_solver


def test_wheelspinner_with_spokes():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 6:
        if rank == 0:
            print(f"ERROR: This test requires exactly 6 MPI ranks, got {size}")
        sys.exit(1)

    import mpisppy.utils.cfg_vanilla as vanilla
    import mpisppy.utils.config as config
    from mpisppy.spin_the_wheel import WheelSpinner
    from mpisppy.tests.examples.farmer import (
        scenario_creator,
        scenario_denouement,
    )
    from mpisppy.utils.cfg_vanilla import aphfw_hub

    # Build cfg with all required attributes
    cfg = config.Config()
    cfg.popular_args()
    cfg.aphfw_args()

    # Hub convergence options
    cfg.add_to_config("rel_gap", description="relative gap", domain=float, default=0.0)
    cfg.add_to_config("abs_gap", description="absolute gap", domain=float, default=0.0)
    cfg.add_to_config("max_stalled_iters", description="max stalled iters", domain=int, default=100)
    cfg.add_to_config("num_scens", description="number of scenarios", domain=int, default=3)

    # Set values
    cfg.solver_name = aphfw_test_solver()
    cfg.default_rho = 1.0
    cfg.max_iterations = 20
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
    cfg.rel_gap = 0.01
    cfg.abs_gap = 0.0
    cfg.max_stalled_iters = 100

    num_scens = 3
    all_scenario_names = [f"Scenario{i}" for i in range(num_scens)]
    scenario_creator_kwargs = {"num_scens": num_scens}
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    hub_dict = aphfw_hub(
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    # Lagrangian outer bound spoke
    lagrangian_spoke = vanilla.lagrangian_spoke(
        *beans,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    if rank == 0:
        print("\n=== WheelSpinner hub+Lagrangian test (farmer, 3 scenarios) ===")

    # Remove stale trace file if present
    trace_file = "LagrangianOuterBound.csv"
    if rank == 0 and os.path.exists(trace_file):
        os.remove(trace_file)
    comm.Barrier()

    wheel = WheelSpinner(hub_dict, [lagrangian_spoke])
    wheel.spin()

    # Clean up trace file
    if rank == 0 and os.path.exists(trace_file):
        os.remove(trace_file)

    # Check results on the hub rank 0
    if wheel.strata_rank == 0 and wheel.cylinder_rank == 0:
        cycle_count = wheel.spcomm.current_iteration()
        print(f"\n  Cycles completed: {cycle_count}")
        assert cycle_count > 0, f"Expected cycles > 0, got {cycle_count}"

    comm.Barrier()

    if rank == 0:
        print("\nPASSED: WheelSpinner hub+Lagrangian test completed successfully")

    return True


if __name__ == "__main__":
    result = test_wheelspinner_with_spokes()
    if not result:
        sys.exit(1)
