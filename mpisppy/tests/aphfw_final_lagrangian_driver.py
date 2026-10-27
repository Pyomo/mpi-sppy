###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Test --aphfw-final-lagrangian: the optional termination Lagrangian in hub mode.

Two independent checks:

  1. A pure check of APHFWHub.publish_outer_bound -- that a final bound only
     replaces the published outer bound when it actually beats what the bound
     spokes already found, and that a winning bound is stamped with the "W"
     trace symbol.  No MPI or solver needed.

  2. A hub-only WheelSpinner run of farmer.  With no spokes the hub is the only
     source of an outer bound, so BestOuterBound isolates exactly what the flag
     does.  With the flag off, C++ never recomputes the Lagrangian bound in hub
     mode, so the published bound must still be the iter-0 (trivial) one; with
     it on, the final solve runs and the published bound must be whichever of
     the two is better.  Note that the Lagrangian bound is not monotone in W,
     so on farmer at rho=1 the final solve usually comes back *worse* than the
     iter-0 bound; the assertion below covers either outcome, and the losing
     case is the one that would expose a final bound that clobbers instead of
     competing.

One case per process (the flag is chosen on the command line) so that no run
inherits another's C++ controller.  tests/final_lagrangian_smoke.bash drives
both cases and compares them.

Must be run under MPI with exactly 3 ranks:
    mpirun -np 3 python -m mpi4py aphfw_final_lagrangian_driver.py [--final-lagrangian]
"""

import sys

from mpi4py import MPI

from mpisppy.tests.utils import aphfw_test_solver


def check_publish_outer_bound():
    """APHFWHub.publish_outer_bound competes rather than clobbers."""
    from mpisppy.cylinders.hub import APHFWHub

    class _StubHub(APHFWHub):
        """Just enough state for OuterBoundUpdate; no cylinder, no windows."""

        def __init__(self, best_outer_bound):
            self.BestOuterBound = best_outer_bound
            # Minimizing: a larger outer bound is a better one.
            self._outer_bound_update = lambda new, old: new > old
            self.strata_rank = 0
            self.latest_ob_char = None
            self.last_ob_idx = None
            self.sends = 0

        def send_boundsout(self):
            self.sends += 1

    assert APHFWHub.converger_spoke_char == "W", (
        f"expected trace symbol W, got {APHFWHub.converger_spoke_char!r}")

    # A final bound worse than the spokes' best is discarded, and leaves the
    # attribution alone -- the winning spoke keeps credit for the bound.
    hub = _StubHub(-100.0)
    assert hub.publish_outer_bound(-200.0) is False
    assert hub.BestOuterBound == -100.0
    assert hub.latest_ob_char is None

    # A final bound that beats them wins, and is attributed to the hub.
    hub = _StubHub(-100.0)
    assert hub.publish_outer_bound(-50.0) is True
    assert hub.BestOuterBound == -50.0
    assert hub.latest_ob_char == "W"

    # Either way the send happens, so write ids stay in step across the
    # cylinder's ranks (readers reject a read whose ids disagree).
    hub = _StubHub(-100.0)
    hub.publish_outer_bound(-200.0)
    hub.publish_outer_bound(-50.0)
    assert hub.sends == 2, f"expected 2 sends, got {hub.sends}"

    # Nothing to offer is not an error.
    hub = _StubHub(-100.0)
    assert hub.publish_outer_bound(None) is False
    assert hub.sends == 0

    print("  publish_outer_bound: competition and 'W' symbol OK")


def run_wheel(final_lagrangian):
    """Hub-only farmer run; returns (trivial_bound, best_outer, cpp_bound)."""
    from unittest.mock import patch

    import mpisppy.utils.config as config
    from mpisppy.opt.aphfw import APHFWOpt
    from mpisppy.spin_the_wheel import WheelSpinner
    from mpisppy.tests.examples.farmer import (
        scenario_creator,
        scenario_denouement,
    )
    from mpisppy.utils.cfg_vanilla import aphfw_hub

    cfg = config.Config()
    cfg.popular_args()
    cfg.aphfw_args()

    # Hub convergence options (not in popular_args)
    cfg.add_to_config("rel_gap", description="relative gap", domain=float, default=0.0)
    cfg.add_to_config("abs_gap", description="absolute gap", domain=float, default=0.0)
    cfg.add_to_config("max_stalled_iters", description="max stalled iters", domain=int, default=100)
    cfg.add_to_config("num_scens", description="number of scenarios", domain=int, default=3)

    cfg.solver_name = aphfw_test_solver()
    cfg.default_rho = 1.0
    cfg.max_iterations = 0
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
    cfg.aphfw_max_cycles = 5
    cfg.aphfw_primal_term_thresh = 0.001
    cfg.aphfw_dual_term_thresh = 0.001
    cfg.aphfw_display_progress = True
    cfg.aphfw_block_sync = True
    cfg.aphfw_verbosity = 0
    cfg.aphfw_final_lagrangian = final_lagrangian

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

    # The hub finalizes the C++ side inside spin(), so read its bound just
    # before that happens.
    cpp_bound = []
    real_finalize = APHFWOpt.aphfw_finalize

    def finalize_and_capture(self):
        if self._cpp is not None:
            cpp_bound.append(self._cpp.get_current_lagrangian_bound())
        real_finalize(self)

    wheel = WheelSpinner(hub_dict, [])
    with patch.object(APHFWOpt, "aphfw_finalize", finalize_and_capture):
        wheel.spin()

    opt = wheel.spcomm.opt
    return (opt.trivial_bound,
            wheel.spcomm.BestOuterBound,
            cpp_bound[0])


def test_final_lagrangian(final_lagrangian):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 3:
        if rank == 0:
            print(f"ERROR: This test requires exactly 3 MPI ranks, got {size}")
        sys.exit(1)

    if rank == 0:
        print("\n=== final-Lagrangian test (farmer, 3 scenarios, hub only, "
              f"aphfw_final_lagrangian={final_lagrangian}) ===")
        check_publish_outer_bound()

    trivial, best_outer, cpp_bound = run_wheel(final_lagrangian)

    if rank == 0:
        print(f"\n  trivial (iter-0) bound : {trivial:.6f}")
        print(f"  C++ Lagrangian bound   : {cpp_bound:.6f}")
        print(f"  published BestOuterBound: {best_outer:.6f}")

        # An outer bound may never move backwards, whichever case we are in.
        assert best_outer >= trivial, (
            f"published bound {best_outer} is worse than the trivial bound "
            f"{trivial}")

        if final_lagrangian:
            # The final solve ran, so C++ is no longer sitting on the iter-0
            # value, and the published bound is whichever of the two won.
            assert cpp_bound != trivial, (
                "expected the final Lagrangian to recompute the bound, but "
                f"C++ still reports the iter-0 value {trivial}")
            assert best_outer == max(trivial, cpp_bound), (
                f"published {best_outer}, expected max({trivial}, {cpp_bound})")
        else:
            # Hub mode skips the final solve, so nothing recomputes the bound.
            assert cpp_bound == trivial, (
                f"final Lagrangian ran without the flag: {cpp_bound} != "
                f"{trivial}")
            assert best_outer == trivial, (
                f"published {best_outer}, expected the trivial bound {trivial}")

        print(f"\nFINAL_LAGRANGIAN_RESULT flag={int(final_lagrangian)} "
              f"trivial={trivial!r} best_outer={best_outer!r}")
        print("PASSED: final-Lagrangian test completed successfully")

    comm.Barrier()
    return True


if __name__ == "__main__":
    flag = "--final-lagrangian" in sys.argv[1:]
    if not test_final_lagrangian(flag):
        sys.exit(1)
