###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unequal-rank integration test for the *strict*-coherence path (DUALS).

Runs farmer with a PH hub and a Lagrangian spoke at unequal rank counts. The
Lagrangian spoke receives DUALS (the PH multipliers W_s) from the hub, which
crosses cylinders and is assembled multi-source. DUALS uses strict coherence:
the dual normalization sum_s p_s W_s = 0 holds only within a single PH
iteration, so a W stitched from mixed iterations would yield an *invalid*
Lagrangian bound (one that need not lower-bound the optimum).

Correctness is pinned two ways, both on fullcomm windows (environment-safe):
  * 4+2 vs 2+4 must produce the same Lagrangian outer bound -- the hub's W
    trajectory is a global reduction independent of the rank split, so the
    reassembled W (and the bound it yields) must match across both splits; and
  * the bound must be a valid lower bound on the extensive-form optimum.

A mixed-iteration (non-strict) assembly would break the second property and/or
make the two splits disagree.

mpiexec -np 6 python -m mpi4py -m pytest mpisppy/tests/test_flexible_rank_duals.py
"""

import unittest

from mpi4py import MPI

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver

comm = MPI.COMM_WORLD

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

# Extensive-form optimum for farmer NUM_SCENS=10 (see test_flexible_rank_cylinders).
EF_OPT = -122146.70


def _build_dicts(num_scens, max_iterations, hub_ratio, spoke_ratio):
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.lagrangian_args()
    cfg.solver_name = solver_name
    cfg.num_scens = num_scens
    cfg.default_rho = 1.0
    cfg.max_iterations = max_iterations
    cfg.rel_gap = 1e-6

    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = farmer.scenario_names_creator(num_scens)
    scenario_creator_kwargs = farmer.kw_creator(cfg)
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
    hub_dict["rank_ratio"] = hub_ratio

    lagrangian_spoke = vanilla.lagrangian_spoke(
        *beans, scenario_creator_kwargs=scenario_creator_kwargs
    )
    lagrangian_spoke["rank_ratio"] = spoke_ratio

    return hub_dict, [lagrangian_spoke]


def _run(num_scens, max_iterations, hub_ratio, spoke_ratio):
    hub_dict, spoke_list = _build_dicts(
        num_scens, max_iterations, hub_ratio, spoke_ratio
    )
    wheel = WheelSpinner(hub_dict, spoke_list)
    wheel.spin()
    if wheel.global_rank == 0:
        payload = wheel.BestOuterBound
    else:
        payload = None
    bound = comm.bcast(payload, root=0)
    # every rank's coherence read-outcome counters (field name -> buckets),
    # allgathered so each rank can assert on the whole picture
    local_counters = {
        field.name: dict(counters)
        for field, counters in wheel.spcomm.coherence_counters.items()
    }
    return bound, comm.allgather(local_counters)


@unittest.skipUnless(comm.size == 6, "needs exactly 6 MPI ranks")
@unittest.skipUnless(solver_available, "no solver is available")
class TestFlexibleRankDuals(unittest.TestCase):

    NUM_SCENS = 10
    MAX_ITERS = 50

    def test_strict_duals_both_directions_agree_and_bound_valid(self):
        # 4-rank hub, 2-rank spoke / 2-rank hub, 4-rank spoke
        ob_42, counters_42 = _run(self.NUM_SCENS, self.MAX_ITERS, 1.0, 0.5)
        ob_24, counters_24 = _run(self.NUM_SCENS, self.MAX_ITERS, 1.0, 2.0)

        # Finite bound (a broken read would give inf or error out).
        self.assertTrue(abs(ob_42) < float("inf"))
        self.assertTrue(abs(ob_24) < float("inf"))

        # The hub's W trajectory is rank-split-independent, so the reassembled
        # W -- and the Lagrangian bound it yields -- must match across splits.
        self.assertLess(
            abs(ob_42 - ob_24), 1e-3 * abs(ob_42),
            msg=f"4+2 Lagrangian bound {ob_42} disagrees with 2+4 {ob_24}",
        )

        # A *valid* Lagrangian bound lower-bounds the (minimizing) optimum.
        # A mixed-iteration W could violate this.
        self.assertLessEqual(ob_42, EF_OPT + 1e-4 * abs(EF_OPT))
        self.assertLessEqual(ob_24, EF_OPT + 1e-4 * abs(EF_OPT))

        # Coherence read-outcome diagnostic. Only reads with >= 2 sources are
        # counted, so presence depends on the split: in 4+2 each 5-scenario
        # spoke rank must straddle the 4-rank hub's <=3-scenario slices
        # (guaranteed multi-source), while in 2+4 the spoke's slices may nest
        # inside the hub's halves and leave every read single-source.
        for all_counters in (counters_42, counters_24):
            for rank_counters in all_counters:
                for counters in rank_counters.values():
                    # the outcome buckets partition the total
                    self.assertGreater(counters["total"], 0)
                    self.assertEqual(
                        counters["total"],
                        counters["new_accepted"] + counters["not_new"]
                        + counters["rejected_incoherent"]
                        + counters["rejected_cross_reader"]
                        + counters["accepted_mixed"],
                    )
                # DUALS is strict: a blended assembly must never be accepted
                if "DUALS" in rank_counters:
                    self.assertEqual(rank_counters["DUALS"]["accepted_mixed"], 0)
        duals_42 = [c["DUALS"] for c in counters_42 if "DUALS" in c]
        self.assertGreater(len(duals_42), 0)
        # the 4+2 bound can only have come from accepted multi-source reads
        self.assertGreater(sum(c["new_accepted"] for c in duals_42), 0)


if __name__ == "__main__":
    unittest.main()
