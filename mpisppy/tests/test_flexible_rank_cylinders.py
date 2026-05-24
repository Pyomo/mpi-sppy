###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""End-to-end integration test for the unequal-rank communication layer.

Runs farmer with a PH hub and an xhatshuffle spoke at *unequal* rank counts.
This is the clean Phase-2 case: of the local-sized per-scenario fields, only
NONANTS_VALS crosses cylinders (hub -> spoke, multi-source assembly). The PH
hub receives only scalar bounds back from the xhatshuffle spoke, so no DUALS
(strict coherence) or xhat circular-buffer assembly is involved -- those are
later phases.

The xhatshuffle spoke reaches a good incumbent only if it reconstructs the
hub's per-scenario nonant values correctly through the multi-source overlap
assembly; a misrouted or torn read would feed it garbage candidates. We pin
correctness two ways, both using fullcomm windows (so neither depends on the
shared-memory-on-a-split-subcommunicator path, which some MPI builds reject):

  * 4+2 vs 2+4: the same problem solved with the hub larger than the spoke and
    vice versa. These exercise opposite asymmetry directions -- a spoke rank
    stitching several hub buffers, and a spoke rank reading a sub-range of one
    -- yet must reach the same inner bound. (No magic constant.)
  * both must reach the extensive-form optimum (ground truth) within a loose
    tolerance.

mpiexec -np 6 python -m mpi4py -m pytest mpisppy/tests/test_flexible_rank_cylinders.py
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

# Extensive-form optimum for farmer with NUM_SCENS=10 (a deterministic LP
# optimum, solver-independent). Regenerate with mpisppy.opt.ef.ExtensiveForm on
# one rank if the farmer example ever changes.
EF_OPT = -122146.70


def _build_dicts(num_scens, max_iterations, hub_ratio, spoke_ratio):
    # Fresh cfg/dicts per run: WheelSpinner.run mutates the dicts (key
    # renaming) and may only run once, so each run needs its own.
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.xhatshuffle_args()
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

    xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
        *beans, scenario_creator_kwargs=scenario_creator_kwargs
    )
    xhatshuffle_spoke["rank_ratio"] = spoke_ratio

    return hub_dict, [xhatshuffle_spoke]


def _run(num_scens, max_iterations, hub_ratio, spoke_ratio):
    hub_dict, spoke_list = _build_dicts(
        num_scens, max_iterations, hub_ratio, spoke_ratio
    )
    wheel = WheelSpinner(hub_dict, spoke_list)
    wheel.spin()
    # Bounds are populated on the hub ranks (strata_rank 0); broadcast the
    # lead hub rank's values so all ranks can assert in lockstep.
    if wheel.global_rank == 0:
        payload = (wheel.BestInnerBound, wheel.BestOuterBound)
    else:
        payload = None
    return comm.bcast(payload, root=0)


@unittest.skipUnless(comm.size == 6, "needs exactly 6 MPI ranks")
@unittest.skipUnless(solver_available, "no solver is available")
class TestFlexibleRankCylinders(unittest.TestCase):

    NUM_SCENS = 10
    MAX_ITERS = 50

    def test_unequal_rank_both_directions_agree_and_are_optimal(self):
        # 4-rank hub, 2-rank spoke: each spoke rank stitches NONANTS_VALS from
        # several hub buffers (multi-segment assembly).
        ib_42, ob_42 = _run(self.NUM_SCENS, self.MAX_ITERS, 1.0, 0.5)
        # 2-rank hub, 4-rank spoke: each spoke rank reads a sub-range of one
        # hub buffer (the other asymmetry direction).
        ib_24, ob_24 = _run(self.NUM_SCENS, self.MAX_ITERS, 1.0, 2.0)

        # Finite incumbents (a torn/garbage read would give inf or error out).
        self.assertTrue(abs(ib_42) < float("inf"))
        self.assertTrue(abs(ib_24) < float("inf"))

        # Both asymmetry directions reach the same inner bound.
        self.assertLess(
            abs(ib_42 - ib_24), 1e-3 * abs(ib_42),
            msg=f"4+2 inner bound {ib_42} disagrees with 2+4 {ib_24}",
        )

        # ...and that bound is the extensive-form optimum (within 1%).
        for ib in (ib_42, ib_24):
            self.assertLess(
                abs(ib - EF_OPT), 1e-2 * abs(EF_OPT),
                msg=f"inner bound {ib} is not near the EF optimum {EF_OPT}",
            )

        # Valid bracketing for the (minimizing) farmer objective.
        self.assertLessEqual(ob_42, ib_42 + 1e-6)
        self.assertLessEqual(ob_24, ib_24 + 1e-6)


if __name__ == "__main__":
    unittest.main()
