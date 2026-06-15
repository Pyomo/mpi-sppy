###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""End-to-end integration test for the unequal-rank XFEAS field (Phase 4a).

XFEAS's only consumer is the CG hub, and a CG hub spanning more than one rank
deadlocked at convergence until the best_solution_obj_val broadcast fix landed.
With that fix in place we can finally exercise the XFEAS multi-source path at
unequal ranks.

Runs farmer with a CG hub and a ph_xfeas (xhat-feasible) spoke at *unequal* rank
counts. The ph_xfeas spoke sends XFEAS -- a per-scenario [first-stage nonants,
obj_val] block of *distinct* candidate iterates (Category-1, relaxed coherence,
no cross-scenario NAC) -- which the multi-rank CG hub assembles across the
spoke's ranks and tries as incumbents. A torn read or misrouted segment would feed the hub garbage
candidates: the inner bound would stop tracking the extensive-form optimum or the
two asymmetry directions would disagree.

We pin correctness with the same two-directional / ground-truth pattern as
test_flexible_rank_xhat, fullcomm windows in both directions:

  * 4+2 vs 2+4: hub larger than spoke and vice versa (opposite asymmetry, so a
    hub rank stitches several spoke buffers in one direction and reads a
    sub-range of one in the other);
  * both must bracket and approach the extensive-form optimum (ground truth).

CG needs a non-persistent master solver (the persistent interface raises
"Please use set_instance ..." on the CG master).

mpiexec -np 6 python -m mpi4py -m pytest mpisppy/tests/test_flexible_rank_xfeas.py
"""

import unittest

from mpi4py import MPI

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver

comm = MPI.COMM_WORLD

# CG needs a non-persistent master solver.
solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver(persistent_OK=False)

# Extensive-form optimum for farmer with NUM_SCENS=10 (a deterministic LP
# optimum, solver-independent); see test_flexible_rank_cylinders.
EF_OPT = -122146.70


def _build_dicts(num_scens, max_iterations, hub_ratio, spoke_ratio):
    # Fresh cfg/dicts per run: WheelSpinner.run mutates the dicts and may only
    # run once, so each run needs its own.
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.cg_args()
    cfg.ph_xfeas_spoke_args()
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

    hub_dict = vanilla.cg_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
    hub_dict["rank_ratio"] = hub_ratio

    ph_xfeas_spoke = vanilla.ph_xfeas_spoke(
        *beans, scenario_creator_kwargs=scenario_creator_kwargs
    )
    ph_xfeas_spoke["rank_ratio"] = spoke_ratio

    return hub_dict, [ph_xfeas_spoke]


def _run(num_scens, max_iterations, hub_ratio, spoke_ratio):
    hub_dict, spoke_list = _build_dicts(
        num_scens, max_iterations, hub_ratio, spoke_ratio
    )
    wheel = WheelSpinner(hub_dict, spoke_list)
    # If the multi-rank CG hub ranks disagreed on termination this would hang.
    wheel.spin()
    # Bounds are populated on the hub's base rank (global rank 0); broadcast so
    # every rank can assert in lockstep.
    if wheel.global_rank == 0:
        payload = (wheel.BestInnerBound, wheel.BestOuterBound)
    else:
        payload = None
    return comm.bcast(payload, root=0)


@unittest.skipUnless(comm.size == 6, "needs exactly 6 MPI ranks")
@unittest.skipUnless(solver_available, "no solver is available")
class TestFlexibleRankXFeas(unittest.TestCase):

    NUM_SCENS = 10
    MAX_ITERS = 25

    def test_cg_hub_unequal_ranks_both_directions(self):
        # 4-rank hub, 2-rank spoke: each hub rank stitches the spoke's XFEAS
        # block from several spoke buffers (multi-segment).
        ib_42, ob_42 = _run(self.NUM_SCENS, self.MAX_ITERS, 1.0, 0.5)
        # 2-rank hub, 4-rank spoke: each hub rank reads a sub-range of one
        # spoke buffer (the other asymmetry direction).
        ib_24, ob_24 = _run(self.NUM_SCENS, self.MAX_ITERS, 1.0, 2.0)

        # Finite bounds (a torn/garbage read would give inf or error out).
        for b in (ib_42, ob_42, ib_24, ob_24):
            self.assertTrue(abs(b) < float("inf"))

        # CG's outer bound is its master LP bound; the two asymmetry directions
        # must agree closely.
        self.assertLess(
            abs(ob_42 - ob_24), 1e-2 * abs(ob_42),
            msg=f"4+2 outer bound {ob_42} disagrees with 2+4 {ob_24}",
        )

        for ib, ob in ((ib_42, ob_42), (ib_24, ob_24)):
            # Minimizing farmer: outer bound is a lower bound (<= opt), inner
            # bound (incumbent from the assembled XFEAS candidates) is an upper
            # bound (>= opt). The outer bound is CG's deterministic master LP
            # bound, checked tightly (1%); the incumbent rides on the XFEAS
            # candidate the hub happens to accept, so it gets a looser 2%
            # near-optimal check.
            self.assertLessEqual(ob, EF_OPT + 1e-2 * abs(EF_OPT))
            self.assertGreaterEqual(ib, EF_OPT - 1e-2 * abs(EF_OPT))
            self.assertLess(
                abs(ob - EF_OPT), 1e-2 * abs(EF_OPT),
                msg=f"outer bound {ob} is not near the EF optimum {EF_OPT}",
            )
            self.assertLess(
                abs(ib - EF_OPT), 2e-2 * abs(EF_OPT),
                msg=f"inner bound {ib} is not near the EF optimum {EF_OPT}",
            )
            # Valid bracketing. At a rel_gap-converged optimum the master LP
            # bound and the incumbent meet and may cross by a hair (LP solver
            # tolerance), so allow a crossing on the order of the convergence
            # gap rather than an absolute 1e-6 that is unrealistic at this
            # objective magnitude (~1.2e5).
            self.assertLessEqual(ob, ib + 1e-6 * abs(EF_OPT))


if __name__ == "__main__":
    unittest.main()
