###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""End-to-end integration test for the unequal-rank xhat fields on a
*multistage* problem (Phase 4b).

Phase 4a handled the two-stage xhat fields, where the entire NAC-redundant
nonant block is one root node shared by every scenario. Multistage is the
general case: non-anticipativity is a per-*node* property -- a node's nonants are
shared only by the scenarios routed through it, while scenarios in different
subtrees hold different later-stage decisions.

The multi-source assembler does not need to know any of that. Each scenario's
block is its whole root->leaf nonant path plus obj_val, routed wholesale from the
rank that holds it (the layout is index-based and stage-agnostic). The xhat ring
is read with *strict* coherence, so an accepted read has every source at one
write_id -- one coherent incumbent in which every scenario through a node already
carries that node's values identically. Faithful routing therefore yields a
per-node NAC-consistent assembly with no post-assembly fix-up, and -- crucially --
without collapsing the distinct subtrees onto one another. This test confirms
that holds inside a real run.

Setup: aircond with branching factors [4, 4] -- a 3-stage tree, 4 second-stage
nodes (subtrees) of 4 scenarios each (16 scenarios) -- with an FWPH hub (which
assembles the xhatshuffle spoke's BEST_XHAT / RECENT_XHATS) and an xhatshuffle
inner-bound spoke, at *unequal* rank counts. The subtree count (4) and per-node
child count (4) are chosen so both rank counts the test uses (2 and 4) give
subtree-aligned scenario splits and satisfy the stage2ef divisibility rule, and
so that a 2-rank cylinder holds *two whole subtrees per rank* -- the case where a
hub rank assembles blocks spanning more than one second-stage node, which a
naive whole-block collapse would get wrong.

A misrouted segment or wrong version interleaving would feed FWPH inconsistent
columns -- its outer bound would stop tracking the extensive-form optimum or the
two asymmetry directions would disagree. We pin correctness with the same
two-directional / ground-truth pattern as test_flexible_rank_xhat, using
fullcomm windows in both directions:

  * 4+2 (hub larger): each hub rank holds one subtree; spoke ranks straddle.
  * 2+4 (spoke larger): each hub rank holds two whole subtrees, so it assembles
    blocks spanning the root plus two distinct stage-2 nodes -- the
    multistage-specific layout.

mpiexec -np 6 python -m mpi4py -m pytest mpisppy/tests/test_flexible_rank_xhat_multistage.py
"""

import unittest

import numpy as np
from mpi4py import MPI

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.sputils as sputils
import mpisppy.tests.examples.aircond as aircond
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver

comm = MPI.COMM_WORLD

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()
# FWPH's master and the xhatshuffle stage-2 EF need a non-persistent solver.
_, np_solver_name, _, _ = get_solver(persistent_OK=False)

BRANCHING_FACTORS = [4, 4]
# Extensive-form optimum for aircond with these branching factors and the
# fixture's default parameters (a deterministic LP optimum, solver-independent).
EF_OPT = 419.51919699962326


def _build_dicts(max_iterations, hub_ratio, spoke_ratio):
    # Fresh cfg/dicts per run: WheelSpinner.run mutates the dicts and may only
    # run once, so each run needs its own.
    cfg = config.Config()
    cfg.multistage()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.fwph_args()
    cfg.xhatshuffle_args()
    aircond.inparser_adder(cfg)
    cfg.solver_name = np_solver_name
    cfg.default_rho = 1.0
    cfg.max_iterations = max_iterations
    cfg.rel_gap = 1e-6
    cfg.branching_factors = BRANCHING_FACTORS
    cfg.max_solver_threads = 2

    num_scens = int(np.prod(BRANCHING_FACTORS))
    all_scenario_names = aircond.scenario_names_creator(num_scens)
    all_nodenames = sputils.create_nodenames_from_branching_factors(BRANCHING_FACTORS)
    scenario_creator_kwargs = aircond.kw_creator(cfg)
    beans = (cfg, aircond.scenario_creator, aircond.scenario_denouement,
             all_scenario_names)

    hub_dict = vanilla.fwph_hub(
        *beans, scenario_creator_kwargs=scenario_creator_kwargs,
        all_nodenames=all_nodenames)
    hub_dict["rank_ratio"] = hub_ratio

    xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
        *beans, scenario_creator_kwargs=scenario_creator_kwargs,
        all_nodenames=all_nodenames)
    xhatshuffle_spoke["opt_kwargs"]["options"]["stage2_ef_solver_name"] = np_solver_name
    xhatshuffle_spoke["opt_kwargs"]["options"]["branching_factors"] = BRANCHING_FACTORS
    xhatshuffle_spoke["rank_ratio"] = spoke_ratio

    return hub_dict, [xhatshuffle_spoke]


def _run(max_iterations, hub_ratio, spoke_ratio):
    hub_dict, spoke_list = _build_dicts(max_iterations, hub_ratio, spoke_ratio)
    wheel = WheelSpinner(hub_dict, spoke_list)
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
class TestFlexibleRankXhatMultistage(unittest.TestCase):

    MAX_ITERS = 20

    def test_fwph_hub_unequal_ranks_both_directions(self):
        # 4-rank hub, 2-rank spoke: each hub rank holds one subtree; spoke ranks
        # straddle and the hub stitches BEST_XHAT/RECENT_XHATS from several.
        ib_42, ob_42 = _run(self.MAX_ITERS, 1.0, 0.5)
        # 2-rank hub, 4-rank spoke: each hub rank holds two whole subtrees, so it
        # assembles blocks spanning the root plus two distinct stage-2 nodes.
        ib_24, ob_24 = _run(self.MAX_ITERS, 1.0, 2.0)

        # Finite bounds (a torn/garbage read would give inf or error out).
        for b in (ib_42, ob_42, ib_24, ob_24):
            self.assertTrue(abs(b) < float("inf"))

        # FWPH's outer bound is its master LP bound, fed by the assembled xhat
        # columns; the two asymmetry directions must agree closely.
        self.assertLess(
            abs(ob_42 - ob_24), 1e-2 * abs(ob_42),
            msg=f"4+2 outer bound {ob_42} disagrees with 2+4 {ob_24}",
        )

        for ib, ob in ((ib_42, ob_42), (ib_24, ob_24)):
            # Minimizing aircond: outer bound is a lower bound (<= opt), inner
            # bound (incumbent) is an upper bound (>= opt). The outer bound is
            # FWPH's deterministic master LP bound, checked tightly (1%); the
            # incumbent comes from the (order-sensitive) xhatshuffle spoke, so
            # it gets a looser 2% near-optimal check.
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
            # Valid bracketing.
            self.assertLessEqual(ob, ib + 1e-6)


if __name__ == "__main__":
    unittest.main()
