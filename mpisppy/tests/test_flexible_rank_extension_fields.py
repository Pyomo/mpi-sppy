###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unequal-rank regression test for per-scenario fields registered by an
*extension* rather than a cylinder's class-level receive_fields.

A spoke's own receive_fields get their multi-source overlap map built in
register_receive_fields; an extension that registers a receive field directly
(via register_recv_field) used to get no map, so at unequal ranks its read fell
to the single-source path and tore -- the receive buffer and the peer's field
have different lengths when the rank counts differ, which aborts the run at
Iter0 (np.size(dest) == count assertion, or a NaN write_id read). Building the
overlap map inside register_recv_field fixes every such extension uniformly.

This exercises that path with the relaxed-PH fixer: a PH hub with the
RelaxedPHFixer extension (which reads RELAXED_NONANTS_VALS from a relaxed-PH
spoke) at unequal rank counts. Without the fix this aborts at Iter0 in both
asymmetry directions; with it the run completes. (grad_rho reading BEST_XHAT is
the same path and the same fix.)

mpiexec -np 6 python -m mpi4py -m pytest mpisppy/tests/test_flexible_rank_extension_fields.py
"""

import math
import unittest

from mpi4py import MPI

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver

comm = MPI.COMM_WORLD

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()


def _build_dicts(num_scens, max_iterations, hub_ratio, spoke_ratio):
    # Fresh cfg/dicts per run: WheelSpinner.run mutates the dicts and may only
    # run once, so each run needs its own.
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.relaxed_ph_args()
    cfg.relaxed_ph_fixer_args()
    cfg.solver_name = solver_name
    cfg.num_scens = num_scens
    cfg.default_rho = 1.0
    cfg.max_iterations = max_iterations
    cfg.rel_gap = 1e-6
    cfg.relaxed_ph = True
    cfg.relaxed_ph_fixer = True

    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = farmer.scenario_names_creator(num_scens)
    scenario_creator_kwargs = farmer.kw_creator(cfg)
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
    hub_dict["rank_ratio"] = hub_ratio
    # RelaxedPHFixer reads RELAXED_NONANTS_VALS -- the extension-registered
    # per-scenario field under test.
    vanilla.add_relaxed_ph_fixer(hub_dict, cfg)

    relaxed_ph_spoke = vanilla.relaxed_ph_spoke(
        *beans, scenario_creator_kwargs=scenario_creator_kwargs
    )
    relaxed_ph_spoke["rank_ratio"] = spoke_ratio

    return hub_dict, [relaxed_ph_spoke]


def _run(num_scens, max_iterations, hub_ratio, spoke_ratio):
    hub_dict, spoke_list = _build_dicts(
        num_scens, max_iterations, hub_ratio, spoke_ratio
    )
    wheel = WheelSpinner(hub_dict, spoke_list)
    # Without the fix this raises at Iter0 inside the fixer's multi-source read;
    # reaching the return means the unequal-rank read assembled cleanly.
    wheel.spin()
    if wheel.global_rank == 0:
        payload = (wheel.BestInnerBound, wheel.BestOuterBound)
    else:
        payload = None
    return comm.bcast(payload, root=0)


@unittest.skipUnless(comm.size == 6, "needs exactly 6 MPI ranks")
@unittest.skipUnless(solver_available, "no solver is available")
class TestFlexibleRankExtensionFields(unittest.TestCase):

    NUM_SCENS = 10
    MAX_ITERS = 10

    def test_extension_field_unequal_ranks_both_directions(self):
        # 4-rank hub, 2-rank spoke: each hub rank stitches the relaxed solution
        # from several spoke buffers (multi-segment).
        ib_42, ob_42 = _run(self.NUM_SCENS, self.MAX_ITERS, 1.0, 0.5)
        # 2-rank hub, 4-rank spoke: each hub rank reads a sub-range of one spoke
        # buffer (the other asymmetry direction).
        ib_24, ob_24 = _run(self.NUM_SCENS, self.MAX_ITERS, 1.0, 2.0)

        # The torn single-source read used to put NaN in the buffer (and abort
        # before this point); a completed run with no NaN confirms the
        # multi-source assembly was wired for the extension-registered field.
        for b in (ib_42, ob_42, ib_24, ob_24):
            self.assertFalse(math.isnan(b))


if __name__ == "__main__":
    unittest.main()
