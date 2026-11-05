###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright information.
###############################################################################
"""Unequal-rank MPI integration test for APH ``ASYNC_LATEST`` dual reads.

The four-rank APH hub publishes W pieces independently. Each rank of the
two-rank Lagrangian spoke must assemble its scenarios from multiple hub ranks,
so this exercises the flexible-rank, multi-source path rather than the simpler
equal-rank receive path.

mpiexec -np 6 python -m mpi4py mpisppy/tests/test_flexible_rank_async_duals.py
"""

import unittest

from mpi4py import MPI

import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver
from mpisppy.utils import config

comm = MPI.COMM_WORLD

solver_available, solver_name, _, _ = get_solver()

# Extensive-form optimum for farmer with ten scenarios.
EF_OPT = -122146.70


def _build_dicts():
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.aph_args()
    cfg.lagrangian_args()
    cfg.solver_name = solver_name
    cfg.num_scens = 10
    cfg.default_rho = 1.0
    cfg.max_iterations = 30
    cfg.max_stalled_iters = 100
    cfg.intra_hub_conv_thresh = 0.0
    cfg.rel_gap = 0.0
    cfg.abs_gap = 0.0
    cfg.aph_frac_needed = 0.5
    cfg.aph_dispatch_frac = 1.0

    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = farmer.scenario_names_creator(cfg.num_scens)
    scenario_creator_kwargs = farmer.kw_creator(cfg)
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    hub_dict = vanilla.aph_hub(
        *beans, scenario_creator_kwargs=scenario_creator_kwargs)
    hub_dict["rank_ratio"] = 1.0

    lagrangian_spoke = vanilla.lagrangian_spoke(
        *beans, scenario_creator_kwargs=scenario_creator_kwargs)
    lagrangian_spoke["rank_ratio"] = 0.5
    return hub_dict, [lagrangian_spoke]


@unittest.skipUnless(comm.size == 6, "needs exactly 6 MPI ranks")
@unittest.skipUnless(solver_available, "no solver is available")
class TestFlexibleRankAsyncDuals(unittest.TestCase):

    def test_async_latest_multisource_read_produces_valid_bound(self):
        hub_dict, spoke_list = _build_dicts()
        wheel = WheelSpinner(hub_dict, spoke_list)
        wheel.spin()

        bound = comm.bcast(
            wheel.BestOuterBound if wheel.global_rank == 0 else None, root=0)

        recv_buf = getattr(
            wheel.spcomm, "_nonant_len_receive_buffer", None)
        source_ids = (None if recv_buf is None
                      else recv_buf._source_write_ids)
        all_source_ids = comm.allgather(source_ids)

        # A finite, honest bound proves that the mixed W was projected and
        # solved successfully. For this minimization problem it cannot exceed
        # the extensive-form optimum (allowing ordinary solver tolerance).
        self.assertTrue(abs(bound) < float("inf"))
        self.assertLessEqual(bound, EF_OPT + 1e-4 * abs(EF_OPT))

        # At least one spoke rank accepted an ASYNC_LATEST snapshot assembled
        # from two or more APH publisher ranks. This rules out accidentally
        # exercising only the equal-rank or single-source path.
        multisource_vectors = [
            ids for ids in all_source_ids
            if ids is not None and len(ids) >= 2
        ]
        self.assertGreater(len(multisource_vectors), 0)
        self.assertTrue(all(max(ids.values()) > 0
                            for ids in multisource_vectors))


if __name__ == "__main__":
    unittest.main()
