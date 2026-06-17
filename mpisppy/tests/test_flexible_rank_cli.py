###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""The flexible-rank CLI flags (--<spoke>-rank-ratio) are declared with their
spokes and flow through build_spoke_list onto each spoke dict's rank_ratio key,
which WheelSpinner reads. Serial; no MPI."""

import unittest

from mpisppy.utils import config
import mpisppy.tests.examples.farmer as farmer
from mpisppy.generic.spokes import build_spoke_list


def _full_cfg():
    # Declare the same arg set the generic driver does (the subset that
    # build_spoke_list reads), so every cfg flag it touches exists.
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.gapper_args("lagrangian")
    cfg.ph_dual_args()
    cfg.relaxed_ph_args()
    cfg.ph_xfeas_spoke_args()
    cfg.subgradient_args()
    cfg.subgradient_bounder_args()
    cfg.xhatshuffle_args()
    cfg.xhatxbar_args()
    cfg.reduced_costs_args()
    cfg.sep_rho_args()
    cfg.coeff_rho_args()
    cfg.sensi_rho_args()
    cfg.gradient_args()
    cfg.gapper_args()
    return cfg


class TestFlexibleRankCLI(unittest.TestCase):

    def test_rank_ratio_args_default_to_one(self):
        cfg = _full_cfg()
        self.assertEqual(cfg.lagrangian_rank_ratio, 1.0)
        self.assertEqual(cfg.xhatshuffle_rank_ratio, 1.0)
        self.assertEqual(cfg.xhatxbar_rank_ratio, 1.0)
        self.assertEqual(cfg.ph_xfeas_spoke_rank_ratio, 1.0)
        self.assertEqual(cfg.fwph_rank_ratio, 1.0)
        self.assertEqual(cfg.ph_dual_rank_ratio, 1.0)
        self.assertEqual(cfg.relaxed_ph_rank_ratio, 1.0)
        self.assertEqual(cfg.subgradient_rank_ratio, 1.0)

    def test_build_spoke_list_injects_rank_ratio(self):
        cfg = _full_cfg()
        cfg.num_scens = 6
        cfg.lagrangian = True
        cfg.lagrangian_rank_ratio = 0.5
        cfg.xhatshuffle = True
        cfg.xhatshuffle_rank_ratio = 0.25
        cfg.xhatxbar = True
        cfg.xhatxbar_rank_ratio = 2.0
        cfg.ph_xfeas_spoke = True
        cfg.ph_xfeas_spoke_rank_ratio = 4.0
        cfg.fwph = True
        cfg.fwph_rank_ratio = 8.0
        cfg.ph_dual = True
        cfg.ph_dual_rank_ratio = 3.0
        cfg.relaxed_ph = True
        cfg.relaxed_ph_rank_ratio = 5.0
        cfg.subgradient = True
        cfg.subgradient_rank_ratio = 0.125

        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(cfg)
        beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

        spokes = build_spoke_list(
            cfg, beans, scenario_creator_kwargs,
            rho_setter=None, all_nodenames=None,
        )

        # exactly the enabled spokes, each carrying its requested ratio
        ratios = sorted(d["rank_ratio"] for d in spokes)
        self.assertEqual(ratios, sorted([0.5, 0.25, 2.0, 4.0,
                                         8.0, 3.0, 5.0, 0.125]))
        # and every spoke dict got an explicit rank_ratio
        self.assertTrue(all("rank_ratio" in d for d in spokes))

    def test_default_run_leaves_ratios_at_one(self):
        # With no ratios set, enabled spokes default to 1.0 (equal-rank path).
        cfg = _full_cfg()
        cfg.num_scens = 6
        cfg.xhatshuffle = True
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(cfg)
        beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

        spokes = build_spoke_list(
            cfg, beans, scenario_creator_kwargs,
            rho_setter=None, all_nodenames=None,
        )
        self.assertEqual([d["rank_ratio"] for d in spokes], [1.0])


if __name__ == "__main__":
    unittest.main()
