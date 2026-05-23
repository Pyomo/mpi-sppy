###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
mpiexec -np 2 python -m mpi4py test_dualcg_with_cylinders.py

"""

import unittest

from mpi4py import MPI

from mpisppy.utils import config

import mpisppy.tests.examples.farmer as farmer
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver


comm = MPI.COMM_WORLD
assert comm.size == 2, "These tests need two ranks"

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver(persistent_OK=False)


def _create_cfg():
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.cg_args()
    cfg.dualcg_args()
    cfg.solver_name = solver_name
    return cfg


class TestFarmerWithDCGCylinders(unittest.TestCase):
    """Test DCGHub with the farmer model."""

    def setUp(self):
        self.cfg = _create_cfg()

    def _create_stuff(self, iters=5):
        self.cfg.num_scens = 3
        self.cfg.max_iterations = iters
        self.cfg.rel_gap = 0.005
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.dualcg_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        return scenario_creator_kwargs, beans, hub_dict

    def _assert_dcg_ran(self, wheel):
        if wheel.global_rank == 0:
            dcg_object = wheel.spcomm.opt
            self.assertIsNotNone(dcg_object)
            self.assertIsNotNone(dcg_object.conv)
            self.assertIsNotNone(dcg_object.best_bound_obj_val)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_dualcg_hub_runs(self):
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()
        wheel = WheelSpinner(hub_dict, [])
        wheel.spin()

        self._assert_dcg_ran(wheel)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_dualcg_xhatshuffler(self):
        self.cfg.xhatshuffle_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        wheel = WheelSpinner(hub_dict, [xhatshuffle_spoke])
        wheel.spin()

        self._assert_dcg_ran(wheel)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_dualcg_fwph(self):
        self.cfg.default_rho = 1
        self.cfg.fwph_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff(iters=15)
        fwph_spoke = vanilla.fwph_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        wheel = WheelSpinner(hub_dict, [fwph_spoke])
        wheel.spin()

        self._assert_dcg_ran(wheel)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_dualcg_ph_xfeas_spoke(self):
        self.cfg.default_rho = 1
        self.cfg.ph_xfeas_spoke_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()
        ph_xfeas_spoke = vanilla.ph_xfeas_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        wheel = WheelSpinner(hub_dict, [ph_xfeas_spoke])
        wheel.spin()

        self._assert_dcg_ran(wheel)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_dualcg_subgradient_bounder(self):
        self.cfg.default_rho = 1
        self.cfg.subgradient_bounder_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()
        subgradient_spoke = vanilla.subgradient_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        wheel = WheelSpinner(hub_dict, [subgradient_spoke])
        wheel.spin()

        self._assert_dcg_ran(wheel)


if __name__ == "__main__":
    unittest.main()
