###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
mpiexec -np 2 python -m mpi4py test_cg_with_cylinders.py

"""

import unittest
from mpisppy.utils import config

import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver

__version__ = 0.1

from mpi4py import MPI
comm = MPI.COMM_WORLD
assert comm.size == 2, "These tests need two ranks"

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver(persistent_OK=False)

def _create_cfg():
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.cg_args()
    cfg.solver_name = solver_name


    return cfg

#*****************************************************************************


class Test_farmer_with_cylinders(unittest.TestCase):
    """ Test the find rho code using farmer."""

    def _create_stuff(self, iters=5):
        # assumes setup has been called; very specific...
        self.cfg.num_scens = 3
        self.cfg.max_iterations = iters
        self.cfg.rel_gap = 0.005
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.cg_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

        return scenario_creator_kwargs, beans, hub_dict


    def setUp(self):
        self.cfg = _create_cfg()

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_cg_hub_runs(self):
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff(iters=3)

        list_of_spoke_dict = []
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()

        if wheel.global_rank == 0:
            cg_object = wheel.spcomm.opt
            self.assertIsNotNone(cg_object)
            self.assertIsNotNone(cg_object.conv)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_xhatshuffler(self):

        self.cfg.xhatshuffle_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()

        list_of_spoke_dict = list()
        # xhat shuffle bound spoke
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                                   scenario_creator_kwargs=scenario_creator_kwargs)
        list_of_spoke_dict.append(xhatshuffle_spoke)
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.global_rank == 1:
            # xhatshuffle evaluates single-scenario plans; its bound depends
            # on which scenario it last evaluated before the CG hub terminated.
            self.assertAlmostEqual(
                wheel.spcomm.bound,
                -109499.5160897,
                delta=10000,
            )

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fwph(self):

        self.cfg.default_rho = 1
        self.cfg.fwph_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff(iters=15)

        list_of_spoke_dict = list()
        fwph_spoke = vanilla.fwph_spoke(*beans,
                                                   scenario_creator_kwargs=scenario_creator_kwargs)
        list_of_spoke_dict.append(fwph_spoke)
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.global_rank == 1:
            self.assertAlmostEqual(
                wheel.spcomm.bound,
                -109499.5160897,
                delta=2000,
            )

    def test_ph_spoke(self):

        self.cfg.default_rho = 1
        self.cfg.ph_spoke_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()

        list_of_spoke_dict = list()
        ph_spoke = vanilla.ph_spoke(*beans,
                                                   scenario_creator_kwargs=scenario_creator_kwargs)
        list_of_spoke_dict.append(ph_spoke)
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.global_rank == 1:
            self.assertAlmostEqual(
                wheel.spcomm.final_bound,
                -109499.5160897,
                delta=6000,
            )

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_subgradient_bounder(self):
        self.cfg.default_rho = 1
        self.cfg.subgradient_bounder_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()

        list_of_spoke_dict = list()
        subgradient_spoke = vanilla.subgradient_spoke(*beans,
                                                   scenario_creator_kwargs=scenario_creator_kwargs,)

        list_of_spoke_dict.append(subgradient_spoke)
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        print(wheel)
        wheel.spin()
        if wheel.global_rank == 1:
            self.assertAlmostEqual(
                wheel.spcomm.bound,
                -109499.5160897,
                delta=15000,
            )



if __name__ == '__main__':
    unittest.main()
