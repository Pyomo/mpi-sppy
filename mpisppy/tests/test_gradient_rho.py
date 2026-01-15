###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Author: Ulysse Naepels and D.L. Woodruff
"""
IMPORTANT:
  Unless we run to convergence, the solver, and even solver
version matter a lot, so we often just do smoke tests.
"""

import unittest
from mpisppy.utils import config

from mpisppy.utils import cfg_vanilla as vanilla, scenario_names_creator
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver
from mpisppy.extensions.grad_rho import GradRho

__version__ = 0.3

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver()

def _create_cfg():
    cfg = config.Config()
    cfg.gradient_args()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.dynamic_rho_args()
    cfg.solver_name = solver_name
    cfg.default_rho = 1
    cfg.grad_order_stat = 0.5
    return cfg

#*****************************************************************************

class Test_gradient_farmer(unittest.TestCase):
    """ Test the gradient code using farmer."""

    def _create_ph_farmer(self):
        # This causes iter zero to execute, which is overkill to just create ph for farmer
        self.cfg.num_scens = 3
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        hub_dict['opt_kwargs']['options']['cfg'] = self.cfg
        list_of_spoke_dict = list()
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.strata_rank == 0:
            ph_object = wheel.spcomm.opt
            return ph_object

    def setUp(self):
        self.cfg = _create_cfg()
        self.cfg.max_iterations = 0
        self.ph_object = self._create_ph_farmer()
        self.ph_object.options["grad_rho_options"] = {"cfg": self.cfg}
        

    def test_grad_rho_init(self):
        self.grad_object = GradRho(self.ph_object)


if __name__ == '__main__':
    unittest.main()
