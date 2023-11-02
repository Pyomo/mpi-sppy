# This software is distributed under the 3-clause BSD License.
# Author:  D.L. Woodruff 2023
"""
mpiexec -np 2 python -m mpi4py test_with_cylinders.py

"""

import os
import unittest
from mpisppy.utils import config

import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.sputils as sputils
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver,round_pos_sig

__version__ = 0.1

from mpi4py import MPI
comm = MPI.COMM_WORLD
assert comm.size == 2, "These tests need two ranks"

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver()

def _create_cfg():
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.solver_name = solver_name
    cfg.default_rho = 1
    return cfg

#*****************************************************************************

        
class Test_farmer_with_cylinders(unittest.TestCase):
    """ Test the find rho code using farmer."""

    def _create_stuff(self):
        # assumes setup has been called; very specific...
        self.cfg.num_scens = 3
        self.cfg.max_iterations = 3
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

        return scenario_creator_kwargs, beans, hub_dict


    def setUp(self):
        self.cfg = _create_cfg()

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_xhatxbar_extended(self):
        from mpisppy.extensions.test_extension import TestExtension
        
        self.cfg.xhatxbar_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()

        list_of_spoke_dict = list()
        # xhat shuffle bound spoke
        xhatxbar_spoke = vanilla.xhatxbar_spoke(*beans,
                                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                                   ph_extensions=TestExtension)
        list_of_spoke_dict.append(xhatxbar_spoke)

        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.global_rank == 1:
            xhat_object = wheel.spcomm.opt
            print(f"{xhat_object._TestExtension_who_is_called =}")
            self.assertIn('post_solve', xhat_object._TestExtension_who_is_called)


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_lagrangian(self):
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()

        self.cfg.lagrangian_args()
        list_of_spoke_dict = list()
        # xhat shuffle bound spoke
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                                    scenario_creator_kwargs=scenario_creator_kwargs,)
        list_of_spoke_dict.append(lagrangian_spoke)

        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.global_rank == 1:
            lagrangian_object = wheel.spcomm.opt
            # TBD: check something
            print(f"{lagrangian_object.extensions= }")


if __name__ == '__main__':
    unittest.main()
