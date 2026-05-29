###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""
Exercise a CG hub that spans more than one MPI rank.

    mpiexec -np 4 python -m mpi4py test_cg_multirank_hub.py

With one hub plus one spoke at np=4, WheelSpinner gives each cylinder two
ranks, so the CG hub's cylinder_comm has rank 0 and rank 1. This is the first
configuration to put more than one rank under a CG hub, and it covers the
convergence-path shutdown that used to deadlock when per-iteration results
(in particular best_solution_obj_val) were computed on rank 0 but not
broadcast across the hub (issue #729): the hub ranks would then disagree on
the termination verdict, with one rank breaking out of the iteration loop
while another blocked forever at the next master-problem broadcast.
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
assert comm.size == 4, "These tests need four ranks (two-rank CG hub)"

# CG needs a non-persistent master solver; the persistent interface raises
# "Please use set_instance ..." on the CG master.
solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver(persistent_OK=False)


def _create_cfg():
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.ph_args()
    cfg.two_sided_args()
    cfg.cg_args()
    cfg.solver_name = solver_name
    return cfg


class Test_cg_multirank_hub(unittest.TestCase):
    """ A CG hub spanning two ranks must reach convergence without deadlocking. """

    def _create_stuff(self, iters=10):
        self.cfg.num_scens = 6
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

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_two_rank_hub_converges(self):
        self.cfg.default_rho = 1
        self.cfg.xhatshuffle_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()

        list_of_spoke_dict = list()
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
            *beans, scenario_creator_kwargs=scenario_creator_kwargs)
        list_of_spoke_dict.append(xhatshuffle_spoke)

        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        # If the hub ranks disagree on termination (issue #729) this hangs.
        wheel.spin()

        if wheel.strata_rank == 0:  # a CG hub rank
            cg_object = wheel.spcomm.opt
            self.assertIsNotNone(cg_object)
            self.assertIsNotNone(cg_object.conv)


if __name__ == '__main__':
    unittest.main()
