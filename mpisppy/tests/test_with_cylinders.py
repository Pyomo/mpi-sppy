###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Author:  D.L. Woodruff 2023
"""
mpiexec -np 2 python -m mpi4py test_with_cylinders.py

"""

import unittest
import pyomo.environ as pyo
from mpisppy.utils import config

import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.sputils as sputils
import mpisppy.tests.examples.farmer as farmer
import mpisppy.tests.examples.hydro.hydro as hydro
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver

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

    def _create_stuff(self, iters=5, sense=pyo.minimize):
        # assumes setup has been called; very specific...
        self.cfg.num_scens = 3
        self.cfg.max_iterations = iters
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        # farmer negates its cost expression for sense=pyo.maximize
        scenario_creator_kwargs["sense"] = sense
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
            self.assertIn('post_solve', xhat_object._TestExtension_who_is_called)


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_xhatshuffle_extended(self):
        print("begin xhatshuffle_extended with test extension")
        from mpisppy.extensions.test_extension import TestExtension

        self.cfg.xhatxbar_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff()

        list_of_spoke_dict = list()
        # xhat shuffle bound spoke
        ext = TestExtension
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                scenario_creator_kwargs=scenario_creator_kwargs,
                                ph_extensions=ext)
        list_of_spoke_dict.append(xhatshuffle_spoke)

        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.global_rank == 1:
            xhat_object = wheel.spcomm.opt
            print(f"{xhat_object._TestExtension_who_is_called =}")
            self.assertIn('post_solve', xhat_object._TestExtension_who_is_called)


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_xhatshuffle_coverage(self):
        print("begin xhatshuffle_coverage")
        from helper_extension import TestHelperExtension

        self.cfg.xhatxbar_args()
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff(iters=2)

        list_of_spoke_dict = list()
        # xhat shuffle bound spoke
        ext = TestHelperExtension
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                scenario_creator_kwargs=scenario_creator_kwargs,
                                ph_extensions=ext)
        list_of_spoke_dict.append(xhatshuffle_spoke)

        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.global_rank == 1:
            xhat_object = wheel.spcomm.opt
            for idx,v in xhat_object._TestHelperExtension_checked.items():
                # print("xhat_object = ", xhat_object)
                # print("idx = ", idx, "    v = ", v)
                self.assertNotEqual(v[0], v[1], f"xhatshuffle does not seem to change things for {idx} (and maybe others)")


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_lagrangian(self):
        print("Start lagrangian")
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
            #print(f"{wheel.spcomm.bound= }")
            self.assertAlmostEqual(wheel.spcomm.bound, -109499.5160897, 1)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_lagrangian_max(self):
        # Same as test_lagrangian but maximizing. farmer negates its cost for
        # sense=pyo.maximize, so the Lagrangian (outer) bound is the negation
        # of the minimize bound. For max the outer bound is an UPPER bound, so
        # it must lie above the optimal expected profit (108390).
        print("Start lagrangian_max")
        scenario_creator_kwargs, beans, hub_dict = self._create_stuff(sense=pyo.maximize)

        self.cfg.lagrangian_args()
        list_of_spoke_dict = list()
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                                    scenario_creator_kwargs=scenario_creator_kwargs,)
        list_of_spoke_dict.append(lagrangian_spoke)

        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.global_rank == 1:
            self.assertAlmostEqual(wheel.spcomm.bound, 109499.5160897, 1)
            self.assertGreaterEqual(wheel.spcomm.bound, 108390.0)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_wheel_sign_flip_equivalence(self):
        # Full-featured sign-flip equivalence: run the SAME PH-hub +
        # xhatshuffle-spoke wheel once minimizing and once maximizing (farmer
        # negates its cost expression for sense=pyo.maximize). The two runs are
        # exact mirror images, so every reported bound must be the negation of
        # its counterpart, and each must bracket the optimum (+/-108390) on the
        # correct, sense-dependent side: for max the incumbent (inner) bound is
        # below the optimum and the outer bound above it; for min the reverse.
        FARMER_OPT = 108390.0

        def run(sense):
            self.cfg = _create_cfg()
            self.cfg.xhatshuffle_args()
            scenario_creator_kwargs, beans, hub_dict = self._create_stuff(
                iters=10, sense=sense)
            xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
                *beans, scenario_creator_kwargs=scenario_creator_kwargs)
            wheel = WheelSpinner(hub_dict, [xhatshuffle_spoke])
            wheel.spin()
            return wheel

        wmin = run(pyo.minimize)
        wmax = run(pyo.maximize)
        if wmin.global_rank == 0:
            for w in (wmin, wmax):
                self.assertIsNotNone(w.BestInnerBound)
                self.assertIsNotNone(w.BestOuterBound)
            # exact mirror equivalence of the full-run results
            self.assertAlmostEqual(wmax.BestInnerBound, -wmin.BestInnerBound, 1)
            self.assertAlmostEqual(wmax.BestOuterBound, -wmin.BestOuterBound, 1)
            # max: incumbent (inner) <= optimum <= outer bound
            self.assertLessEqual(wmax.BestInnerBound, FARMER_OPT + 1.0)
            self.assertGreaterEqual(wmax.BestOuterBound, FARMER_OPT - 1.0)
            # min: outer bound <= optimum <= incumbent (inner)
            self.assertGreaterEqual(wmin.BestInnerBound, -FARMER_OPT - 1.0)
            self.assertLessEqual(wmin.BestOuterBound, -FARMER_OPT + 1.0)


#*****************************************************************************


class Test_hydro_with_cylinders(unittest.TestCase):
    """Test multistage (hydro) with stage2_ef_solver_name."""

    def setUp(self):
        # Use branching_factors=[2,2] so 2 second-stage nodes works with 2 ranks
        self.branching_factors = [2, 2]
        self.num_scens = self.branching_factors[0] * self.branching_factors[1]
        self.all_scenario_names = [f"Scen{i+1}" for i in range(self.num_scens)]
        self.all_nodenames = sputils.create_nodenames_from_branching_factors(
            self.branching_factors)

        self.cfg = config.Config()
        self.cfg.popular_args()
        self.cfg.two_sided_args()
        self.cfg.ph_args()
        self.cfg.xhatshuffle_args()
        self.cfg.lagrangian_args()
        self.cfg.add_branching_factors()
        self.cfg.solver_name = solver_name
        self.cfg.default_rho = 1
        self.cfg.max_iterations = 5
        self.cfg.branching_factors = self.branching_factors

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_xhatshuffle_stage2ef(self):
        """Test xhatshuffle with stage2_ef_solver_name on a multistage problem."""
        scenario_creator_kwargs = {"branching_factors": self.branching_factors}
        beans = (self.cfg, hydro.scenario_creator,
                 hydro.scenario_denouement, self.all_scenario_names)

        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  all_nodenames=self.all_nodenames)

        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
            all_nodenames=self.all_nodenames)
        xhatshuffle_spoke["opt_kwargs"]["options"]["stage2_ef_solver_name"] = solver_name
        xhatshuffle_spoke["opt_kwargs"]["options"]["branching_factors"] = self.branching_factors

        list_of_spoke_dict = [xhatshuffle_spoke]

        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()

        if wheel.global_rank == 0:
            self.assertTrue(wheel.BestInnerBound is not None)
            self.assertTrue(wheel.BestOuterBound is not None)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_xhatshuffle_stage2ef_asymmetric_bf(self):
        """Regression test: asymmetric branching factors must use bf[0] as the
        second-stage node count, not bf[1].

        With branching_factors=[2,3] and 2 ranks, the count of second-stage
        nodes is 2 (=bf[0]); the buggy code used bf[1]=3 and tripped the
        `stage2cnt % rankcnt == 0` assertion (3 % 2 != 0)."""
        branching_factors = [2, 3]
        num_scens = branching_factors[0] * branching_factors[1]
        all_scenario_names = [f"Scen{i+1}" for i in range(num_scens)]
        all_nodenames = sputils.create_nodenames_from_branching_factors(
            branching_factors)
        self.cfg.branching_factors = branching_factors

        scenario_creator_kwargs = {"branching_factors": branching_factors}
        beans = (self.cfg, hydro.scenario_creator,
                 hydro.scenario_denouement, all_scenario_names)

        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  all_nodenames=all_nodenames)

        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
            *beans,
            scenario_creator_kwargs=scenario_creator_kwargs,
            all_nodenames=all_nodenames)
        xhatshuffle_spoke["opt_kwargs"]["options"]["stage2_ef_solver_name"] = solver_name
        xhatshuffle_spoke["opt_kwargs"]["options"]["branching_factors"] = branching_factors

        wheel = WheelSpinner(hub_dict, [xhatshuffle_spoke])
        wheel.spin()

        if wheel.global_rank == 0:
            self.assertTrue(wheel.BestInnerBound is not None)
            self.assertTrue(wheel.BestOuterBound is not None)


if __name__ == '__main__':
    unittest.main()
