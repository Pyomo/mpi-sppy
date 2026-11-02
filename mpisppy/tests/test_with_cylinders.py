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

import time
import unittest
from unittest.mock import patch

import pyomo.environ as pyo

import mpisppy.tests.examples.farmer as farmer
import mpisppy.tests.examples.hydro.hydro as hydro
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.cvar as cvar
import mpisppy.utils.sputils as sputils
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver
from mpisppy.utils import config

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

    def _create_stuff(self, iters=5, sense=pyo.minimize, num_scens=3):
        # assumes setup has been called; very specific...
        self.cfg.num_scens = num_scens
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
        # The assertion below compares the value recorded on iteration 0 with
        # the one recorded on iteration 1, so the hub has to actually run both
        # iterations.  cfg.rel_gap defaults to 0.05, and the hub stops as soon
        # as the inter-cylinder gap falls under it -- if the xhatshuffle spoke
        # lands an incumbent before the iteration-1 convergence check, PH
        # terminates after one iteration and only one value is recorded.  That
        # is a race (it depends on how fast the spoke is relative to the hub),
        # and it fires on some MPI/host combinations and not others.  Pin the
        # gap to 0 so the two iterations always happen.
        self.cfg.rel_gap = 0.0
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
        # negates its cost expression for sense=pyo.maximize). The hub's PH
        # iterates are exact mirror images, so its outer bound must be the
        # negation of its counterpart. Each bound must bracket the optimum
        # (+/-108390) on the correct, sense-dependent side: for max the
        # incumbent (inner) bound is below the optimum and the outer bound
        # above it; for min the reverse.
        # The inner bounds are not compared with each other: xhatshuffle runs
        # asynchronously, so which incumbent each run ends on depends on
        # timing (issue #936).
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
            # exact mirror equivalence of the hub's bound
            self.assertAlmostEqual(wmax.BestOuterBound, -wmin.BestOuterBound, 1)
            # max: incumbent (inner) <= optimum <= outer bound
            self.assertLessEqual(wmax.BestInnerBound, FARMER_OPT + 1.0)
            self.assertGreaterEqual(wmax.BestOuterBound, FARMER_OPT - 1.0)
            # min: outer bound <= optimum <= incumbent (inner)
            self.assertGreaterEqual(wmin.BestInnerBound, -FARMER_OPT - 1.0)
            self.assertLessEqual(wmin.BestOuterBound, -FARMER_OPT + 1.0)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_cross_scenario_cuts(self):
        # The cut spoke has to keep sending cuts, and the bound the hub
        # computes from them has to stay valid. Before issue #933 the spoke
        # sent only its first message, and with cuts flowing the bound came
        # out above the optimum for 4 or more scenarios, from eta's huge
        # default lower bound.
        from mpisppy.extensions.cross_scen_extension import (
            CrossScenarioExtension,
        )
        # The 4-scenario farmer optimum, from ExtensiveForm.
        FARMER4_OPT = -100975.99397650975
        real_enditer = CrossScenarioExtension.enditer
        timed_out = []

        def waiting_enditer(ext):
            # How soon the spoke's first cut arrives depends on the machine.
            # Wait for one at iteration 1, so the rest of the run has cuts to
            # compute the bound from. The spoke's setup takes seconds; the
            # limit only has to end a run whose spoke never cuts inside CI's
            # 10-minute step.
            real_enditer(ext)
            if ext.opt._PHIter != 1:
                return
            # Both decisions are agreed across the hub's ranks, and a time
            # out ends the run normally rather than raising: an exception on
            # the hub would leave the spoke waiting for a kill signal.
            start = time.time()
            while True:
                ext.opt.spcomm.sync_nonants()
                ext.sync_with_spokes()
                have = int(all(_has_a_real_cut(s)
                               for s in ext.opt.local_scenarios.values()))
                if ext.opt.mpicomm.allreduce(have, op=MPI.MIN):
                    return
                late = int(time.time() - start > 120)
                if ext.opt.mpicomm.allreduce(late, op=MPI.MAX):
                    timed_out.append(True)
                    return
                time.sleep(0.01)

        # Both runs first, then the assertions: an assertion that fails on
        # rank 0 between runs would leave the other rank in the next run.
        wheels = {}
        for sense in (pyo.minimize, pyo.maximize):
            self.cfg = _create_cfg()
            self.cfg.cross_scenario_cuts_args()
            scenario_creator_kwargs, beans, hub_dict = self._create_stuff(
                iters=20, sense=sense, num_scens=4)
            vanilla.add_cross_scenario_cuts(hub_dict, self.cfg)
            cut_spoke = vanilla.cross_scenario_cuts_spoke(
                *beans, scenario_creator_kwargs=scenario_creator_kwargs)
            wheel = WheelSpinner(hub_dict, [cut_spoke])
            with patch.object(CrossScenarioExtension, "enditer",
                              waiting_enditer):
                wheel.spin()
            wheels[sense] = wheel
        if wheels[pyo.minimize].global_rank != 0:
            return
        self.assertFalse(timed_out, msg="no cut arrived in 120 seconds")
        tol = 1e-9 * abs(FARMER4_OPT)
        for sense, wheel in wheels.items():
            s = next(iter(wheel.spcomm.opt.local_scenarios.values()))
            # A Benders cut, not the eta lower bounds the spoke sends first,
            # which have no nonant term.
            self.assertTrue(_has_a_real_cut(s),
                            msg="the hub got no Benders cut")
            # Better than PH's iteration-0 bound, so the extension did
            # compute one from the cuts.
            trivial = wheel.spcomm.opt.trivial_bound
            if sense == pyo.minimize:
                self.assertLessEqual(wheel.BestOuterBound, FARMER4_OPT + tol)
                self.assertGreater(wheel.BestOuterBound, trivial)
            else:
                self.assertGreaterEqual(wheel.BestOuterBound,
                                        -FARMER4_OPT - tol)
                self.assertLess(wheel.BestOuterBound, trivial)


def _has_a_real_cut(s):
    """Whether a scenario holds a cut with a nonzero nonant coefficient."""
    from pyomo.repn.standard_repn import generate_standard_repn
    nonants = {id(v) for v in s._mpisppy_data.nonant_indices.values()}
    for c in s._mpisppy_model.benders_cuts.values():
        repn = generate_standard_repn(c.body)
        if any(coef != 0 and id(v) in nonants
               for coef, v in zip(repn.linear_coefs, repn.linear_vars)):
            return True
    return False


#*****************************************************************************


class Test_farmer_cvar_with_cylinders(unittest.TestCase):
    """Risk-averse (CVaR) farmer through the cylinders.

    eta is just another first-stage variable, so PH/Lagrangian/xhat run with no
    algorithm changes.  The spoke bounds must bracket the EF-CVaR optimum; this
    is a rho-independent guarantee (valid bounds regardless of PH convergence
    rate), so we assert the bracket, not a specific value.
    """

    CVAR_WEIGHT = 2.0
    CVAR_ALPHA = 0.8

    def setUp(self):
        self.cfg = _create_cfg()
        self.cfg.num_scens = 3
        self.cfg.max_iterations = 10
        self.creator = cvar.cvar_scenario_creator(
            farmer.scenario_creator,
            cvar_weight=self.CVAR_WEIGHT, cvar_alpha=self.CVAR_ALPHA)
        self.names = farmer.scenario_names_creator(self.cfg.num_scens)
        self.kwargs = farmer.kw_creator(self.cfg)

    def _ef_cvar_opt(self):
        ef = sputils.create_EF(self.names, self.creator,
                               scenario_creator_kwargs=self.kwargs,
                               suppress_warnings=True)
        solver = pyo.SolverFactory(solver_name)
        if "persistent" in solver_name:
            solver.set_instance(ef)
        solver.solve(ef)
        return pyo.value(ef.EF_Obj)

    def _hub(self):
        beans = (self.cfg, self.creator, farmer.scenario_denouement, self.names)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=self.kwargs)
        return beans, hub_dict

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_lagrangian_outer_bound(self):
        self.cfg.lagrangian_args()
        beans, hub_dict = self._hub()
        lagrangian_spoke = vanilla.lagrangian_spoke(
            *beans, scenario_creator_kwargs=self.kwargs)
        wheel = WheelSpinner(hub_dict, [lagrangian_spoke])
        wheel.spin()
        if wheel.global_rank == 0:
            ef_opt = self._ef_cvar_opt()
            self.assertIsNotNone(wheel.BestOuterBound)
            # outer (lower) bound for this minimization (small epsilon for solver round-off)
            self.assertLessEqual(wheel.BestOuterBound, ef_opt + 1e-8 * abs(ef_opt))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_xhatshuffle_inner_bound(self):
        self.cfg.xhatshuffle_args()
        beans, hub_dict = self._hub()
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(
            *beans, scenario_creator_kwargs=self.kwargs)
        wheel = WheelSpinner(hub_dict, [xhatshuffle_spoke])
        wheel.spin()
        if wheel.global_rank == 0:
            ef_opt = self._ef_cvar_opt()
            self.assertIsNotNone(wheel.BestInnerBound)
            # inner (upper) bound for this minimization (small epsilon for solver round-off)
            self.assertGreaterEqual(wheel.BestInnerBound, ef_opt - 1e-8 * abs(ef_opt))


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
