# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Provide some test for aph under mpi-sppy.
# Author: David L. Woodruff (started circa September 2019)
#         add farmer tests April 2021
"""
IMPORTANT:
  Unless we run to convergence and check xhat, the solver, and even solver
version matter a lot, so we often just do smoke tests for sizes.
"""

import sys
import unittest
from math import log10, floor
import pyomo.environ as pyo
import mpisppy.opt.aph
import mpisppy.phbase
from mpisppy.tests.examples.sizes.sizes import scenario_creator, \
                                               scenario_denouement, \
                                               _rho_setter
import mpisppy.tests.examples.farmer as farmer

__version__ = 0.5
solvers = ["xpress_persistent", "gurobi_persistent", "cplex"]

for solvername in solvers:
    solver_available = pyo.SolverFactory(solvername).available(exception_flag=False)
    if solver_available:
        break

import mpi4py.MPI as mpi
fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

#*****************************************************************************
class Test_aph_sizes(unittest.TestCase):
    """ Test the aph mpisppy code using sizes."""

    def setUp(self):
        self.BasePHoptions = {}
        self.BasePHoptions["asynchronousPH"] = False
        self.BasePHoptions["solvername"] = solvername
        self.BasePHoptions["PHIterLimit"] = 10
        self.BasePHoptions["defaultPHrho"] = 1
        self.BasePHoptions["convthresh"] = 0.001
        self.BasePHoptions["subsolvedirectives"] = None
        self.BasePHoptions["verbose"] = False
        self.BasePHoptions["display_timing"] = False
        self.BasePHoptions["display_progress"] = False
        self.BasePHoptions["iter0_solver_options"] = {"mipgap": 0.1}
        self.BasePHoptions["iterk_solver_options"] = {"mipgap": 0.02}

        self.BasePHoptions["display_progress"] = False

        self.all3_scenario_names = list()
        for sn in range(3):
            self.all3_scenario_names.append("Scenario"+str(sn+1))

        self.all10_scenario_names = list()
        for sn in range(10):
            self.all10_scenario_names.append("Scenario"+str(sn+1))

    def _copy_of_base_options(self):
        retval = {}
        for k,v in self.BasePHoptions.items():
            retval[k] = v
        return retval

    def round_pos_sig(self, x, sig=1):
        return round(x, sig-int(floor(log10(abs(x))))-1)
        
    def test_make_aph(self):
        """ Just smoke to verify construction"""
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["async_frac_needed"] = 0.5
        PHoptions["async_sleep_secs"] = 0.5
        aph = mpisppy.opt.aph.APH(
            PHoptions,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_aph_basic(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["async_frac_needed"] = 0.5
        PHoptions["async_sleep_secs"] = 0.5
        aph = mpisppy.opt.aph.APH(
            PHoptions,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )

        conv, obj, tbound = aph.APH_main(spcomm=None)
        print ("objthing={}, (was=-2435908)".format(obj))
        print ("tbound ={} (was=224106)".format(tbound))

    def test_bundles(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["bundles_per_rank"] = 2
        PHoptions["async_frac_needed"] = 0.5
        PHoptions["async_sleep_secs"] = 0.5
        aph = mpisppy.opt.aph.APH(
            PHoptions,
            self.all10_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 10},
        )
        conv, obj, tbound = aph.APH_main(spcomm=None)
        print ("bundle objthing={}, (was=224712.9)".format(obj))
        print ("bundle tbound ={} (was=223168.5)".format(tbound))

        
    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_APHgamma(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["async_frac_needed"] = 0.5
        PHoptions["async_sleep_secs"] = 0.5
        a = 2
        PHoptions["APHgamma"] = a
        aph = mpisppy.opt.aph.APH(
            PHoptions,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )

        conv, obj, tbound = aph.APH_main(spcomm=None)
        print (f"APHgamma={a}; objthing={obj}")
        print ("tbound ={}".format(tbound))


    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_use_lag(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["async_frac_needed"] = 0.5
        PHoptions["async_sleep_secs"] = 0.5
        a = 2
        PHoptions["APHuse_lag"] = True
        aph = mpisppy.opt.aph.APH(
            PHoptions,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )

        conv, obj, tbound = aph.APH_main(spcomm=None)
        print (f"use lag objthing={obj}")
        print ("tbound ={}".format(tbound))


    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_running_dump(self):
        # just see if "display_convergence_detail" causes a crash
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["display_convergence_detail"] = True
        PHoptions["async_frac_needed"] = 1
        PHoptions["async_sleep_secs"] = 0.5
        aph = mpisppy.opt.aph.APH(
            PHoptions,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
        conv, obj, tbound = aph.APH_main(spcomm=None)


    def test_lags_bundles(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["bundles_per_rank"] = 2
        PHoptions["async_frac_needed"] = 0.5
        PHoptions["async_sleep_secs"] = 0.5
        PHoptions["APHuse_lag"] = True
        aph = mpisppy.opt.aph.APH(
            PHoptions,
            self.all10_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 10},
        )
        conv, obj, tbound = aph.APH_main(spcomm=None)
        print ("uselag bundle objthing={}, (pre-lag was=224712.9)".format(obj))
        print ("bundle tbound ={} (was=223168.5)".format(tbound))

#*****************************************************************************
class Test_aph_farmer(unittest.TestCase):
    """ Test the aph mpisppy code using farmer."""

    def setUp(self):
        self.BasePHoptions = {}
        self.BasePHoptions["asynchronousPH"] = False
        self.BasePHoptions["solvername"] = solvername
        self.BasePHoptions["PHIterLimit"] = 10
        self.BasePHoptions["defaultPHrho"] = 1
        self.BasePHoptions["convthresh"] = 0.001
        self.BasePHoptions["subsolvedirectives"] = None
        self.BasePHoptions["verbose"] = False
        self.BasePHoptions["display_timing"] = False
        self.BasePHoptions["display_progress"] = False
        self.BasePHoptions["iter0_solver_options"] = None
        self.BasePHoptions["iterk_solver_options"] = None

        self.BasePHoptions["display_progress"] = False

    def _copy_of_base_options(self):
        retval = {}
        for k,v in self.BasePHoptions.items():
            retval[k] = v
        return retval

    def round_pos_sig(self, x, sig=1):
        return round(x, sig-int(floor(log10(abs(x))))-1)

    def _make_scenario_names(self, cnt):
        return [f"Scenario{i+1}" for i in range(cnt)]
        
    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_aph_farmer_basic30(self):
        APHoptions = self._copy_of_base_options()
        APHoptions["PHIterLimit"] = 2
        APHoptions["aph_dispatch_frac"] = 1
        APHoptions["async_frac_needed"] = 1
        APHoptions["async_sleep_secs"] = 0.01
        APHoptions["aph_gamma"] = 1
        aph = mpisppy.opt.aph.APH(
            APHoptions,
            self._make_scenario_names(30),
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs={"crops_multiplier": 1},
        )

        conv, obj, tbound = aph.APH_main(spcomm=None)
        digits=3
        tbtarget = self.round_pos_sig(137846, digits)
        boundgot = self.round_pos_sig(-tbound, digits)
        self.assertEqual(tbtarget, boundgot)
        # Note: this is not a valid xhat obj, it is weighted
        objtarget = self.round_pos_sig(134796, digits)
        objgot = self.round_pos_sig(-obj, digits)
        self.assertEqual(objgot, objtarget)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_aph_farmer_dispatch(self):
        APHoptions = self._copy_of_base_options()
        APHoptions["PHIterLimit"] = 2
        APHoptions["dispatch_frac"] = 0.25
        APHoptions["async_frac_needed"] = 1
        APHoptions["async_sleep_secs"] = 0.01
        APHoptions["aph_gamma"] = 1
        aph = mpisppy.opt.aph.APH(
            APHoptions,
            self._make_scenario_names(30),
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs={"crops_multiplier": 1},
        )

        conv, obj, tbound = aph.APH_main(spcomm=None)
        digits=3
        tbtarget = self.round_pos_sig(137846, digits)
        boundgot = self.round_pos_sig(-tbound, digits)
        self.assertEqual(tbtarget, boundgot)
        # Note: this is not a valid xhat obj, it is weighted
        objtarget = self.round_pos_sig(134680, digits)
        objgot = self.round_pos_sig(-obj, digits)
        self.assertEqual(objgot, objtarget)


    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_aph_farmer_dispatch_bundles(self):
        APHoptions = self._copy_of_base_options()
        APHoptions["PHIterLimit"] = 2
        APHoptions["dispatch_frac"] = 0.25
        APHoptions["async_frac_needed"] = 1
        APHoptions["async_sleep_secs"] = 0.01
        APHoptions["aph_gamma"] = 1
        APHoptions["bundles_per_rank"] = 5
        
        aph = mpisppy.opt.aph.APH(
            APHoptions,
            self._make_scenario_names(30),
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs={"crops_multiplier": 1},
        )

        conv, obj, tbound = aph.APH_main(spcomm=None)
        digits=3
        tbtarget = self.round_pos_sig(133005, digits)
        boundgot = self.round_pos_sig(-tbound, digits)
        self.assertEqual(tbtarget, boundgot)
        # Note: this is not a valid xhat obj, it is weighted
        objtarget = self.round_pos_sig(132959, digits)
        objgot = self.round_pos_sig(-obj, digits)
        self.assertEqual(objgot, objtarget)


if __name__ == '__main__':
    unittest.main()
