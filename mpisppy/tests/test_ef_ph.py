# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Provide some test for mpisppy.0
# Author: David L. Woodruff (circa January 2019)
"""
IMPORTANT:
  Unless we run to convergence, the solver, and even solver
version matter a lot, so we often just do smoke tests.
"""

import os
import pandas as pd
import unittest

from math import log10, floor
import pyomo.environ as pyo
import mpisppy.opt.ph
import mpisppy.phbase
import mpisppy.utils.sputils as sputils
from mpisppy.tests.examples.sizes.sizes import scenario_creator, \
                                               scenario_denouement, \
                                               _rho_setter
import mpisppy.tests.examples.hydro.hydro as hydro
from mpisppy.extensions.xhatspecific import XhatSpecific

__version__ = 0.52

solvers = ["gurobi","cplex"]

for solvername in solvers:
    solver_available = pyo.SolverFactory(solvername).available()
    if solver_available:
        break

persistentsolvername = solvername+"_persistent"
try:
    persistent_available = pyo.SolverFactory(persistentsolvername).available()
except:
    persistent_available = False
# TBD: cplex persistent does not work with some versions of Pyomo (June 2020)
if solvername == "cplex":
    persistent_available = False

def _get_ph_base_options():
    BasePHoptions = {}
    BasePHoptions["asynchronousPH"] = False
    BasePHoptions["solvername"] = solvername
    BasePHoptions["PHIterLimit"] = 10
    BasePHoptions["defaultPHrho"] = 1
    BasePHoptions["convthresh"] = 0.001
    BasePHoptions["subsolvedirectives"] = None
    BasePHoptions["verbose"] = False
    BasePHoptions["display_timing"] = False
    BasePHoptions["display_progress"] = False
    BasePHoptions["iter0_solver_options"] = {"mipgap": 0.001}
    BasePHoptions["iterk_solver_options"] = {"mipgap": 0.00001}

    BasePHoptions["display_progress"] = False

    return BasePHoptions

def round_pos_sig(x, sig=1):
    return round(x, sig-int(floor(log10(abs(x))))-1)


#*****************************************************************************
class Test_sizes(unittest.TestCase):
    """ Test the mpisppy code using sizes."""

    def setUp(self):
        self.all3_scenario_names = list()
        for sn in range(3):
            self.all3_scenario_names.append("Scenario"+str(sn+1))

        self.all10_scenario_names = list()
        for sn in range(10):
            self.all10_scenario_names.append("Scenario"+str(sn+1))

    def _copy_of_base_options(self):
        return _get_ph_base_options()

    def _fix_creator(self, scenario_name, node_names=None, cb_data=None):
        """ For the test of fixed vars"""
        model = scenario_creator(scenario_name,
                                 node_names=node_names,
                                 cb_data=cb_data)
        model.NumProducedFirstStage[5].fix(1134) # 3scen -> 45k
        return model
        
    def test_disable_tictoc(self):
        import mpisppy.utils.sputils as utils
        print("disabling tictoc output")
        utils.disable_tictoc_output()
        # now just do anything that would make a little tictoc output
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 0

        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=3)
        print("reeanabling tictoc ouput")
        utils.reenable_tictoc_output()


    def test_ph_constructor(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 0

        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=3)

    def test_ef_constructor(self):
        ScenCount = 3
        ef = mpisppy.utils.sputils.create_EF(self.all3_scenario_names,
                                    scenario_creator,
                                    creator_options={"cb_data": ScenCount},
                                    suppress_warnings=True)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_solve(self):
        PHoptions = self._copy_of_base_options()
        solver = pyo.SolverFactory(PHoptions["solvername"])
        ScenCount = 3
        ef = mpisppy.utils.sputils.create_EF(self.all3_scenario_names,
                                    scenario_creator,
                                    creator_options={"cb_data": ScenCount},
                                    suppress_warnings=True)
        results = solver.solve(ef, tee=False)
        sig2eobj = round_pos_sig(pyo.value(ef.EF_Obj),2)
        self.assertEqual(220000.0, sig2eobj)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fix_ef_solve(self):
        PHoptions = self._copy_of_base_options()
        solver = pyo.SolverFactory(PHoptions["solvername"])
        ScenCount = 3
        ef = mpisppy.utils.sputils.create_EF(self.all3_scenario_names,
                                       self._fix_creator,
                                       creator_options={"cb_data": ScenCount})
        results = solver.solve(ef, tee=False)
        sig2eobj = round_pos_sig(pyo.value(ef.EF_Obj),2)
        # the fix creator fixed num prod first stage for size 5 to 1134
        self.assertEqual(ef.Scenario1.NumProducedFirstStage[5], 1134)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_iter0(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 0

        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=3)
    
        conv, obj, tbound = ph.ph_main()

        sig2obj = round_pos_sig(obj,2)
        #self.assertEqual(1400000000, sig2obj)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fix_ph_iter0(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 0

        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    self._fix_creator, scenario_denouement,
                                    cb_data=3)
    
        conv, obj, tbound = ph.ph_main()

        # Actually, if it is still fixed for one scenario, probably fixed forall.
        for k,s in ph.local_scenarios.items():
            self.assertEqual(pyo.value(s.NumProducedFirstStage[5]), 1134)
            self.assertTrue(s.NumProducedFirstStage[5].is_fixed())
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_basic(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=3)
        conv, obj, tbound = ph.ph_main()


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fix_ph_basic(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    self._fix_creator, scenario_denouement,
                                    cb_data=3)
        conv, obj, tbound = ph.ph_main()
        for k,s in ph.local_scenarios.items():
            self.assertTrue(s.NumProducedFirstStage[5].is_fixed())
            self.assertEqual(pyo.value(s.NumProducedFirstStage[5]), 1134)
        

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_rhosetter(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=3, rho_setter=_rho_setter)
        conv, obj, tbound = ph.ph_main()
        sig2obj = round_pos_sig(obj,2)
        #self.assertEqual(220000.0, sig2obj)
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_bundles(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["bundles_per_rank"] = 2
        ph = mpisppy.opt.ph.PH(PHoptions, self.all10_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=10)
        conv, obj, tbound = ph.ph_main()

    @unittest.skipIf(not persistent_available,
                     "%s solver is not available" % (persistentsolvername,))
    def test_persistent_basic(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 10
        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=3)
        conv, basic_obj, tbound = ph.ph_main()

        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 10
        PHoptions["solvername"] = persistentsolvername
        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=3)
        conv, pobj, tbound = ph.ph_main()

        sig2basic = round_pos_sig(basic_obj,2)
        sig2pobj = round_pos_sig(pobj,2)
        self.assertEqual(sig2basic, sig2pobj)
        
    @unittest.skipIf(not persistent_available,
                     "%s solver is not available" % (persistentsolvername,))
    def test_persistent_bundles(self):
        """ This excercises complicated code.
        """
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["bundles_per_rank"] = 2
        ph = mpisppy.opt.ph.PH(PHoptions, self.all10_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=10)
        conv, basic_obj, tbound = ph.ph_main()

        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        PHoptions["bundles_per_rank"] = 2
        PHoptions["solvername"] = persistentsolvername
        ph = mpisppy.opt.ph.PH(PHoptions, self.all10_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=10)
        conv, pbobj, tbound = ph.ph_main()

        sig2basic = round_pos_sig(basic_obj,2)
        sig2pbobj = round_pos_sig(pbobj,2)
        self.assertEqual(sig2basic, sig2pbobj)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_xhat_extension(self):
        """ Make sure least one of the xhat extensions runs.
        """
        from mpisppy.extensions.xhatlooper import XhatLooper
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 1
        PHoptions["xhat_looper_options"] =  {"xhat_solver_options":\
                                             PHoptions["iterk_solver_options"],
                                             "scen_limit": 3}

        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=3, PH_extensions=XhatLooper)
        conv, basic_obj, tbound = ph.ph_main()
        # in this particular case, the extobject is an xhatter
        xhatobj1 = round_pos_sig(ph.extobject._xhat_looper_obj_final, 1)
        self.assertEqual(xhatobj1, 200000)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fix_xhat_extension(self):
        """ Make sure that ph and xhat does not unfix a fixed Var
        """
        from mpisppy.extensions.xhatlooper import XhatLooper
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 1
        PHoptions["xhat_looper_options"] =  {"xhat_solver_options":\
                                             PHoptions["iterk_solver_options"],
                                             "scen_limit": 3}

        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    self._fix_creator, scenario_denouement,
                                    cb_data=3, PH_extensions=XhatLooper)
        conv, basic_obj, tbound = ph.ph_main()

        for k,s in ph.local_scenarios.items():
            self.assertTrue(s.NumProducedFirstStage[5].is_fixed())
            self.assertEqual(pyo.value(s.NumProducedFirstStage[5]), 1134)

            
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_lagrangian_bound(self):
        """ Make sure the lagrangian bound is at least a bound
        """
        from mpisppy.extensions.xhatlooper import XhatLooper
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 1
        PHoptions["xhat_looper_options"] =  {"xhat_solver_options":\
                                             PHoptions["iterk_solver_options"],
                                             "scen_limit": 3}
        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    cb_data=3, PH_extensions=XhatLooper)
        conv, basic_obj, tbound = ph.ph_main()
        xhatobj = ph.extobject._xhat_looper_obj_final
        dopts = sputils.option_string_to_dict("mipgap=0.0001")
        objbound = ph.post_solve_bound(solver_options=dopts, verbose=False)
        self.assertGreaterEqual(xhatobj, objbound)
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def smoke_for_extensions(self):
        """ Make sure the example extensions can at least run.
        """
        from mpisppy.extensions.extension import MultiPHExtension
        from mpisppy.extensions.fixer import Fixer
        from mpisppy.extensions.mipgapper import Gapper
        from mpisppy.extensions.xhatlooper import XhatLooper
        from mpisppy.extensions.xhatclosest import XhatClosest

        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    PH_extensions=MultiPHExtension,
                                    PH_extension_kwargs=multi_ext,
        )
        multi_ext = {"ext_classes": [Fixer, Gapper, XhatLooper, XhatClosest]}
        conv, basic_obj, tbound = ph.ph_main()


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def fix_for_extensions(self):
        """ Make sure the example extensions don't destroy fixedness
        """
        from mpisppy.extensions.extensions import MultiPHExtension
        from mpisppy.extensions.fixer import Fixer
        from mpisppy.extensions.mipgapper import Gapper
        from mpisppy.extensions.xhatlooper import XhatLooper
        from mpisppy.extensions.xhatclosest import XhatClosest
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(PHoptions, self.all3_scenario_names,
                                    self._fix_creator, scenario_denouement,
                                    PH_extensions=MultiPHExtension,
                                    PH_extension_kwargs=multi_ext,
        )
        multi_ext = {"ext_classes": [Fixer, Gapper, XhatLooper, XhatClosest]}
        conv, basic_obj, tbound = ph.ph_main( )
        for k,s in ph.local_scenarios.items():
            self.assertTrue(s.NumProducedFirstStage[5].is_fixed())
            self.assertEqual(pyo.value(s.NumProducedFirstStage[5]), 1134)

#*****************************************************************************
class Test_hydro(unittest.TestCase):
    """ Test the mpisppy code using hydro (three stages)."""

    def _copy_of_base_options(self):
        retval = _get_ph_base_options()
        retval["branching_factors"] = [3, 3]
        return retval
    

    def setUp(self):
        self.PHoptions = self._copy_of_base_options()
        # branching factor (3 stages is hard-wired)
        self.BFs = self.PHoptions["branching_factors"]
        self.ScenCount = self.BFs[0] * self.BFs[1]
        self.all_scenario_names = list()
        for sn in range(self.ScenCount):
            self.all_scenario_names.append("Scen"+str(sn+1))
        self.all_nodenames = ["ROOT"] # all trees must have this node
        for b in range(self.BFs[0]):
            self.all_nodenames.append("ROOT_"+str(b))
        # end hardwire

    def test_ph_constructor(self):
        PHoptions = self._copy_of_base_options()

        ph = mpisppy.opt.ph.PH(PHoptions, self.all_scenario_names,
                                    hydro.scenario_creator,
                                    hydro.scenario_denouement,
                                    all_nodenames=self.all_nodenames,
                                    cb_data=self.BFs)
    def test_ef_constructor(self):
        ef = mpisppy.utils.sputils.create_EF(self.all_scenario_names,
                                       hydro.scenario_creator,
                                       creator_options={"cb_data": self.BFs})

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_solve(self):
        PHoptions = self._copy_of_base_options()
        solver = pyo.SolverFactory(PHoptions["solvername"])
        ef = mpisppy.utils.sputils.create_EF(self.all_scenario_names,
                                         hydro.scenario_creator,
                                         creator_options={"cb_data": self.BFs})
        results = solver.solve(ef, tee=False)
        mpisppy.utils.sputils.ef_nonants_csv(ef, "delme.csv")

        df = pd.read_csv("delme.csv", index_col=1)
        val2 = round_pos_sig(df.loc[" Scen7.Pgt[2]"].at[" Value"])
        self.assertEqual(val2, 60)
        os.remove("delme.csv")

        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_csv(self):
        PHoptions = self._copy_of_base_options()
        solver = pyo.SolverFactory(PHoptions["solvername"])
        ef = mpisppy.utils.sputils.create_EF(self.all_scenario_names,
                                         hydro.scenario_creator,
                                         creator_options={"cb_data": self.BFs})
        results = solver.solve(ef, tee=False)
    

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_solve(self):
        PHoptions = self._copy_of_base_options()

        ph = mpisppy.opt.ph.PH(PHoptions, self.all_scenario_names,
                                  hydro.scenario_creator,
                                  hydro.scenario_denouement,
                                    all_nodenames=self.all_nodenames,
                                    cb_data=self.BFs)
        conv, obj, tbound = ph.ph_main()

        sig2tbnd = round_pos_sig(tbound, 2)
        self.assertEqual(180, sig2tbnd)

        ph._disable_W_and_prox()
        sig2eobj = round_pos_sig(ph.Eobjective(), 2)
        self.assertEqual(190, sig2eobj)

        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_xhat(self):
        PHoptions = self._copy_of_base_options()
        PHoptions["PHIterLimit"] = 10  # xhat is feasible
        PHoptions["xhat_specific_options"] = {"xhat_solver_options":
                                              PHoptions["iterk_solver_options"],
                                              "xhat_scenario_dict": \
                                              {"ROOT": "Scen1",
                                               "ROOT_0": "Scen1",
                                               "ROOT_1": "Scen4",
                                               "ROOT_2": "Scen7"},
                                              "csvname": "specific.csv"}

        ph = mpisppy.opt.ph.PH(PHoptions, self.all_scenario_names,
                                  hydro.scenario_creator,
                                  hydro.scenario_denouement,
                                    all_nodenames=self.all_nodenames,
                                    cb_data=self.BFs, PH_extensions = XhatSpecific)
        conv, obj, tbound = ph.ph_main()
        sig2xhatobj = round_pos_sig(ph.extobject._xhat_specific_obj_final, 2)
        self.assertEqual(190, sig2xhatobj)


if __name__ == '__main__':
    unittest.main()
