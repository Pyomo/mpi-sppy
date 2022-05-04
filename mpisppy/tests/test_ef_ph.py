# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Author: David L. Woodruff (circa January 2019)
"""
IMPORTANT:
  Unless we run to convergence, the solver, and even solver
version matter a lot, so we often just do smoke tests.
"""

import os
import pandas as pd
import unittest

import pyomo.environ as pyo
import mpisppy.opt.ph
import mpisppy.phbase
import mpisppy.utils.sputils as sputils
from mpisppy.tests.examples.sizes.sizes import scenario_creator, \
                                               scenario_denouement, \
                                               _rho_setter
import mpisppy.tests.examples.hydro.hydro as hydro
from mpisppy.extensions.xhatspecific import XhatSpecific
from mpisppy.tests.utils import get_solver,round_pos_sig

__version__ = 0.54

solver_available,solvername, persistent_available, persistentsolvername= get_solver()

def _get_ph_base_options():
    Baseoptions = {}
    Baseoptions["asynchronousPH"] = False
    Baseoptions["solvername"] = solvername
    Baseoptions["PHIterLimit"] = 10
    Baseoptions["defaultPHrho"] = 1
    Baseoptions["convthresh"] = 0.001
    Baseoptions["subsolvedirectives"] = None
    Baseoptions["verbose"] = False
    Baseoptions["display_timing"] = False
    Baseoptions["display_progress"] = False
    if "cplex" in solvername:
        Baseoptions["iter0_solver_options"] = {"mip_tolerances_mipgap": 0.001}
        Baseoptions["iterk_solver_options"] = {"mip_tolerances_mipgap": 0.00001}
    else:
        Baseoptions["iter0_solver_options"] = {"mipgap": 0.001}
        Baseoptions["iterk_solver_options"] = {"mipgap": 0.00001}

    Baseoptions["display_progress"] = False

    return Baseoptions




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

    def _fix_creator(self, scenario_name, scenario_count=None):
        """ For the test of fixed vars"""
        model = scenario_creator(
            scenario_name, scenario_count=scenario_count
        )
        model.NumProducedFirstStage[5].fix(1134) # 3scen -> 45k
        return model
        
    def test_disable_tictoc(self):
        import mpisppy.utils.sputils as utils
        print("disabling tictoc output")
        utils.disable_tictoc_output()
        # now just do anything that would make a little tictoc output
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 0

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
        print("reeanabling tictoc ouput")
        utils.reenable_tictoc_output()


    def test_ph_constructor(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 0

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )

    def test_ef_constructor(self):
        ScenCount = 3
        ef = mpisppy.utils.sputils.create_EF(
            self.all3_scenario_names,
            scenario_creator,
            scenario_creator_kwargs={"scenario_count": ScenCount},
            suppress_warnings=True
        )

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_solve(self):
        options = self._copy_of_base_options()
        solver = pyo.SolverFactory(options["solvername"])
        ScenCount = 3
        ef = mpisppy.utils.sputils.create_EF(
            self.all3_scenario_names,
            scenario_creator,
            scenario_creator_kwargs={"scenario_count": ScenCount},
            suppress_warnings=True
        )
        if '_persistent' in options["solvername"]:
            solver.set_instance(ef)
        results = solver.solve(ef, tee=False)
        sig2eobj = round_pos_sig(pyo.value(ef.EF_Obj),2)
        self.assertEqual(220000.0, sig2eobj)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fix_ef_solve(self):
        options = self._copy_of_base_options()
        solver = pyo.SolverFactory(options["solvername"])
        ScenCount = 3
        ef = mpisppy.utils.sputils.create_EF(
            self.all3_scenario_names,
            self._fix_creator,
            scenario_creator_kwargs={"scenario_count": ScenCount},
        )
        if '_persistent' in options["solvername"]:
            solver.set_instance(ef)
        results = solver.solve(ef, tee=False)
        sig2eobj = round_pos_sig(pyo.value(ef.EF_Obj),2)
        # the fix creator fixed num prod first stage for size 5 to 1134
        self.assertEqual(pyo.value(ef.Scenario1.NumProducedFirstStage[5]), 1134)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_iter0(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 0

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
    
        conv, obj, tbound = ph.ph_main()

        sig2obj = round_pos_sig(obj,2)
        #self.assertEqual(1400000000, sig2obj)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fix_ph_iter0(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 0

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            self._fix_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
    
        conv, obj, tbound = ph.ph_main()

        # Actually, if it is still fixed for one scenario, probably fixed forall.
        for k,s in ph.local_scenarios.items():
            self.assertEqual(pyo.value(s.NumProducedFirstStage[5]), 1134)
            self.assertTrue(s.NumProducedFirstStage[5].is_fixed())
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_basic(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
        conv, obj, tbound = ph.ph_main()


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fix_ph_basic(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            self._fix_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
        conv, obj, tbound = ph.ph_main()
        for k,s in ph.local_scenarios.items():
            self.assertTrue(s.NumProducedFirstStage[5].is_fixed())
            self.assertEqual(pyo.value(s.NumProducedFirstStage[5]), 1134)
        

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_rhosetter(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
            rho_setter=_rho_setter,
        )
        conv, obj, tbound = ph.ph_main()
        sig2obj = round_pos_sig(obj,2)
        #self.assertEqual(220000.0, sig2obj)
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_bundles(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 2
        options["bundles_per_rank"] = 2
        ph = mpisppy.opt.ph.PH(
            options,
            self.all10_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 10},
        )
        conv, obj, tbound = ph.ph_main()

    @unittest.skipIf(not persistent_available,
                     "%s solver is not available" % (persistentsolvername,))
    def test_persistent_basic(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 10
        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
        conv, basic_obj, tbound = ph.ph_main()

        options = self._copy_of_base_options()
        options["PHIterLimit"] = 10
        options["solvername"] = persistentsolvername
        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
        conv, pobj, tbound = ph.ph_main()

        sig2basic = round_pos_sig(basic_obj,2)
        sig2pobj = round_pos_sig(pobj,2)
        self.assertEqual(sig2basic, sig2pobj)
        
    @unittest.skipIf(not persistent_available,
                     "%s solver is not available" % (persistentsolvername,))
    def test_persistent_bundles(self):
        """ This excercises complicated code.
        """
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 2
        options["bundles_per_rank"] = 2
        ph = mpisppy.opt.ph.PH(
            options,
            self.all10_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 10},
        )
        conv, basic_obj, tbound = ph.ph_main()

        options = self._copy_of_base_options()
        options["PHIterLimit"] = 2
        options["bundles_per_rank"] = 2
        options["solvername"] = persistentsolvername
        ph = mpisppy.opt.ph.PH(
            options,
            self.all10_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 10},
        )
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
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 1
        options["xhat_looper_options"] =  {"xhat_solver_options":\
                                             options["iterk_solver_options"],
                                             "scen_limit": 3}

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
            extensions=XhatLooper,
        )
        conv, basic_obj, tbound = ph.ph_main()
        # in this particular case, the extobject is an xhatter
        xhatobj1 = round_pos_sig(ph.extobject._xhat_looper_obj_final, 1)
        self.assertEqual(xhatobj1, 200000)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fix_xhat_extension(self):
        """ Make sure that ph and xhat do not unfix a fixed Var
        """
        from mpisppy.extensions.xhatlooper import XhatLooper
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 1
        options["xhat_looper_options"] =  {"xhat_solver_options":\
                                             options["iterk_solver_options"],
                                             "scen_limit": 3}

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            self._fix_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
            extensions=XhatLooper,
        )
        conv, basic_obj, tbound = ph.ph_main()

        for k,s in ph.local_scenarios.items():
            self.assertTrue(s.NumProducedFirstStage[5].is_fixed())
            self.assertEqual(pyo.value(s.NumProducedFirstStage[5]), 1134)

            
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_xhatlooper_bound(self):
        """ smoke test for the looper as an extension
        """
        from mpisppy.extensions.xhatlooper import XhatLooper
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 1
        options["xhat_looper_options"] =  {"xhat_solver_options":\
                                             options["iterk_solver_options"],
                                             "scen_limit": 3}
        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
            extensions=XhatLooper,
        )
        conv, basic_obj, tbound = ph.ph_main()
        xhatobj = ph.extobject._xhat_looper_obj_final
        dopts = sputils.option_string_to_dict("mipgap=0.0001")
        objbound = ph.post_solve_bound(solver_options=dopts, verbose=False)
        self.assertGreaterEqual(xhatobj+1.e-6, objbound)
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def smoke_for_extensions(self):
        """ Make sure the example extensions can at least run.
        """
        from mpisppy.extensions.extension import MultiExtension
        from mpisppy.extensions.fixer import Fixer
        from mpisppy.extensions.mipgapper import Gapper
        from mpisppy.extensions.xhatlooper import XhatLooper
        from mpisppy.extensions.xhatclosest import XhatClosest

        options = self._copy_of_base_options()
        options["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(options, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    extensions=MultiExtension,
                                    extension_kwargs=multi_ext,
        )
        multi_ext = {"ext_classes": [Fixer, Gapper, XhatLooper, XhatClosest]}
        conv, basic_obj, tbound = ph.ph_main()


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def fix_for_extensions(self):
        """ Make sure the example extensions don't destroy fixedness
        """
        from mpisppy.extensions.extensions import MultiExtension
        from mpisppy.extensions.fixer import Fixer
        from mpisppy.extensions.mipgapper import Gapper
        from mpisppy.extensions.xhatlooper import XhatLooper
        from mpisppy.extensions.xhatclosest import XhatClosest
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 2
        ph = mpisppy.opt.ph.PH(options, self.all3_scenario_names,
                                    self._fix_creator, scenario_denouement,
                                    extensions=MultiExtension,
                                    extension_kwargs=multi_ext,
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
        self.options = self._copy_of_base_options()
        # branching factor (3 stages is hard-wired)
        self.branching_factors = self.options["branching_factors"]
        self.ScenCount = self.branching_factors[0] * self.branching_factors[1]
        self.all_scenario_names = list()
        for sn in range(self.ScenCount):
            self.all_scenario_names.append("Scen"+str(sn+1))
        self.all_nodenames = sputils.create_nodenames_from_branching_factors(self.branching_factors)
        # end hardwire

    def test_ph_constructor(self):
        options = self._copy_of_base_options()

        ph = mpisppy.opt.ph.PH(
            options,
            self.all_scenario_names,
            hydro.scenario_creator,
            hydro.scenario_denouement,
            all_nodenames=self.all_nodenames,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
        )

    def test_ef_constructor(self):
        ef = mpisppy.utils.sputils.create_EF(
            self.all_scenario_names,
            hydro.scenario_creator,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
        )

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_solve(self):
        options = self._copy_of_base_options()
        solver = pyo.SolverFactory(options["solvername"])
        ef = mpisppy.utils.sputils.create_EF(
            self.all_scenario_names,
            hydro.scenario_creator,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
        )
        if '_persistent' in options["solvername"]:
            solver.set_instance(ef)
        results = solver.solve(ef, tee=False)
        mpisppy.utils.sputils.ef_nonants_csv(ef, "delme.csv")

        df = pd.read_csv("delme.csv", index_col=1)
        val2 = round_pos_sig(df.loc[" Scen7.Pgt[2]"].at[" Value"])
        self.assertEqual(val2, 60)
        os.remove("delme.csv")

        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_csv(self):
        options = self._copy_of_base_options()
        solver = pyo.SolverFactory(options["solvername"])
        ef = mpisppy.utils.sputils.create_EF(
            self.all_scenario_names,
            hydro.scenario_creator,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
        )
        if '_persistent' in options["solvername"]:
            solver.set_instance(ef)
        results = solver.solve(ef, tee=False)
    

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_solve(self):
        options = self._copy_of_base_options()

        ph = mpisppy.opt.ph.PH(
            options,
            self.all_scenario_names,
            hydro.scenario_creator,
            hydro.scenario_denouement,
            all_nodenames=self.all_nodenames,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
        )
        conv, obj, tbound = ph.ph_main()

        sig2tbnd = round_pos_sig(tbound, 2)
        self.assertEqual(180, sig2tbnd)

        ph.disable_W_and_prox()
        sig2eobj = round_pos_sig(ph.Eobjective(), 2)
        self.assertEqual(190, sig2eobj)

        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_xhat(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 10  # xhat is feasible
        options["xhat_specific_options"] = {"xhat_solver_options":
                                              options["iterk_solver_options"],
                                              "xhat_scenario_dict": \
                                              {"ROOT": "Scen1",
                                               "ROOT_0": "Scen1",
                                               "ROOT_1": "Scen4",
                                               "ROOT_2": "Scen7"},
                                              "csvname": "specific.csv"}

        ph = mpisppy.opt.ph.PH(
            options,
            self.all_scenario_names,
            hydro.scenario_creator,
            hydro.scenario_denouement,
            all_nodenames=self.all_nodenames,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
            extensions = XhatSpecific
        )
        conv, obj, tbound = ph.ph_main()
        sig2xhatobj = round_pos_sig(ph.extobject._xhat_specific_obj_final, 2)
        self.assertEqual(190, sig2xhatobj)


if __name__ == '__main__':
    unittest.main()
