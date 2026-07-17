###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Author: David L. Woodruff (circa January 2019)
"""
IMPORTANT:
  Unless we run to convergence, the solver, and even solver
version matter a lot, so we often just do smoke tests.
"""

import os
import glob
import io
import contextlib
import re
import json
import shutil
import unittest
import pandas as pd
import pyomo.environ as pyo
import mpisppy.opt.ph
import mpisppy.phbase
import mpisppy.utils.sputils as sputils
import mpisppy.utils.rho_utils as rho_utils
from mpisppy.tests.examples.sizes.sizes import scenario_creator, \
                                               scenario_denouement, \
                                               _rho_setter
import mpisppy.tests.examples.hydro.hydro as hydro
from mpisppy.extensions.mult_rho_updater import MultRhoUpdater
from mpisppy.extensions.xhatspecific import XhatSpecific
from mpisppy.extensions.xhatxbar import XhatXbar
from mpisppy.tests.utils import get_solver,round_pos_sig,limit_solver_threads

__version__ = 0.56

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver()

def _get_ph_base_options():
    Baseoptions = {}
    Baseoptions["asynchronousPH"] = False
    Baseoptions["solver_name"] = solver_name
    Baseoptions["PHIterLimit"] = 10
    Baseoptions["defaultPHrho"] = 1
    Baseoptions["convthresh"] = 0.001
    Baseoptions["verbose"] = False
    Baseoptions["display_timing"] = False
    Baseoptions["display_progress"] = False
    if "cplex" in solver_name:
        Baseoptions["iter0_solver_options"] = {"mip_tolerances_mipgap": 0.001, "threads": 1}
        Baseoptions["iterk_solver_options"] = {"mip_tolerances_mipgap": 0.00001, "threads": 1}
    else:
        Baseoptions["iter0_solver_options"] = {"mipgap": 0.001, "threads": 1}
        Baseoptions["iterk_solver_options"] = {"mipgap": 0.00001, "threads": 1}

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

        mpisppy.opt.ph.PH(
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
        assert ph is not None

    def test_bundles_per_rank_raises(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 0
        options["bundles_per_rank"] = 1

        with self.assertRaises(RuntimeError) as cm:
            mpisppy.opt.ph.PH(
                options,
                self.all3_scenario_names,
                scenario_creator,
                scenario_denouement,
                scenario_creator_kwargs={"scenario_count": 3},
            )
        self.assertIn("--scenarios-per-bundle", str(cm.exception))

    def test_ef_constructor(self):
        ScenCount = 3
        ef = mpisppy.utils.sputils.create_EF(
            self.all3_scenario_names,
            scenario_creator,
            scenario_creator_kwargs={"scenario_count": ScenCount},
            suppress_warnings=True
        )
        assert ef is not None

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_solve(self):
        options = self._copy_of_base_options()
        solver = pyo.SolverFactory(options["solver_name"])
        limit_solver_threads(solver, options["solver_name"])
        ScenCount = 3
        ef = mpisppy.utils.sputils.create_EF(
            self.all3_scenario_names,
            scenario_creator,
            scenario_creator_kwargs={"scenario_count": ScenCount},
            suppress_warnings=True
        )
        if '_persistent' in options["solver_name"]:
            solver.set_instance(ef)
        results = solver.solve(ef, tee=False)
        pyo.assert_optimal_termination(results)
        sig2eobj = round_pos_sig(pyo.value(ef.EF_Obj),2)
        self.assertEqual(220000.0, sig2eobj)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_fix_ef_solve(self):
        options = self._copy_of_base_options()
        solver = pyo.SolverFactory(options["solver_name"])
        limit_solver_threads(solver, options["solver_name"])
        ScenCount = 3
        ef = mpisppy.utils.sputils.create_EF(
            self.all3_scenario_names,
            self._fix_creator,
            scenario_creator_kwargs={"scenario_count": ScenCount},
        )
        if '_persistent' in options["solver_name"]:
            solver.set_instance(ef)
        results = solver.solve(ef, tee=False)
        pyo.assert_optimal_termination(results)
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

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_display_timing_emits_nonzero_solve_times(self):
        # End-to-end check that display_timing actually reaches the PH
        # solve loop and produces a non-zero solve-time report. Guards
        # the user-facing CLI flag (issue #290): if a future refactor
        # quietly drops the option from self.options or the print path,
        # this test fails.
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 0
        options["display_timing"] = True

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ph.ph_main()
        out = buf.getvalue()

        self.assertIn("Pyomo solve times (seconds):", out,
                      msg=f"display_timing=True but timing header not in output:\n{out}")
        m = re.search(
            r"min=([0-9.]+)@\d+ mean=([0-9.]+) max=([0-9.]+)@\d+",
            out,
        )
        self.assertIsNotNone(
            m,
            msg=f"display_timing stats line not found in output:\n{out}",
        )
        mn, me, mx = float(m.group(1)), float(m.group(2)), float(m.group(3))
        # max across scenarios must be strictly positive — any real
        # subproblem solve takes measurable wallclock time. min/mean
        # could round to 0.00 under %4.2f if subproblems are tiny.
        self.assertGreater(
            mx, 0.0,
            msg=f"max solve time reported as zero: min={mn} mean={me} max={mx}",
        )

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_display_timing_off_suppresses_solve_times(self):
        # Companion: confirm the timing report is NOT printed when the
        # flag is False, so we know the assertion above isn't picking up
        # output emitted unconditionally somewhere.
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 0
        options["display_timing"] = False

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ph.ph_main()
        self.assertNotIn("Pyomo solve times (seconds):", buf.getvalue())

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
        options["PHIterLimit"] = 4
        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
        conv, obj, tbound = ph.ph_main()
        print(f"basic ph {obj=}")

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_basic_warmstart(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 4
        options["warmstart_subproblems"] = True
        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
        )
        conv, obj, tbound = ph.ph_main()
        print(f"warmstart_subproblems ph {obj=}")
        #assert(math.isclose(obj, -16418049.260124115, rel_tol=0.001))

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
        sig1obj = round_pos_sig(obj,1)
        self.assertEqual(200000.0, sig1obj)

        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_write_read(self):
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
        # The rho_setter is called after iter0
        fname = "__1134__.csv"
        s = ph.local_scenarios[list(ph.local_scenarios.keys())[0]]
        rholen = len(s._mpisppy_model.rho)
        rho_utils.rhos_to_csv(s, fname)
        rholist = rho_utils.rho_list_from_csv(s, fname)
        os.remove(fname)
        self.assertEqual(len(rholist), rholen)  # not a deep test...

        
    @unittest.skipIf(not persistent_available,
                     "%s solver is not available" % (persistent_solver_name,))
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
        options["solver_name"] = persistent_solver_name
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


    @unittest.skipIf(not pyo.SolverFactory('glpk').available(),
                     "glpk is not available")
    def test_scenario_lpwriter_extension(self):
        from mpisppy.extensions.scenario_lp_mps_files import Scenario_lp_mps_files
        tdir = "_delme_test_write_mp_mps_dir"
        if os.path.exists(tdir):
            shutil.rmtree(tdir)
        options = self._copy_of_base_options()
        options["iter0_solver_options"] = {"mipgap": 0.1}    
        options["PHIterLimit"] = 0
        options["solver_name"] = "glpk"
        options["write_lp_mps_extension_options"]\
            = {"write_scenario_lp_mps_files_dir": tdir}

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
            extensions=Scenario_lp_mps_files,
        )
        conv, basic_obj, tbound = ph.ph_main()
        # The idea is to detect a change in Pyomo's writing of lp files
        with open(os.path.join(tdir, "Scenario1_nonants.json"), "r") as jfile:
            nonants_by_node = json.load(jfile)
        # vname is first name in the file            
        vname = nonants_by_node["treeData"]["ROOT"]["nonAnts"][0]
        gotit = False
        with open(os.path.join(tdir, "Scenario1.lp"), 'r') as lpfile:
            for line in lpfile:
                if vname in line:
                    gotit = True
                    break
        assert gotit, f"The first nonant in Scenario1_nonants.json ({vname}) not found in Scenario1.lp"
        print(f"   deleting f{tdir}")
        
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_wtracker_extension(self):
        """ Make sure the wtracker at least does not cause a stack trace
        """
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 4
        options["wtracker_options"] ={"wlen": 3,
                                      "reportlen": 6,
                                      "stdevthresh": 0.1,
                                      "file_prefix": "__1134__"}
        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
            extensions=Wtracker_extension,
        )
        conv, basic_obj, tbound = ph.ph_main()
        fileList = glob.glob('__1134__*.csv')
        for filePath in fileList:
            os.remove(filePath)

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_wtracker_lacks_iters(self):
        """ Make sure the wtracker is graceful with not enough data
        """
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 4
        options["wtracker_options"] ={"wlen": 10,
                                      "reportlen": 6,
                                      "stdevthresh": 0.1}

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
            extensions=Wtracker_extension,
        )
        conv, basic_obj, tbound = ph.ph_main()
        

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
        multi_ext = {"ext_classes": [Fixer, Gapper, XhatLooper, XhatClosest]}
        ph = mpisppy.opt.ph.PH(options, self.all3_scenario_names,
                                    scenario_creator, scenario_denouement,
                                    extensions=MultiExtension,
                                    extension_kwargs=multi_ext,
        )
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
        multi_ext = {"ext_classes": [Fixer, Gapper, XhatLooper, XhatClosest]}
        ph = mpisppy.opt.ph.PH(options, self.all3_scenario_names,
                                    self._fix_creator, scenario_denouement,
                                    extensions=MultiExtension,
                                    extension_kwargs=multi_ext,
        )
        conv, basic_obj, tbound = ph.ph_main( )
        for k,s in ph.local_scenarios.items():
            self.assertTrue(s.NumProducedFirstStage[5].is_fixed())
            self.assertEqual(pyo.value(s.NumProducedFirstStage[5]), 1134)


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_xhat_xbar(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 5
        options["xhat_xbar_options"] = {"xhat_solver_options":
                                        options["iterk_solver_options"],
                                        "csvname": "xhatxbar.csv"}

        ph = mpisppy.opt.ph.PH(
            options,
            self.all3_scenario_names,
            scenario_creator,
            scenario_denouement,
            scenario_creator_kwargs={"scenario_count": 3},
            extensions = XhatXbar
        )
        conv, obj, tbound = ph.ph_main()
        sig2obj = round_pos_sig(obj,2)
        self.assertEqual(sig2obj, 230000)

            
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
        assert ph is not None

    def test_ef_constructor(self):
        ef = mpisppy.utils.sputils.create_EF(
            self.all_scenario_names,
            hydro.scenario_creator,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
        )
        assert ef is not None

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_solve(self):
        options = self._copy_of_base_options()
        solver = pyo.SolverFactory(options["solver_name"])
        limit_solver_threads(solver, options["solver_name"])
        ef = mpisppy.utils.sputils.create_EF(
            self.all_scenario_names,
            hydro.scenario_creator,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
        )
        if '_persistent' in options["solver_name"]:
            solver.set_instance(ef)
        results = solver.solve(ef, tee=False)
        pyo.assert_optimal_termination(results)
        mpisppy.utils.sputils.ef_nonants_csv(ef, "delme.csv")

        # ef_nonants_csv writes the canonical xhat CSV: '#'-comment header,
        # columns node_name, variable_name, value, with node-local variable
        # names (the scenario-block prefix is stripped, so the second-stage
        # Pgt[2] is shared across its node by non-anticipativity).
        df = pd.read_csv("delme.csv", comment="#", header=None,
                         names=["node", "var", "value"], skipinitialspace=True)
        pgt2_vals = [round_pos_sig(v) for v in df[df["var"] == "Pgt[2]"]["value"]]
        self.assertIn(60, pgt2_vals)
        os.remove("delme.csv")

        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_csv(self):
        options = self._copy_of_base_options()
        solver = pyo.SolverFactory(options["solver_name"])
        limit_solver_threads(solver, options["solver_name"])
        ef = mpisppy.utils.sputils.create_EF(
            self.all_scenario_names,
            hydro.scenario_creator,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
        )
        if '_persistent' in options["solver_name"]:
            solver.set_instance(ef)
        results = solver.solve(ef, tee=False)
        pyo.assert_optimal_termination(results)
    

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
    def test_ph_xhat_specific(self):
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


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ph_mult_rho_updater(self):
        options = self._copy_of_base_options()
        options["PHIterLimit"] = 5
        options["mult_rho_options"] = {'convergence_tolerance' : 1e-4,
                                       'rho_update_stop_iteration' : 4,
                                       'rho_update_start_iteration' : 1,
                                       'verbose' : False,
                                       }

        ph = mpisppy.opt.ph.PH(
            options,
            self.all_scenario_names,
            hydro.scenario_creator,
            hydro.scenario_denouement,
            all_nodenames=self.all_nodenames,
            scenario_creator_kwargs={"branching_factors": self.branching_factors},
            extensions = MultRhoUpdater,
        )
        conv, obj, tbound = ph.ph_main()
        obj2 = round_pos_sig(obj, 2)
        self.assertEqual(210, obj2)


# MultRhoUpdater


class Test_mutable_probability(unittest.TestCase):
    """ Mutable scenario probabilities on the ExtensiveForm (issue #797). """

    def setUp(self):
        import mpisppy.tests.examples.farmer as farmer
        from mpisppy.opt.ef import ExtensiveForm
        self.farmer = farmer
        self.ExtensiveForm = ExtensiveForm
        self.snames = ["scen0", "scen1", "scen2"]
        self.sck = {"num_scens": 3, "sense": pyo.minimize}

    def _pv(self, p_bad):
        rest = (1.0 - p_bad) / 2.0
        return {"scen0": p_bad, "scen1": rest, "scen2": rest}

    def _make_ef(self, mutable_probability=False):
        return self.ExtensiveForm(
            options={"solver": solver_name},
            all_scenario_names=self.snames,
            scenario_creator=self.farmer.scenario_creator,
            scenario_creator_kwargs=self.sck,
            mutable_probability=mutable_probability,
        )

    def _rebuild_obj(self, pv):
        # baked-in EF at the given probabilities (correctness oracle)
        scen_dict = {sn: self.farmer.scenario_creator(sn, **self.sck)
                     for sn in self.snames}
        for sn, scen in scen_dict.items():
            scen._mpisppy_probability = pv[sn]
        ef = sputils._create_EF_from_scen_dict(scen_dict, EF_name="oracle")
        solver = pyo.SolverFactory(solver_name)
        if '_persistent' in solver_name:
            solver.set_instance(ef)
        solver.solve(ef)
        return pyo.value(ef.EF_Obj)

    def test_mutable_prob_builds_param(self):
        ef = self._make_ef(mutable_probability=True)
        self.assertTrue(ef.mutable_probability)
        self.assertTrue(hasattr(ef.ef._mpisppy_model, "prob"))
        # defaults to the scenario_creator probabilities (uniform here)
        for sn in self.snames:
            self.assertAlmostEqual(pyo.value(ef.ef._mpisppy_model.prob[sn]),
                                   1.0 / 3.0)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_matches_rebuild_across_sweep(self):
        ef = self._make_ef(mutable_probability=True)
        for i, p_bad in enumerate([0.0, 0.2, 0.5, 0.9]):
            pv = self._pv(p_bad)
            ef.set_scenario_probabilities(pv)
            ef.solve_extensive_form(reuse_instance=(i > 0))
            self.assertAlmostEqual(ef.get_objective_value(),
                                   self._rebuild_obj(pv), places=4)

    @unittest.skipIf(not persistent_available,
                     "no persistent solver is available")
    def test_reuse_instance_loads_once(self):
        options = {"solver": persistent_solver_name}
        ef = self.ExtensiveForm(
            options=options, all_scenario_names=self.snames,
            scenario_creator=self.farmer.scenario_creator,
            scenario_creator_kwargs=self.sck, mutable_probability=True)
        calls = {"n": 0}
        orig = ef.solver.set_instance
        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        ef.solver.set_instance = counting
        for i, p_bad in enumerate([0.2, 0.5, 0.9]):
            ef.set_scenario_probabilities(self._pv(p_bad))
            ef.solve_extensive_form(reuse_instance=(i > 0))
        self.assertEqual(calls["n"], 1)

    def test_guards(self):
        ef = self._make_ef(mutable_probability=True)
        # non-mutable EF rejects the setter
        plain = self._make_ef(mutable_probability=False)
        with self.assertRaises(RuntimeError):
            plain.set_scenario_probabilities(self._pv(0.2))
        # unknown scenario name
        with self.assertRaises(KeyError):
            ef.set_scenario_probabilities({"nope": 0.5})
        # probabilities not summing to 1, and the failed call is transactional
        before = pyo.value(ef.ef._mpisppy_model.prob["scen0"])
        with self.assertRaises(ValueError):
            ef.set_scenario_probabilities({"scen0": before + 0.25})
        self.assertAlmostEqual(pyo.value(ef.ef._mpisppy_model.prob["scen0"]),
                               before)


class Test_mutable_probability_ph(unittest.TestCase):
    """ Mutable scenario probabilities on the PH path (issue #797, phase 2). """

    def setUp(self):
        import mpisppy.tests.examples.farmer as farmer
        self.farmer = farmer
        self.snames = ["scen0", "scen1", "scen2"]
        self.sck = {"num_scens": 3, "sense": pyo.minimize}

    def _pv(self, p0):
        rest = (1.0 - p0) / 2.0
        return {"scen0": p0, "scen1": rest, "scen2": rest}

    def _denouement(self, *a, **k):
        pass

    def _make_ph(self, iters=0):
        options = _get_ph_base_options()
        options["PHIterLimit"] = iters
        return mpisppy.opt.ph.PH(
            options, self.snames, self.farmer.scenario_creator,
            self._denouement, scenario_creator_kwargs=self.sck)

    def _ef_first_stage(self, pv):
        # solved EF at the given probabilities: the nonanticipative first-stage
        # DevotedAcreage (independent oracle; the EF path is verified in
        # Test_mutable_probability against a rebuild).
        scen_dict = {sn: self.farmer.scenario_creator(sn, **self.sck)
                     for sn in self.snames}
        for sn, scen in scen_dict.items():
            scen._mpisppy_probability = pv[sn]
        ef = sputils._create_EF_from_scen_dict(scen_dict, EF_name="oracle")
        solver = pyo.SolverFactory(solver_name)
        if '_persistent' in solver_name:
            solver.set_instance(ef)
        solver.solve(ef)
        block = getattr(ef, self.snames[0])
        return {c: pyo.value(block.DevotedAcreage[c])
                for c in block.DevotedAcreage}

    def _ph_first_stage(self, ph):
        # xbar is the consensus first-stage value; read it off any local scenario
        s = ph.local_scenarios[self.snames[0]]
        return {c: pyo.value(v) for c, v in
                zip(s.DevotedAcreage, s.DevotedAcreage.values())}

    def test_prob_coeff_refreshes(self):
        # Two-stage: prob_coeff["ROOT"] == _mpisppy_probability after an update.
        ph = self._make_ph(iters=0)
        for s in ph.local_scenarios.values():
            self.assertAlmostEqual(s._mpisppy_data.prob_coeff["ROOT"], 1.0 / 3.0)
        pv = self._pv(0.5)
        ph.set_scenario_probabilities(pv)
        for sname, s in ph.local_scenarios.items():
            self.assertAlmostEqual(s._mpisppy_probability, pv[sname])
            self.assertAlmostEqual(s._mpisppy_data.prob_coeff["ROOT"], pv[sname])

    def test_guards(self):
        ph = self._make_ph(iters=0)
        with self.assertRaises(KeyError):
            ph.set_scenario_probabilities({"nope": 0.5})
        with self.assertRaises(ValueError):
            ph.set_scenario_probabilities({"scen0": 0.9})  # sum != 1

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ph_matches_ef_oracle(self):
        # A fresh PH at each probability vector should converge to the EF
        # solution at that vector, whether the vector was set at construction
        # (uniform) or via set_scenario_probabilities before the first solve.
        for p0 in (1.0 / 3.0, 0.6):
            pv = self._pv(p0)
            ph = self._make_ph(iters=100)
            if abs(p0 - 1.0 / 3.0) > 1e-12:
                ph.set_scenario_probabilities(pv)
            ph.ph_main()
            got = self._ph_first_stage(ph)
            want = self._ef_first_stage(pv)
            for c in want:
                self.assertAlmostEqual(got[c], want[c], places=2)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ph_reuse_after_prob_change(self):
        # In-place probability change on an already-solved PH object: re-solving
        # must track the new probabilities (the probability-sensitivity use
        # case). The default reset_ph_duals=True breaks the stale consensus.
        ph = self._make_ph(iters=100)
        ph.ph_main()
        pv = self._pv(0.6)
        ph.set_scenario_probabilities(pv)
        ph.ph_main()
        got = self._ph_first_stage(ph)
        want = self._ef_first_stage(pv)
        for c in want:
            self.assertAlmostEqual(got[c], want[c], places=2)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ph_reuse_without_dual_reset_stays_stuck(self):
        # Documents the warm-start caveat: keeping stale W (reset_ph_duals=False)
        # after convergence leaves PH at the old consensus, wrong for new probs.
        ph = self._make_ph(iters=100)
        ph.ph_main()
        uniform = self._ph_first_stage(ph)
        pv = self._pv(0.6)
        ph.set_scenario_probabilities(pv, reset_ph_duals=False)
        ph.ph_main()
        stuck = self._ph_first_stage(ph)
        want = self._ef_first_stage(pv)
        # still at the uniform solution, not the (different) skewed optimum
        self.assertNotAlmostEqual(stuck["WHEAT0"], want["WHEAT0"], places=2)
        for c in uniform:
            self.assertAlmostEqual(stuck[c], uniform[c], places=2)


if __name__ == '__main__':
    unittest.main()
