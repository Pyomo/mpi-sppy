# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Provide some test for confidence intervals
"""

"""

import os
import tempfile
import numpy as np
import unittest
import subprocess
import importlib

import mpisppy.MPI as mpi

from mpisppy.tests.utils import get_solver, round_pos_sig
import mpisppy.utils.sputils as sputils
import mpisppy.tests.examples.aircond as aircond

import mpisppy.confidence_intervals.mmw_ci as MMWci
import mpisppy.confidence_intervals.zhat4xhat as zhat4xhat
import mpisppy.utils.amalgamator as ama
from mpisppy.utils.xhat_eval import Xhat_Eval
import mpisppy.confidence_intervals.seqsampling as seqsampling
import mpisppy.confidence_intervals.multi_seqsampling as multi_seqsampling
import mpisppy.confidence_intervals.confidence_config as confidence_config
import mpisppy.confidence_intervals.ciutils as ciutils
from mpisppy.utils import config
import pyomo.common.config as pyofig

__version__ = 0.4

solver_available, solver_name, persistent_available, persistent_solver_name= get_solver()
module_dir = os.path.dirname(os.path.abspath(__file__))

global_BFs = [4,3,2]

#*****************************************************************************
class Test_confint_aircond(unittest.TestCase):
    """ Test the confint code using aircond."""

    @classmethod
    def setUpClass(self):
        self.refmodelname ="mpisppy.tests.examples.aircond"  # amalgamator compatible
        # TBD: maybe this code should create the file
        self.xhatpath = "farmer_cyl_nonants.spy.npy"      

    def _get_base_options(self):
        # Base option has batch options
        # plus kwoptions to pass to kw_creator
        # and kwargs, which is the result of that.
        cfg = config.Config()
        cfg.quick_assign("EF_solver_name", str, solver_name)
        cfg.quick_assign("start_ups", bool, False)
        cfg.quick_assign("num_scens", int, 24)
        cfg.quick_assign("start_seed", int, 0)
        cfg.quick_assign("EF_mstage", bool, True)
        cfg.quick_assign("branching_factors", pyofig.ListOf(int), global_BFs)
        cfg.quick_assign("num_batches", int, 5)
        cfg.quick_assign("batch_size", int, 6)

        scenario_creator_kwargs = aircond.kw_creator(cfg)
        return cfg, scenario_creator_kwargs

    def _get_xhatEval_options(self):
        cfg, scenario_creator_kwargs = self._get_base_options()
        options = {"iter0_solver_options": None,
                 "iterk_solver_options": None,
                 "display_timing": False,
                 "solver_name": solver_name,
                 "verbose": False,
                 "solver_options":None}

        options.update(cfg)
        return options

    def _get_xhat_gen_options(self, BFs):
        num_scens = np.prod(BFs)
        scenario_names = aircond.scenario_names_creator(num_scens)
        xhat_gen_options = {"scenario_names": scenario_names,
                            "solver_name": solver_name,
                            "solver_options": None,
                            "branching_factors": BFs,
                            "mu_dev": 0,
                            "sigma_dev": 40,
                            "start_ups": False,
                            "start_seed": 0,
                            }
        return xhat_gen_options

    def _make_full_xhat(self, BFs):
        ndns = sputils.create_nodenames_from_branching_factors(BFs)
        retval = {i: [200.0,0] for i in ndns}
        return retval

        
    def setUp(self):
        self.xhat = {'ROOT': np.array([200.0, 0.0])}  # really xhat_1
        tmpxhat = tempfile.mkstemp(prefix="xhat",suffix=".npy")
        self.xhat_path =  tmpxhat[1]  # create an empty .npy file
        ciutils.write_xhat(self.xhat,self.xhat_path)


    def tearDown(self):
         os.remove(self.xhat_path)

        
    def _eval_creator(self):
        xh_options = self._get_xhatEval_options()
        
        MMW_options, scenario_creator_kwargs = self._get_base_options()
        branching_factors= global_BFs
        scen_count = np.prod(branching_factors)
        all_scenario_names = aircond.scenario_names_creator(scen_count)
        all_nodenames = sputils.create_nodenames_from_branching_factors(branching_factors)
        ev = Xhat_Eval(xh_options,
                       all_scenario_names,
                       aircond.scenario_creator,
                       scenario_denouement=None,
                       all_nodenames=all_nodenames,
                       scenario_creator_kwargs=scenario_creator_kwargs
                       )
        return ev


    def test_ama_creator(self):
        options, scenario_creator_kwargs = self._get_base_options()
        ama_object = ama.from_module(self.refmodelname,
                                     cfg=options,
                                     use_command_line=False)   
        
    def test_MMW_constructor(self):
        options, scenario_creator_kwargs = self._get_base_options()
        xhat = ciutils.read_xhat(self.xhat_path)

        MMW = MMWci.MMWConfidenceIntervals(self.refmodelname,
                          options,
                          xhat,
                          options['num_batches'], batch_size = options["batch_size"], start = options['num_scens'])

    def test_seqsampling_creator(self):
        BFs = [4, 3, 2]
        xhat_gen_options = self._get_xhat_gen_options(BFs)
        optionsBM = config.Config()
        confidence_config.confidence_config(optionsBM)
        confidence_config.sequential_config(optionsBM)
        optionsBM.quick_assign('BM_h', float, 0.2)
        optionsBM.quick_assign('BM_hprime', float, 0.015,)
        optionsBM.quick_assign('BM_eps', float, 0.5,)
        optionsBM.quick_assign('BM_eps_prime', float, 0.4,)
        optionsBM.quick_assign("BM_p", float, 0.2)
        optionsBM.quick_assign("BM_q", float, 1.2)
        optionsBM.quick_assign("solver_name", str, solver_name)
        optionsBM.quick_assign("stopping", str, "BM")  # TBD use this and drop stopping_criterion from the constructor
        optionsBM.quick_assign("solving_type", str, "EF_mstage")
        optionsBM.quick_assign("branching_factors", pyofig.ListOf(int), global_BFs)

        seqsampling.SeqSampling(self.refmodelname,
                                aircond.xhat_generator_aircond,
                                optionsBM,
                                stochastic_sampling=False,
                                stopping_criterion="BM",
                                solving_type="EF_mstage",
                                )

        
    def test_indepscens_seqsampling_creator(self):
        options, scenario_creator_kwargs = self._get_base_options()
        branching_factors= global_BFs
        xhat_gen_options = self._get_xhat_gen_options(branching_factors)
        
        # We want a very small instance for testing on GitHub.
        optionsBM = config.Config()
        confidence_config.confidence_config(optionsBM)
        confidence_config.sequential_config(optionsBM)
        optionsBM.quick_assign('BM_h', float, 1.75)
        optionsBM.quick_assign('BM_hprime', float, 0.5,)
        optionsBM.quick_assign('BM_eps', float, 0.2,)
        optionsBM.quick_assign('BM_eps_prime', float, 0.1,)
        optionsBM.quick_assign("BM_p", float, 0.1)
        optionsBM.quick_assign("BM_q", float, 1.2)
        optionsBM.quick_assign("solver_name", str, solver_name)
        optionsBM.quick_assign("stopping", str, "BM")  # TBD use this and drop stopping_criterion from the constructor
        optionsBM.quick_assign("solving_type", str, "EF_mstage")
        optionsBM.quick_assign("branching_factors", pyofig.ListOf(int), global_BFs)

        multi_seqsampling.IndepScens_SeqSampling(self.refmodelname,
                                aircond.xhat_generator_aircond,
                                optionsBM,
                                stochastic_sampling=False,
                                stopping_criterion="BM",
                                solving_type="EF_mstage",
                                )


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ama_running(self):
        options, scenario_creator_kwargs = self._get_base_options()
        ama_object = ama.from_module(self.refmodelname,
                                     options, use_command_line=False)
        ama_object.run()
        obj = round_pos_sig(ama_object.EF_Obj,2)
        self.assertEqual(obj, 970.)


    @unittest.skipIf(not solver_available,
                     "no solver is available")      
    def test_xhat_eval_creator(self):
        ev = self._eval_creator()

        
    @unittest.skipIf(not solver_available,
                     "no solver is available")      
    def test_xhat_eval_evaluate(self):
        ev = self._eval_creator()

        base_options, scenario_creator_kwargs = self._get_base_options()
        branching_factors= global_BFs
        full_xhat = self._make_full_xhat(branching_factors)
        obj = round_pos_sig(ev.evaluate(full_xhat),2)
        self.assertEqual(obj, 1000.0)  # rebaselined feb 2022

    @unittest.skipIf(not solver_available,
                     "no solver is available")  
    def test_xhat_eval_evaluate_one(self):
        ev = self._eval_creator()
        options, scenario_creator_kwargs = self._get_base_options()
        branching_factors = global_BFs
        full_xhat = self._make_full_xhat(branching_factors)

        num_scens = np.prod(branching_factors)
        scenario_creator_kwargs['num_scens'] = num_scens
        all_scenario_names = aircond.scenario_names_creator(num_scens)

        k = all_scenario_names[0]
        obj = ev.evaluate_one(full_xhat,k,ev.local_scenarios[k])
        obj = round_pos_sig(obj,2)
        self.assertEqual(obj, 1100.0) # rebaselined feb 2022

    @unittest.skipIf(not solver_available,
                     "no solver is available")  
    def test_MMW_running(self):
        options, scenario_creator_kwargs = self._get_base_options()
        xhat = ciutils.read_xhat(self.xhat_path)
        MMW = MMWci.MMWConfidenceIntervals(self.refmodelname,
                                           options,
                                           xhat,
                                           options['num_batches'],
                                           batch_size = options["batch_size"],
                                           start = options['num_scens'])
        r = MMW.run() 
        s = round_pos_sig(r['std'],2)
        bound = round_pos_sig(r['gap_inner_bound'],2)
        self.assertEqual((s,bound), (33.0, 130.0))

   
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_gap_estimators(self):
        options, scenario_creator_kwargs = self._get_base_options()
        branching_factors= global_BFs
        scen_count = np.prod(branching_factors)
        scenario_names = aircond.scenario_names_creator(scen_count, start=1000)
        sample_options = {"seed": 0,
                          "branching_factors": global_BFs}
        estim = ciutils.gap_estimators(self.xhat,
                                       self.refmodelname,
                                       solving_type="EF_mstage",
                                       sample_options=sample_options,
                                       cfg = options,
                                       solver_name=solver_name,
                                       scenario_names=scenario_names,
                                       )
        G = estim['G']
        s = estim['s']
        G,s = round_pos_sig(G,3),round_pos_sig(s,3)
        #self.assertEqual((G,s), (69.6, 53.8))  # rebaselined April 2022
        # April 2022: s seems to depend a little on the solver. TBD: look into this
        self.assertEqual(G, 69.6)

    
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_indepscens_seqsampling_running(self):
        options, scenario_creator_kwargs = self._get_base_options()
        branching_factors= global_BFs
        xhat_gen_options = self._get_xhat_gen_options(branching_factors)
        
        # We want a very small instance for testing on GitHub.
        optionsBM = config.Config()
        confidence_config.confidence_config(optionsBM)
        confidence_config.sequential_config(optionsBM)
        optionsBM.quick_assign('BM_h', float, 1.75)
        optionsBM.quick_assign('BM_hprime', float, 0.5,)
        optionsBM.quick_assign('BM_eps', float, 0.2,)
        optionsBM.quick_assign('BM_eps_prime', float, 0.1,)
        optionsBM.quick_assign("BM_p", float, 0.1)
        optionsBM.quick_assign("BM_q", float, 1.2)
        optionsBM.quick_assign("solver_name", str, solver_name)
        optionsBM.quick_assign("stopping", str, "BM")  # TBD use this and drop stopping_criterion from the constructor
        optionsBM.quick_assign("solving_type", str, "EF_mstage")
        optionsBM.quick_assign("EF_mstage", bool, True)   # TBD: we should not need both
        optionsBM.quick_assign("start_seed", str, 0)
        optionsBM.quick_assign("branching_factors", pyofig.ListOf(int), [4, 3, 2])
        optionsBM.quick_assign("xhat_gen_kwargs", dict, xhat_gen_options)

        seq_pb = multi_seqsampling.IndepScens_SeqSampling(self.refmodelname,
                                                          aircond.xhat_generator_aircond,
                                                          optionsBM,
                                                          stochastic_sampling=False,
                                                          stopping_criterion="BM",
                                                          solving_type="EF_mstage",
                                                          )
        x = seq_pb.run(maxit=50)
        T = x['T']
        ub = round_pos_sig(x['CI'][1],2)
        self.assertEqual((T,ub), (15, 180.0))


if __name__ == '__main__':
    unittest.main()
    
