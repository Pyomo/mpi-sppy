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

import pyomo.environ as pyo
import mpisppy.MPI as mpi

from mpisppy.tests.utils import get_solver, round_pos_sig
import mpisppy.tests.examples.farmer as farmer

import mpisppy.confidence_intervals.mmw_ci as MMWci
import mpisppy.confidence_intervals.zhat4xhat as zhat4xhat
import mpisppy.utils.amalgamator as ama
from mpisppy.utils.xhat_eval import Xhat_Eval
import mpisppy.confidence_intervals.seqsampling as seqsampling
import mpisppy.confidence_intervals.ciutils as ciutils
from mpisppy.utils import config
import mpisppy.confidence_intervals.confidence_config as confidence_config

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

__version__ = 0.6

solver_available, solver_name, persistent_available, persistent_solver_name= get_solver()
module_dir = os.path.dirname(os.path.abspath(__file__))


#*****************************************************************************
class Test_confint_farmer(unittest.TestCase):
    """ Test the confint code using farmer."""

    @classmethod
    def setUpClass(self):
        self.refmodelname ="mpisppy.tests.examples.farmer"
        self.arefmodelname ="mpisppy.tests.examples.farmer"  # amalgamator compatible


    def _get_base_options(self):
        cfg = config.Config()
        cfg.quick_assign("EF_solver_name", str, solver_name)
        cfg.quick_assign("use_integer", bool, False)
        cfg.quick_assign("crops_multiplier", int, 1)
        cfg.quick_assign("num_scens", int, 12)
        cfg.quick_assign("EF_2stage", bool, True)
        cfg.quick_assign("num_batches", int, 2)
        cfg.quick_assign("batch_size", int, 10)
        scenario_creator_kwargs = farmer.kw_creator(cfg)
        cfg.quick_assign('kwargs', dict,scenario_creator_kwargs)
        return cfg

    def _get_xhatEval_options(self):
        options = {"iter0_solver_options": None,
                 "iterk_solver_options": None,
                 "display_timing": False,
                 "solver_name": solver_name,
                 "verbose": False,
                 "solver_options":None}
        return options

    def setUp(self):
        self.xhat = {'ROOT': np.array([74.0,245.0,181.0])}
        tmpxhat = tempfile.mkstemp(prefix="xhat",suffix=".npy")
        self.xhat_path =  tmpxhat[1]  # create an empty .npy file
        ciutils.write_xhat(self.xhat,self.xhat_path)

    def tearDown(self):
        os.remove(self.xhat_path)

    def test_MMW_constructor(self):
        cfg = self._get_base_options()
        xhat = ciutils.read_xhat(self.xhat_path)

        MMW = MMWci.MMWConfidenceIntervals(self.refmodelname,
                          cfg,
                          xhat,
                          cfg['num_batches'], batch_size = cfg["batch_size"], start = cfg['num_scens'])
    
    def test_xhat_read_write(self):
        path = tempfile.mkstemp(prefix="xhat",suffix=".npy")[1]
        ciutils.write_xhat(self.xhat,path=path)
        x = ciutils.read_xhat(path, delete_file=True)
        self.assertEqual(list(x['ROOT']), list(self.xhat['ROOT']))
    
    def test_xhat_read_write_txt(self):
        path = tempfile.mkstemp(prefix="xhat",suffix=".npy")[1]
        ciutils.writetxt_xhat(self.xhat,path=path)
        x = ciutils.readtxt_xhat(path, delete_file=True)
        self.assertEqual(list(x['ROOT']), list(self.xhat['ROOT']))
    
    def test_ama_creator(self):
        cfg = self._get_base_options()
        ama_options = cfg()
        ama_options.quick_assign("EF_2stage", bool, True)
        ama_object = ama.from_module(self.refmodelname,
                                     cfg=ama_options,
                                     use_command_line=False)   
        
    def test_seqsampling_creator(self):
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
        optionsBM.quick_assign("solving_type", str, "EF_2stage")
        seqsampling.SeqSampling("mpisppy.tests.examples.farmer",
                                seqsampling.xhat_generator_farmer,
                                optionsBM,
                                stochastic_sampling=False,
                                stopping_criterion="BM",
                                solving_type="EF_2stage",
        )


    def test_pyomo_opt_sense(self):
        cfg = self._get_base_options()
        module = self.refmodelname
        ciutils.pyomo_opt_sense(module, cfg)
        assert cfg.pyomo_opt_sense == pyo.minimize  # minimize is 1


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ama_running(self):
        cfg = self._get_base_options()
        ama_options = cfg()
        ama_options.quick_assign("EF_2stage", bool, True)
        ama_object = ama.from_module(self.refmodelname,
                                     ama_options, use_command_line=False)
        ama_object.run()
        obj = round_pos_sig(ama_object.EF_Obj,2)
        self.assertEqual(obj, -130000)
    

    @unittest.skipIf(not solver_available,
                     "no solver is available")      
    def test_xhat_eval_creator(self):
        options = self._get_xhatEval_options()
        
        MMW_options = self._get_base_options()
        scenario_creator_kwargs = MMW_options['kwargs']
        scenario_creator_kwargs['num_scens'] = MMW_options['batch_size']
        ev = Xhat_Eval(options,
                       farmer.scenario_names_creator(100),
                       farmer.scenario_creator,
                       scenario_denouement=None,
                       scenario_creator_kwargs=scenario_creator_kwargs
                       )
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")      
    def test_xhat_eval_evaluate(self):
        options = self._get_xhatEval_options()
        MMW_options = self._get_base_options()
        scenario_creator_kwargs = MMW_options['kwargs']
        scenario_creator_kwargs['num_scens'] = MMW_options['batch_size']
        ev = Xhat_Eval(options,
                   farmer.scenario_names_creator(100),
                   farmer.scenario_creator,
                   scenario_denouement=None,
                   scenario_creator_kwargs=scenario_creator_kwargs
                   )
        
        xhat = ciutils.read_xhat(self.xhat_path)
        obj = round_pos_sig(ev.evaluate(xhat),2)
        self.assertEqual(obj, -1300000.0)
 
    @unittest.skipIf(not solver_available,
                     "no solver is available")  
    def test_xhat_eval_evaluate_one(self):
        options = self._get_xhatEval_options()
        MMW_options = self._get_base_options()
        xhat = ciutils.read_xhat(self.xhat_path)
        scenario_creator_kwargs = MMW_options['kwargs']
        scenario_creator_kwargs['num_scens'] = MMW_options['batch_size']
        scenario_names = farmer.scenario_names_creator(100)
        ev = Xhat_Eval(options,
                   scenario_names,
                   farmer.scenario_creator,
                   scenario_denouement=None,
                   scenario_creator_kwargs=scenario_creator_kwargs
                   )
        k = scenario_names[0]
        obj = ev.evaluate_one(xhat,k,ev.local_scenarios[k])
        obj = round_pos_sig(obj,2)
        self.assertEqual(obj, -48000.0)
      
    @unittest.skipIf(not solver_available,
                     "no solver is available")  
    def test_MMW_running(self):
        cfg = self._get_base_options()
        xhat = ciutils.read_xhat(self.xhat_path)
        MMW = MMWci.MMWConfidenceIntervals(self.refmodelname,
                                           cfg,
                                           xhat,
                                           cfg['num_batches'],
                                           batch_size = cfg["batch_size"],
                                           start = cfg['num_scens'])
        r = MMW.run() 
        s = round_pos_sig(r['std'],2)
        bound = round_pos_sig(r['gap_inner_bound'],2)
        self.assertEqual((s,bound), (1.5,96.0))
   
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_gap_estimators(self):
        scenario_names = farmer.scenario_names_creator(50,start=1000)
        estim = ciutils.gap_estimators(self.xhat,
                                       self.refmodelname,
                                       cfg=self._get_base_options(),
                                       solver_name=solver_name,
                                       scenario_names=scenario_names,
                                       )
        G = estim['G']
        s = estim['s']
        G,s = round_pos_sig(G,3),round_pos_sig(s,3)
        self.assertEqual((G,s), (456.0,944.0))
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    @unittest.skipIf(True, "too big for community solvers")
    def test_seqsampling_running(self):
        #We need a very small instance for testing on GitHub.
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
        optionsBM.quick_assign("solving_type", str, "EF_2stage")
        seq_pb = seqsampling.SeqSampling("mpisppy.tests.examples.farmer",
                                         seqsampling.xhat_generator_farmer,
                                         optionsBM,
                                         stochastic_sampling=False,
                                         stopping_criterion="BM",
                                         solving_type="EF-2stage",
        )
        x = seq_pb.run(maxit=50)
        T = x['T']
        ub = round_pos_sig(x['CI'][1],2)
        self.assertEqual((T,ub), (3, 13.0))


if __name__ == '__main__':
    unittest.main()
    
