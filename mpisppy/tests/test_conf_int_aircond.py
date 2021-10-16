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

import mpi4py.MPI as mpi

from mpisppy.tests.test_utils import get_solver, round_pos_sig
import mpisppy.tests.examples.aircond_submodels as aircond


import mpisppy.confidence_intervals.mmw_ci as MMWci
import mpisppy.confidence_intervals.zhat4xhat as zhat4xhat
import mpisppy.utils.amalgomator as ama
from mpisppy.utils.xhat_eval import Xhat_Eval
import mpisppy.confidence_intervals.seqsampling as seqsampling
import mpisppy.confidence_intervals.ciutils as ciutils

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

__version__ = 0.1

solver_available, solvername, persistent_available, persistentsolvername= get_solver()
module_dir = os.path.dirname(os.path.abspath(__file__))

#*****************************************************************************
class Test_confint_aircond(unittest.TestCase):
    """ Test the confint code using aircond."""

    @classmethod
    def setUpClass(self):
        self.refmodelname ="mpisppy.tests.examples.aircond_submodels"  # amalgomator compatible

    def _get_base_options(self):
        options = { "EF_solver_name": solvername,
                    "start_ups": False,
                    "branching_factors": [4, 3, 2],
                    "num_scens": 12,
                    "EF-mstage": True}
        Baseoptions =  {"num_batches": 5,
                        "batch_size": 6,
                        "opt":options}
        scenario_creator_kwargs = aircond.kw_creator(options)
        Baseoptions['kwargs'] = scenario_creator_kwargs
        return Baseoptions

    def _get_xhatEval_options(self):
        options = {"iter0_solver_options": None,
                 "iterk_solver_options": None,
                 "display_timing": False,
                 "solvername": solvername,
                 "verbose": False,
                 "solver_options":None}
        return options

    def _make_xhat_gen_options(self, BFs):
        num_scens = np.prod(BFs)
        scenario_names = aircond.scenario_names_creator(num_scens)
        xhat_gen_options = {"scenario_names": scenario_names,
                            "solvername": solvername,
                            "solver_options": None,
                            "branching_factors": BFs,
                            "mudev": 0,
                            "sigmadev": 40,
                            "start_ups": False,
                            "start_seed": 0,
                            }
        return xhat_gen_options
    
    def setUp(self):
        self.xhat = {'ROOT': np.array([200.0, 0.0])}
        tmpxhat = tempfile.mkstemp(prefix="xhat",suffix=".npy")
        self.xhat_path =  tmpxhat[1]  # create an empty .npy file
        ciutils.write_xhat(self.xhat,self.xhat_path)

    def tearDown(self):
        os.remove(self.xhat_path)

    def test_ama_creator(self):
        options = self._get_base_options()
        ama_options = options['opt']
        ama_object = ama.from_module(self.refmodelname,
                                     options=ama_options,
                                     use_command_line=False)   
        
    def test_MMW_constructor(self):
        options = self._get_base_options()
        xhat = ciutils.read_xhat(self.xhat_path)

        MMW = MMWci.MMWConfidenceIntervals(self.refmodelname,
                          options['opt'],
                          xhat,
                          options['num_batches'], batch_size = options["batch_size"], start = options['opt']['num_scens'])

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

    def test_seqsampling_creator(self):
        BFs = [4, 3, 2]
        xhat_gen_options = self._make_xhat_gen_options(BFs)
        optionsBM = {'h':0.2,
                     'hprime':0.015, 
                     'eps':0.5, 
                     'epsprime':0.4, 
                     "p":2,
                     "q":1.2,
                     "solvername":solvername,
                     "xhat_gen_options": xhat_gen_options,
                     }
        seqsampling.SeqSampling(self.refmodelname,
                                aircond.xhat_generator_aircond,
                                optionsBM,
                                stochastic_sampling=False,
                                stopping_criterion="BM",
                                )

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ama_running(self):
        options = self._get_base_options()
        ama_options = options['opt']
        ama_object = ama.from_module(self.refmodelname,
                                     ama_options, use_command_line=False)
        ama_object.run()
        obj = round_pos_sig(ama_object.EF_Obj,2)
        self.assertEqual(obj, 720.)
    

    @unittest.skipIf(not solver_available,
                     "no solver is available")      
    def test_xhat_eval_creator(self):
        options = self._get_xhatEval_options()
        
        MMW_options = self._get_base_options()
        scenario_creator_kwargs = MMW_options['kwargs']
        scenario_creator_kwargs['num_scens'] = MMW_options['batch_size']
        all_names = aircond.scenario_names_creator(MMW_options['batch_size'])  # ???
        ev = Xhat_Eval(options,
                       all_names,
                       aircond.scenario_creator,
                       scenario_denouement=None,
                       scenario_creator_kwargs=scenario_creator_kwargs
                       )

    """    
    @unittest.skipIf(not solver_available,
                     "no solver is available")      
    def test_xhat_eval_evaluate(self):
        options = self._get_xhatEval_options()
        MMW_options = self._get_base_options()
        scenario_creator_kwargs = MMW_options['kwargs']
        scenario_creator_kwargs['num_scens'] = MMW_options['batch_size']
        ev = Xhat_Eval(options,
                   aircond.scenario_names_creator(100),
                   aircond.scenario_creator,
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
        scenario_names = aircond.scenario_names_creator(100)
        ev = Xhat_Eval(options,
                   scenario_names,
                   aircond.scenario_creator,
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
        options = self._get_base_options()
        xhat = ciutils.read_xhat(self.xhat_path)
        MMW = MMWci.MMWConfidenceIntervals(self.refmodelname,
                                        options['opt'],
                                        xhat,
                                        options['num_batches'],
                                        batch_size = options["batch_size"],
                                         start = options['opt']['num_scens'])
        r = MMW.run() 
        s = round_pos_sig(r['std'],2)
        bound = round_pos_sig(r['gap_inner_bound'],2)
        self.assertEqual((s,bound), (1.5,96.0))
       
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_gap_estimators(self):
        scenario_names = aircond.scenario_names_creator(50,start=1000)
        estim = ciutils.gap_estimators(self.xhat,
                                       self.refmodelname,
                                       solvername=solvername,
                                       scenario_names=scenario_names,
                                       )
        G = estim['G']
        s = estim['s']
        G,s = round_pos_sig(G,3),round_pos_sig(s,3)
        self.assertEqual((G,s), (110.0,426.0))
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_seqsampling_running(self):
            # We want a very small instance for testing on GitHub.
            optionsBM = {'h':1.75,
                         'hprime':0.5, 
                         'eps':0.2, 
                         'epsprime':0.1, 
                         "p":0.1,
                         "q":1.2,
                         "solvername":solvername,
                         "xhat_gen_options": self.xhat_gen_options,
                         }
            seq_pb = seqsampling.SeqSampling(self.refmodelname,
                                            aircond.xhat_generator_aircond,
                                            optionsBM,
                                            stochastic_sampling=False,
                                            stopping_criterion="BM",
                                            )
            x = seq_pb.run(maxit=50)
            T = x['T']
            ub = round_pos_sig(x['CI'][1],2)
            self.assertEqual((T,ub), (1,7400.0))


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_zhat4xhat(self):
        cmdline = [self.refmodelname, self.xhat_path, "--solver-name", solvername, "--branching-factors", [4, 3, 2]]  # mainly defaults
        parser = zhat4xhat._parser_setup()
        args = parser.parse_args(cmdline)
        model_module = importlib.import_module(self.refmodelname)
        zhatbar, eps_z = zhat4xhat._main_body(args, model_module)

        z2 = round_pos_sig(zhatbar, 2)
        self.assertEqual(z2, 360.)
        e2 = round_pos_sig(eps_z, 2)
        self.assertEqual(e2, 11.)
        print(f"*** {z2 =} {e2 =}")
"""
if __name__ == '__main__':
    unittest.main()
    
