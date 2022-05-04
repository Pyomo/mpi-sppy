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
import mpisppy.confidence_intervals.ciutils as ciutils

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

__version__ = 0.21

solver_available, solvername, persistent_available, persistentsolvername= get_solver()
module_dir = os.path.dirname(os.path.abspath(__file__))

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
        kwoptions = { "EF_solver_name": solvername,
                    "start_ups": False,
                    "branching_factors": [4, 3, 2],
                    "num_scens": 24,
                    "start_seed": 0,
                    "EF-mstage": True}
        Baseoptions =  {"from_Baseoptions": True,
                        "num_batches": 5,
                        "batch_size": 6,
                        "kwoptions":kwoptions}
        scenario_creator_kwargs = aircond.kw_creator(kwoptions)
        Baseoptions['kwargs'] = scenario_creator_kwargs
        return Baseoptions

    def _get_xhatEval_options(self):
        Baseoptions = self._get_base_options()
        options = {"iter0_solver_options": None,
                 "iterk_solver_options": None,
                 "display_timing": False,
                 "solvername": solvername,
                 "verbose": False,
                 "solver_options":None}
        options.update(Baseoptions["kwoptions"])
        return options

    def _get_xhat_gen_options(self, BFs):
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
        options = self._get_xhatEval_options()
        
        MMW_options = self._get_base_options()
        branching_factors= MMW_options['kwoptions']['branching_factors']
        scen_count = np.prod(branching_factors)
        scenario_creator_kwargs = MMW_options['kwargs']
        ###scenario_creator_kwargs['num_scens'] = MMW_options['batch_size']
        all_scenario_names = aircond.scenario_names_creator(scen_count)
        all_nodenames = sputils.create_nodenames_from_branching_factors(branching_factors)
        ev = Xhat_Eval(options,
                       all_scenario_names,
                       aircond.scenario_creator,
                       scenario_denouement=None,
                       all_nodenames=all_nodenames,
                       scenario_creator_kwargs=scenario_creator_kwargs
                       )
        return ev

    def test_ama_creator(self):
        options = self._get_base_options()
        ama_options = options['kwoptions']
        ama_object = ama.from_module(self.refmodelname,
                                     options=ama_options,
                                     use_command_line=False)   
        
    def test_MMW_constructor(self):
        options = self._get_base_options()
        xhat = ciutils.read_xhat(self.xhat_path)

        MMW = MMWci.MMWConfidenceIntervals(self.refmodelname,
                          options['kwoptions'],
                          xhat,
                          options['num_batches'], batch_size = options["batch_size"], start = options['kwoptions']['num_scens'])

    def test_seqsampling_creator(self):
        BFs = [4, 3, 2]
        xhat_gen_options = self._get_xhat_gen_options(BFs)
        optionsBM = {'h':0.2,
                     'hprime':0.015, 
                     'eps':0.5, 
                     'epsprime':0.4, 
                     "p":2,
                     "q":1.2,
                     "solvername":solvername,
                     "xhat_gen_options": xhat_gen_options,
                     "branching_factors": BFs,
                     }
        seqsampling.SeqSampling(self.refmodelname,
                                aircond.xhat_generator_aircond,
                                optionsBM,
                                stochastic_sampling=False,
                                stopping_criterion="BM",
                                solving_type="EF-mstage",
                                )

        
    def test_indepscens_seqsampling_creator(self):
        options = self._get_base_options()
        branching_factors= options['kwoptions']['branching_factors']
        xhat_gen_options = self._get_xhat_gen_options(branching_factors)
        
        # We want a very small instance for testing on GitHub.
        optionsBM = {'h':1.75,
                     'hprime':0.5, 
                     'eps':0.2, 
                     'epsprime':0.1, 
                     "p":0.1,
                     "q":1.2,
                     "branching_factors": branching_factors,
                     "xhat_gen_options": xhat_gen_options,
                     }

        multi_seqsampling.IndepScens_SeqSampling(self.refmodelname,
                                aircond.xhat_generator_aircond,
                                optionsBM,
                                stochastic_sampling=False,
                                stopping_criterion="BM",
                                solving_type="EF-mstage",
                                )


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ama_running(self):
        options = self._get_base_options()
        ama_options = options['kwoptions']
        ama_object = ama.from_module(self.refmodelname,
                                     ama_options, use_command_line=False)
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

        base_options = self._get_base_options()
        branching_factors= base_options['kwoptions']['branching_factors']
        full_xhat = self._make_full_xhat(branching_factors)
        obj = round_pos_sig(ev.evaluate(full_xhat),2)
        self.assertEqual(obj, 1000.0)  # rebaselined feb 2022

    @unittest.skipIf(not solver_available,
                     "no solver is available")  
    def test_xhat_eval_evaluate_one(self):
        ev = self._eval_creator()
        options = self._get_base_options()
        branching_factors= options['kwoptions']['branching_factors']
        full_xhat = self._make_full_xhat(branching_factors)

        num_scens = np.prod(branching_factors)
        scenario_creator_kwargs = options['kwargs']
        scenario_creator_kwargs['num_scens'] = num_scens
        all_scenario_names = aircond.scenario_names_creator(num_scens)

        k = all_scenario_names[0]
        obj = ev.evaluate_one(full_xhat,k,ev.local_scenarios[k])
        obj = round_pos_sig(obj,2)
        self.assertEqual(obj, 1100.0) # rebaselined feb 2022

    @unittest.skipIf(not solver_available,
                     "no solver is available")  
    def test_MMW_running(self):
        options = self._get_base_options()
        xhat = ciutils.read_xhat(self.xhat_path)
        MMW = MMWci.MMWConfidenceIntervals(self.refmodelname,
                                        options['kwoptions'],
                                        xhat,
                                        options['num_batches'],
                                        batch_size = options["batch_size"],
                                         start = options['kwoptions']['num_scens'])
        r = MMW.run() 
        s = round_pos_sig(r['std'],2)
        bound = round_pos_sig(r['gap_inner_bound'],2)
        self.assertEqual((s,bound), (33.0, 130.0))

   
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_gap_estimators(self):
        options = self._get_base_options()
        branching_factors= options['kwoptions']['branching_factors']
        scen_count = np.prod(branching_factors)
        scenario_names = aircond.scenario_names_creator(scen_count, start=1000)
        sample_options = {"seed": 0,
                          "branching_factors": options['kwoptions']['branching_factors']}
        estim = ciutils.gap_estimators(self.xhat,
                                       self.refmodelname,
                                       solving_type="EF-mstage",
                                       sample_options=sample_options,
                                       options = options,
                                       solvername=solvername,
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
        options = self._get_base_options()
        branching_factors= options['kwoptions']['branching_factors']
        xhat_gen_options = self._get_xhat_gen_options(branching_factors)
        
        # We want a very small instance for testing on GitHub.
        optionsBM = {'h':1.75,
                     'hprime':0.5, 
                     'eps':0.2, 
                     'epsprime':0.1, 
                     "p":0.1,
                     "q":1.2,
                     "branching_factors": branching_factors,
                     "xhat_gen_options": xhat_gen_options,
                     "start_ups": False,
                     "start_seed": 0,
                     "solvername":solvername,
                     }
        seq_pb = multi_seqsampling.IndepScens_SeqSampling(self.refmodelname,
                                                          aircond.xhat_generator_aircond,
                                                          optionsBM,
                                                          stochastic_sampling=False,
                                                          stopping_criterion="BM",
                                                          solving_type="EF-mstage",
                                                          )
        x = seq_pb.run(maxit=50)
        T = x['T']
        ub = round_pos_sig(x['CI'][1],2)
        self.assertEqual((T,ub), (15, 180.0))


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_zhat4xhat(self):
        cmdline = [self.refmodelname, self.xhat_path, "--solver-name",
                   solvername, "--branching-factors",  "4" ]   # mainly using defaults
        parser = zhat4xhat._parser_setup()
        aircond.inparser_adder(parser)
        args = parser.parse_args(cmdline)
        # I couldnt't figure out how to get the branching factors parsed from this list, so
        args.branching_factors = [4, 3, 2]
                      
        model_module = importlib.import_module(self.refmodelname)
        zhatbar, eps_z = zhat4xhat._main_body(args, model_module)

        z2 = round_pos_sig(zhatbar, 2)
        self.assertEqual(z2, 850.)
        e2 = round_pos_sig(eps_z, 2)
        self.assertEqual(e2, 68.)
        #print(f"*** {z2 =} {e2 =}")

if __name__ == '__main__':
    unittest.main()
    
