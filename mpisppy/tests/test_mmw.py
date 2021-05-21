# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Provide some test for confidence intervals
"""

"""

import os
import tempfile
import numpy as np
import unittest


import mpi4py.MPI as mpi

from mpisppy.tests.test_utils import get_solver, round_pos_sig
import mpisppy.tests.examples.farmer as farmer 


import mpisppy.confidence_intervals.mmw_ci as MMWci
import mpisppy.utils.amalgomator as ama
from mpisppy.utils.xhat_eval import Xhat_Eval

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

__version__ = 0.2

solver_available,solvername, persistent_available, persistentsolvername= get_solver()

def _get_base_options():
    options = { "EF_solver_name": solvername,
                     "use_integer": False,
                     "crops_multiplier": 1,
                     'num_scens': 12,
                     'start': 0}
    Baseoptions =  {"num_batches": 2,
                     "batch_size": 10,
                     "opt":options}
    scenario_creator_kwargs = farmer.kw_creator(options)
    Baseoptions['kwargs'] = scenario_creator_kwargs
    return Baseoptions

def _get_xhatEval_options():
    options = {"iter0_solver_options": None,
             "iterk_solver_options": None,
             "display_timing": False,
             "solvername": solvername,
             "verbose": False,
             "solver_options":None}
    return options



refmodelname ="mpisppy.tests.examples.farmer"
#*****************************************************************************
class Test_MMW_farmer(unittest.TestCase):
    """ Test the MMWci code using farmer."""

    def setUp(self):
        self.xhat = {'ROOT': np.array([74.0,245.0,181.0])}
        tmpxhat = tempfile.mkstemp(prefix="xhat",suffix=".npy")
        self.xhat_path =  tmpxhat[1]#create an empty .npy file
        
        MMWci.write_xhat(self.xhat,self.xhat_path)

    def tearDown(self):
        os.remove(self.xhat_path)

    def test_MMW_constructor(self):
        options = _get_base_options()
        xhat = MMWci.read_xhat(self.xhat_path)

        MMW = MMWci.MMWConfidenceIntervals(refmodelname,
                          options['opt'],
                          xhat,
                          options['num_batches'])
    
    def test_xhat_read_write(self):
        path = tempfile.mkstemp(prefix="xhat",suffix=".npy")[1]
        MMWci.write_xhat(self.xhat,path=path)
        x = MMWci.read_xhat(path, delete_file=True)
        self.assertEqual(list(x['ROOT']), list(self.xhat['ROOT']))
    
    def test_xhat_read_write_txt(self):
        path = tempfile.mkstemp(prefix="xhat",suffix=".npy")[1]
        MMWci.writetxt_xhat(self.xhat,path=path)
        x = MMWci.readtxt_xhat(path, delete_file=True)
        self.assertEqual(list(x['ROOT']), list(self.xhat['ROOT']))
        
        
    
    def test_ama_creator(self):
        options = _get_base_options()
        ama_options = {"EF-2stage": True,}
        ama_options.update(options['opt'])
        ama_object = ama.from_module(refmodelname,
                                     options=ama_options,
                                     use_command_line=False)
        #ama_object.run()

        

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ama_running(self):
        options = _get_base_options()
        ama_options = {"EF-2stage": True}
        ama_options.update(options['opt'])
        ama_object = ama.from_module(refmodelname,
                                     ama_options, use_command_line=False)
        ama_object.run()
        obj = round_pos_sig(ama_object.EF_Obj,2)
        self.assertEqual(obj, -130000)
        #TODO : check if we get the right result using round_pos_sig
    

    @unittest.skipIf(not solver_available,
                     "no solver is available")      
    def test_xhat_eval_creator(self):
        options = _get_xhatEval_options()
        
        MMW_options = _get_base_options()
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
        options = _get_xhatEval_options()
        MMW_options = _get_base_options()
        scenario_creator_kwargs = MMW_options['kwargs']
        scenario_creator_kwargs['num_scens'] = MMW_options['batch_size']
        ev = Xhat_Eval(options,
                   farmer.scenario_names_creator(100),
                   farmer.scenario_creator,
                   scenario_denouement=None,
                   scenario_creator_kwargs=scenario_creator_kwargs
                   )
        
        xhat = MMWci.read_xhat(self.xhat_path)
        obj = round_pos_sig(ev.evaluate(xhat),2)
        self.assertEqual(obj, -1300000.0)
 
    @unittest.skipIf(not solver_available,
                     "no solver is available")  
    def test_xhat_eval_evaluate_one(self):
        options = _get_xhatEval_options()
        MMW_options = _get_base_options()
        xhat = MMWci.read_xhat(self.xhat_path)
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
        options = _get_base_options()
        xhat = MMWci.read_xhat(self.xhat_path)
        MMW = MMWci.MMWConfidenceIntervals(refmodelname,
                                        options['opt'],
                                        xhat,
                                        options['num_batches'])
        r = MMW.run() 
        s = round_pos_sig(r['std'],2)
        bound = round_pos_sig(r['gap_inner_bound'],2)
        self.assertEqual((s,bound), (43.0,280.0))
    
if __name__ == '__main__':
    unittest.main()
    
