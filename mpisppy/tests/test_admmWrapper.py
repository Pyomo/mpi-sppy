###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import unittest
import mpisppy.utils.admmWrapper as admmWrapper
import examples.distr as distr
from mpisppy.utils import config
from mpisppy.tests.utils import get_solver
from mpisppy import MPI
import subprocess
import os

solver_available, solver_name, persistent_available, persistent_solver_name= get_solver()

class TestAdmmWrapper(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def _cfg_creator(self, num_scens):
        cfg = config.Config()

        cfg.num_scens_required()
        cfg.num_scens = num_scens
        return cfg 


    def _make_admm(self, num_scens,verbose=False):
        cfg = self._cfg_creator(num_scens)
        options = {}
        all_scenario_names = distr.scenario_names_creator(num_scens=cfg.num_scens)
        scenario_creator = distr.scenario_creator
        scenario_creator_kwargs = distr.kw_creator(cfg)
        consensus_vars = distr.consensus_vars_creator(cfg.num_scens)
        n_cylinders = 1 #distr_admm_cylinders._count_cylinders(cfg)
        return admmWrapper.AdmmWrapper(options,
                            all_scenario_names, 
                            scenario_creator,
                            consensus_vars,
                            n_cylinders=n_cylinders,
                            mpicomm=MPI.COMM_WORLD,
                            scenario_creator_kwargs=scenario_creator_kwargs,
                            verbose=verbose,
                            )
    
    def test_constructor(self):
        self._make_admm(2,verbose=True)
        for i in range(3,5):
            self._make_admm(i)

    def test_variable_probability(self):        
        admm = self._make_admm(3)
        q = dict()
        for sname, s in admm.local_scenarios.items():
            q[sname] = admm.var_prob_list(s)
        self.assertEqual(q["Region1"][0][1], 0.5)
        self.assertEqual(q["Region3"][0][1], 0)

    def test_admmWrapper_scenario_creator(self):
        admm = self._make_admm(3)
        sname = "Region3"
        q = admm.admmWrapper_scenario_creator(sname)
        self.assertTrue(q.y__DC1DC2__.is_fixed())
        self.assertFalse(q.y["DC3_1DC1"].is_fixed())
    
    def _slack_name(self, dummy_node):
        return f"y[{dummy_node}]"

    def test_assign_variable_probs_error1(self):
        admm = self._make_admm(3)
        admm.consensus_vars["Region1"].append(self._slack_name("DC2DC3"))
        self.assertRaises(RuntimeError, admm.assign_variable_probs)
        
    def test_assign_variable_probs_error2(self):
        admm = self._make_admm(3)
        admm.consensus_vars["Region1"].remove(self._slack_name("DC3_1DC1"))
        self.assertRaises(RuntimeError, admm.assign_variable_probs)

    
    def _extracting_output(self, line):
        import re
        pattern = r'\[\s*\d+\.\d+\]\s+\d+\s+(?:L\s*B?|B\s*L?)?\s+([-.\d]+)\s+([-.\d]+)'

        match = re.search(pattern, line)

        if match:
            outer_bound = match.group(1)
            inner_bound = match.group(2)
            return float(outer_bound), float(inner_bound)
        else:
            raise RuntimeError("The test is probably not correctly adapted: can't match the format of the line")

    def test_values(self):
        command_line_pairs = [(f"mpiexec -np 3 python -u -m mpi4py distr_admm_cylinders.py --num-scens 3 --default-rho 10 --solver-name {solver_name} --max-iterations 50 --xhatxbar --lagrangian --rel-gap 0.01 --ensure-xhat-feas" \
                         , f"python distr_ef.py --solver-name {solver_name} --num-scens 3 --ensure-xhat-feas"), \
                         (f"mpiexec -np 6 python -u -m mpi4py distr_admm_cylinders.py --num-scens 5 --default-rho 10 --solver-name {solver_name} --max-iterations 50 --xhatxbar --lagrangian --mnpr 6 --rel-gap 0.05 --scalable --ensure-xhat-feas" \
                         , f"python distr_ef.py --solver-name {solver_name} --num-scens 5 --ensure-xhat-feas --mnpr 6 --scalable")]
        original_dir = os.getcwd()
        for command_line_pair in command_line_pairs:
            target_directory = '../../examples/distr'
            os.chdir(target_directory)
            objectives = {}
            command = command_line_pair[0].split()
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.stderr:
                print("Error output:")
                print(result.stderr)

            # Check the standard output
            if result.stdout:
                result_by_line = result.stdout.strip().split('\n')
                
            target_line = "Iter.           Best Bound  Best Incumbent      Rel. Gap        Abs. Gap"
            precedent_line_target = False
            i = 0
            for line in result_by_line:
                if precedent_line_target:
                    if i%2 == 1:
                        outer_bound, inner_bound = self._extracting_output(line)
                        objectives["outer bound"] = outer_bound
                        objectives["inner bound"] = inner_bound
                    precedent_line_target = False 
                    i += 1
                elif target_line in line:
                    precedent_line_target = True
            
            # For the EF
            command = command_line_pair[1].split()
            result = subprocess.run(command, capture_output=True, text=True)
            result_by_line = result.stdout.strip().split('\n')
            for i in range(len(result_by_line)):
                if "EF objective" in result_by_line[-i-1]: #should be on last line but we can check
                    decomposed_line = result_by_line[-i-1].split(': ')
                    objectives["EF objective"] = float(decomposed_line[1])
            try:
                correct_order = objectives["outer bound"] <= objectives["EF objective"] <= objectives["inner bound"]
            except Exception:
                raise RuntimeError("The output could not be read to capture the values")
            assert correct_order, f' We obtained {objectives["outer bound"]=}, {objectives["EF objective"]=}, {objectives["inner bound"]=}'
            os.chdir(original_dir)


if __name__ == '__main__':
    unittest.main()
