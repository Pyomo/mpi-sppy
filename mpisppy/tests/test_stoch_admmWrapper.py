###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import unittest
import subprocess
import examples.stoch_distr.stoch_distr_admm_cylinders as stoch_distr_admm_cylinders
import examples.stoch_distr.stoch_distr as stoch_distr
from mpisppy.utils import config
from mpisppy.tests.utils import get_solver
import os

solver_available, solver_name, persistent_available, persistent_solver_name= get_solver()

class TestStochAdmmWrapper(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def _cfg_creator(self, num_stoch_scens, num_admm_subproblems):
        # Needs to be modified
        cfg = config.Config()
        stoch_distr.inparser_adder(cfg)

        cfg.num_stoch_scens = num_stoch_scens
        cfg.num_admm_subproblems = num_admm_subproblems

        return cfg
    
    def _make_admm(self, num_stoch_scens, num_admm_subproblems, verbose=False):
        cfg = self._cfg_creator(num_stoch_scens, num_admm_subproblems)
        admm, all_admm_stoch_subproblem_scenario_names = stoch_distr_admm_cylinders._make_admm(cfg,n_cylinders=1,verbose=verbose)
        return admm

    def _get_base_options(self):
        cfg = config.Config()
        cfg.quick_assign("run_async", bool, False)
        cfg.quick_assign("num_stoch_scens", int, 4)
        cfg.quick_assign("num_admm_subproblems", int, 2)
        cfg.quick_assign("default_rho", int, 10)
        cfg.quick_assign("EF_2stage", bool, True)
        cfg.quick_assign("num_batches", int, 2)
        cfg.quick_assign("batch_size", int, 10)

    
    def test_constructor(self):
        for num_stoch_scens in range(3,5):
            for num_admm_subproblems in range(3,5):
                self._make_admm(num_stoch_scens, num_admm_subproblems)

    def test_variable_probability(self):
        admm = self._make_admm(4,3)
        q = dict()
        for sname, s in admm.local_admm_stoch_subproblem_scenarios.items():
            q[sname] = admm.var_prob_list(s)
        self.assertEqual(q["ADMM_STOCH_Region1_StochasticScenario1"][0][1], 0.5)
        self.assertEqual(q["ADMM_STOCH_Region3_StochasticScenario1"][0][1], 0)

    def test_admmWrapper_scenario_creator(self):
        admm = self._make_admm(4,3)
        sname = "ADMM_STOCH_Region3_StochasticScenario1"
        q = admm.admmWrapper_scenario_creator(sname)
        self.assertTrue(q.y__DC1DC2__.is_fixed())
        self.assertFalse(q.y["DC3_1DC1"].is_fixed())
    
    def _slack_name(self, dummy_node):
        return f"y[{dummy_node}]"

    def test_assign_variable_probs_error1(self):
        admm = self._make_admm(2,3)
        admm.consensus_vars["Region1"].append((self._slack_name("DC2DC3"),2)) # The variable is added in the second stage
        self.assertRaises(RuntimeError, admm.assign_variable_probs, admm)
        
    def test_assign_variable_probs_error2(self):
        admm = self._make_admm(2,3)
        admm.consensus_vars["Region1"].remove((self._slack_name("DC3_1DC1"),2))
        self.assertRaises(RuntimeError, admm.assign_variable_probs, admm)

    
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
        command_line_pairs = [(f"mpiexec -np 3 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 10 --num-admm-subproblems 2 --default-rho 10 --solver-name {solver_name} --max-iterations 50 --xhatxbar --lagrangian --rel-gap 0.001 --num-stage 3" \
                         , f"python stoch_distr_ef.py --solver-name {solver_name} --num-stoch-scens 10 --num-admm-subproblems 2 --num-stage 3"), \
                         (f"mpiexec -np 3 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 5 --num-admm-subproblems 3 --default-rho 5 --solver-name {solver_name} --max-iterations 50 --xhatxbar --lagrangian --rel-gap 0.01 --ensure-xhat-feas" \
                         , f"python stoch_distr_ef.py --solver-name {solver_name} --num-stoch-scens 5 --num-admm-subproblems 3 --ensure-xhat-feas"), \
                              (f"mpiexec -np 6 python -u -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 4 --num-admm-subproblems 5 --default-rho 15 --solver-name {solver_name} --max-iterations 30 --xhatxbar --lagrangian --mnpr 5 --scalable --ensure-xhat-feas" \
                         , f"python stoch_distr_ef.py --solver-name {solver_name} --num-stoch-scens 4 --num-admm-subproblems 5 --mnpr 5 --scalable --ensure-xhat-feas")  ]
        #command_line = f"mpiexec -np 6 python -m mpi4py examples/stoch_distr/stoch_distr_admm_cylinders.py --num-admm-subproblems 2 --num-stoch-scens 4 --default-rho 10 --solver-name {solver_name} --max-iterations 100 --xhatxbar --lagrangian"
        original_dir = os.getcwd()
        for j in range(len(command_line_pairs)):
            if j == 0: # The first line is executed in the test directory because it has a 3-stage problem. This one does not insure xhatfeasibility but luckily works
                target_directory = 'examples/stoch_distr'
            if j != 0: # The other lines are executed in the real directory because it ensures xhat feasibility
                target_directory = '../../examples/stoch_distr'
            os.chdir(target_directory)
            objectives = {}
            command = command_line_pairs[j][0].split()
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.stderr:
                print("Error output:")
                print(result.stderr)
                raise RuntimeError("Error encountered as shown above.")
            # Check the standard output
            if result.stdout:
                result_by_line = result.stdout.strip().split('\n')
            else:
                raise RuntimeError(f"No results in stdout for {command=}.")
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
            command = command_line_pairs[j][1].split()
            result = subprocess.run(command, capture_output=True, text=True)
            result_by_line = result.stdout.strip().split('\n')

            for i in range(len(result_by_line)):
                if "EF objective" in result_by_line[-i-1]: #should be on last line but we can check
                    decomposed_line = result_by_line[-i-1].split(': ')
                    objectives["EF objective"] = float(decomposed_line[1])#math.ceil(float(decomposed_line[1]))
            try:
                correct_order = objectives["outer bound"] <= objectives["EF objective"] <= objectives["inner bound"]
            except Exception:
                raise RuntimeError("The output could not be read to capture the values")
            assert correct_order, f' We obtained {objectives["outer bound"]=}, {objectives["EF objective"]=}, {objectives["inner bound"]=}'
            os.chdir(original_dir)
    


if __name__ == '__main__':
    unittest.main()
