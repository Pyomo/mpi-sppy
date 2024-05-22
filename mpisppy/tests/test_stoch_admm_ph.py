import unittest
import subprocess
import mpisppy.utils.stoch_admm_ph as stoch_admm_ph
import examples.stoch_distr.stoch_distr_admm_cylinders as stoch_distr_admm_cylinders
import examples.stoch_distr.stoch_distr as stoch_distr
from mpisppy.utils import config
import math

class TestSTOCHADMMPH(unittest.TestCase):
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

    
    def test_constructor(self):
        for num_stoch_scens in range(3,5):
            for num_admm_subproblems in range(3,5):
                self._make_admm(num_stoch_scens, num_admm_subproblems)

    def test_variable_probability(self):
        admm = self._make_admm(4,3)
        q = dict()
        for sname, s in admm.local_admm_stoch_subproblem_scenarios.items():
            q[sname] = admm.var_prob_list_fct(s)
        self.assertEqual(q["ADMM_STOCH_Region1_StochasticScenario1"][0][1], 0.5)
        self.assertEqual(q["ADMM_STOCH_Region3_StochasticScenario1"][0][1], 0)

    def test_admm_ph_scenario_creator(self):
        admm = self._make_admm(4,3)
        sname = "ADMM_STOCH_Region3_StochasticScenario1"
        q = admm.admm_ph_scenario_creator(sname)
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

    def test_values(self):
        command_line = "mpiexec -np 6 python -m mpi4py examples/stoch_distr/stoch_distr_admm_cylinders.py --num-admm-subproblems 2 --num-stoch-scens 4 --default-rho 10 --solver-name cplex_direct --max-iterations 10 --xhatxbar --lagrangian"
        command = command_line.split()
        #command = ["bash examples/stoch_distr/bash_script_test.sh"]
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)
        result_by_line = result.stdout.strip().split('\n')
        for i in range(len(result_by_line)):
            if "best_objective" in result_by_line[-i-1]: #should be on line 2 but we can check
                decomposed_line = result_by_line[-i-1].split('=')
                best_objective = math.ceil(float(decomposed_line[1]))
                assert best_objective == -27305, f"the script doesn't return the expected value {-27305} but rather {best_objective}"
                return
        return RuntimeError, "no 'best_objective' found in the output"


if __name__ == '__main__':
    unittest.main()