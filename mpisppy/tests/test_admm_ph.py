import unittest
import mpisppy.utils.admm_ph as admm_ph
import examples.distr as distr
from mpisppy.utils import config
from mpisppy import MPI

class TestADMMPH(unittest.TestCase):
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
        return admm_ph.ADMM_PH(options,
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
            q[sname] = admm.var_prob_list_fct(s)
        self.assertEqual(q["Region1"][0][1], 0.5)
        self.assertEqual(q["Region3"][0][1], 0)

    def test_admm_ph_scenario_creator(self):
        admm = self._make_admm(3)
        sname = "Region3"
        q = admm.admm_ph_scenario_creator(sname)
        self.assertTrue(q.y__DC1DC2__.is_fixed())
        self.assertFalse(q.y["DC3_1DC1"].is_fixed())
    
    def _slack_name(self, dummy_node):
        return f"y[{dummy_node}]"

    def test_assign_variable_probs_error1(self):
        admm = self._make_admm(3)
        admm.consensus_vars["Region1"].append(self._slack_name("DC2DC3"))
        self.assertRaises(RuntimeError, admm.assign_variable_probs, admm)
        
    def test_assign_variable_probs_error2(self):
        admm = self._make_admm(3)
        admm.consensus_vars["Region1"].remove(self._slack_name("DC3_1DC1"))
        self.assertRaises(RuntimeError, admm.assign_variable_probs, admm)


if __name__ == '__main__':
    unittest.main()
