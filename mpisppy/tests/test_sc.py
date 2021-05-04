import unittest
import sys
import os


class TestSC(unittest.TestCase):
    def setUp(self):
        self.original_path = sys.path
        example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples', 'farmer')
        sys.path.append(example_dir)

    def tearDown(self):
        sys.path = self.original_path
    
    def test_farmer_example(self):
        import schur_complement as sc_example
        ef_opt = sc_example.solve_with_extensive_form(scen_count=3)
        sc_opt = sc_example.solve_with_sc(scen_count=3)

        ef_sol = ef_opt.gather_var_values_to_rank0()
        sc_sol = sc_opt.gather_var_values_to_rank0()

        for k, v in ef_sol.items():
            self.assertAlmostEqual(v, sc_sol[k])
