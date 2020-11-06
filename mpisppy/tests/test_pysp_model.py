# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import pyutilib.th as unittest
import pyomo.environ as pyo
import os.path

from mpisppy.utils.pysp_model import PySPModel
from mpisppy.tests.test_ef_ph import _get_ph_base_options,\
                                     solver_available,\
                                     round_pos_sig

import mpisppy.opt.ef
import mpisppy.opt.ph

file_dir = os.path.dirname(os.path.abspath(__file__))
sizes_dir = os.path.join(file_dir,'examples','sizes')

class Test_sizes_abstract(unittest.TestCase):
    """ Test PySPModel using abstract sizes case """

    def setUp(self):
        self.pysp_sizes3 = PySPModel(os.path.join(sizes_dir,
                                     'ReferenceModel.py'),
                                     os.path.join(sizes_dir,
                                     'SIZES3', 'ScenarioStructure.dat')
                                    )

    def test_ph_constructor(self):
        pysp_sizes = self.pysp_sizes3
        PHoptions = _get_ph_base_options()
        PHoptions['PHIterLimit'] = 0
        ph = mpisppy.opt.ph.PH(PHoptions,
                                  pysp_sizes.all_scenario_names,
                                  pysp_sizes.scenario_creator,
                                  lambda *args : None,
                                  )

    def test_ef_constructor(self):
        pysp_sizes = self.pysp_sizes3
        options = {"solver": "cplex"}
        ef = mpisppy.opt.ef.ExtensiveForm(
            options,
            pysp_sizes.all_scenario_names,
            pysp_sizes.scenario_creator,
        )

    
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_ef_solve(self):
        pysp_sizes = self.pysp_sizes3
        PHoptions = _get_ph_base_options()
        options = {"solver": PHoptions["solvername"]}
        ef = mpisppy.opt.ef.ExtensiveForm(
            options,
            pysp_sizes.all_scenario_names,
            pysp_sizes.scenario_creator,
        )
        results = ef.solve_extensive_form(tee=False)
        sig2eobj = round_pos_sig(pyo.value(ef.ef.EF_Obj),2)
        self.assertEqual(220000.0, sig2eobj)


if __name__ == '__main__':
    unittest.main()
