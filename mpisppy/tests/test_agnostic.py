# This software is distributed under the 3-clause BSD License.


import os
import sys
import unittest
import pyomo.environ as pyo
from mpisppy.tests.utils import get_solver, round_pos_sig
import mpisppy.utils.config as config
import mpisppy.utils.agnostic as agnostic

sys.path.insert(0, '../../examples/farmer')
import farmer_pyomo_agnostic
import farmer_ampl_agnostic


__version__ = 0.1

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver()

def _farmer_parse_args():
    cfg = config.Config()

    farmer_pyomo_agnostic.inparser_adder(cfg)    


#*****************************************************************************

class Test_Agnostic_pyomo(unittest.TestCase):
    def test_agnostic_pyomo_constructor(self):
        cfg = _farmer_parse_args()
        Ag = agnostic.Agnostic(farmer_pyomo_agnostic, cfg)


class Test_Agnostic_AMPL(unittest.TestCase):
    def test_agnostic_AMPL_constructor(self):
        cfg = _farmer_parse_args()
        Ag = agnostic.Agnostic(farmer_ampl_agnostic, cfg)
        

if __name__ == '__main__':
    unittest.main()
