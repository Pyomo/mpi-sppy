# This software is distributed under the 3-clause BSD License.


import os
import sys
import unittest
import pyomo.environ as pyo
import mpisppy.opt.ph
from mpisppy.tests.utils import get_solver, round_pos_sig
import mpisppy.utils.config as config
import mpisppy.agnostic.agnostic as agnostic

sys.path.insert(0, '../../examples/farmer')
import farmer_pyomo_agnostic
import farmer_ampl_agnostic


__version__ = 0.1

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver()

# NOTE Gurobi is hardwired for the AMPL test, so don't install it on github
# (and, if you have gurobi installed the ampl test will fail)

def _farmer_cfg():
    cfg = config.Config()
    cfg.popular_args()
    cfg.ph_args()
    cfg.default_rho = 1
    farmer_pyomo_agnostic.inparser_adder(cfg)
    return cfg

def _get_ph_base_options():
    Baseoptions = {}
    Baseoptions["asynchronousPH"] = False
    Baseoptions["solver_name"] = solver_name
    Baseoptions["PHIterLimit"] = 3
    Baseoptions["defaultPHrho"] = 1
    Baseoptions["convthresh"] = 0.001
    Baseoptions["subsolvedirectives"] = None
    Baseoptions["verbose"] = False
    Baseoptions["display_timing"] = False
    Baseoptions["display_progress"] = False
    if "cplex" in solver_name:
        Baseoptions["iter0_solver_options"] = {"mip_tolerances_mipgap": 0.001}
        Baseoptions["iterk_solver_options"] = {"mip_tolerances_mipgap": 0.00001}
    else:
        Baseoptions["iter0_solver_options"] = {"mipgap": 0.001}
        Baseoptions["iterk_solver_options"] = {"mipgap": 0.00001}

    Baseoptions["display_progress"] = False

    return Baseoptions

#*****************************************************************************

class Test_Agnostic_pyomo(unittest.TestCase):
    
    def test_agnostic_pyomo_constructor(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_pyomo_agnostic, cfg)

        
    def test_agnostic_pyomo_scenario_creator(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_pyomo_agnostic, cfg)
        s0 = Ag.scenario_creator("scen0")
        s2 = Ag.scenario_creator("scen2")

        
    def test_agnostic_pyomo_PH_constructor(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_pyomo_agnostic, cfg)
        s1 = Ag.scenario_creator("scen1")  # average case
        phoptions = _get_ph_base_options()
        ph = mpisppy.opt.ph.PH(
            phoptions,
            farmer_pyomo_agnostic.scenario_names_creator(num_scens=3),
            Ag.scenario_creator,
            farmer_pyomo_agnostic.scenario_denouement,
            scenario_creator_kwargs=None,   # agnostic.py takes care of this
            extensions=None
        )

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_agnostic_pyomo_PH(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_pyomo_agnostic, cfg)
        s1 = Ag.scenario_creator("scen1")  # average case
        phoptions = _get_ph_base_options()
        phoptions["Ag"] = Ag  # this is critical
        scennames = farmer_pyomo_agnostic.scenario_names_creator(num_scens=3)
        ph = mpisppy.opt.ph.PH(
            phoptions,
            scennames,
            Ag.scenario_creator,
            farmer_pyomo_agnostic.scenario_denouement,
            scenario_creator_kwargs=None,   # agnostic.py takes care of this
            extensions=None
        )
        conv, obj, tbound = ph.ph_main()
        self.assertAlmostEqual(-115405.5555, tbound, places=2)
        self.assertAlmostEqual(-110433.4007, obj, places=2)

class Test_Agnostic_AMPL(unittest.TestCase):
    # HEY (Sept 2023), when we go to a more generic cylinders for
    # agnostic, move the model file name to cfg and remove the model
    # file from the test directory TBD
    def test_agnostic_AMPL_constructor(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_ampl_agnostic, cfg)

    
    def test_agnostic_AMPL_scenario_creator(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_ampl_agnostic, cfg)
        s0 = Ag.scenario_creator("scen0")
        s2 = Ag.scenario_creator("scen2")
        
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_agnostic_ampl_PH(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_ampl_agnostic, cfg)
        s1 = Ag.scenario_creator("scen1")  # average case
        phoptions = _get_ph_base_options()
        phoptions["Ag"] = Ag  # this is critical
        phoptions["solver_name"] = "gurobi"  # need an ampl solver
        scennames = farmer_ampl_agnostic.scenario_names_creator(num_scens=3)
        ph = mpisppy.opt.ph.PH(
            phoptions,
            scennames,
            Ag.scenario_creator,
            farmer_ampl_agnostic.scenario_denouement,
            scenario_creator_kwargs=None,   # agnostic.py takes care of this
            extensions=None
        )
        conv, obj, tbound = ph.ph_main()
        print(f"{obj =}, {tbound}")
        message = """ NOTE if you are getting zeros it is because Gurobi is
        hardwired for AMPL tests, so don't install it on github (and, if you have
        gurobi generally installed on your machine, then the ampl
        test will fail on your machine).  """
        self.assertAlmostEqual(-115405.5555, tbound, 2, message)
        self.assertAlmostEqual(-110433.4007, obj, 2, message)        

if __name__ == '__main__':
    unittest.main()
