# This software is distributed under the 3-clause BSD License.


import os
import sys
import unittest
import pyomo.environ as pyo
import mpisppy.opt.ph
from mpisppy.tests.utils import get_solver, round_pos_sig
import mpisppy.utils.config as config
import mpisppy.agnostic.agnostic as agnostic
import mpisppy.agnostic.agnostic_cylinders as agnostic_cylinders
import mpisppy.utils.sputils as sputils

sys.path.insert(0, "../../examples/farmer")
import farmer_pyomo_agnostic
import farmer_ampl_agnostic
try:
    import farmer_gurobipy_agnostic
    have_gurobipy = True
except:
    have_gurobipy = False    

__version__ = 0.2

solver_available, solver_name, persistent_available, persistent_solver_name = (
    get_solver()
)

# NOTE Gurobi is hardwired for the AMPL and GAMS tests, so don't install it on github
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


# *****************************************************************************

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
    def test_agnostic_AMPL_constructor(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_ampl_agnostic, cfg)

    def test_agnostic_AMPL_scenario_creator(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_ampl_agnostic, cfg)
        s0 = Ag.scenario_creator("scen0")
        s2 = Ag.scenario_creator("scen2")

    @unittest.skipIf(not solver_available, "no solver is available")
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
            scenario_creator_kwargs=None,  # agnostic.py takes care of this
            extensions=None,
        )
        conv, obj, tbound = ph.ph_main()
        print(f"{obj =}, {tbound}")
        print(f"{solver_name=}")
        print(f"{tbound=}")
        print(f"{obj=}")
        message = """ NOTE if you are getting zeros it is because Gurobi is
        hardwired for AMPL tests, so don't install it on github (and, if you have
        gurobi generally installed on your machine, then the ampl
        test will fail on your machine).  """
        self.assertAlmostEqual(-115405.5555, tbound, 2, message)
        self.assertAlmostEqual(-110433.4007, obj, 2, message)

    def test_agnostic_cylinders_ampl(self):
        # just make sure PH runs
        # TBD: Nov 2024
        # this test (or the code) needs work: why does it stop after 3 iterations
        # why are spbase and phbase initialized at the end
        model_fname = "mpisppy.agnostic.examples.farmer_ampl_model"
        module = sputils.module_name_to_module(model_fname)
        cfg = agnostic_cylinders._setup_args(module)
        cfg.module_name = model_fname
        cfg.default_rho = 1
        cfg.num_scens = 3
        cfg.solver_name= "gurobi"
        cfg.guest_language = "AMPL"
        cfg.max_iterations = 5
        cfg.ampl_model_file = "../agnostic/examples/farmer.mod"
        agnostic_cylinders.main(model_fname, module, cfg)
        
        
print("*********** hack out gurobipy  ***********")
have_gurobipy = False
@unittest.skipIf(not have_gurobipy, "skipping gurobipy")
class Test_Agnostic_gurobipy(unittest.TestCase):
    def test_agnostic_gurobipy_constructor(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_gurobipy_agnostic, cfg)

    def test_agnostic_gurobipy_scenario_creator(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_gurobipy_agnostic, cfg)
        s0 = Ag.scenario_creator("scen0")
        s2 = Ag.scenario_creator("scen2")

    def test_agnostic_gurobipy_PH_constructor(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_gurobipy_agnostic, cfg)
        s1 = Ag.scenario_creator("scen1")  # average case
        phoptions = _get_ph_base_options()
        ph = mpisppy.opt.ph.PH(
            phoptions,
            farmer_gurobipy_agnostic.scenario_names_creator(num_scens=3),
            Ag.scenario_creator,
            farmer_gurobipy_agnostic.scenario_denouement,
            scenario_creator_kwargs=None,  # agnostic.py takes care of this
            extensions=None,
        )


    @unittest.skipIf(not solver_available, "no solver is available")
    # Test Case is failing
    def test_agnostic_gurobipy_PH(self):
        cfg = _farmer_cfg()
        Ag = agnostic.Agnostic(farmer_gurobipy_agnostic, cfg)
        s1 = Ag.scenario_creator("scen1")  # average case
        phoptions = _get_ph_base_options()
        phoptions["Ag"] = Ag  # this is critical
        scennames = farmer_gurobipy_agnostic.scenario_names_creator(num_scens=3)
        ph = mpisppy.opt.ph.PH(
            phoptions,
            scennames,
            Ag.scenario_creator,
            farmer_gurobipy_agnostic.scenario_denouement,
            scenario_creator_kwargs=None,  # agnostic.py takes care of this
            extensions=None,
        )
        conv, obj, tbound = ph.ph_main()
        print(f"{solver_name=}")
        print(f"{tbound=}")
        print(f"{obj=}")
        self.assertAlmostEqual(-110433.4007, obj, places=2)
        self.assertAlmostEqual(-115405.5555, tbound, places=2)


if __name__ == "__main__":
    unittest.main()
