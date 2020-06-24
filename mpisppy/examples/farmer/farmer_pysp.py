import sys

from pyomo.environ import SolverFactory
from mpisppy.utils.pysp_model import PySPModel
from mpisppy.opt.ph import PH

def _print_usage():
    print('Usage: "farmer_pysp.py solver" where solver is a pyomo solver name')
if len(sys.argv) < 2:
    _print_usage()
    sys.exit()
try:
    solver_avail = SolverFactory(sys.argv[1]).available()
    if not solver_avail:
        print(f"Cannot find solver {sys.argv[1]}")
        sys.exit()
except:
    print(f"Cannot find solver {sys.argv[1]}")
    _print_usage()
    sys.exit()

farmer = PySPModel(scenario_creator='./PySP/concrete/ReferenceModel.py',
                   tree_model='./PySP/ScenarioStructure.dat')

phoptions = {'defaultPHrho': 1.0,
             'solvername':sys.argv[1],
             'PHIterLimit': 50,
             'convthresh': 0.01,
             'verbose': False,
             'display_progress': True,
             'display_timing': False,
             'iter0_solver_options': None,
             'iterk_solver_options': None
             }

ph = PH( PHoptions = phoptions,
         all_scenario_names = farmer.all_scenario_names,
         scenario_creator = farmer.scenario_creator,
         scenario_denouement = farmer.scenario_denouement,
        )

ph.ph_main()
