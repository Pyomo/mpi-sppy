# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys

from pyomo.environ import SolverFactory
from mpisppy.utils.pysp_model import PySPModel
from mpisppy.opt.ef import ExtensiveForm

def _print_usage():
    print('Usage: "sizes_pysp.py num_scen solver" where num_scen is 3 or 10 and solver is a pyomo solver name')

if len(sys.argv) < 3:
    _print_usage()
    sys.exit()
elif int(sys.argv[1]) not in [3,10]:
    _print_usage()
    sys.exit()

try:
    solver_avail = SolverFactory(sys.argv[2]).available()
    if not solver_avail:
        print(f"Cannot find solver {sys.argv[2]}")
        sys.exit()
except:
    print(f"Cannot find solver {sys.argv[2]}")
    _print_usage()
    sys.exit()

num_scen = int(sys.argv[1])
solver = sys.argv[2]

sizes = PySPModel(scenario_creator='./models/ReferenceModel.py',
                  tree_model=f'./SIZES{num_scen}/ScenarioStructure.dat',
                  )


ef = ExtensiveForm(options={'solver':solver}, 
                   all_scenario_names=sizes.all_scenario_names,
                   scenario_creator=sizes.scenario_creator,
                   model_name='sizes_EF')

ef.solve_extensive_form(tee=True)
