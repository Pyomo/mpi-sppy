# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# from PySP where it was an AbstractModel, with AMPL format data

import os
import sys

from pyomo.environ import SolverFactory
from mpisppy.utils.pysp_model import PySPModel
from mpisppy.opt.ph import PH
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo

def _print_usage():
    print('Usage: "abstract.py solver" where solver is a pyomo solver name')
if len(sys.argv) < 2:
    _print_usage()
    sys.exit()

solver_name = sys.argv[1]
try:
    solver_avail = SolverFactory(solver_name).available()
    if not solver_avail:
        print(f"Cannot find solver {solver_name}")
        sys.exit()
except:
    print(f"Cannot find solver {solver_name}")
    _print_usage()
    sys.exit()

pref = os.path.join("..","PySP","abstract")
farmer = PySPModel(model=os.path.join(pref,"ReferenceModel.py"),
                   scenario_tree=os.path.join(pref,"ScenarioStructure.dat"),
                   data_dir=pref)

phoptions = {'defaultPHrho': 1.0,
             'solver_name':solver_name,
             'PHIterLimit': 50,
             'convthresh': 0.01,
             'verbose': False,
             'display_progress': True,
             'display_timing': False,
             'iter0_solver_options': None,
             'iterk_solver_options': None
             }

ph = PH( options = phoptions,
         all_scenario_names = farmer.all_scenario_names,
         scenario_creator = farmer.scenario_creator,
         scenario_denouement = farmer.scenario_denouement,
        )

ph.ph_main()

ef = sputils.create_EF(farmer.all_scenario_names,
                       farmer.scenario_creator)
solver = pyo.SolverFactory(solver_name)
if 'persistent' in solver_name:
    solver.set_instance(ef, symbolic_solver_labels=True)
    solver.solve(tee=True)
else:
    solver.solve(ef, tee=True, symbolic_solver_labels=True,)

print(f"EF objective: {pyo.value(ef.EF_Obj)}")
farmer.close()
