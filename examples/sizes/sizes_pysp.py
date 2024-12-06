###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
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
except Exception:
    print(f"Cannot find solver {sys.argv[2]}")
    _print_usage()
    sys.exit()

num_scen = int(sys.argv[1])
solver = sys.argv[2]

sizes = PySPModel(model='./models/ReferenceModel.py',
                  scenario_tree=f'./SIZES{num_scen}/ScenarioStructure.dat',
                  )


ef = ExtensiveForm(options={'solver':solver}, 
                   all_scenario_names=sizes.all_scenario_names,
                   scenario_creator=sizes.scenario_creator,
                   model_name='sizes_EF')

ef.solve_extensive_form(tee=True)

sizes.close()
