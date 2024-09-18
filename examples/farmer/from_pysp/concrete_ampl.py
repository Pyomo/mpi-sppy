###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# from PySP where it was a concrete model with AMPL format scenario tree data
import os
import sys

from pyomo.environ import SolverFactory
from mpisppy.utils.pysp_model import PySPModel
from mpisppy.opt.ph import PH
from mpisppy.extensions.xhatclosest import XhatClosest

def _print_usage():
    print('Usage: "concrete_ampl.py solver" where solver is a pyomo solver name')
if len(sys.argv) < 2:
    _print_usage()
    sys.exit()
try:
    solver_avail = SolverFactory(sys.argv[1]).available()
    if not solver_avail:
        print(f"Cannot find solver {sys.argv[1]}")
        sys.exit()
except Exception:
    print(f"Cannot find solver {sys.argv[1]}")
    _print_usage()
    sys.exit()

pref = os.path.join("..","PySP")
farmer = PySPModel(model=os.path.join(pref,"concrete",
                                                 "ReferenceModel.py"),
                   scenario_tree=os.path.join(pref,"ScenarioStructure.dat"))

phoptions = {'defaultPHrho': 1.0,
             'solver_name':sys.argv[1],
             'PHIterLimit': 50,
             'convthresh': 0.01,
             'verbose': False,
             'display_progress': True,
             'display_timing': False,
             'iter0_solver_options': None,
             'iterk_solver_options': None,
             'xhat_closest_options': {'xhat_solver_options': {}, 'keep_solution':True},
             }

ph = PH( options = phoptions,
         all_scenario_names = farmer.all_scenario_names,
         scenario_creator = farmer.scenario_creator,
         scenario_denouement = farmer.scenario_denouement,
         extensions = XhatClosest,
        )

ph.ph_main()

if ph.tree_solution_available:
    print(f"Final objective from XhatClosest: {ph.extobject._final_xhat_closest_obj}")

farmer.close()
