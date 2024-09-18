###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import sys
import mpisppy.phbase
import pyomo.environ as pyo
from sizes import scenario_creator


ScenCount = 3

solver = pyo.SolverFactory(sys.argv[1])

all_scenario_names = list()
for sn in range(ScenCount):
    all_scenario_names.append("Scenario"+str(sn+1))

ef = mpisppy.utils.sputils.create_EF(
    all_scenario_names,
    scenario_creator,
    scenario_creator_kwargs={"scenario_count": ScenCount},
)
if 'persistent' in solver.name: 
    solver.set_instance(ef, symbolic_solver_labels=True)
solver.options["mipgap"] = 0.0001
results = solver.solve(ef, tee=True)
print('EF objective value:', pyo.value(ef.EF_Obj))
