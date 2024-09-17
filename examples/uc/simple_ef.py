###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import sys
import uc_funcs as uc
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils


""" UC """
assert len(sys.argv) == 2, "Supply the solver name as the first argument"
solver_name = sys.argv[1]
scen_count = 3
scenario_names = [f"Scenario{i+1}" for i in range(scen_count)]
scenario_creator_kwargs = {"path": f"{scen_count}scenarios_r1/"}
options = {"solver": solver_name}
ef = sputils.create_EF(
    scenario_names,
    uc.scenario_creator,
    scenario_creator_kwargs=scenario_creator_kwargs,
)

solver = pyo.SolverFactory(solver_name)
if 'persistent' in solver_name:
    solver.set_instance(ef, symbolic_solver_labels=True)
    results = solver.solve(tee=True)
else:
    results = solver.solve(ef, tee=True, symbolic_solver_labels=True,)
###results = ef.solve_extensive_form(tee=True)
pyo.assert_optimal_termination(results)

print(f"{scen_count}-scenario UC objective value:", pyo.value(ef.EF_Obj))
npyfile = "uc_cyl_nonants.npy"
print(f"About to write {npyfile}")
sputils.ef_ROOT_nonants_npy_serializer(ef, npyfile)
