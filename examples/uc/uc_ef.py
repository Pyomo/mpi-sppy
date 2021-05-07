# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys
import uc_funcs as uc
import pyomo.environ as pyo

from mpisppy.opt.ef import ExtensiveForm

""" UC """
solver_name = sys.argv[1]
scen_count = 3
scenario_names = [f"Scenario{i+1}" for i in range(scen_count)]
scenario_creator_kwargs = {"path": f"{scen_count}scenarios_r1/"}
options = {"solver": solver_name}
ef = ExtensiveForm(
    options,
    scenario_names,
    uc.scenario_creator,
    model_name="TestEF",
    scenario_creator_kwargs=scenario_creator_kwargs,
)
results = ef.solve_extensive_form(tee=True)
print(f"{scen_count}-scenario UC objective value:", pyo.value(ef.ef.EF_Obj))
if ef.tree_solution_available:
    print(f"Writing tree solution")
    ef.write_tree_solution(f"{scen_count}scenario_EF_solution", uc.scenario_tree_solution_writer)
