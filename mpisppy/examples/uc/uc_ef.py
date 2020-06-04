# This software is distributed under the 3-clause BSD License.
import mpisppy.examples.uc.uc_funcs as uc
import pyomo.environ as pyo

from mpisppy.opt.ef import ExtensiveForm

""" UC """
scen_count = 3
scenario_names = [f"Scenario{i+1}" for i in range(scen_count)]
scenario_creator_options = {
    "cb_data": {"path": f"./examples/uc/{scen_count}scenarios_r1/"}
}
options = {"solver": "gurobi"}
ef = ExtensiveForm(
    options,
    scenario_names,
    uc.scenario_creator,
    model_name="TestEF",
    scenario_creator_options=scenario_creator_options,
)
results = ef.solve_extensive_form(tee=True)
print(f"{scen_count}-scenario UC objective value:", pyo.value(ef.ef.EF_Obj))
