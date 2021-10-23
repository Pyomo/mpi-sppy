# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys
import hydro
import pyomo.environ as pyo
from mpisppy.opt.ef import ExtensiveForm
import mpisppy.utils.sputils as sputils

options = {"solver": sys.argv[1]}
num_scenarios = 9
BFs = [3, 3]
all_scenario_names = [f"Scen{i+1}" for i in range(num_scenarios)]

# This is multi-stage, so we need to supply node names
all_nodenames = sputils.create_nodenames_from_branching_factors(BFs)

options["branching_factors"] = BFs
ef = ExtensiveForm(
    options,
    all_scenario_names,
    hydro.scenario_creator,
    scenario_creator_kwargs={"branching_factors": BFs},
    all_nodenames=all_nodenames
)
ef.solve_extensive_form(tee=True)
print(f'hydro objective value {pyo.value(ef.ef.EF_Obj)}')
ef.report_var_values_at_rank0("Hydro")
