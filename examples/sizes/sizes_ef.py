import sys
import mpisppy.phbase
import pyomo.environ as pyo
from sizes import scenario_creator


ScenCount = 3

options = {}
options["solver_name"] = sys.argv[1]
options["convthresh"] = 0.001
options["subsolvedirectives"] = None
options["verbose"] = True
options["display_timing"] = True
options["display_progress"] = True
options["iter0_solver_options"] = {"mipgap": 0.01}
options["iterk_solver_options"] = {"mipgap": 0.005}

solver = pyo.SolverFactory(options["solver_name"])

all_scenario_names = list()
for sn in range(ScenCount):
    all_scenario_names.append("Scenario"+str(sn+1))

ef = mpisppy.utils.sputils.create_EF(
    all_scenario_names,
    scenario_creator,
    scenario_creator_kwargs={"scenario_count": ScenCount},
)
if 'persistent' in options["solver_name"]:
    solver.set_instance(ef, symbolic_solver_labels=True)
solver.options["mipgap"] = 0.0001
results = solver.solve(ef, tee=options["verbose"])
print('EF objective value:', pyo.value(ef.EF_Obj))