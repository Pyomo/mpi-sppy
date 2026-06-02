###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
###############################################################################
''' Solve the EF of the scmnd problems
'''

from mpisppy.opt.ef import ExtensiveForm
from scmnd import scenario_creator
import pyomo.environ as pyo
import sys
import os

def main():
    # inst = "scmnd_20_120_40_20"
    if len(sys.argv) == 1:
        inst = "scmnd_20_120_40_20"
    elif len(sys.argv) == 2:
        inst = sys.argv[1]
    else:
        print("Invalid input.")
        quit()
    num_scen = int(inst.split("_")[-1])
    all_scenario_names = list()
    for sn in range(num_scen):
        all_scenario_names.append("Scenario" + str(sn + 1))
    data_dir = f'./examples/scmnd/data/{inst}/scenariodata'
    options = {"solver": "xpress"}
    ef = ExtensiveForm(
        options,
        all_scenario_names,
        scenario_creator,
        model_name=f"{inst}-EF",
        scenario_creator_kwargs={"data_dir": data_dir},
    )
    results = ef.solve_extensive_form()
    if not pyo.check_optimal_termination(results):
        print("Warning: Non-optimal termination condition from Pyomo")
    print("scmnd objective value:", pyo.value(ef.ef.EF_Obj))

if __name__=="__main__":
    main()
