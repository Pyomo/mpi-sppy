''' Solve the EF of the sslp problems
'''
from mpisppy.opt.ef import ExtensiveForm
from sslp import scenario_creator
import pyomo.environ as pyo
import sys
import os

def main():
    # inst = "sslp_15_45_5"
    if len(sys.argv) == 1:
        inst = "sslp_15_45_10"
    elif len(sys.argv) == 2:
        inst = sys.argv[1]
    else:
        print("Invalid input.")
        quit()
    num_scen = int(inst.split("_")[-1])
    all_scenario_names = list()
    for sn in range(num_scen):
        all_scenario_names.append("Scenario" + str(sn + 1))
    data_dir = "data" + os.sep + inst + os.sep + "scenariodata"
    options = {"solver": "xpress"}
    ef = ExtensiveForm(
        options,
        all_scenario_names,
        scenario_creator,
        model_name=f"{inst}-EF",
        scenario_creator_kwargs={"data_dir": data_dir},
    )
    results = ef.solve_extensive_form()
    print("sslp objective value:", pyo.value(ef.ef.EF_Obj))

if __name__=="__main__":
    main()
