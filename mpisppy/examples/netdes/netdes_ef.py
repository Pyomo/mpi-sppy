# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Solve the EF of the network problems
'''
from mpisppy.opt.ef import ExtensiveForm
from mpisppy.examples.netdes.netdes import scenario_creator
import pyomo.environ as pyo

def main():
    inst = "network-10-20-L-01"
    num_scen = int(inst.split("-")[-3])
    scenario_names = [f"Scen{i}" for i in range(num_scen)]
    scenario_creator_options = {
        "cb_data": f"./data/{inst}.dat"
    }
    options = {"solver": "gurobi"}
    ef = ExtensiveForm(
        options,
        scenario_names,
        scenario_creator,
        model_name=f"{inst}-EF",
        scenario_creator_options=scenario_creator_options
    )
    results = ef.solve_extensive_form()
    print("Netdes objective value:", pyo.value(ef.ef.EF_Obj))

if __name__=="__main__":
    main()
