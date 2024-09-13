''' Solve the EF of the network problems
'''
from mpisppy.opt.ef import ExtensiveForm
from netdes import scenario_creator
import pyomo.environ as pyo
import sys

def main():
    # inst = "network-10-20-L-01"
    if len(sys.argv) == 1:
        inst = "network-10-20-L-01"
    elif len(sys.argv) == 2:
        inst = sys.argv[1]
    else:
        print("Invalid input.")
        quit()
    num_scen = int(inst.split("-")[-3])
    scenario_names = [f"Scen{i}" for i in range(num_scen)]
    path = f"./data/{inst}.dat"
    options = {"solver": "gurobi"}
    ef = ExtensiveForm(
        options,
        scenario_names,
        scenario_creator,
        model_name=f"{inst}-EF",
        scenario_creator_kwargs={"path": path},
    )
    results = ef.solve_extensive_form(tee=True)
    print("Netdes objective value:", pyo.value(ef.ef.EF_Obj))

if __name__=="__main__":
    main()
