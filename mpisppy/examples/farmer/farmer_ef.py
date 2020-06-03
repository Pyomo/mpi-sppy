# This software is distributed under the 3-clause BSD License.
import mpisppy.examples.farmer.farmer as farmer
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
import sys

def main():

    scenario_creator = farmer.scenario_creator

    CropsMult = int(sys.argv[1])
    scen_count = int(sys.argv[2])

    cb_data={'use_integer': False, "CropsMult": CropsMult}

    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]

    ef = sputils.create_EF(scenario_names, scenario_creator, creator_options={'cb_data':cb_data})

    solver = pyo.SolverFactory('gurobi')
    solver.solve(ef, tee=True, symbolic_solver_labels=True,)

    print(f"EF objective: {pyo.value(ef.EF_Obj)}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage python farmer_ef.py {CropsMult} {scen_count}")
        print("e.g., python farmer_ef.py 1 3")
        quit()
    main()
