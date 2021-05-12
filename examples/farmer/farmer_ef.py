# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import farmer
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
import sys

def main():

    scenario_creator = farmer.scenario_creator

    crops_multiplier = int(sys.argv[1])
    scen_count = int(sys.argv[2])
    solver_name = sys.argv[3]
    
    scenario_creator_kwargs = {
        "use_integer": False,
        "crops_multiplier": crops_multiplier,
    }

    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]

    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    solver = pyo.SolverFactory(solver_name)
    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        solver.solve(tee=True)
    else:
        solver.solve(ef, tee=True, symbolic_solver_labels=True,)

    return ef

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("usage python farmer_ef.py {crops_multiplier} {scen_count} {solver_name}")
        print("e.g., python farmer_ef.py 1 3 gurobi")
        quit()
    main_ef = main()
    print(f"EF objective: {pyo.value(main_ef.EF_Obj)}")
    sputils.ef_ROOT_nonants_npy_serializer(main_ef, "farmer_root_nonants.npy")

