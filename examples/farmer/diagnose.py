# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Demonstrate the use of the scenario diagnostic software
# Intended to be run "from" the farmer example directory

import sys
import farmer
import mpisppy.utils.scenario_diagnostics as diag

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage python diagnose.py {crops_multiplier} {scen_count} {solver_name}")
        print("e.g., python diagnose.py 1 3 gurobi")
        quit()
    
    scenario_creator = farmer.scenario_creator

    crops_multiplier = int(sys.argv[1])
    scen_count = int(sys.argv[2])
    solver_name = sys.argv[3]

    scenario_creator_kwargs = {
        "use_integer": False,
        "crops_multiplier": crops_multiplier,
    }

    solve_kwargs = {"tee": True}

    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]

    scenario_name, pyomodel = diag.find_problematic_scenario(solver_name,
                                                             scenario_names,
                                                             scenario_creator,
                                                             scenario_creator_kwargs = scenario_creator_kwargs,
                                                             solve_kwargs=solve_kwargs)
    print(f"Done. This should be None={None}")

