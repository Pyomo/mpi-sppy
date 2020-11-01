# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# updated April 2020 so it executes, but this needs a three driver
from mpisppy.opt.ph import PH
from mpisppy.examples.battery.battery import scenario_creator, scenario_denouement

###from mpisppy.examples.battery.batteryext import BatteryExtension

def main():
    cb_data = {
        'solar_filename' : 'solar.csv',
        'use_LP'         : False,
        'lam'            : 467, # Dual weight for dualized chance constr
    }
    PH_options = {
        'solvername' : 'gurobi',
        'PHIterLimit' : 250,
        'defaultPHrho' : 1e-1,
        'convthresh' : 1e-3,
        'verbose' : False,
        'display_timing' : False,
        'display_progress' : True,
        'iter0_solver_options' : dict(),
        'iterk_solver_options' : dict(),
        'tee-rank0-solves': False,
    }

    # extensions?

    names = ['s' + str(i) for i in range(50)] 
    ph = PH(PH_options, names, scenario_creator, scenario_denouement,
            cb_data=cb_data)

    ph.ph_main()

    res = get_solutions(ph)
    for (sname, sol) in res.items():
        z = sol['z']
        print(sname, z)

def get_solutions(ph):
    result = dict()
    for (sname, scenario) in ph.local_scenarios.items():
        result[sname] = dict()
        for node in scenario._PySPnode_list:
            for var in node.nonant_vardata_list:
                result[sname][var.name] = var.value
        result[sname]['z'] = scenario.z[0].value
    return result

if __name__=='__main__':
    main()
