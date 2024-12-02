###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# updated April 2020 so it executes, but this needs a three driver
from mpisppy.opt.ph import PH
from battery import scenario_creator, scenario_denouement

###from batteryext import BatteryExtension

def main():
    scenario_creator_kwargs = {
        'solar_filename' : 'solar.csv',
        'use_LP'         : False,
        'lam'            : 467, # Dual weight for dualized chance constr
    }
    PH_options = {
        'solver_name' : 'gurobi',
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
            scenario_creator_kwargs=scenario_creator_kwargs)

    ph.ph_main()

    res = get_solutions(ph)
    for (sname, sol) in res.items():
        z = sol['z']
        print(sname, z)

def get_solutions(ph):
    result = dict()
    for (sname, scenario) in ph.local_scenarios.items():
        result[sname] = dict()
        for node in scenario._mpisppy_node_list:
            for var in node.nonant_vardata_list:
                result[sname][var.name] = var.value
        result[sname]['z'] = scenario.z[0].value
    return result

if __name__=='__main__':
    main()
