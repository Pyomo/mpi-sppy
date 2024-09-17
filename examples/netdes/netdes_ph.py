###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' Solve the EF of the network problems
'''
from netdes import scenario_creator, scenario_denouement
from mpisppy.opt.ph import PH
from mpisppy.convergers.primal_dual_converger import PrimalDualConverger
from mpisppy.extensions.xhatclosest import XhatClosest
import os
import sys


def main():
    msg = (
        "Give instance name, then PH maxiters "
        + "then rho, then smooth_type, then pvalue_or_pratio, then beta \n" 
        + "(e.g., network-10-20-L-01 100 10000 0 0.0 1.0)"
    )
    if len(sys.argv) != 7:
        print(msg)
        quit()
    # solver_name = 'gurobi'
    fname = "data" + os.sep + sys.argv[1] + ".dat"
    num_scen = int(fname.split('-')[2])
    scenario_names = ['Scen' + str(i) for i in range(num_scen)]
    scenario_creator_kwargs = {"path": fname}

    ''' Now solve with PH to see what happens (very little, I imagine) '''
    PH_options = {
        'solver_name'           : 'gurobi_persistent',
        'PHIterLimit'          : int(sys.argv[2]),
        'defaultPHrho'         : float(sys.argv[3]),
        'convthresh'           : -1e-8,
        'verbose'              : False,
        'display_progress'     : True,
        'display_timing'       : True,
        'iter0_solver_options' : {"threads": 1},
        'iterk_solver_options' : {"threads": 1},
        'bundles_per_rank'     : 0, # 0 = no bundles
        "display_convergence_detail": False,
        'xhat_closest_options' : {'xhat_solver_options': {}, 'keep_solution':True},
        "smoothed": int(sys.argv[4]),
        "defaultPHp": float(sys.argv[5]),
        "defaultPHbeta": float(sys.argv[6]),
        "primal_dual_converger_options" : {"tol" : 1e-5}
    }
    
    ph = PH(
        PH_options, 
        scenario_names,
        scenario_creator,
        scenario_denouement,
        # extensions=NetworkDesignTracker,
        extensions=XhatClosest,
        ph_converger = PrimalDualConverger,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    conv, obj, triv = ph.ph_main()
    # Obj includes prox (only ok if we find a non-ant soln)
    
    ph.post_solve_bound(verbose=False)

    variables = ph.gather_var_values_to_rank0()
    for (scenario_name, variable_name) in variables:
        variable_value = variables[scenario_name, variable_name]
        print(scenario_name, variable_name, variable_value)

    if ph.tree_solution_available:
        print(f"Final objective from XhatClosest: {ph.extobject._final_xhat_closest_obj}")
    else:
        print(f"Final objective from XhatClosest: {float('inf')}")

if __name__=='__main__':
    main()
