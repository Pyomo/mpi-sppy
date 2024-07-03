# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Solve the EF of the network problems
'''
from netdes import scenario_creator, scenario_denouement
from mpisppy.opt.ph import PH
from netdes_extension import NetworkDesignTracker
from mpisppy.convergers.primal_dual_converger import PrimalDualConverger


def main():
    solver_name = 'gurobi'
    fname = 'data/network-10-20-L-01.dat'
    num_scen = int(fname.split('-')[2])
    scenario_names = ['Scen' + str(i) for i in range(num_scen)]
    scenario_creator_kwargs = {"path": fname}

    ''' Now solve with PH to see what happens (very little, I imagine) '''
    PH_options = {
        'solver_name'           : 'xpress_persistent',
        'PHIterLimit'          : 200,
        'defaultPHrho'         : 100000,
        'convthresh'           : -1e-8,
        'verbose'              : True,
        'display_progress'     : True,
        'display_timing'       : False,
        'iter0_solver_options' : dict(),
        'iterk_solver_options' : dict(),
        'bundles_per_rank'     : 2, # 0 = no bundles
        "display_convergence_detail": False,
        "smoothed": False,
        "defaultPHp": 1000,
        "defaultPHbeta": 0.1,
        "primal_dual_converger_options" : {"tol" : 1e-6}
    }
    
    ph = PH(
        PH_options, 
        scenario_names,
        scenario_creator,
        scenario_denouement,
        # extensions=NetworkDesignTracker,
        ph_converger = PrimalDualConverger,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    conv, obj, triv = ph.ph_main()
    # Obj includes prox (only ok if we find a non-ant soln)
    if (conv < 1e-8):
        print(f'Objective value: {obj:.2f}')
    else:
        print('Did not find a non-anticipative solution '
             f'(conv = {conv:.1e})')
    
    ph.post_solve_bound(verbose=False)

if __name__=='__main__':
    main()
