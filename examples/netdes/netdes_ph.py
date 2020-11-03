# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Solve the EF of the network problems
'''
from mpisppy.examples.netdes.netdes import scenario_creator, scenario_denouement
from mpisppy.opt.ph import PH
from netdes_extension import NetworkDesignTracker

def main():
    solver_name = 'gurobi'
    fname = 'data/network-10-20-L-01.dat'
    num_scen = int(fname.split('-')[2])
    scenario_names = ['Scen' + str(i) for i in range(num_scen)]

    ''' Now solve with PH to see what happens (very little, I imagine) '''
    PH_options = {
        'solvername'           : 'gurobi_persistent',
        'PHIterLimit'          : 20,
        'defaultPHrho'         : 10000,
        'convthresh'           : 1e-8,
        'verbose'              : False,
        'display_progress'     : False,
        'display_timing'       : False,
        'iter0_solver_options' : dict(),
        'iterk_solver_options' : dict(),
        'bundles_per_rank'     : 2, # 0 = no bundles
    }
    
    ph = PH(
        PH_options, 
        scenario_names,
        scenario_creator,
        scenario_denouement,
        PH_extensions=NetworkDesignTracker,
        cb_data=fname,
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
