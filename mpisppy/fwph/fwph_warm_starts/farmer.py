# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Test warm-starting the farmer example (no variable fixing here)
'''
import mpisppy.examples.farmer.farmer as gf
from mpisppy.fwph.fwph_warm_starts.fext import FarmerExtension
from mpisppy.opt.ph import PH

def _print_nonants(nonants):
    ''' Can also be used to print dual weights '''
    header = f'{"":10s}' + \
        ' '.join([f'{key.split("[")[1].rstrip("]"):>12s}' 
                for key in nonants['Scenario0'].keys()])
    print(header)
    for (sname, var) in nonants.items():
        row = f'{sname:10s}' + \
            ' '.join([f'{val:12.4f}' for val in var.values()])
        print(row)

def _get_nonants(ph):
    nonants = dict()
    for (name, scenario) in ph.local_scenarios.items():
        vs = {var.name: var.value for node in scenario._PySPnode_list
                                  for var  in node.nonant_vardata_list}
        nonants[name] = vs
    return nonants

def _get_weights(ph):
    weights = dict()
    for (name, scenario) in ph.local_scenarios.items():
        vs = {var.name: scenario._Ws[node.name,ix].value
                        for node in scenario._PySPnode_list
                        for (ix,var) in enumerate(node.nonant_vardata_list)}
        weights[name] = vs
    return weights

def main():
    scen_count          = 3
    integer             = False
    true                = -108390
    scenario_creator    = gf.scenario_creator
    scenario_denouement = gf.scenario_denouement

    cb_data = {'use_integer': integer}
    PH_options_base = {
        'solvername': 'gurobi_persistent',
        'PHIterLimit': 1500,
        'defaultPHrho': 1.5,
        'convthresh': 1e-6,
        'verbose': False,
        'display_timing': False,
        'display_progress': False,
        'iter0_solver_options': {'mipgap': 1e-4},
        'iterk_solver_options': {'mipgap': 1e-4},
        'use_lagrangian' : False,
    }

    if PH_options_base['solvername'].endswith('persistent'):
        print('Using persistent solver')
    else:
        print('Using standard solver')

    inst = 'farmer/intfarmer_3' if integer else 'farmer/farmer_3'
    schema = {
        '0-init': dict(),
        '*-init-w':   {'init_W_fname':    f'{inst}.opt.weights'},
        '*-init-x':   {'init_Xbar_fname': f'{inst}.opt.xbar'},
        '*-init':     {'init_W_fname':    f'{inst}.opt.weights', 
                       'init_Xbar_fname': f'{inst}.opt.xbar'},
        '5/1-init-w': {'init_W_fname':    f'{inst}.5itr.weights'},
        '5/1-init-x': {'init_Xbar_fname': f'{inst}.5itr.xbar'},
        '5/1-init':   {'init_W_fname':    f'{inst}.5itr.weights', 
                       'init_Xbar_fname': f'{inst}.5itr.xbar'},
    }
    names = ['Scenario' + str(i) for i in range(scen_count)]

    for rho in [0.5, 0.75, 1.0, 1.25, 1.50]:
        for (name, opts) in schema.items():
            PH_options = {**PH_options_base, **opts}
            PH_options['defaultPHrho'] = rho
            ph = PH(PH_options, names, scenario_creator, scenario_denouement)
            conv, obj, bound = ph.ph_main(PH_extensions=FarmerExtension, cb_data=cb_data)
            itr = ph._PHIter
            gap = 100 * (obj - true) / abs(true)
            print(f'{name},{rho},{itr},{gap}')
            if (ph.rank == 0):
                with open('continuous_farmer', 'a') as f:
                    f.write(f'{name},{rho},{itr},{gap}\n')

main()
