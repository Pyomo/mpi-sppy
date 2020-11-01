# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Test warm-starting the farmer example (no variable fixing here)
'''
import os
import mpisppy.examples.sslp.sslp as ref_model
from mpisppy.fwph.fwph_warm_starts.sslpext import SSLPExtension
from mpisppy.opt.ph import PH

def _print_nonants(nonants):
    key = list(nonants.keys())[0]
    header = f'{"":10s}' + \
        ' '.join([f'{key.split("[")[1].rstrip("]"):>12s}' 
                for key in nonants[key].keys()])
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

def main():
    def nonsense(arg1, arg2, arg3):
        pass

    inst = 'sslp_5_25_50'
    sslp_path = os.path.dirname(ref_model.__file__)
    path = os.sep.join([sslp_path, 'data', inst, 'scenariodata'])
    scen_count = int(inst.split('_')[-1])
    scenario_creator    = ref_model.scenario_creator
    scenario_denouement = ref_model.scenario_denouement

    PH_options_base = {
        'solvername': 'gurobi_persistent',
        'PHIterLimit': 500,
        'defaultPHrho': 20,
        'convthresh': 1e-6,
        'verbose': False,
        'display_timing': False,
        'display_progress': False,
        'iter0_solver_options': dict(),
        'iterk_solver_options': dict(),
        'use_lagrangian': False,
    }

    names = ['Scenario' + str(i+1) for i in range(scen_count)]
    true = -121.6

    schema = {
        '0-init': dict(),
        '*-init-w':    {'init_W_fname':    f'sslp/{inst}.opt.weights'},
        '*-init-x':    {'init_Xbar_fname': f'sslp/{inst}.opt.xbar'},
        '*-init':      {'init_W_fname':    f'sslp/{inst}.opt.weights', 
                        'init_Xbar_fname': f'sslp/{inst}.opt.xbar'},
        '5/30-init-w': {'init_W_fname':    f'sslp/{inst}.5-30.weights'},
        '5/30-init-x': {'init_Xbar_fname': f'sslp/{inst}.5-30.xbar'},
        '5/30-init':   {'init_W_fname':    f'sslp/{inst}.5-30.weights', 
                        'init_Xbar_fname': f'sslp/{inst}.5-30.xbar'},
    }

    for rho in [10, 20, 30, 40, 50]:
        for (name, opts) in schema.items():
            PH_options = {**PH_options_base, **opts}
            PH_options['defaultPHrho'] = rho
            ph = PH(PH_options, names, scenario_creator, scenario_denouement)
            conv, obj, bound = ph.ph_main(PH_extensions=SSLPExtension, cb_data=path)
            itr = ph._PHIter
            gap = 100 * (obj - true) / abs(true)
            if (ph.rank == 0):
                with open('tmp', 'a') as f:
                    f.write(f'{name},{rho},{itr},{gap}\n')
            print('Gap =', gap)

main()
