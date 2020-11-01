# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import mpisppy.examples.uc.uc_funcs as uc_funcs
import mpisppy.fwph.fwph_warm_starts.uc.ucext import UCExtension
import os

from mpisppy.opt.ph import PH

def empty_denouement(*args):
    pass

def main():
    scen_count          = 25
    scenario_creator    = uc_funcs.pysp2_callback
    scenario_denouement = empty_denouement
    scenario_rhosetter  = uc_funcs.scenario_rhos
    scenario_fixer      = uc_funcs.id_fix_list_fct
    os.environ[uc_funcs.UC_NUMSCENS_ENV_VAR] = '../../uc/' + str(scen_count)

    fixer_options = {
        'verbose': False,
        'boundtol': 1e-2,
        'id_fix_list_fct': scenario_fixer,
    }

    rho_setter_kwargs = {
        'rho_scale_factor': 1.0,
    }

    PH_options = {
        'solvername': 'gurobi_persistent',
        'PHIterLimit': 25,
        'defaultPHrho': 1,
        'convthresh': 1e-6,
        'verbose': False,
        'display_timing': False,
        'display_progress': False,
        'iter0_solver_options': dict(),
        'iterk_solver_options': dict(),
        'rho_setter_kwargs': rho_setter_kwargs,
        # Comment either of these two lines out to disable fixing/initialization
        # 'fixeroptions': fixer_options,
        # 'init_W_fname': 'uc_25.5itr.weights',
        # 'init_Xbar_fname': 'uc_25.5itr_nobundle.xbar',
    }

    names = ['Scenario' + str(i+1) for i in range(scen_count)]

    ph = PH(PH_options, names, scenario_creator, scenario_denouement)

    conv, obj, bound = ph.ph_main(PH_extensions=UCExtension,
                                    rho_setter=scenario_rhosetter)
    print('Objective:', obj)

if __name__=='__main__':
    main()
