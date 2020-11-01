# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' Driver script to test the w/xbar read/write extensions using UC 
'''

import mpisppy.examples.uc.uc_funcs as uc_funcs
from mpisppy.utils.wxbarwriter import WXBarWriter
from mpisppy.utils.wxbarreader import WXBarReader
import os

from mpisppy.opt.ph import PH

def nonsense(arg1, arg2, arg3): # Empty scenario_denouement
    pass

def read_test():
    scen_count          = 3
    scenario_creator    = uc_funcs.pysp2_callback
    scenario_denouement = nonsense
    scenario_rhosetter  = uc_funcs.scenario_rhos
    os.environ[uc_funcs.UC_NUMSCENS_ENV_VAR] = 'examples/uc/' + str(scen_count)

    PH_options = {
        'solvername': 'gurobi',
        'PHIterLimit': 2,
        'defaultPHrho': 1,
        'convthresh': 1e-6,
        'verbose': False,
        'display_timing': False,
        'display_progress': False,
        'iter0_solver_options': dict(),
        'iterk_solver_options': dict(),
        'init_W_fname': 'david/weights.csv', # Path to the weight files
        'init_separate_W_files': False,
        'init_Xbar_fname': 'david/somexbars.csv',
    }

    names = ['Scenario' + str(i+1) for i in range(scen_count)]

    ph = PH(PH_options, names, scenario_creator, scenario_denouement)

    conv, obj, bound = ph.ph_main(PH_extensions=WXBarReader,
                                    rho_setter=scenario_rhosetter)

def write_test():
    scen_count          = 3
    scenario_creator    = uc_funcs.pysp2_callback
    scenario_denouement = nonsense
    scenario_rhosetter  = uc_funcs.scenario_rhos
    os.environ[uc_funcs.UC_NUMSCENS_ENV_VAR] = 'examples/uc/' + str(scen_count)

    PH_options = {
        'solvername': 'gurobi',
        'PHIterLimit': 2,
        'defaultPHrho': 1,
        'convthresh': 1e-6,
        'verbose': False,
        'display_timing': False,
        'display_progress': False,
        'iter0_solver_options': dict(),
        'iterk_solver_options': dict(),
        'W_fname': 'david/weights.csv',
        'separate_W_files': False,
        'Xbar_fname': 'somexbars.csv',
    }

    names = ['Scenario' + str(i+1) for i in range(scen_count)]

    ph = PH(PH_options, names, scenario_creator, scenario_denouement)

    conv, obj, bound = ph.ph_main(PH_extensions=WXBarWriter,
                                    rho_setter=scenario_rhosetter)

if __name__=='__main__':
    read_test()
