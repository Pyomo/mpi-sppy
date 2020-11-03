# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
## includes fwph
# updated 23April2020
# dlw August 2019: 3driver caller for the uc example; unix path names

import datetime
import logging
import sys
import os
import mpi4py.MPI as mpi

# Hub and spoke SPBase classes
from mpisppy.phbase import PHBase
from mpisppy.opt.ph import PH
from mpisppy.fwph.fwph import FWPH
# Hub and spoke SPCommunicator classes
from mpisppy.cylinders.fwph_spoke import FrankWolfeOuterBound
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.cylinders.xhatlooper_bounder import XhatLooperInnerBound
from mpisppy.cylinders.hub import PHHub
# extensions for the hub
from mpisppy.extensions.extension import MultiPHExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.mipgapper import Gapper
# Make it all go
from mpisppy.utils.sputils import spin_the_wheel
# the problem
from mpisppy.examples.uc.uc_funcs import scenario_creator, \
                                       scenario_denouement, \
                                       _rho_setter, \
                                       id_fix_list_fct
from mpisppy.log import setup_logger

# mpi setup
fullcomm = mpi.COMM_WORLD
rank_global = fullcomm.Get_rank()

def _usage():
    print("usage: mpiexec -np {N} python -m mpi4py uc4wood.py {ScenCount} {bundles_per_rank} {PHIterLimit} {fixer|nofixer")
    print("e.g., mpiexec -np 4 python -m mpi4py uc4wood.py 10 0 5 fixer")
    sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename='dlw.log',
                        filemode='w', format='(%(threadName)-10s) %(message)s')
    setup_logger(f'dtm{rank_global}', f'dtm{rank_global}.log')
    dtm = logging.getLogger(f'dtm{rank_global}')

    if len(sys.argv) != 5:
        _usage()

    print("Start time={} for global rank={}".\
          format(datetime.datetime.now(), rank_global))

    try:
        ScenCount = int(sys.argv[1])
        bundles_per_rank = int(sys.argv[2])
        PHIterLimit = int(sys.argv[3])
    except:
        _usage()
    if sys.argv[4] == "fixer":
        usefixer = True
    elif sys.argv[4] == "nofixer":
        usefixer = False
    else:
        print ("The last arg must be fixer or else nofixer.")
        _usage()
        
    assert(ScenCount in [3,5,10,25,50])
    all_scenario_names = ['Scenario{}'.format(sn+1) for sn in range(ScenCount)]
    # sent to the scenario creator
    cb_data = {"scenario-count": ScenCount,
               "path": str(ScenCount)+"scenarios_r1"}
    
    hub_ph_options = {
        "solvername": "gurobi_persistent",
        'bundles_per_rank': bundles_per_rank, # 0 = no bundles
        "asynchronousPH": False,
        "PHIterLimit": PHIterLimit,
        "defaultPHrho": 1,
        "convthresh": 0.001,
        "subsolvedirectives": None,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "tee-rank0-solves": False,
        "iter0_solver_options" : {
            "mipgap": 1e-7, # So the trivial bound is accurate (but wastes time in hub)
            "threads": 2,
        },
        "iterk_solver_options": {
            "mipgap": 0.001,
            "threads": 2,
        },
        "fixeroptions": {
            "verbose": False,
            "boundtol": 0.01,
            "id_fix_list_fct": id_fix_list_fct,
        },
        "gapperoptions": {
            "verbose": False,
            "mipgapdict": dict() , # Setting this changes iter0_solver_options
        },
    }
    if usefixer==True:
        multi_ext = {"ext_classes": [Fixer, Gapper]}
    else:
        multi_ext = {"ext_classes": [Gapper]}

    # PH hub
    hub_dict = {
        "hub_class": PHHub,
        "hub_kwargs": dict(),
        "opt_class": PH,
        "opt_kwargs": {
            "PHoptions": hub_ph_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
            "rho_setter": _rho_setter,
            "PH_extensions": MultiPHExtension,
            "PH_extension_kwargs": multi_ext,
        }
    }

    # FWPH spoke
    fw_options = {
        "FW_iter_limit": 10,
        "FW_weight": 0., # Or 1.? I forget what this does, honestly
        "FW_conv_thresh": 1e-4,
        "solvername": "gurobi_persistent",
        "FW_verbose": False,
    }
    fw_spoke = {
        "spoke_class": FrankWolfeOuterBound,
        "spoke_kwargs": dict(),
        "opt_class": FWPH,
        "opt_kwargs": {
            "PH_options": hub_ph_options,  # many are ignored
            "FW_options": fw_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        }
    }

    lagrangian_spoke = {
        "spoke_class": LagrangianOuterBound,
        "spoke_kwargs": dict(),
        "opt_class": PHBase,   
        'opt_kwargs': {
            'PHoptions': hub_ph_options,
            'all_scenario_names': all_scenario_names,
            'scenario_creator': scenario_creator,
            "cb_data": cb_data,
            "rho_setter": _rho_setter,
            'scenario_denouement': scenario_denouement,
        },
    }

    # maybe we should copy the options sometimes
    xhat_options = hub_ph_options.copy()
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhat_options["xhat_looper_options"] =\
    {
        "xhat_solver_options": {
            "mipgap": 0.001,
            "threads": 2,
        },
        "scen_limit": 3,
        "dump_prefix": "delme",
        "csvname": "looper.csv",
    }
    """
        "xhat_closest_options": {
            "xhat_solver_options": {
                "mipgap": 0.001,
                "threads": 2,
            },
            "csvname": "closest.csv",
        },
        "xhat_specific_options": {
            "xhat_solver_options": {
                "mipgap": 0.001,
                "threads": 2,
            },
            "xhat_scenario_name": "Scenario3",
            "csvname": "specific.csv",
        },
    """

    ub_spoke = {
        'spoke_class': XhatLooperInnerBound,
        "spoke_kwargs": dict(),
        "opt_class": PHBase,   
        'opt_kwargs': {
            'PHoptions': xhat_options,
            'all_scenario_names': all_scenario_names,
            'scenario_creator': scenario_creator,
            'scenario_denouement': scenario_denouement,
            "cb_data": cb_data,
        },
    }

    list_of_spoke_dict = (fw_spoke, lagrangian_spoke, ub_spoke)

    spcomm, opt_dict = spin_the_wheel(hub_dict, list_of_spoke_dict)
    # there are ways to get the answer sooner
    if "hub_class" in opt_dict:  # we are hub rank
        if spcomm.opt.rank == spcomm.opt.rank0:  # we are the reporting hub rank
            print("BestInnerBound={} and BestOuterBound={}".\
                  format(spcomm.BestInnerBound, spcomm.BestOuterBound))
    print("End time={} for global rank={}".\
          format(datetime.datetime.now(), rank_global))
