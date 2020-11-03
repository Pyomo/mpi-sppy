# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import datetime
import logging
import sys
import os
import mpi4py.MPI as mpi
import copy

# Hub and spoke SPBase classes
from mpisppy.phbase import PHBase
from mpisppy.opt.ph import PH

# Hub and spoke SPCommunicator classes
from mpisppy.cylinders.xhatshufflelooper_bounder import XhatShuffleInnerBound

# extensions for the hub
from mpisppy.extensions.extension import MultiPHExtension
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.cylinders.hub import PHHub 
from mpisppy.cylinders.cross_scen_hub import CrossScenarioHub
from mpisppy.cylinders.cross_scen_spoke import CrossScenarioCutSpoke
from mpisppy.opt.lshaped import LShapedMethod

from mpisppy.utils.xhat_tryer import XhatTryer



# Make it all go
from mpisppy.utils.sputils import spin_the_wheel
# the problem
from mpisppy.log import setup_logger
import mpisppy.examples.farmer.farmer as farmer

import pyomo.environ as pyo

# mpi setup
fullcomm = mpi.COMM_WORLD
rank_global = fullcomm.Get_rank()


def _usage():
    print("usage mpiexec -np {N} python -m mpi4py cs_farmer.py {CropsMult} {scen_count} {bundles_per_rank} {PHIterLimit}")
    print("e.g., mpiexec -np 3 python -m mpi4py cs_farmer.py 1 3 0 50")
    sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename='dlw.log',
                        filemode='w', format='(%(threadName)-10s) %(message)s')
    setup_logger(f'dtm{rank_global}', f'dtm{rank_global}.log')
    dtm = logging.getLogger(f'dtm{rank_global}')

    if len(sys.argv) != 5:
        _usage()

    CropsMult = int(sys.argv[1])
    scen_count = int(sys.argv[2])
    bundles_per_rank = int(sys.argv[3])
    PHIterLimit = int(sys.argv[4])

    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement

    all_scenario_names = ['scen{}'.format(sn) for sn in range(scen_count)]
    rho_setter = farmer._rho_setter if hasattr(farmer, '_rho_setter') else None
    cb_data={'use_integer': True, "CropsMult": CropsMult, "sense":pyo.maximize}

    hub_ph_options = {
        "solvername": "xpress_persistent",
        'bundles_per_rank': bundles_per_rank,  # 0 = no bundles
        "asynchronousPH": False,
        "PHIterLimit": PHIterLimit,
        "defaultPHrho": 0.5,
        "convthresh": 0.0,
        "subsolvedirectives": None,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "tee-rank0-solves": False,
        "iter0_solver_options": None,
        "iterk_solver_options": None,
        "cross_scen_options":{"check_bound_improve_iterations":2}, 
    }

    multi_ext = { 'ext_classes' : [CrossScenarioExtension] }
    #multi_ext = { 'ext_classes' : [] }

    # PH hub
    hub_dict = {
        "hub_class": CrossScenarioHub,
        "hub_kwargs": dict(),
        "opt_class": PH,
        "opt_kwargs": {
            "PHoptions": hub_ph_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
            "rho_setter": rho_setter,
            "PH_extensions": MultiPHExtension,
            "PH_extension_kwargs": multi_ext,
        }
    }

    # TBD: this should have a different rho setter than the optimizer

    ph_options = copy.deepcopy(hub_ph_options)  # many will not be used
    ph_options['bundles_per_rank'] = 0 # no bundles for xhat
    ph_options["xhat_looper_options"] = \
        {
            "xhat_solver_options": None,
            "scen_limit": 3,
            "dump_prefix": "delme",
            "csvname": "looper.csv",
        }

    ub_spoke = {
        'spoke_class': XhatShuffleInnerBound,
        "spoke_kwargs": dict(),
        "opt_class": XhatTryer,
        'opt_kwargs': {
            'PHoptions': ph_options,
            'all_scenario_names': all_scenario_names,
            'scenario_creator': scenario_creator,
            'scenario_denouement': scenario_denouement,
            "cb_data": cb_data,
        },
    }

    ls_options = {
        "master_solver": "xpress_persistent",
        "sp_solver": "xpress_persistent",
        "sp_solver_options": {"threads":1},
       }
    cut_spoke = {
        'spoke_class': CrossScenarioCutSpoke,
        "spoke_kwargs": dict(),
        "opt_class": LShapedMethod,
        'opt_kwargs': {
            'options': ls_options,
            'all_scenario_names': all_scenario_names,
            'scenario_creator': scenario_creator,
            'scenario_denouement': scenario_denouement,
            "cb_data": cb_data
        },
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
            'scenario_denouement': scenario_denouement,
        },
    }

    list_of_spoke_dict = (ub_spoke, cut_spoke, ) #lagrangian_spoke)
    #list_of_spoke_dict = (ub_spoke, )

    spcomm, opt_dict = spin_the_wheel(hub_dict, list_of_spoke_dict)

    # there are ways to get the answer sooner
    if "hub_class" in opt_dict:  # we are hub rank
        if spcomm.opt.rank == spcomm.opt.rank0:  # we are the reporting hub rank
            print("BestInnerBound={} and BestOuterBound={}". \
                  format(spcomm.BestInnerBound, spcomm.BestOuterBound))
    print("End time={} for global rank={}". \
          format(datetime.datetime.now(), rank_global))
