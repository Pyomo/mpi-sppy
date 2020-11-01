# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# DLW version: add Xhat
import mpisppy.examples.netdes.netdes as netdes

# Hub and spoke SPBase classes
from mpisppy.phbase import PHBase
from mpisppy.opt.ph import PH
from mpisppy.fwph.fwph import FWPH
# Hub and spoke SPCommunicator classes
from mpisppy.cylinders.fwph_spoke import FrankWolfeOuterBound
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.cylinders.xhatlooper_bounder import XhatLooperInnerBound
from mpisppy.cylinders.hub import PHHub
# Make it all go
from mpisppy.utils.sputils import spin_the_wheel


if __name__=="__main__":
    """ For testing and debugging only 
    
        Race a Lagrangian spoke against an FWPH spoke
        to compute and outer bound for a network design problem
    """

    # Use netdes as an example
    inst = "network-10-20-L-01"
    num_scen = int(inst.split("-")[-3])
    scenario_names = [f"Scen{i}" for i in range(num_scen)]
    cb_data = f"{netdes.__file__[:-10]}/data/{inst}.dat"
    scenario_creator = netdes.scenario_creator

    hub_ph_options = {
        "solvername": "gurobi_persistent",
        "PHIterLimit": 1000,
        "defaultPHrho": 10000, # Big for netdes
        "convthresh": 0.,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
    }

    # Vanilla PH hub
    hub_dict = {
        "hub_class": PHHub,
        "hub_kwargs": {
            "options": {
                "rel_gap": 0.05,
                "abs_gap": 10000,
            }
        },
        "opt_class": PH,
        "opt_kwargs": {
            "PHoptions": hub_ph_options,
            "all_scenario_names": scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        }
    }

    ph_options = {
        "solvername": "gurobi_persistent",
        "PHIterLimit": 1000,
        "defaultPHrho": 10000, # Big for netdes
        "convthresh": 0., # To prevent FWPH from terminating
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
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
            "PH_options": ph_options,
            "FW_options": fw_options,
            "all_scenario_names": scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        }
    }

    # Standard Lagrangian bound spoke
    lagrangian_spoke = {
        "spoke_class": LagrangianOuterBound,
        "spoke_kwargs": dict(),
        "opt_class": PHBase,
        "opt_kwargs": {
            "PHoptions": ph_options,
            "all_scenario_names": scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        }
    }

    # xhat looper bound spoke
    xhat_options = hub_ph_options.copy()
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhat_options["xhat_looper_options"] =  {"xhat_solver_options":\
                                          ph_options["iterk_solver_options"],
                                          "scen_limit": 3,
                                          "dump_prefix": "delme",
                                          "csvname": "looper.csv"}
    xhatlooper_spoke = {
        "spoke_class": XhatLooperInnerBound,
        "spoke_kwargs": dict(),
        "opt_class": PHBase,
        "opt_kwargs": {
            "PHoptions": xhat_options,
            "all_scenario_names": scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        }
    }
    list_of_spoke_dict = (fw_spoke, lagrangian_spoke, xhatlooper_spoke)

    spin_the_wheel(hub_dict, list_of_spoke_dict)
