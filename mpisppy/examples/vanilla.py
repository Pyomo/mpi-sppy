""" Plain versions of dictionaries that can be modified for each example
    as needed.
    ASSUME the corresponding args have been set up.
    IDIOM: we feel free to have unused dictionary entries."""

import copy
# Hub and spoke SPBase classes
from mpisppy.phbase import PHBase
from mpisppy.opt.ph import PH
from mpisppy.fwph.fwph import FWPH
from mpisppy.cylinders.fwph_spoke import FrankWolfeOuterBound
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.cylinders.xhatlooper_bounder import XhatLooperInnerBound
from mpisppy.cylinders.xhatspecific_bounder import XhatSpecificInnerBound
from mpisppy.cylinders.hub import PHHub


def shared_options(args):
    shoptions = {
        "solvername": args.solver_name,
        "defaultPHrho": args.default_rho,
        "convthresh": 0,
        "PHIterLimit": args.max_iterations,  # not needed by all
        "verbose": args.with_verbose,
        "display_progress": args.with_display_progress,
        "display_timing": args.with_display_timing,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
    }
    return shoptions


def ph_hub(args,
           scenario_creator,
           scenario_denouement,
           all_scenario_names,
           cb_data=None,
           ph_extensions=None,
           rho_setter=None):
    shoptions = shared_options(args)
    PHoptions = copy.deepcopy(shoptions)
    PHoptions["convthresh"] = args.intra_hub_conv_thresh
    PHoptions["bundles_per_rank"] = args.bundles_per_rank

    hub_dict = {
        "hub_class": PHHub,
        "hub_kwargs": {"options": {"rel_gap": args.rel_gap,
                                   "abs_gap": args.abs_gap}},
        "opt_class": PH,
        "opt_kwargs": {
            "PHoptions": PHoptions,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
            "rho_setter": rho_setter,
            "PH_extensions": ph_extensions,
        }
    }
    return hub_dict


def fwph_spoke(args,
               scenario_creator,
               scenario_denouement,
               all_scenario_names,
               cb_data=None):
    shoptions = shared_options(args)
    fw_options = {
        "FW_iter_limit": args.fwph_iter_limit,
        "FW_weight": args.fwph_weight,
        "FW_conv_thresh": args.fwph_conv_thresh,
        "stop_check_tol": args.fwph_stop_check_tol,
        "solvername": args.solver_name,
        "FW_verbose": args.with_verbose,
    }
    fw_dict = {
        "spoke_class": FrankWolfeOuterBound,
        "spoke_kwargs": dict(),
        "opt_class": FWPH,
        "opt_kwargs": {
            "PH_options": shoptions,  # be sure convthresh is zero for fwph
            "FW_options": fw_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        },
    }
    return fw_dict

def lagrangian_spoke(args,
                  scenario_creator,
                  scenario_denouement,
                  all_scenario_names,
                  cb_data=None,
                  rho_setter=None):
    shoptions = shared_options(args)
    lagrangian_spoke = {
        "spoke_class": LagrangianOuterBound,
        "spoke_kwargs": dict(),
        "opt_class": PHBase,
        "opt_kwargs": {
            "PHoptions": shoptions,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
            "rho_setter": rho_setter,
            'scenario_denouement': scenario_denouement,
        }
    }
    return lagrangian_spoke

        
def xhatlooper_spoke(args,
               scenario_creator,
               scenario_denouement,
               all_scenario_names,
               cb_data=None):
    
    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhat_options["xhat_looper_options"] = {
        "xhat_solver_options": shoptions["iterk_solver_options"],
        "scen_limit": args.xhat_scen_limit,
        "dump_prefix": "delme",
        "csvname": "looper.csv",
    }
    xhatlooper_dict = {
        "spoke_class": XhatLooperInnerBound,
        "spoke_kwargs": dict(),
        "opt_class": PHBase,
        "opt_kwargs": {
            "PHoptions": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        },
    }
    return xhatlooper_dict


# this xhatspecific_spoke is for multistage
def xhatspecific_spoke(args,
                       scenario_creator,
                       scenario_denouement,
                       all_scenario_names,
                       scenario_dict,
                       all_nodenames,
                       BFs,
                       cb_data=None):
    
    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options\
        ["xhat_specific_options"] = {"xhat_solver_options":
                                     shoptions["iterk_solver_options"],
                                     "xhat_scenario_dict": scenario_dict,
                                     "csvname": "specific.csv"}
    xhat_options["branching_factors"] = BFs

    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhatspecific_dict = {
        "spoke_class": XhatSpecificInnerBound,
        "spoke_kwargs": dict(),
        "opt_class": PHBase,
        "opt_kwargs": {
            "PHoptions": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
            "all_nodenames": all_nodenames,
        },
    }
    return xhatspecific_dict
