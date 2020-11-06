# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
""" Plain versions of dictionaries that can be modified for each example
    as needed.
    ASSUME the corresponding args have been set up.
    IDIOM: we feel free to have unused dictionary entries."""

import copy
# Hub and spoke SPBase classes
from mpisppy.phbase import PHBase
from mpisppy.opt.ph import PH
from mpisppy.opt.lshaped import LShapedMethod
from mpisppy.fwph.fwph import FWPH
from mpisppy.utils.xhat_tryer import XhatTryer
from mpisppy.cylinders.fwph_spoke import FrankWolfeOuterBound
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.cylinders.lagranger_bounder import LagrangerOuterBound
from mpisppy.cylinders.xhatlooper_bounder import XhatLooperInnerBound
from mpisppy.cylinders.xhatspecific_bounder import XhatSpecificInnerBound
from mpisppy.cylinders.xhatshufflelooper_bounder import XhatShuffleInnerBound
from mpisppy.cylinders.lshaped_bounder import XhatLShapedInnerBound
from mpisppy.cylinders.slam_heuristic import SlamUpHeuristic, SlamDownHeuristic
from mpisppy.cylinders.cross_scen_spoke import CrossScenarioCutSpoke
from mpisppy.cylinders.cross_scen_hub import CrossScenarioHub
from mpisppy.cylinders.hub import PHHub

def _hasit(args, argname):
    return hasattr(args, argname) and getattr(args, argname) is not None

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
        "tee-rank0-solves": args.tee_rank0_solves,
        "trace_prefix" : args.trace_prefix,
    }
    if _hasit(args, "max_solver_threads"):
        shoptions["iter0_solver_options"]["threads"] = args.max_solver_threads
        shoptions["iterk_solver_options"]["threads"] = args.max_solver_threads
    if _hasit(args, "iter0_mipgap"):
        shoptions["iter0_solver_options"]["mipgap"] = args.iter0_mipgap
    if _hasit(args, "iterk_mipgap"):
        shoptions["iterk_solver_options"]["mipgap"] = args.iterk_mipgap
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

    if _hasit(args, "with_cross_scenario_cuts") and args.with_cross_scenario_cuts:
        hub_class = CrossScenarioHub
    else:
        hub_class = PHHub

    hub_dict = {
        "hub_class": hub_class,
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

    mip_solver_options, qp_solver_options = dict(), dict()
    if _hasit(args, "max_solver_threads"):
        mip_solver_options["threads"] = args.max_solver_threads
        qp_solver_options["threads"] = args.max_solver_threads
    if _hasit(args, "fwph_mipgap"):
        mip_solver_options["mipgap"] = args.fwph_mipgap

    fw_options = {
        "FW_iter_limit": args.fwph_iter_limit,
        "FW_weight": args.fwph_weight,
        "FW_conv_thresh": args.fwph_conv_thresh,
        "stop_check_tol": args.fwph_stop_check_tol,
        "solvername": args.solver_name,
        "FW_verbose": args.with_verbose,
        "mip_solver_options" : mip_solver_options,
        "qp_solver_options" : qp_solver_options,
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
    if args.lagrangian_iter0_mipgap is not None:
        lagrangian_spoke["opt_kwargs"]["PHoptions"]["iter0_solver_options"]\
            ["mipgap"] = args.lagrangian_iter0_mipgap
    if args.lagrangian_iterk_mipgap is not None:
        lagrangian_spoke["opt_kwargs"]["PHoptions"]["iterk_solver_options"]\
            ["mipgap"] = args.lagrangian_iterk_mipgap
    return lagrangian_spoke


# special lagrangian that computes its own xhat and W
def lagranger_spoke(args,
                  scenario_creator,
                  scenario_denouement,
                  all_scenario_names,
                  cb_data=None,
                  rho_setter=None):
    shoptions = shared_options(args)
    lagranger_spoke = {
        "spoke_class": LagrangerOuterBound,
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
    if args.lagranger_iter0_mipgap is not None:
        lagranger_spoke["opt_kwargs"]["PHoptions"]["iter0_solver_options"]\
            ["mipgap"] = args.lagranger_iter0_mipgap
    if args.lagranger_iterk_mipgap is not None:
        lagranger_spoke["opt_kwargs"]["PHoptions"]["iterk_solver_options"]\
            ["mipgap"] = args.lagranger_iterk_mipgap
    if args.lagranger_rho_rescale_factors_json is not None:
        lagranger_spoke["opt_kwargs"]["PHoptions"]\
            ["lagranger_rho_rescale_factors_json"]\
            = args.lagranger_rho_rescale_factors_json
        
    return lagranger_spoke

        
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


def xhatshuffle_spoke(args,
               scenario_creator,
               scenario_denouement,
               all_scenario_names,
               cb_data=None):

    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhat_options["xhat_looper_options"] = {
        "xhat_solver_options": shoptions["iterk_solver_options"],
        "dump_prefix": "delme",
        "csvname": "looper.csv",
    }
    xhatlooper_dict = {
        "spoke_class": XhatShuffleInnerBound,
        "spoke_kwargs": dict(),
        "opt_class": XhatTryer,
        "opt_kwargs": {
            "PHoptions": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        },
    }
    return xhatlooper_dict


def xhatspecific_spoke(args,
                       scenario_creator,
                       scenario_denouement,
                       all_scenario_names,
                       scenario_dict,
                       all_nodenames=None,
                       BFs=None,
                       cb_data=None):
    
    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options\
        ["xhat_specific_options"] = {"xhat_solver_options":
                                     shoptions["iterk_solver_options"],
                                     "xhat_scenario_dict": scenario_dict,
                                     "csvname": "specific.csv"}
    if BFs:
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

def xhatlshaped_spoke(args,
                      scenario_creator,
                      scenario_denouement,
                      all_scenario_names,
                      cb_data=None):
    
    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat

    xhatlshaped_dict = {
        "spoke_class": XhatLShapedInnerBound,
        "spoke_kwargs": dict(),
        "opt_class": XhatTryer,
        "opt_kwargs": {
            "PHoptions": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        },
    }
    return xhatlshaped_dict

def slamup_spoke(args,
               scenario_creator,
               scenario_denouement,
               all_scenario_names,
               cb_data=None):

    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhatlooper_dict = {
        "spoke_class": SlamUpHeuristic,
        "spoke_kwargs": dict(),
        "opt_class": XhatTryer,
        "opt_kwargs": {
            "PHoptions": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        },
    }
    return xhatlooper_dict

def slamdown_spoke(args,
               scenario_creator,
               scenario_denouement,
               all_scenario_names,
               cb_data=None):

    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhatlooper_dict = {
        "spoke_class": SlamDownHeuristic,
        "spoke_kwargs": dict(),
        "opt_class": XhatTryer,
        "opt_kwargs": {
            "PHoptions": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        },
    }
    return xhatlooper_dict

def cross_scenario_cut_spoke(args, 
               scenario_creator,
               scenario_denouement,
               all_scenario_names,
               cb_data=None):

    if _hasit(args, "max_solver_threads"):
        sp_solver_options = {"threads":args.max_solver_threads}
    else:
        sp_solver_options = dict() 

    if _hasit(args, "eta_bounds_mipgap"):
        sp_solver_options["mipgap"] = args.eta_bounds_mipgap

    ls_options = { "master_solver" : args.solver_name,
                   "sp_solver": args.solver_name,
                   "sp_solver_options" : sp_solver_options,
                    "verbose": args.with_verbose,
                 }
    cut_spoke = {
        "spoke_class": CrossScenarioCutSpoke,
        "spoke_kwargs": dict(),
        "opt_class": LShapedMethod,
        "opt_kwargs": {
            "options": ls_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_denouement": scenario_denouement,
            "cb_data": cb_data,
            },
        }

    return cut_spoke
