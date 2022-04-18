# If possible, use cfg_vanilla and config.py instead of this and baseparsers.py.
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
from mpisppy.opt.aph import APH
from mpisppy.opt.lshaped import LShapedMethod
from mpisppy.fwph.fwph import FWPH
from mpisppy.utils.xhat_eval import Xhat_Eval
import mpisppy.utils.sputils as sputils
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
from mpisppy.cylinders.hub import APHHub
from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension

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
        "display_convergence_detail": args.with_display_convergence_detail,
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

def add_multistage_options(cylinder_dict,all_nodenames,branching_factors):
    cylinder_dict = copy.deepcopy(cylinder_dict)
    if branching_factors is not None:
        if hasattr(cylinder_dict["opt_kwargs"], "options"):
            cylinder_dict["opt_kwargs"]["options"]["branching_factors"] = branching_factors
        if all_nodenames is None:
            all_nodenames = sputils.create_nodenames_from_branching_factors(branching_factors)
    if all_nodenames is not None:
        print("Hello, surprise !!")
        cylinder_dict["opt_kwargs"]["all_nodenames"] = all_nodenames
    print("Hello,",cylinder_dict)
    return cylinder_dict

def ph_hub(
        args,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=None,
        ph_extensions=None,
        ph_converger=None,
        rho_setter=None,
        variable_probability=None,
        all_nodenames=None,
        spoke_sleep_time=None,
):
    shoptions = shared_options(args)
    options = copy.deepcopy(shoptions)
    options["convthresh"] = args.intra_hub_conv_thresh
    options["bundles_per_rank"] = args.bundles_per_rank
    options["linearize_binary_proximal_terms"] = args.linearize_binary_proximal_terms
    options["linearize_proximal_terms"] = args.linearize_proximal_terms
    options["proximal_linearization_tolerance"] = args.proximal_linearization_tolerance

    if _hasit(args, "with_cross_scenario_cuts") and args.with_cross_scenario_cuts:
        hub_class = CrossScenarioHub
    else:
        hub_class = PHHub

    hub_dict = {
        "hub_class": hub_class,
        "hub_kwargs": {"options": {"spoke_sleep_time": spoke_sleep_time,
                                   "rel_gap": args.rel_gap,
                                   "abs_gap": args.abs_gap,
                                   "max_stalled_iters": args.max_stalled_iters}},
        "opt_class": PH,
        "opt_kwargs": {
            "options": options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement,
            "rho_setter": rho_setter,
            "variable_probability": variable_probability,
            "extensions": ph_extensions,
            "ph_converger": ph_converger,
            "all_nodenames": all_nodenames
        }
    }
    
    return hub_dict


def aph_hub(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    ph_extensions=None,
    rho_setter=None,
    variable_probability=None,
    all_nodenames=None,
    spoke_sleep_time=None,
):
    hub_dict = ph_hub(args,
                      scenario_creator,
                      scenario_denouement,
                      all_scenario_names,
                      scenario_creator_kwargs=scenario_creator_kwargs,
                      ph_extensions=ph_extensions,
                      rho_setter=rho_setter,
                      variable_probability=variable_probability,
                      all_nodenames = all_nodenames,
                      spoke_sleep_time = spoke_sleep_time)

    hub_dict['hub_class'] = APHHub
    hub_dict['opt_class'] = APH    

    hub_dict['opt_kwargs']['options']['APHgamma'] = args.aph_gamma
    hub_dict['opt_kwargs']['options']['APHnu'] = args.aph_nu
    hub_dict['opt_kwargs']['options']['async_frac_needed'] = args.aph_frac_needed
    hub_dict['opt_kwargs']['options']['dispatch_frac'] = args.aph_dispatch_frac
    hub_dict['opt_kwargs']['options']['async_sleep_secs'] = args.aph_sleep_seconds

    return hub_dict


def extension_adder(hub_dict,ext_class):
    if "extensions" not in hub_dict["opt_kwargs"] or \
        hub_dict["opt_kwargs"]["extensions"] is None:
        hub_dict["opt_kwargs"]["extensions"] = ext_class
    elif hub_dict["opt_kwargs"]["extensions"] == MultiExtension:
        if not ext_class in  hub_dict["opt_kwargs"]["ext_classes"]:
            hub_dict["opt_kwargs"]["ext_classes"].append(ext_class)
    elif hub_dict["opt_kwargs"]["extensions"] != ext_class: 
        #ext_class is the second extension
        if not "extensions_kwargs" in hub_dict["opt_kwargs"]:
            hub_dict["opt_kwargs"]["extension_kwargs"] = {
                "ext_classes": [hub_dict["opt_kwargs"]["extensions"],
                                                 ext_class]}
        else:
            hub_dict["opt_kwargs"]["extension_kwargs"]["ext_classes"] = \
                [hub_dict["opt_kwargs"]["extensions"],
                                                 ext_class]
        hub_dict["opt_kwargs"]["extensions"] = MultiExtension
    return hub_dict
    

def add_fixer(hub_dict,
              args,
              ):
    hub_dict = extension_adder(hub_dict,Fixer)
    hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {"verbose":False,
                                              "boundtol": args.fixer_tol,
                                              "id_fix_list_fct": args.id_fix_list_fct}
    return hub_dict

def add_cross_scenario_cuts(hub_dict,
                            args,
                            ):
    #WARNING: Do not use without a cross_scenario_cuts spoke
    hub_dict = extension_adder(hub_dict, CrossScenarioExtension)
    hub_dict["opt_kwargs"]["options"]["cross_scen_options"]\
            = {"check_bound_improve_iterations" : args.cross_scenario_iter_cnt}
    return hub_dict

def fwph_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    all_nodenames=None,
    spoke_sleep_time=None,
):
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
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": FWPH,
        "opt_kwargs": {
            "PH_options": shoptions,  # be sure convthresh is zero for fwph
            "FW_options": fw_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement,
            "all_nodenames": all_nodenames            
        },
    }
    
    return fw_dict


def lagrangian_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    rho_setter=None,
    all_nodenames=None,
    spoke_sleep_time=None,
):
    shoptions = shared_options(args)
    lagrangian_spoke = {
        "spoke_class": LagrangianOuterBound,
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": PHBase,
        "opt_kwargs": {
            "options": shoptions,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            'scenario_denouement': scenario_denouement,            
            "rho_setter": rho_setter,
            "all_nodenames": all_nodenames

        }
    }
    if args.lagrangian_iter0_mipgap is not None:
        lagrangian_spoke["opt_kwargs"]["options"]["iter0_solver_options"]\
            ["mipgap"] = args.lagrangian_iter0_mipgap
    if args.lagrangian_iterk_mipgap is not None:
        lagrangian_spoke["opt_kwargs"]["options"]["iterk_solver_options"]\
            ["mipgap"] = args.lagrangian_iterk_mipgap
    
    return lagrangian_spoke


# special lagrangian that computes its own xhat and W
def lagranger_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    rho_setter=None,
    all_nodenames = None,
    spoke_sleep_time=None,
):
    shoptions = shared_options(args)
    lagranger_spoke = {
        "spoke_class": LagrangerOuterBound,
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": PHBase,
        "opt_kwargs": {
            "options": shoptions,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            'scenario_denouement': scenario_denouement,            
            "rho_setter": rho_setter,
            "all_nodenames": all_nodenames
        }
    }
    if args.lagranger_iter0_mipgap is not None:
        lagranger_spoke["opt_kwargs"]["options"]["iter0_solver_options"]\
            ["mipgap"] = args.lagranger_iter0_mipgap
    if args.lagranger_iterk_mipgap is not None:
        lagranger_spoke["opt_kwargs"]["options"]["iterk_solver_options"]\
            ["mipgap"] = args.lagranger_iterk_mipgap
    if args.lagranger_rho_rescale_factors_json is not None:
        lagranger_spoke["opt_kwargs"]["options"]\
            ["lagranger_rho_rescale_factors_json"]\
            = args.lagranger_rho_rescale_factors_json
    
    return lagranger_spoke

        
def xhatlooper_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):
    
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
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": Xhat_Eval,
        "opt_kwargs": {
            "options": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement            
        },
    }
    return xhatlooper_dict


def xhatshuffle_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    all_nodenames=None,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):

    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhat_options["xhat_looper_options"] = {
        "xhat_solver_options": shoptions["iterk_solver_options"],
        "dump_prefix": "delme",
        "csvname": "looper.csv",
    }
    if _hasit(args,"add_reversed_shuffle"):
        xhat_options["xhat_looper_options"]["reverse"] = args.add_reversed_shuffle
    if _hasit(args,"add_reversed_shuffle"):
        xhat_options["xhat_looper_options"]["xhatshuffle_iter_step"] = args.xhatshuffle_iter_step
    
    xhatlooper_dict = {
        "spoke_class": XhatShuffleInnerBound,
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": Xhat_Eval,
        "opt_kwargs": {
            "options": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement,   
            "all_nodenames": all_nodenames                    
        },
    }

    return xhatlooper_dict


def xhatspecific_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_dict,
    all_nodenames=None,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):
    
    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options["xhat_specific_options"] = {
        "xhat_solver_options": shoptions["iterk_solver_options"],
        "xhat_scenario_dict": scenario_dict,
        "csvname": "specific.csv",
    }

    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhatspecific_dict = {
        "spoke_class": XhatSpecificInnerBound,
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": Xhat_Eval,
        "opt_kwargs": {
            "options": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement,
            "all_nodenames": all_nodenames
        },
    }

    return xhatspecific_dict

def xhatlshaped_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):
    
    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat

    xhatlshaped_dict = {
        "spoke_class": XhatLShapedInnerBound,
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": Xhat_Eval,
        "opt_kwargs": {
            "options": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement            
        },
    }
    return xhatlshaped_dict

def slamup_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):

    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhatlooper_dict = {
        "spoke_class": SlamUpHeuristic,
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": Xhat_Eval,
        "opt_kwargs": {
            "options": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement
        },
    }
    return xhatlooper_dict

def slamdown_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):

    shoptions = shared_options(args)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhatlooper_dict = {
        "spoke_class": SlamDownHeuristic,
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": Xhat_Eval,
        "opt_kwargs": {
            "options": xhat_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement            
        },
    }
    return xhatlooper_dict

def cross_scenario_cuts_spoke(
    args,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    all_nodenames=None,
    spoke_sleep_time=None,
):

    if _hasit(args, "max_solver_threads"):
        sp_solver_options = {"threads":args.max_solver_threads}
    else:
        sp_solver_options = dict() 

    if _hasit(args, "eta_bounds_mipgap"):
        sp_solver_options["mipgap"] = args.eta_bounds_mipgap

    ls_options = { "root_solver" : args.solver_name,
                   "sp_solver": args.solver_name,
                   "sp_solver_options" : sp_solver_options,
                    "verbose": args.with_verbose,
                 }
    cut_spoke = {
        "spoke_class": CrossScenarioCutSpoke,
        "spoke_kwargs": {"options":{"spoke_sleep_time":spoke_sleep_time}},
        "opt_class": LShapedMethod,
        "opt_kwargs": {
            "options": ls_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement,
            "all_nodenames": all_nodenames
            },
        }

    return cut_spoke

        
