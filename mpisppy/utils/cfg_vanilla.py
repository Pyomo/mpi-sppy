# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
""" **** cfg version of vanilla that uses the Pyomo configuration system
    Plain versions of dictionaries that can be modified for each example as needed.
    ASSUME the corresponding args have been set up.
    IDIOM: we feel free to have unused dictionary entries."""

import copy
from mpisppy.utils import config

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

def _hasit(argname):
    # aside: config.global_config functions as a dict or object
    return config.global_config.get(argname) is not None and config.global_config[argname] is not None

def shared_options(cfg):
    shoptions = {
        "solvername": cfg.solver_name,
        "defaultPHrho": cfg.default_rho,
        "convthresh": 0,
        "PHIterLimit": cfg.max_iterations,  # not needed by all
        "verbose": cfg.verbose,
        "display_progress": cfg.display_progress,
        "display_convergence_detail": cfg.display_convergence_detail,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
        "tee-rank0-solves": cfg.tee_rank0_solves,
        "trace_prefix" : cfg.trace_prefix,
    }
    if _hasit("max_solver_threads"):
        shoptions["iter0_solver_options"]["threads"] = cfg.max_solver_threads
        shoptions["iterk_solver_options"]["threads"] = cfg.max_solver_threads
    if _hasit("iter0_mipgap"):
        shoptions["iter0_solver_options"]["mipgap"] = cfg.iter0_mipgap
    if _hasit("iterk_mipgap"):
        shoptions["iterk_solver_options"]["mipgap"] = cfg.iterk_mipgap
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
        cfg,
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
    shoptions = shared_options(cfg)
    options = copy.deepcopy(shoptions)
    options["convthresh"] = cfg.intra_hub_conv_thresh
    options["bundles_per_rank"] = cfg.bundles_per_rank
    options["linearize_binary_proximal_terms"] = cfg.linearize_binary_proximal_terms
    options["linearize_proximal_terms"] = cfg.linearize_proximal_terms
    options["proximal_linearization_tolerance"] = cfg.proximal_linearization_tolerance

    if _hasit("cross_scenario_cuts") and cfg.cross_scenario_cuts:
        hub_class = CrossScenarioHub
    else:
        hub_class = PHHub

    hub_dict = {
        "hub_class": hub_class,
        "hub_kwargs": {"options": {"spoke_sleep_time": spoke_sleep_time,
                                   "rel_gap": cfg.rel_gap,
                                   "abs_gap": cfg.abs_gap,
                                   "max_stalled_iters": cfg.max_stalled_iters}},
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


def aph_hub(cfg,
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
    hub_dict = ph_hub(cfg,
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

    hub_dict['opt_kwargs']['options']['APHgamma'] = cfg.aph_gamma
    hub_dict['opt_kwargs']['options']['APHnu'] = cfg.aph_nu
    hub_dict['opt_kwargs']['options']['async_frac_needed'] = cfg.aph_frac_needed
    hub_dict['opt_kwargs']['options']['dispatch_frac'] = cfg.aph_dispatch_frac
    hub_dict['opt_kwargs']['options']['async_sleep_secs'] = cfg.aph_sleep_seconds

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
              cfg,
              ):
    hub_dict = extension_adder(hub_dict,Fixer)
    hub_dict["opt_kwargs"]["options"]["fixeroptions"] = {"verbose":False,
                                              "boundtol": cfg.fixer_tol,
                                              "id_fix_list_fct": cfg.id_fix_list_fct}
    return hub_dict

def add_cross_scenario_cuts(hub_dict,
                            cfg,
                            ):
    #WARNING: Do not use without a cross_scenario_cuts spoke
    hub_dict = extension_adder(hub_dict, CrossScenarioExtension)
    hub_dict["opt_kwargs"]["options"]["cross_scen_options"]\
            = {"check_bound_improve_iterations" : cfg.cross_scenario_iter_cnt}
    return hub_dict

def fwph_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    all_nodenames=None,
    spoke_sleep_time=None,
):
    shoptions = shared_options(cfg)

    mip_solver_options, qp_solver_options = dict(), dict()
    if _hasit("max_solver_threads"):
        mip_solver_options["threads"] = cfg.max_solver_threads
        qp_solver_options["threads"] = cfg.max_solver_threads
    if _hasit("fwph_mipgap"):
        mip_solver_options["mipgap"] = cfg.fwph_mipgap

    fw_options = {
        "FW_iter_limit": cfg.fwph_iter_limit,
        "FW_weight": cfg.fwph_weight,
        "FW_conv_thresh": cfg.fwph_conv_thresh,
        "stop_check_tol": cfg.fwph_stop_check_tol,
        "solvername": cfg.solver_name,
        "FW_verbose": cfg.verbose,
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
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    rho_setter=None,
    all_nodenames=None,
    spoke_sleep_time=None,
):
    shoptions = shared_options(cfg)
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
    if cfg.lagrangian_iter0_mipgap is not None:
        lagrangian_spoke["opt_kwargs"]["options"]["iter0_solver_options"]\
            ["mipgap"] = cfg.lagrangian_iter0_mipgap
    if cfg.lagrangian_iterk_mipgap is not None:
        lagrangian_spoke["opt_kwargs"]["options"]["iterk_solver_options"]\
            ["mipgap"] = cfg.lagrangian_iterk_mipgap
    
    return lagrangian_spoke


# special lagrangian that computes its own xhat and W
def lagranger_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    rho_setter=None,
    all_nodenames = None,
    spoke_sleep_time=None,
):
    shoptions = shared_options(cfg)
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
    if cfg.lagranger_iter0_mipgap is not None:
        lagranger_spoke["opt_kwargs"]["options"]["iter0_solver_options"]\
            ["mipgap"] = cfg.lagranger_iter0_mipgap
    if cfg.lagranger_iterk_mipgap is not None:
        lagranger_spoke["opt_kwargs"]["options"]["iterk_solver_options"]\
            ["mipgap"] = cfg.lagranger_iterk_mipgap
    if cfg.lagranger_rho_rescale_factors_json is not None:
        lagranger_spoke["opt_kwargs"]["options"]\
            ["lagranger_rho_rescale_factors_json"]\
            = cfg.lagranger_rho_rescale_factors_json
    
    return lagranger_spoke

        
def xhatlooper_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):
    
    shoptions = shared_options(cfg)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhat_options["xhat_looper_options"] = {
        "xhat_solver_options": shoptions["iterk_solver_options"],
        "scen_limit": cfg.xhat_scen_limit,
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
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    all_nodenames=None,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):

    shoptions = shared_options(cfg)
    xhat_options = copy.deepcopy(shoptions)
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhat_options["xhat_looper_options"] = {
        "xhat_solver_options": shoptions["iterk_solver_options"],
        "dump_prefix": "delme",
        "csvname": "looper.csv",
    }
    if _hasit("add_reversed_shuffle"):
        xhat_options["xhat_looper_options"]["reverse"] = cfg.add_reversed_shuffle
    if _hasit("add_reversed_shuffle"):
        xhat_options["xhat_looper_options"]["xhatshuffle_iter_step"] = cfg.xhatshuffle_iter_step
    
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
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_dict,
    all_nodenames=None,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):
    
    shoptions = shared_options(cfg)
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
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):
    
    shoptions = shared_options(cfg)
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
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):

    shoptions = shared_options(cfg)
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
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    spoke_sleep_time=None,
):

    shoptions = shared_options(cfg)
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
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    all_nodenames=None,
    spoke_sleep_time=None,
):

    if _hasit("max_solver_threads"):
        sp_solver_options = {"threads":cfg.max_solver_threads}
    else:
        sp_solver_options = dict() 

    if _hasit("eta_bounds_mipgap"):
        sp_solver_options["mipgap"] = cfg.eta_bounds_mipgap

    ls_options = { "root_solver" : cfg.solver_name,
                   "sp_solver": cfg.solver_name,
                   "sp_solver_options" : sp_solver_options,
                    "verbose": cfg.verbose,
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

        
