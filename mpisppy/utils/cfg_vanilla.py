###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
""" **** cfg version of vanilla that uses the Pyomo configuration system
    Plain versions of dictionaries that can be modified for each example as needed.
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
from mpisppy.cylinders.subgradient_bounder import SubgradientOuterBound
from mpisppy.cylinders.ph_ob import PhOuterBound
from mpisppy.cylinders.xhatlooper_bounder import XhatLooperInnerBound
from mpisppy.cylinders.xhatxbar_bounder import XhatXbarInnerBound
from mpisppy.cylinders.xhatspecific_bounder import XhatSpecificInnerBound
from mpisppy.cylinders.xhatshufflelooper_bounder import XhatShuffleInnerBound
from mpisppy.cylinders.lshaped_bounder import XhatLShapedInnerBound
from mpisppy.cylinders.slam_heuristic import SlamMaxHeuristic, SlamMinHeuristic
from mpisppy.cylinders.cross_scen_spoke import CrossScenarioCutSpoke
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke
from mpisppy.cylinders.hub import PHHub
from mpisppy.cylinders.hub import APHHub
from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer
from mpisppy.extensions.integer_relax_then_enforce import IntegerRelaxThenEnforce
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension
from mpisppy.extensions.reduced_costs_fixer import ReducedCostsFixer
from mpisppy.extensions.reduced_costs_rho import ReducedCostsRho
from mpisppy.extensions.sep_rho import SepRho
from mpisppy.extensions.coeff_rho import CoeffRho
from mpisppy.extensions.sensi_rho import SensiRho
from mpisppy.utils.wxbarreader import WXBarReader
from mpisppy.utils.wxbarwriter import WXBarWriter

def _hasit(cfg, argname):
    # aside: Config objects act like a dict or an object TBD: so why the and?
    return cfg.get(argname) is not None and cfg[argname] is not None

def shared_options(cfg):
    shoptions = {
        "solver_name": cfg.solver_name,
        "defaultPHrho": cfg.default_rho,
        "convthresh": 0,
        "PHIterLimit": cfg.max_iterations,  # not needed by all
        "time_limit": cfg.time_limit,  # not needed by all
        "verbose": cfg.verbose,
        "display_progress": cfg.display_progress,
        "display_convergence_detail": cfg.display_convergence_detail,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
        "tee-rank0-solves": cfg.tee_rank0_solves,
        "trace_prefix" : cfg.trace_prefix,
        "presolve" : cfg.presolve,
    }
    if _hasit(cfg, "max_solver_threads"):
        shoptions["iter0_solver_options"]["threads"] = cfg.max_solver_threads
        shoptions["iterk_solver_options"]["threads"] = cfg.max_solver_threads
    if _hasit(cfg, "iter0_mipgap"):
        shoptions["iter0_solver_options"]["mipgap"] = cfg.iter0_mipgap
    if _hasit(cfg, "iterk_mipgap"):
        shoptions["iterk_solver_options"]["mipgap"] = cfg.iterk_mipgap
    if _hasit(cfg, "reduced_costs"):
        shoptions["rc_bound_tol"] = cfg.rc_bound_tol

    return shoptions

def add_multistage_options(cylinder_dict,all_nodenames,branching_factors):
    cylinder_dict = copy.deepcopy(cylinder_dict)
    if branching_factors is not None:
        if hasattr(cylinder_dict["opt_kwargs"], "options"):
            cylinder_dict["opt_kwargs"]["options"]["branching_factors"] = branching_factors
        if all_nodenames is None:
            all_nodenames = sputils.create_nodenames_from_branching_factors(branching_factors)
    if all_nodenames is not None:
        cylinder_dict["opt_kwargs"]["all_nodenames"] = all_nodenames
    return cylinder_dict

def ph_hub(
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=None,
        ph_extensions=None,
        extension_kwargs=None,
        ph_converger=None,
        rho_setter=None,
        variable_probability=None,
        all_nodenames=None,
):
    shoptions = shared_options(cfg)
    options = copy.deepcopy(shoptions)
    options["convthresh"] = cfg.intra_hub_conv_thresh
    options["bundles_per_rank"] = cfg.bundles_per_rank
    options["linearize_binary_proximal_terms"] = cfg.linearize_binary_proximal_terms
    options["linearize_proximal_terms"] = cfg.linearize_proximal_terms
    options["proximal_linearization_tolerance"] = cfg.proximal_linearization_tolerance
    options["smoothed"] = 2 if cfg.smoothing else 0
    options["defaultPHp"] = cfg.smoothing_rho_ratio
    options["defaultPHbeta"] = cfg.smoothing_beta

    hub_dict = {
        "hub_class": PHHub,
        "hub_kwargs": {"options": {"rel_gap": cfg.rel_gap,
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
            "extension_kwargs": extension_kwargs,
            "ph_converger": ph_converger,
            "all_nodenames": all_nodenames
        }
    }
    add_wxbar_read_write(hub_dict, cfg)
    add_ph_tracking(hub_dict, cfg)
    return hub_dict


def aph_hub(cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    ph_extensions=None,
    extension_kwargs=None,
    rho_setter=None,
    variable_probability=None,
    all_nodenames=None,
):
    # TBD: March 2023: multiple extensions needs work
    hub_dict = ph_hub(cfg,
                      scenario_creator,
                      scenario_denouement,
                      all_scenario_names,
                      scenario_creator_kwargs=scenario_creator_kwargs,
                      ph_extensions=ph_extensions,
                      extension_kwargs=extension_kwargs,
                      rho_setter=rho_setter,
                      variable_probability=variable_probability,
                      all_nodenames = all_nodenames,
                    )

    hub_dict['hub_class'] = APHHub
    hub_dict['opt_class'] = APH

    hub_dict['opt_kwargs']['options']['APHgamma'] = cfg.aph_gamma
    hub_dict['opt_kwargs']['options']['APHnu'] = cfg.aph_nu
    hub_dict['opt_kwargs']['options']['async_frac_needed'] = cfg.aph_frac_needed
    hub_dict['opt_kwargs']['options']['dispatch_frac'] = cfg.aph_dispatch_frac
    hub_dict['opt_kwargs']['options']['async_sleep_secs'] = cfg.aph_sleep_seconds

    return hub_dict


def extension_adder(hub_dict,ext_class):
    # TBD March 2023: this is not really good enough
    if "extensions" not in hub_dict["opt_kwargs"] or \
        hub_dict["opt_kwargs"]["extensions"] is None:
        hub_dict["opt_kwargs"]["extensions"] = ext_class
    elif hub_dict["opt_kwargs"]["extensions"] == MultiExtension:
        if hub_dict["opt_kwargs"]["extension_kwargs"] is None:
            hub_dict["opt_kwargs"]["extension_kwargs"] = {"ext_classes": []}
        if ext_class not in hub_dict["opt_kwargs"]["extension_kwargs"]["ext_classes"]:
            hub_dict["opt_kwargs"]["extension_kwargs"]["ext_classes"].append(ext_class)
    elif hub_dict["opt_kwargs"]["extensions"] != ext_class:
        #ext_class is the second extension
        if "extensions_kwargs" not in hub_dict["opt_kwargs"]:
            hub_dict["opt_kwargs"]["extension_kwargs"] = {}
        hub_dict["opt_kwargs"]["extension_kwargs"]["ext_classes"] = \
            [hub_dict["opt_kwargs"]["extensions"], ext_class]
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

def add_integer_relax_then_enforce(hub_dict,
              cfg,
              ):
    hub_dict = extension_adder(hub_dict,IntegerRelaxThenEnforce)
    hub_dict["opt_kwargs"]["options"]["integer_relax_then_enforce_options"] = {"ratio":cfg.integer_relax_then_enforce_ratio}
    return hub_dict

def add_reduced_costs_rho(hub_dict, cfg):
    hub_dict = extension_adder(hub_dict,ReducedCostsRho)
    hub_dict["opt_kwargs"]["options"]["reduced_costs_rho_options"] = {"multiplier" : cfg.reduced_costs_rho_multiplier, "cfg": cfg}

def add_sep_rho(hub_dict, cfg):
    hub_dict = extension_adder(hub_dict,SepRho)
    hub_dict["opt_kwargs"]["options"]["sep_rho_options"] = {"multiplier" : cfg.sep_rho_multiplier, "cfg": cfg}

def add_coeff_rho(hub_dict, cfg):
    hub_dict = extension_adder(hub_dict,CoeffRho)
    hub_dict["opt_kwargs"]["options"]["coeff_rho_options"] = {"multiplier" : cfg.coeff_rho_multiplier}

def add_sensi_rho(hub_dict, cfg):
    hub_dict = extension_adder(hub_dict,SensiRho)
    hub_dict["opt_kwargs"]["options"]["sensi_rho_options"] = {"multiplier" : cfg.sensi_rho_multiplier, "cfg": cfg}

def add_cross_scenario_cuts(hub_dict,
                            cfg,
                            ):
    #WARNING: Do not use without a cross_scenario_cuts spoke
    hub_dict = extension_adder(hub_dict, CrossScenarioExtension)
    hub_dict["opt_kwargs"]["options"]["cross_scen_options"]\
            = {"check_bound_improve_iterations" : cfg.cross_scenario_iter_cnt}
    return hub_dict

def add_reduced_costs_fixer(hub_dict,
                            cfg,
                            ):
    #WARNING: Do not use without a reduced_costs_spoke spoke
    hub_dict = extension_adder(hub_dict, ReducedCostsFixer)

    hub_dict["opt_kwargs"]["options"]["rc_options"] = {
            "verbose": cfg.rc_verbose,
            "debug": cfg.rc_debug,
            "use_rc_fixer": cfg.rc_fixer,
            "zero_rc_tol": cfg.rc_zero_tol,
            "fix_fraction_target_pre_iter0": cfg.rc_fix_fraction_pre_iter0,
            "fix_fraction_target_iter0": cfg.rc_fix_fraction_iter0,
            "fix_fraction_target_iterK": cfg.rc_fix_fraction_iterk,
            "use_rc_bt": cfg.rc_bound_tightening,
            "rc_bound_tol": cfg.rc_bound_tol,
            "rc_fixer_require_improving_lagrangian": cfg.rc_fixer_require_improving_lagrangian,
        }

    return hub_dict

def add_wxbar_read_write(hub_dict, cfg):
    """
    Add the wxbar read and write extensions to the hub_dict

    NOTE
    At the moment, the options are not stored in a extension options dict()
    but are 'loose' in the hub options dict
    """
    if _hasit(cfg, 'init_W_fname') or _hasit(cfg, 'init_Xbar_fname'):
        hub_dict = extension_adder(hub_dict, WXBarReader)
        hub_dict["opt_kwargs"]["options"].update(
            {"init_W_fname" : cfg.init_W_fname,
             "init_Xbar_fname" : cfg.init_Xbar_fname,
             "init_separate_W_files" : cfg.init_separate_W_files
            })
    if _hasit(cfg, 'W_fname') or _hasit(cfg, 'Xbar_fname'):
        hub_dict = extension_adder(hub_dict, WXBarWriter)
        hub_dict["opt_kwargs"]["options"].update(
            {"W_fname" : cfg.W_fname,
             "Xbar_fname" : cfg.Xbar_fname,
             "separate_W_files" : cfg.separate_W_files
            })
    return hub_dict

def add_ph_tracking(cylinder_dict, cfg, spoke=False):
    """ Manage the phtracker extension and bridge gap between config and ph options dict
        Args:
            cylinder_dict (dict): the hub or spoke dictionary
            cfg (dict): the configuration dictionary
            spoke (bool, optional): Whether the cylinder is a spoke. Defaults to False.
        Returns:
            cylinder_dict (dict): the updated hub or spoke dictionary

        for cfg.track_* flags, the ints are mapped as followed:

        0: do not track
        1: track for all cylinders
        2: track for hub only
        3: track for spokes only
        4: track and plot for all cylinders
        5: track and plot for hub
        6: track and plot for spokes

        If 'ph_track_progress' is True in the cfg dictionary, this function adds the
        ph tracking extension to the cylinder dict with the specified tracking options.
    """
    if _hasit(cfg, 'ph_track_progress') and cfg.ph_track_progress:
        from mpisppy.extensions.phtracker import PHTracker
        cylinder_dict = extension_adder(cylinder_dict, PHTracker)
        phtrackeroptions = {"results_folder": cfg.tracking_folder}

        t_vars = ['convergence', 'xbars', 'duals', 'nonants', 'scen_gaps', 'reduced_costs']
        for t_var in t_vars:
            if _hasit(cfg, f'track_{t_var}'):
                trval = cfg[f'track_{t_var}']
                if ((trval in {1, 4} or \
                    (not spoke and trval in {2, 5}) or \
                    (spoke and trval in {3, 6}))):
                    phtrackeroptions[f'track_{t_var}'] = True

                    if trval in {4, 5, 6}:
                        phtrackeroptions[f'plot_{t_var}'] = True

        # because convergence maps to multiple tracking options
        if phtrackeroptions.get('track_convergence'):
            phtrackeroptions['track_bounds'] = True
            phtrackeroptions['track_gaps'] = True
        if phtrackeroptions.get('plot_convergence'):
            phtrackeroptions['plot_bounds'] = True
            phtrackeroptions['plot_gaps'] = True

        cylinder_dict["opt_kwargs"]["options"]["phtracker_options"] = phtrackeroptions

    return cylinder_dict

def fwph_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    all_nodenames=None,
    rho_setter=None,
):
    shoptions = shared_options(cfg)

    mip_solver_options, qp_solver_options = dict(), dict()
    if _hasit(cfg, "max_solver_threads"):
        mip_solver_options["threads"] = cfg.max_solver_threads
        qp_solver_options["threads"] = cfg.max_solver_threads
    if _hasit(cfg, "fwph_mipgap"):
        mip_solver_options["mipgap"] = cfg.fwph_mipgap

    fw_options = {
        "FW_iter_limit": cfg.fwph_iter_limit,
        "FW_weight": cfg.fwph_weight,
        "FW_conv_thresh": cfg.fwph_conv_thresh,
        "stop_check_tol": cfg.fwph_stop_check_tol,
        "solver_name": cfg.solver_name,
        "FW_verbose": cfg.verbose,
        "mip_solver_options": mip_solver_options,
        "qp_solver_options": qp_solver_options,
    }
    fw_dict = {
        "spoke_class": FrankWolfeOuterBound,
        "opt_class": FWPH,
        "opt_kwargs": {
            "PH_options": shoptions,  # be sure convthresh is zero for fwph
            "FW_options": fw_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "scenario_denouement": scenario_denouement,
            "all_nodenames": all_nodenames,
            "rho_setter" : rho_setter,
        },
    }
    return fw_dict


# The next function is to provide some standardization
def _PHBase_spoke_foundation(
        spoke_class,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=None,
        rho_setter=None,
        all_nodenames=None,
        ph_extensions=None,
        extension_kwargs=None,
        ):
    # only the shared options
    shoptions = shared_options(cfg)
    my_options = copy.deepcopy(shoptions)  # extra safe...    
    spoke_dict = {
        "spoke_class": spoke_class,
        "opt_class": PHBase,
        "opt_kwargs": {
            "options": my_options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            'scenario_denouement': scenario_denouement,
        }
    }
    if all_nodenames is not None:
        spoke_dict["opt_kwargs"]["all_nodenames"] = all_nodenames
    if rho_setter is not None:
        spoke_dict["opt_kwargs"]["rho_setter"] = rho_setter
    if ph_extensions is not None:
        spoke_dict["opt_kwargs"]["extensions"] = ph_extensions
    if extension_kwargs is not None:
        spoke_dict["opt_kwargs"]["extension_kwargs"] = extension_kwargs

    return spoke_dict

def _Xhat_Eval_spoke_foundation(
        spoke_class,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=None,
        rho_setter=None,
        all_nodenames=None,
        ph_extensions=None,
        extension_kwargs=None,
        ):
    spoke_dict = _PHBase_spoke_foundation(
        spoke_class,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        rho_setter=rho_setter,
        all_nodenames=all_nodenames,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
        )
    spoke_dict["opt_class"] = Xhat_Eval
    return spoke_dict


def lagrangian_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    rho_setter=None,
    all_nodenames=None,
    ph_extensions=None,
    extension_kwargs=None,
):
    lagrangian_spoke = _PHBase_spoke_foundation(
        LagrangianOuterBound,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        rho_setter=rho_setter,
        all_nodenames=all_nodenames,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
    )
    if cfg.lagrangian_iter0_mipgap is not None:
        lagrangian_spoke["opt_kwargs"]["options"]["iter0_solver_options"]\
            ["mipgap"] = cfg.lagrangian_iter0_mipgap
    if cfg.lagrangian_iterk_mipgap is not None:
        lagrangian_spoke["opt_kwargs"]["options"]["iterk_solver_options"]\
            ["mipgap"] = cfg.lagrangian_iterk_mipgap
    add_ph_tracking(lagrangian_spoke, cfg, spoke=True)

    return lagrangian_spoke


def reduced_costs_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    rho_setter=None,
    all_nodenames=None,
    ph_extensions=None,
    extension_kwargs=None,
):
    rc_spoke = _PHBase_spoke_foundation(
        ReducedCostsSpoke,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        rho_setter=rho_setter,
        all_nodenames=all_nodenames,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
    )

    add_ph_tracking(rc_spoke, cfg, spoke=True)

    return rc_spoke


# special lagrangian: computes its own xhat and W (does not seem to work well)
# ph_ob_spoke is probably better
def lagranger_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    rho_setter=None,
    all_nodenames = None,
    ph_extensions=None,
    extension_kwargs=None,
):
    lagranger_spoke = _PHBase_spoke_foundation(
        LagrangerOuterBound,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        rho_setter=rho_setter,
        all_nodenames=all_nodenames,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
    )
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
    add_ph_tracking(lagranger_spoke, cfg, spoke=True)
    return lagranger_spoke


def subgradient_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    rho_setter=None,
    all_nodenames=None,
    ph_extensions=None,
    extension_kwargs=None,
):
    subgradient_spoke = _PHBase_spoke_foundation(
        SubgradientOuterBound,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        rho_setter=rho_setter,
        all_nodenames=all_nodenames,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
    )
    if cfg.subgradient_iter0_mipgap is not None:
        subgradient_spoke["opt_kwargs"]["options"]["iter0_solver_options"]\
            ["mipgap"] = cfg.subgradient_iter0_mipgap
    if cfg.subgradient_iterk_mipgap is not None:
        subgradient_spoke["opt_kwargs"]["options"]["iterk_solver_options"]\
            ["mipgap"] = cfg.subgradient_iterk_mipgap
    if cfg.subgradient_rho_multiplier is not None:
        subgradient_spoke["opt_kwargs"]["options"]["subgradient_rho_multiplier"]\
            = cfg.subgradient_rho_multiplier
    add_ph_tracking(subgradient_spoke, cfg, spoke=True)

    return subgradient_spoke


def xhatlooper_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    ph_extensions=None,
    extension_kwargs=None,
):

    xhatlooper_dict = _Xhat_Eval_spoke_foundation(
        XhatLooperInnerBound,        
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
    )

    xhatlooper_dict["opt_kwargs"]["options"]['bundles_per_rank'] = 0 #  no bundles for xhat
    xhatlooper_dict["opt_kwargs"]["options"]["xhat_looper_options"] = {
        "xhat_solver_options": xhatlooper_dict["opt_kwargs"]["options"]["iterk_solver_options"],
        "scen_limit": cfg.xhat_scen_limit,
        "dump_prefix": "delme",
        "csvname": "looper.csv",
    }
    
    return xhatlooper_dict


def xhatxbar_spoke(
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=None,
        variable_probability=None,
        ph_extensions=None,
        extension_kwargs=None,
        all_nodenames=None,
):
    xhatxbar_dict = _Xhat_Eval_spoke_foundation(
        XhatXbarInnerBound,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
        all_nodenames=all_nodenames,
    )

    xhatxbar_dict["opt_kwargs"]["options"]['bundles_per_rank'] = 0  # no bundles for xhat
    xhatxbar_dict["opt_kwargs"]["options"]["xhat_xbar_options"] = {
        "xhat_solver_options": xhatxbar_dict["opt_kwargs"]["options"]["iterk_solver_options"],
        "dump_prefix": "delme",
        "csvname": "looper.csv",
    }
    
    xhatxbar_dict["opt_kwargs"]["variable_probability"] = variable_probability

    return xhatxbar_dict


def xhatshuffle_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    all_nodenames=None,
    scenario_creator_kwargs=None,
    ph_extensions=None,
    extension_kwargs=None,
):

    xhatshuffle_dict = _Xhat_Eval_spoke_foundation(
        XhatShuffleInnerBound,        
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        all_nodenames=all_nodenames,
        scenario_creator_kwargs=scenario_creator_kwargs,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
    )
    xhatshuffle_dict["opt_kwargs"]["options"]['bundles_per_rank'] = 0  # no bundles for xhat
    xhatshuffle_dict["opt_kwargs"]["options"]["xhat_looper_options"] = {
        "xhat_solver_options": xhatshuffle_dict["opt_kwargs"]["options"]["iterk_solver_options"],
        "dump_prefix": "delme",
        "csvname": "looper.csv",
    }
    if _hasit(cfg, "add_reversed_shuffle"):
        xhatshuffle_dict["opt_kwargs"]["options"]["xhat_looper_options"]["reverse"] = cfg.add_reversed_shuffle
    if _hasit(cfg, "add_reversed_shuffle"):
        xhatshuffle_dict["opt_kwargs"]["options"]["xhatshuffle_iter_step"] = cfg.xhatshuffle_iter_step

    return xhatshuffle_dict


def xhatspecific_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_dict,
    all_nodenames=None,
    scenario_creator_kwargs=None,
    ph_extensions=None,
    extension_kwargs=None,
):

    xhatspecific_dict = _Xhat_Eval_spoke_foundation(
        XhatSpecificInnerBound,        
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
    )
    xhatspecific_dict["opt_kwargs"]["options"]['bundles_per_rank'] = 0  # no bundles for xhat    
    return xhatspecific_dict

def xhatlshaped_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    ph_extensions=None,
    extension_kwargs=None,
):

    xhatlshaped_dict = _Xhat_Eval_spoke_foundation(
        XhatLShapedInnerBound,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        ph_extensions=ph_extensions,
        extension_kwargs=extension_kwargs,
    )
    xhatlshaped_dict["opt_kwargs"]["options"]['bundles_per_rank'] = 0  # no bundles for xhat    

    return xhatlshaped_dict

def slammax_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    ph_extensions=None,
):

    slammax_dict = _Xhat_Eval_spoke_foundation(
        SlamMaxHeuristic,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        ph_extensions=ph_extensions,
    )
    slammax_dict["opt_kwargs"]["options"]['bundles_per_rank'] = 0  # no bundles for slamming
    return slammax_dict

def slammin_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    ph_extensions=None,
):
    slammin_dict = _Xhat_Eval_spoke_foundation(
        SlamMinHeuristic,
        cfg,
        scenario_creator,
        scenario_denouement,
        all_scenario_names,
        scenario_creator_kwargs=scenario_creator_kwargs,
        ph_extensions=ph_extensions,
    )
    slammin_dict["opt_kwargs"]["options"]['bundles_per_rank'] = 0  # no bundles for slamming
    return slammin_dict


def cross_scenario_cuts_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    all_nodenames=None,
):

    if _hasit(cfg, "max_solver_threads"):
        sp_solver_options = {"threads":cfg.max_solver_threads}
    else:
        sp_solver_options = dict()

    if _hasit(cfg, "eta_bounds_mipgap"):
        sp_solver_options["mipgap"] = cfg.eta_bounds_mipgap

    ls_options = { "root_solver" : cfg.solver_name,
                   "sp_solver": cfg.solver_name,
                   "sp_solver_options" : sp_solver_options,
                    "verbose": cfg.verbose,
                 }
    cut_spoke = {
        "spoke_class": CrossScenarioCutSpoke,
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

# run PH with smaller rho to compute LB
def ph_ob_spoke(
    cfg,
    scenario_creator,
    scenario_denouement,
    all_scenario_names,
    scenario_creator_kwargs=None,
    rho_setter=None,
    all_nodenames=None,
    variable_probability=None,
):
    shoptions = shared_options(cfg)
    ph_ob_spoke = {
        "spoke_class": PhOuterBound,
        "opt_class": PHBase,
        "opt_kwargs": {
            "options": shoptions,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            'scenario_denouement': scenario_denouement,
            "rho_setter": rho_setter,
            "all_nodenames": all_nodenames,
            "variable_probability": variable_probability,
        }
    }
    if cfg.ph_ob_rho_rescale_factors_json is not None:
        ph_ob_spoke["opt_kwargs"]["options"]\
            ["ph_ob_rho_rescale_factors_json"]\
            = cfg.ph_ob_rho_rescale_factors_json
    ph_ob_spoke["opt_kwargs"]["options"]["ph_ob_initial_rho_rescale_factor"]\
        = cfg.ph_ob_initial_rho_rescale_factor
    if cfg.ph_ob_gradient_rho:
        ph_ob_spoke["opt_kwargs"]["options"]\
            ["ph_ob_gradient_rho"]\
            = dict()
        ph_ob_spoke["opt_kwargs"]["options"]\
            ["ph_ob_gradient_rho"]["cfg"]\
            = cfg

    return ph_ob_spoke
