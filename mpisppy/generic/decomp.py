###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Decomposition orchestrator for generic_cylinders."""

import mpisppy.utils.sputils as sputils
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy import global_toc

from mpisppy.generic.parsing import name_lists
from mpisppy.generic.hub import build_hub_dict
from mpisppy.generic.extensions import configure_extensions
from mpisppy.generic.spokes import build_spoke_list


def do_decomp(module, cfg, scenario_creator, scenario_creator_kwargs,
              scenario_denouement, bundle_wrapper=None):
    """Essentially, the main program for decomposition.

    Args:
       module (Python module or class): the model file with required functions
           or a class with the required methods.
       cfg (Pyomo config object): parsed arguments with perhaps a few attachments
       scenario_creator (function): note: this might be a wrapper and
           therefore not in the module
       scenario_creator_kwargs (dict): args for the scenario creator function
       scenario_denouement (function): some things (e.g. PH) call this for
           each scenario at the end
       bundle_wrapper (ProperBundler): wraps the module for proper bundle creation

    Returns:
        wheel (WheelSpinner): the container used for the spokes
            (so callers can query results)
    """
    if cfg.get("scenarios_per_bundle") is not None and cfg.scenarios_per_bundle == 1:
        raise RuntimeError("To get one scenarios-per-bundle=1, you need to write then read the 'bundles'")

    rho_setter = _get_rho_setter(module, cfg)
    ph_converger = _get_converger(cfg)

    all_scenario_names, all_nodenames = name_lists(module, cfg,
                                                   bundle_wrapper=bundle_wrapper)

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    variable_probability = getattr(cfg, "_admm_variable_probability", None)

    hub_dict = build_hub_dict(cfg, beans, scenario_creator_kwargs,
                              rho_setter, all_nodenames, ph_converger,
                              variable_probability=variable_probability)
    configure_extensions(hub_dict, module, cfg)

    # reduced cost fixer options setup (needs hub_dict before building spokes)
    if cfg.reduced_costs:
        from mpisppy.utils import cfg_vanilla as vanilla
        vanilla.add_reduced_costs_fixer(hub_dict, cfg)

    list_of_spoke_dict = build_spoke_list(cfg, beans, scenario_creator_kwargs,
                                          rho_setter, all_nodenames,
                                          variable_probability=variable_probability)

    # if the user dares, let them mess with the hubdict prior to solve
    if hasattr(module, 'hub_and_spoke_dict_callback'):
        module.hub_and_spoke_dict_callback(hub_dict, list_of_spoke_dict, cfg)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    _write_solutions(wheel, module, cfg)

    return wheel


def _get_rho_setter(module, cfg):
    """Get the rho_setter from the module, or None."""
    rho_setter = module._rho_setter if hasattr(module, '_rho_setter') else None
    if cfg.default_rho is None and rho_setter is None:
        if cfg.sep_rho or cfg.coeff_rho or cfg.sensi_rho:
            cfg.default_rho = 1
        else:
            raise RuntimeError("No rho_setter so a default must be specified via --default-rho")
    return rho_setter


def _get_converger(cfg):
    """Get the converger class based on cfg, or None."""
    if cfg.use_norm_rho_converger:
        from mpisppy.convergers.norm_rho_converger import NormRhoConverger
        if not cfg.use_norm_rho_updater:
            raise RuntimeError("--use-norm-rho-converger requires --use-norm-rho-updater")
        return NormRhoConverger
    elif cfg.primal_dual_converger:
        from mpisppy.convergers.primal_dual_converger import PrimalDualConverger
        return PrimalDualConverger
    return None


def _write_solutions(wheel, module, cfg):
    """Write solution output if configured."""
    if cfg.solution_base_name is not None:
        root_writer = getattr(module, "first_stage_solution_writer",
                                     sputils.first_stage_nonant_npy_serializer)
        tree_writer = getattr(module, "tree_solution_writer", None)

        wheel.write_first_stage_solution(f'{cfg.solution_base_name}.csv')
        wheel.write_first_stage_solution(f'{cfg.solution_base_name}.npy',
                first_stage_solution_writer=root_writer)
        if tree_writer is not None:
            wheel.write_tree_solution(f'{cfg.solution_base_name}_soldir',
                                      scenario_tree_solution_writer=tree_writer)
        else:
            wheel.write_tree_solution(f'{cfg.solution_base_name}_soldir')
        global_toc("Wrote solution data.")

    if hasattr(module, "custom_writer"):
        module.custom_writer(wheel, cfg)
