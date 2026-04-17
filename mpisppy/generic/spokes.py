###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Spoke dict construction for generic_cylinders."""

import copy

import mpisppy.utils.cfg_vanilla as vanilla


def build_spoke_list(cfg, beans, scenario_creator_kwargs,
                     rho_setter, all_nodenames,
                     variable_probability=None):
    """Build and return the list of spoke dicts for WheelSpinner.

    Args:
        cfg: Config object
        beans: tuple of (cfg, scenario_creator, scenario_denouement, all_scenario_names)
        scenario_creator_kwargs: dict for scenario creator
        rho_setter: rho setter function or None
        all_nodenames: list of node names or None
        variable_probability: variable probability list or None (used by ADMM)

    Returns:
        list: list of spoke dicts
    """
    # FWPH spoke
    if cfg.fwph:
        fw_spoke = vanilla.fwph_spoke(*beans,
                                      scenario_creator_kwargs=scenario_creator_kwargs,
                                      all_nodenames=all_nodenames,
                                      rho_setter=rho_setter,
                                      )

    # Standard Lagrangian bound spoke
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                                rho_setter=rho_setter,
                                                all_nodenames=all_nodenames,
                                                )
        if cfg.lagrangian_starting_mipgap is not None:
            vanilla.add_gapper(lagrangian_spoke, cfg, "lagrangian")

    # dual ph spoke
    if cfg.ph_dual:
        ph_dual_spoke = vanilla.ph_dual_spoke(*beans,
                                          scenario_creator_kwargs=scenario_creator_kwargs,
                                          rho_setter=rho_setter,
                                          all_nodenames=all_nodenames,
                                          )
        if cfg.sep_rho or cfg.coeff_rho or cfg.sensi_rho or cfg.grad_rho:
            # Note that this deepcopy might be expensive if certain wrappers were used.
            # (Could we do the modification to cfg in ph_dual to obviate the need?)
            modified_cfg = copy.deepcopy(cfg)
            modified_cfg["grad_rho_multiplier"] = cfg.ph_dual_rho_multiplier
        if cfg.sep_rho:
            vanilla.add_sep_rho(ph_dual_spoke, modified_cfg)
        if cfg.coeff_rho:
            vanilla.add_coeff_rho(ph_dual_spoke, modified_cfg)
        if cfg.sensi_rho:
            vanilla.add_sensi_rho(ph_dual_spoke, modified_cfg)
        if cfg.grad_rho:
            modified_cfg["grad_order_stat"] = cfg.ph_dual_grad_order_stat
            vanilla.add_grad_rho(ph_dual_spoke, modified_cfg)

    # relaxed ph spoke
    if cfg.relaxed_ph:
        relaxed_ph_spoke = vanilla.relaxed_ph_spoke(*beans,
                                          scenario_creator_kwargs=scenario_creator_kwargs,
                                          rho_setter=rho_setter,
                                          all_nodenames=all_nodenames,
                                          )
        if cfg.sep_rho:
            vanilla.add_sep_rho(relaxed_ph_spoke, cfg)
        if cfg.coeff_rho:
            vanilla.add_coeff_rho(relaxed_ph_spoke, cfg)
        if cfg.sensi_rho:
            vanilla.add_sensi_rho(relaxed_ph_spoke, cfg)

    # subgradient outer bound spoke
    if cfg.subgradient:
        subgradient_spoke = vanilla.subgradient_spoke(*beans,
                                          scenario_creator_kwargs=scenario_creator_kwargs,
                                          rho_setter=rho_setter,
                                          all_nodenames=all_nodenames,
                                          )
        if cfg.sep_rho:
            vanilla.add_sep_rho(subgradient_spoke, cfg)
        if cfg.coeff_rho:
            vanilla.add_coeff_rho(subgradient_spoke, cfg)
        if cfg.sensi_rho:
            vanilla.add_sensi_rho(subgradient_spoke, cfg)

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans,
                                                      scenario_creator_kwargs=scenario_creator_kwargs,
                                                      all_nodenames=all_nodenames)
        # special code for multi-stage (e.g., hydro)
        if cfg.get("stage2_ef_solver_name") is not None:
            xhatshuffle_spoke["opt_kwargs"]["options"]["stage2_ef_solver_name"] = cfg["stage2_ef_solver_name"]
            xhatshuffle_spoke["opt_kwargs"]["options"]["branching_factors"] = cfg["branching_factors"]

    if cfg.xhatxbar:
        xhatxbar_spoke = vanilla.xhatxbar_spoke(*beans,
                                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                                   variable_probability=variable_probability,
                                                   all_nodenames=all_nodenames)

    # reduced cost fixer options setup
    if cfg.reduced_costs:
        reduced_costs_spoke = vanilla.reduced_costs_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              all_nodenames=all_nodenames,
                                              rho_setter=None)

    list_of_spoke_dict = list()
    if cfg.fwph:
        list_of_spoke_dict.append(fw_spoke)
    if cfg.lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if cfg.ph_dual:
        list_of_spoke_dict.append(ph_dual_spoke)
    if cfg.relaxed_ph:
        list_of_spoke_dict.append(relaxed_ph_spoke)
    if cfg.subgradient:
        list_of_spoke_dict.append(subgradient_spoke)
    if cfg.xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)
    if cfg.xhatxbar:
        list_of_spoke_dict.append(xhatxbar_spoke)
    if cfg.reduced_costs:
        list_of_spoke_dict.append(reduced_costs_spoke)

    return list_of_spoke_dict
