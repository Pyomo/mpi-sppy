###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Extension wiring for the hub dict."""

import os

import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.sputils as sputils
from mpisppy.extensions.extension import MultiExtension, Extension


def configure_extensions(hub_dict, module, cfg):
    """Add extensions to hub_dict based on cfg flags.

    Args:
        hub_dict: the hub dictionary to modify in place
        module: the model module
        cfg: Config object
    """
    ext_classes = list()
    # TBD: add cross_scenario_cuts, which also needs a cylinder

    if cfg.mipgaps_json is not None or cfg.starting_mipgap is not None:
        vanilla.add_gapper(hub_dict, cfg)

    if cfg.fixer:  # cfg_vanilla takes care of the fixer_tol?
        assert hasattr(module, "id_fix_list_fct"), "id_fix_list_fct required for --fixer"
        vanilla.add_fixer(hub_dict, cfg, module)

    if cfg.rc_fixer:
        vanilla.add_reduced_costs_fixer(hub_dict, cfg)

    if cfg.get("xhat_feasibility_cuts_count", 0) or 0:
        vanilla.add_xhat_feasibility_cuts(hub_dict, cfg)

    if cfg.relaxed_ph_fixer:
        vanilla.add_relaxed_ph_fixer(hub_dict, cfg)

    if cfg.integer_relax_then_enforce:
        vanilla.add_integer_relax_then_enforce(hub_dict, cfg)

    if cfg.grad_rho:
        from mpisppy.extensions.grad_rho import GradRho
        ext_classes.append(GradRho)
        hub_dict['opt_kwargs']['options']['grad_rho_options'] = {'cfg': cfg}

    if cfg.write_scenario_lp_mps_files_dir is not None:
        raise RuntimeError("write_scenario_lp_mps_files_dir is not currently supported in _do_decomp")

    if cfg.W_and_xbar_reader:
        from mpisppy.utils.w_utils.wxbarreader import WXBarReader
        ext_classes.append(WXBarReader)

    if cfg.W_and_xbar_writer:
        from mpisppy.utils.w_utils.wxbarwriter import WXBarWriter
        ext_classes.append(WXBarWriter)

    if cfg.user_defined_extensions is not None:
        import json
        for ext_name in cfg.user_defined_extensions:
            ext_module = sputils.module_name_to_module(ext_name)
            user_ext_classes = []
            # Collect all valid Extension instances in the module to ensure no valid extensions are missed.
            for name in dir(ext_module):
                if isinstance(getattr(ext_module, name), Extension):
                    user_ext_classes.append(getattr(ext_module, name))
            if not user_ext_classes:
                raise RuntimeError(f"Could not find an mpisppy extension in module {ext_name}")
            # Add all found extensions to the hub_dict
            for ext_class in user_ext_classes:
                vanilla.extension_adder(hub_dict, ext_class)
            # grab JSON for this module's option dictionary
            json_filename = ext_name+".json"
            if os.path.exists(json_filename):
                ext_options = json.load(json_filename)
                hub_dict['opt_kwargs']['options'][ext_name] = ext_options
            else:
                raise RuntimeError(f"JSON options file {json_filename} for user defined extension not found")

    if cfg.sep_rho:
        vanilla.add_sep_rho(hub_dict, cfg)

    if cfg.coeff_rho:
        vanilla.add_coeff_rho(hub_dict, cfg)

    # these should be after sep rho and coeff rho
    # as they will use existing rho values if the
    # sensitivity is too small
    if cfg.sensi_rho:
        vanilla.add_sensi_rho(hub_dict, cfg)

    if cfg.reduced_costs_rho:
        vanilla.add_reduced_costs_rho(hub_dict, cfg)

    if len(ext_classes) != 0:
        hub_dict['opt_kwargs']['extensions'] = MultiExtension
        hub_dict["opt_kwargs"]["extension_kwargs"] = {"ext_classes" : ext_classes}
    if cfg.primal_dual_converger:
        hub_dict['opt_kwargs']['options']\
            ['primal_dual_converger_options'] = {
                'verbose': True,
                'tol': cfg.primal_dual_converger_tol,
                'tracking': True}

    # norm rho adaptive rho (not the gradient version)
    if cfg.use_norm_rho_updater:
        from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
        vanilla.extension_adder(hub_dict, NormRhoUpdater)
        hub_dict['opt_kwargs']['options']['norm_rho_options'] = {'verbose': cfg.verbose}

    if cfg.use_primal_dual_rho_updater:
        from mpisppy.extensions.primal_dual_rho import PrimalDualRho
        vanilla.extension_adder(hub_dict, PrimalDualRho)
        hub_dict['opt_kwargs']['options']['primal_dual_rho_options'] = {
                'verbose': cfg.verbose,
                'rho_update_threshold': cfg.primal_dual_rho_update_threshold,
                'primal_bias': cfg.primal_dual_rho_primal_bias,
            }
