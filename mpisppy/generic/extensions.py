###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Extension wiring for the hub dict."""

import inspect
import json
import os

import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.sputils as sputils
from mpisppy.extensions.extension import Extension


def load_user_extension_module(ext_name):
    """Import a module named by --user-defined-extensions.

    Args:
        ext_name (str): the module name, as given on the command line

    Returns:
        (list, dict or None): every subclass of ``Extension`` defined in the
        module (not ones it imports), in the order they appear in the source,
        and the contents of ``<module>.json`` from the module's directory, or
        None when there is no such file.
    """
    ext_module = sputils.module_name_to_module(ext_name)
    found = [
        attr for attr in vars(ext_module).values()
        if inspect.isclass(attr)
        and issubclass(attr, Extension)
        and attr.__module__ == ext_module.__name__
    ]
    if not found:
        raise RuntimeError(
            f"Could not find a subclass of mpisppy.extensions.extension.Extension "
            f"defined in module {ext_name}"
        )
    ext_options = None
    module_file = getattr(ext_module, "__file__", None)
    if module_file is not None:
        json_filename = os.path.splitext(module_file)[0] + ".json"
        if os.path.exists(json_filename):
            with open(json_filename) as f:
                ext_options = json.load(f)
    return found, ext_options


def configure_extensions(hub_dict, module, cfg):
    """Add extensions to hub_dict based on cfg flags.

    Args:
        hub_dict: the hub dictionary to modify in place
        module: the model module
        cfg: Config object
    """
    ext_classes = list()
    # TBD: add cross_scenario_cuts, which also needs a cylinder

    final_xhat = cfg.get("final_xhat", "none")
    if final_xhat in ("xbar", "closest"):
        # These are deliberately hub extensions, not the inner-bound spokes.
        # PHHub.finalize calls post_everything, where the selected extension
        # evaluates one final candidate and registers the hub as an incumbent
        # source for WheelSpinner's solution writers.
        from mpisppy.opt.aph import APH
        from mpisppy.opt.ph import PH
        if hub_dict.get("opt_class") not in (PH, APH):
            raise RuntimeError(
                f"--final-xhat {final_xhat} currently supports only PH and APH hubs; "
                "remove --final-xhat or select a PH-based hub."
            )
        hub_options = hub_dict["opt_kwargs"]["options"]
        final_options = {
            # Resolve through SPOpt._effective_solver_options only when the
            # final evaluation runs, after scheduled and dynamic layers have
            # reached their final state.
            "xhat_solver_options": None,
            "use_ph_solver_options": True,
            "keep_solution": True,
        }
        if final_xhat == "xbar":
            from mpisppy.extensions.xhatxbar import XhatXbar
            ext_classes.append(XhatXbar)
            hub_options["xhat_xbar_options"] = final_options
        else:
            from mpisppy.extensions.xhatclosest import XhatClosest
            ext_classes.append(XhatClosest)
            hub_options["xhat_closest_options"] = final_options

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

    if cfg.slamming_directives_file is not None:
        vanilla.add_slammer(hub_dict, cfg)

    if cfg.detect_W_oscillations is not None \
            or cfg.interrupt_W_oscillations is not None:
        vanilla.add_w_oscillation(hub_dict, cfg)

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

    if cfg.get("wtracker", ifmissing=False):
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        ext_classes.append(Wtracker_extension)
        hub_dict["opt_kwargs"]["options"]["wtracker_options"] = {
            "wlen": cfg.wtracker_wlen,
            "reportlen": cfg.wtracker_reportlen,
            "stdevthresh": cfg.wtracker_stdevthresh,
            "file_prefix": cfg.wtracker_file_prefix,
        }

    if cfg.user_defined_extensions is not None:
        for ext_name in cfg.user_defined_extensions:
            user_ext_classes, ext_options = load_user_extension_module(ext_name)
            for ext_class in user_ext_classes:
                vanilla.extension_adder(hub_dict, ext_class)
            if ext_options is not None:
                hub_dict['opt_kwargs']['options'][ext_name] = ext_options

    if cfg.sep_rho:
        vanilla.add_sep_rho(hub_dict, cfg)

    if cfg.coeff_rho:
        vanilla.add_coeff_rho(hub_dict, cfg)

    # these should be after sep rho and coeff rho
    # as they will use existing rho values if the
    # sensitivity is too small
    if cfg.sensi_rho:
        vanilla.add_sensi_rho(hub_dict, cfg)

    # reduced_costs_rho was deprecated and removed; Config.checker() rejects it.

    # Compose with whatever is already attached (the Checkpointer, say) rather
    # than rebuilding the extension list from scratch, which silently dropped
    # every earlier extension.
    for ext_class in ext_classes:
        vanilla.extension_adder(hub_dict, ext_class)
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
