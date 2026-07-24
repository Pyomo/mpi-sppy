###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""CLI arg parsing and module loading for generic_cylinders."""

import sys
import os
import importlib
import inspect

import numpy as np
import pyomo.common.config as pyofig

import mpisppy.utils.config as config
import mpisppy.utils.sputils as sputils


_IMPLICIT_MODULES = {
    "--smps-dir": "mpisppy.problem_io.smps_module",
    "--mps-files-directory": "mpisppy.problem_io.mps_module",
}

def model_fname():
    """Extract the module name from the CLI arguments (--module-name).

    ``--module-name`` may appear in any position on the command line, in
    either the ``--module-name foo`` or ``--module-name=foo`` form.

    As an exception, ``--smps-dir`` and ``--mps-files-directory`` may be
    used instead of ``--module-name``; the appropriate module is inferred
    automatically.  Using ``--module-name`` together with one of these
    implicit-module flags is an error.
    """
    def _bad_news():
        raise RuntimeError("Unable to parse module name from the command line"
                           " (for module foo.py, we want, e.g.\n"
                           "--module-name foo\n"
                           "or\n"
                           "--module-name=foo\n"
                           "Alternatively, use --smps-dir or"
                           " --mps-files-directory.")

    args = sys.argv[1:]

    # Scan the whole command line for an explicit --module-name (any position).
    module_name = None
    for i, arg in enumerate(args):
        if arg == "--module-name":
            if i + 1 >= len(args):
                _bad_news()
            module_name = args[i + 1]
            break
        if arg.startswith("--module-name="):
            module_name = arg.split("=", 1)[1]
            if not module_name:
                _bad_news()
            break

    # Scan for an implicit-module flag (--smps-dir, --mps-files-directory).
    # Handle both --flag value and --flag=value forms.
    implicit_flag = None
    for arg in args:
        if arg.split("=")[0] in _IMPLICIT_MODULES:
            implicit_flag = arg.split("=")[0]
            break

    if implicit_flag is not None:
        if module_name is not None:
            raise RuntimeError(
                f"Cannot use both {implicit_flag} and --module-name."
                f" {implicit_flag} implies --module-name"
                f" {_IMPLICIT_MODULES[implicit_flag]}")
        return _IMPLICIT_MODULES[implicit_flag]

    if module_name is None:
        _bad_news()
    return module_name


def load_module(model_fname):
    """Import and return a module given its name or path."""
    if inspect.ismodule(model_fname):
        return model_fname
    dpath = os.path.dirname(model_fname)
    fname = os.path.basename(model_fname)
    sys.path.append(dpath)
    return importlib.import_module(fname)


def add_decomp_args(cfg):
    """Register every CLI arg consumed by hub-and-spoke decomposition.

    This is the single source of truth for the decomposition (cylinder and
    spoke) options.  Both generic_cylinders (via parse_args) and mrp_generic
    (via parse_mrp_args, used when --xhat-method=cylinders) call it, so a
    command line accepted by one driver is accepted by the other.  Every option
    read by mpisppy.generic.decomp.do_decomp and the hub/spoke builders it calls
    must be registered here.
    """
    # There are some arguments here that will not make sense for all models
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.lshaped_args()
    cfg.xhatlshaped_args()    
    cfg.ph_args()
    cfg.aph_args()
    cfg.cg_args()
    cfg.dualcg_args()
    cfg.subgradient_args()
    cfg.fixer_args()
    cfg.cvar_args()
    cfg.relaxed_ph_fixer_args()
    cfg.integer_relax_then_enforce_args()
    cfg.slamming_args()
    cfg.w_oscillation_args()
    cfg.gapper_args()
    cfg.gapper_args(name="lagrangian")
    cfg.ph_primal_args()
    cfg.ph_dual_args()
    cfg.relaxed_ph_args()
    cfg.ph_xfeas_spoke_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.subgradient_bounder_args()
    cfg.xhatshuffle_args()
    cfg.xhatxbar_args()
    cfg.xhat_from_file_args()
    cfg.write_xhat_file_args()
    cfg.xhat_feasibility_cut_args()
    cfg.norm_rho_args()
    cfg.primal_dual_rho_args()
    cfg.converger_args()
    cfg.wxbar_read_write_args()
    cfg.wtracker_args()
    cfg.tracking_args()
    cfg.gradient_args()
    cfg.dynamic_rho_args()
    cfg.reduced_costs_args()
    cfg.sep_rho_args()
    cfg.coeff_rho_args()
    cfg.sensi_rho_args()
    cfg.reduced_costs_rho_args()

    cfg.add_to_config("user_defined_extensions",
                      description="Space-delimited module names for user extensions",
                      domain=pyofig.ListOf(str),
                      default=None)


def parse_args(m):
    """Parse CLI args given the model module m. Returns a Config object."""
    cfg = config.Config()
    cfg.proper_bundle_config()
    cfg.pickle_scenarios_config()
    cfg.pre_pickle_args()

    cfg.add_to_config(name="module_name",
                      description="Name of the file that has the scenario creator, etc.",
                      domain=str,
                      default=None,
                      argparse=True)
    assert hasattr(m, "inparser_adder"), "The model file must have an inparser_adder function"
    cfg.add_to_config(name="solution_base_name",
                      description="The string used for a directory of ouput along with a csv and an npv file (default None, which means no soltion output)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="write_scenario_lp_mps_files_dir",
                      description="Invokes an extension that writes an model lp file, mps file and a nonants json file for each scenario before iteration 0",
                      domain=str,
                      default=None)

    m.inparser_adder(cfg)
    # many models, e.g., farmer, need num_scens_required
    #  in which case, it should go in the inparser_adder function
    # cfg.num_scens_required()
    # On the other hand, this program really wants cfg.num_scens somehow so
    # maybe it should just require it.

    cfg.EF_base()  # If EF is slected, most other options will be moot
    cfg.chance_constraint_args()  # EF-only (see config.checker)
    # Hub-and-spoke decomposition args (shared with mrp_generic)
    add_decomp_args(cfg)
    # TBD - think about adding directory for json options files

    cfg.mmw_args()

    from mpisppy.generic.admm import admm_args
    admm_args(cfg)

    cfg.parse_command_line(f"mpi-sppy for {cfg.module_name}")

    cfg.checker()  # looks for inconsistencies
    return cfg


def name_lists(module, cfg, bundle_wrapper=None):
    """Build all_scenario_names and all_nodenames from module and cfg.

    Returns:
        tuple: (all_scenario_names, all_nodenames)
    """
    # ADMM wrappers provide their own scenario names and nodenames
    admm_names = getattr(cfg, "_admm_scenario_names", None)
    if admm_names is not None:
        all_nodenames = getattr(cfg, "_admm_nodenames", None)
        return admm_names, all_nodenames

    # Note: high level code like this assumes there are branching factors for
    # multi-stage problems. For other trees, you will need lower-level code
    if cfg.get("branching_factors") is not None:
        all_nodenames = sputils.create_nodenames_from_branching_factors(
                                    cfg.branching_factors)
        num_scens = np.prod(cfg.branching_factors)
        if cfg.xhatshuffle and cfg.get("stage2_ef_solver_name") is None:
            import warnings
            warnings.warn(
                "stage2_ef_solver_name is recommended for multistage xhatshuffle",
                UserWarning,
                stacklevel=2,
            )
    else:
        all_nodenames = None
        num_scens = cfg.get("num_scens")  # maybe None is OK

    # proper bundles should be almost magic
    if cfg.unpickle_bundles_dir or cfg.scenarios_per_bundle is not None:
        # bundle_names_creator needs cfg.num_scens, but it is not always set yet:
        #  - multi-stage: num_scens was computed locally above from branching
        #    factors but never written back to cfg.
        #  - file-based paths (e.g. --mps-files-directory): the scenario count is
        #    implied by the files, so there is no --num-scens; recover it from the
        #    module's scenario_names_creator (None => every scenario).
        if num_scens is None:
            num_scens = len(module.scenario_names_creator(None))
        if cfg.get("num_scens") is None:
            cfg.quick_assign("num_scens", int, int(num_scens))
        num_buns = cfg.num_scens // cfg.scenarios_per_bundle
        all_scenario_names = bundle_wrapper.bundle_names_creator(num_buns, cfg=cfg)
        all_nodenames = None  # This is seldom used; also, proper bundles result in two stages
    else:
        all_scenario_names = module.scenario_names_creator(num_scens)

    return all_scenario_names, all_nodenames


def proper_bundles(cfg):
    """Return True if proper bundles are configured."""
    return cfg.get("pickle_bundles_dir", ifmissing=False)\
        or cfg.get("unpickle_bundles_dir", ifmissing=False)\
        or cfg.get("scenarios_per_bundle", ifmissing=False)
