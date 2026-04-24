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
    """Extract the module name from the first CLI argument (--module-name).

    As an exception, ``--smps-dir`` and ``--mps-files-directory`` may be
    used instead of ``--module-name``; the appropriate module is inferred
    automatically.  Using ``--module-name`` together with one of these
    implicit-module flags is an error.
    """
    def _bad_news():
        raise RuntimeError("Unable to parse module name from first argument"
                           " (for module foo.py, we want, e.g.\n"
                           "--module-name foo\n"
                           "or\n"
                           "--module-name=foo\n"
                           "Alternatively, use --smps-dir or"
                           " --mps-files-directory as the first argument.")
    def _len_check(needed_length):
        if len(sys.argv) <= needed_length:
            _bad_news()
        else:
            return True

    _len_check(1)

    # Check for implicit module flags (--smps-dir, --mps-files-directory)
    first_arg = sys.argv[1]
    # Handle both --flag value and --flag=value forms
    first_flag = first_arg.split("=")[0]

    if first_flag in _IMPLICIT_MODULES:
        # Error if --module-name also appears on the command line
        for arg in sys.argv[2:]:
            if arg.startswith("--module-name"):
                raise RuntimeError(
                    f"Cannot use both {first_flag} and --module-name."
                    f" {first_flag} implies --module-name"
                    f" {_IMPLICIT_MODULES[first_flag]}")
        return _IMPLICIT_MODULES[first_flag]

    if not first_arg.startswith("--module-name"):
        _bad_news()
    if first_arg == "--module-name":
        _len_check(2)
        return sys.argv[2]
    else:
        parts = first_arg.split("=")
        if len(parts) != 2:
            _bad_news()
        return parts[1]


def load_module(model_fname):
    """Import and return a module given its name or path."""
    if inspect.ismodule(model_fname):
        return model_fname
    dpath = os.path.dirname(model_fname)
    fname = os.path.basename(model_fname)
    sys.path.append(dpath)
    return importlib.import_module(fname)


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
    # There are some arguments here that will not make sense for all models
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.aph_args()
    cfg.subgradient_args()
    cfg.fixer_args()
    cfg.relaxed_ph_fixer_args()
    cfg.integer_relax_then_enforce_args()
    cfg.gapper_args()
    cfg.gapper_args(name="lagrangian")
    cfg.ph_primal_args()
    cfg.ph_dual_args()
    cfg.relaxed_ph_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.subgradient_bounder_args()
    cfg.xhatshuffle_args()
    cfg.xhatxbar_args()
    cfg.norm_rho_args()
    cfg.primal_dual_rho_args()
    cfg.converger_args()
    cfg.wxbar_read_write_args()
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
        # When branching_factors are used (multi-stage), num_scens is computed locally
        # above but cfg.num_scens is never set from it.  Fill it in so that
        # bundle_names_creator (which asserts cfg.num_scens is not None) works.
        if cfg.get("num_scens") is None and num_scens is not None:
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
