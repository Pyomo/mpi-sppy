###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""ADMM setup for generic_cylinders."""

from mpisppy import MPI
from mpisppy.utils.admmWrapper import AdmmWrapper
from mpisppy.utils.stoch_admmWrapper import (
    Stoch_AdmmWrapper,
    default_combining_names,
    default_split_admm_stoch_subproblem_scenario_name,
    default_admm_stoch_subproblem_scenario_names_creator,
)


def admm_args(cfg):
    """Register ADMM-specific config args.

    Note: num_admm_subproblems and num_stoch_scens may already be registered
    by the model's inparser_adder; we only add them if not already present.
    """
    cfg.add_to_config("admm", description="Use ADMM decomposition",
                      domain=bool, default=False)
    cfg.add_to_config("stoch_admm", description="Use stochastic ADMM decomposition",
                      domain=bool, default=False)
    if "num_admm_subproblems" not in cfg:
        cfg.add_to_config("num_admm_subproblems",
                          description="Number of ADMM subproblems (stoch-admm only)",
                          domain=int, default=None)
    if "num_stoch_scens" not in cfg:
        cfg.add_to_config("num_stoch_scens",
                          description="Number of stochastic scenarios (stoch-admm only)",
                          domain=int, default=None)
    
    cfg.add_branching_factors()
    cfg.add_stage2_ef_solver_name_arg()


def _count_cylinders(cfg):
    """Count the number of cylinders (hub + spokes) from cfg flags.

    This must be called before creating the ADMM wrapper, because
    the wrapper needs n_cylinders for rank assignment.
    """
    count = 1  # the hub
    spoke_flags = [
        "fwph", "lagrangian", "ph_dual", "relaxed_ph",
        "subgradient", "xhatshuffle", "xhatxbar", "reduced_costs",
    ]
    for flag in spoke_flags:
        if cfg.get(flag, ifmissing=False):
            count += 1
    return count


def _check_admm_compatibility(cfg):
    """Raise errors for incompatible option combinations."""
    if cfg.get("admm", ifmissing=False) and cfg.get("stoch_admm", ifmissing=False):
        raise RuntimeError("Cannot use both --admm and --stoch-admm")
    if cfg.get("fwph", ifmissing=False):
        raise RuntimeError("FWPH does not work with ADMM (variable_probability)")
    if cfg.get("scenarios_per_bundle") is not None:
        if cfg.get("admm", ifmissing=False):
            raise RuntimeError("Proper bundles are not supported with deterministic ADMM")
        # stoch_admm + bundles is OK for in-memory operation — handled by AdmmBundler
        if cfg.get("stoch_admm", ifmissing=False):
            # The generic pickling paths assume standard bundle names like
            # "Bundle_<i>_<j>", which are incompatible with AdmmBundler's
            # "Bundle_ADMM_<sub>_<k>" naming and can cause internal errors.
            pickling_opts = (
                "pickle_bundles_dir",
                "unpickle_bundles_dir",
                "pickle_scenarios_dir",
                "unpickle_scenarios_dir",
            )
            for opt in pickling_opts:
                if cfg.get(opt, ifmissing=None) is not None:
                    raise RuntimeError(
                        "Option '--{}' is not supported together with "
                        "'--stoch-admm' and '--scenarios-per-bundle'. "
                        "ADMM bundling uses a different naming scheme for "
                        "bundles/scenarios, which is incompatible with the "
                        "generic pickling/unpickling paths.".format(
                            opt.replace("_", "-")
                        )
                    )
    # Pickling is also unsupported for ADMM without bundles (the IO paths
    # use module.scenario_names_creator which won't match ADMM wrapper names).
    pickling_opts = ("pickle_bundles_dir", "unpickle_bundles_dir",
                     "pickle_scenarios_dir", "unpickle_scenarios_dir")
    for opt in pickling_opts:
        if cfg.get(opt, ifmissing=None) is not None:
            raise RuntimeError(
                f"--{opt.replace('_', '-')} is not supported with ADMM"
            )
    # xhatshuffle without stage2_ef_solver_name is invalid for stoch-admm:
    # the picked scenario's xhats only fix nonants along its own tree path,
    # leaving ADMM consensus variables in other stochastic outcomes
    # unconstrained.  The resulting "inner bound" violates the problem's
    # ADMM consensus constraints and has no valid interpretation as a
    # relaxation, so it must not be silently produced.
    if (cfg.get("stoch_admm", ifmissing=False)
            and cfg.get("xhatshuffle", ifmissing=False)
            and cfg.get("stage2_ef_solver_name") is None):
        raise RuntimeError(
            "--xhatshuffle with --stoch-admm requires --stage2-ef-solver-name. "
            "Without it, xhatshuffle fixes nonants only along one scenario's "
            "tree path, leaving the ADMM consensus variables in other "
            "stochastic outcomes unconstrained and producing an invalid "
            "(over-optimistic) inner bound.  Pass --stage2-ef-solver-name "
            "(typically the same solver as --solver-name), or use "
            "--xhatxbar instead."
        )


def setup_admm(module, cfg, n_cylinders):
    """Create AdmmWrapper for deterministic ADMM.

    Modifies cfg by attaching variable_probability.

    Returns:
        tuple: (scenario_creator, scenario_creator_kwargs,
                all_scenario_names, all_nodenames)
    """
    all_scenario_names = module.scenario_names_creator(cfg.num_scens)
    scenario_creator_kwargs = module.kw_creator(cfg)
    consensus_vars = module.consensus_vars_creator(
        cfg.num_scens, all_scenario_names, **scenario_creator_kwargs
    )

    admm = AdmmWrapper(
        options={},
        all_scenario_names=all_scenario_names,
        scenario_creator=module.scenario_creator,
        consensus_vars=consensus_vars,
        n_cylinders=n_cylinders,
        mpicomm=MPI.COMM_WORLD,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    # Store on cfg as plain attributes (Pyomo Config can't handle these types)
    object.__setattr__(cfg, "_admm_variable_probability", admm.var_prob_list)
    object.__setattr__(cfg, "_admm_scenario_names", all_scenario_names)

    return (admm.admmWrapper_scenario_creator, {},
            all_scenario_names, None)


def _discover_naming_helpers(module):
    """Discover the model module's stoch-admm naming inverse-pair and
    validate the both-or-neither contract.

    ``combining_names(admm_sub, stoch_scen) -> name`` and
    ``split_admm_stoch_subproblem_scenario_name(name) -> (admm_sub,
    stoch_scen)`` must agree.  If the user defines one without the
    other, we cannot pair a default with a custom function safely --
    the pairing would silently produce mismatched names -- so we
    raise.  When both are absent, callers fall back to the module
    defaults in ``mpisppy.utils.stoch_admmWrapper``.

    Returns:
        dict with keys ``combining_names`` and
        ``split_admm_stoch_subproblem_scenario_name``; either-both are
        the module's callables or both are None (signalling "fall back
        to defaults" to the caller).
    """
    combining = getattr(module, "combining_names", None)
    split = getattr(module, "split_admm_stoch_subproblem_scenario_name", None)
    if (combining is None) != (split is None):
        present = "combining_names" if combining is not None else "split_admm_stoch_subproblem_scenario_name"
        missing = "split_admm_stoch_subproblem_scenario_name" if combining is not None else "combining_names"
        raise RuntimeError(
            f"Module {module.__name__!r} defines {present} but not "
            f"{missing}.  These naming helpers form an inverse pair "
            f"and must be defined together (or both omitted to use "
            f"the defaults from mpisppy.utils.stoch_admmWrapper).  "
            f"See doc/src/generic_admm.rst."
        )
    return {
        "combining_names": combining,
        "split_admm_stoch_subproblem_scenario_name": split,
    }


def _discover_first_stage_hooks(module):
    """Discover the four optional first-stage hooks on a stoch-admm
    model module and validate the both-or-neither contract.

    The two core hooks (first_stage_cost, first_stage_varlist) must
    be defined together or both omitted; mixing them is an error.

    The two advanced hooks (first_stage_surrogate_nonant_list,
    first_stage_nonant_ef_suppl_list) are each independent of the
    other; defining either alone is fine, but only when the two
    core hooks are also defined -- they forward to
    sputils.attach_root_node's optional surrogate_nonant_list /
    nonant_ef_suppl_list parameters and have nothing to attach to
    otherwise.

    Returns:
        dict: keyword arguments suitable for forwarding to
        Stoch_AdmmWrapper / AdmmBundler.  Hooks that the module did
        not define are passed as None.
    """
    first_stage_cost = getattr(module, "first_stage_cost", None)
    first_stage_varlist = getattr(module, "first_stage_varlist", None)
    if (first_stage_cost is None) != (first_stage_varlist is None):
        present = "first_stage_cost" if first_stage_cost is not None else "first_stage_varlist"
        missing = "first_stage_varlist" if first_stage_cost is not None else "first_stage_cost"
        raise RuntimeError(
            f"Module {module.__name__!r} defines {present} but not "
            f"{missing}.  These hooks must be defined together "
            f"(or both omitted).  See doc/src/generic_admm.rst."
        )

    first_stage_surrogate_nonant_list = getattr(
        module, "first_stage_surrogate_nonant_list", None)
    first_stage_nonant_ef_suppl_list = getattr(
        module, "first_stage_nonant_ef_suppl_list", None)
    advanced = {
        "first_stage_surrogate_nonant_list": first_stage_surrogate_nonant_list,
        "first_stage_nonant_ef_suppl_list": first_stage_nonant_ef_suppl_list,
    }
    present_advanced = [n for n, h in advanced.items() if h is not None]
    if present_advanced and first_stage_cost is None:
        raise RuntimeError(
            f"Module {module.__name__!r} defines the advanced hook(s) "
            f"{present_advanced} but not first_stage_cost / "
            f"first_stage_varlist.  Advanced hooks forward to "
            f"sputils.attach_root_node's optional parameters and only "
            f"make sense when the core hooks are also defined.  See "
            f"doc/src/generic_admm.rst."
        )

    return {
        "first_stage_cost": first_stage_cost,
        "first_stage_varlist": first_stage_varlist,
        "first_stage_surrogate_nonant_list": first_stage_surrogate_nonant_list,
        "first_stage_nonant_ef_suppl_list": first_stage_nonant_ef_suppl_list,
    }


def setup_stoch_admm(module, cfg, n_cylinders):
    """Create Stoch_AdmmWrapper for stochastic ADMM.

    Modifies cfg by attaching variable_probability.

    Returns:
        tuple: (scenario_creator, scenario_creator_kwargs,
                all_scenario_names, all_nodenames)
    """
    admm_subproblem_names = module.admm_subproblem_names_creator(cfg)
    stoch_scenario_names = module.stoch_scenario_names_creator(cfg)

    naming = _discover_naming_helpers(module)
    # Fall back to the default scen-names creator (which uses the
    # module's combining_names if defined, otherwise the package
    # default).  Same nesting order in both paths so MPI rank
    # assignment is unaffected by the fallback.
    if hasattr(module, "admm_stoch_subproblem_scenario_names_creator"):
        # The wrapper decodes each wrapped-scenario name via
        # split_admm_stoch_subproblem_scenario_name; if the user
        # ships custom names without the matching split, decoding
        # silently falls back to the default split and ValueErrors
        # at runtime.  Catch the inconsistency here instead.
        if naming["split_admm_stoch_subproblem_scenario_name"] is None:
            raise RuntimeError(
                f"Module {module.__name__!r} defines "
                f"admm_stoch_subproblem_scenario_names_creator but "
                f"not the combining_names / "
                f"split_admm_stoch_subproblem_scenario_name inverse "
                f"pair.  The wrapper needs the split function to "
                f"decode the custom names; provide both, or omit "
                f"all three to use the defaults."
            )
        all_names = module.admm_stoch_subproblem_scenario_names_creator(
            admm_subproblem_names, stoch_scenario_names)
    else:
        combining = naming["combining_names"] or default_combining_names
        all_names = default_admm_stoch_subproblem_scenario_names_creator(
            admm_subproblem_names, stoch_scenario_names,
            combining_fn=combining)

    scenario_creator_kwargs = module.kw_creator(cfg)
    stoch_scenario_name = stoch_scenario_names[0]
    consensus_vars = module.consensus_vars_creator(
        admm_subproblem_names, stoch_scenario_name, **scenario_creator_kwargs)

    first_stage_hooks = _discover_first_stage_hooks(module)

    admm = Stoch_AdmmWrapper(
        options={},
        all_admm_stoch_subproblem_scenario_names=all_names,
        split_admm_stoch_subproblem_scenario_name=naming[
            "split_admm_stoch_subproblem_scenario_name"],
        admm_subproblem_names=admm_subproblem_names,
        stoch_scenario_names=stoch_scenario_names,
        scenario_creator=module.scenario_creator,
        consensus_vars=consensus_vars,
        n_cylinders=n_cylinders,
        mpicomm=MPI.COMM_WORLD,
        scenario_creator_kwargs=scenario_creator_kwargs,
        BFs=cfg.get("branching_factors"),
        **first_stage_hooks,
    )

    # Store on cfg as plain attributes (Pyomo Config can't handle these types)
    object.__setattr__(cfg, "_admm_variable_probability", admm.var_prob_list)
    object.__setattr__(cfg, "_admm_scenario_names", all_names)
    object.__setattr__(cfg, "_admm_nodenames", admm.all_nodenames)

    # Publish the augmented branching factors so downstream consumers
    # (notably xhatshuffle's stage2ef path in extensions/xhatbase.py) see
    # the wrapper's true tree shape without the user having to hand-encode
    # the wrapper's append convention.
    cfg.quick_assign("branching_factors", list, list(admm.BFs))

    return (admm.admmWrapper_scenario_creator, {},
            all_names, admm.all_nodenames)


def setup_stoch_admm_with_bundles(module, cfg, n_cylinders):
    """Create AdmmBundler for stochastic ADMM with proper bundles.

    The AdmmBundler creates scenarios on-the-fly (like ProperBundler),
    so PH can distribute bundles across ranks independently.

    Modifies cfg by attaching variable_probability and bundle names.

    Returns:
        tuple: (scenario_creator, scenario_creator_kwargs,
                all_bundle_names, all_nodenames)
    """
    from mpisppy.utils.admm_bundler import AdmmBundler

    admm_subproblem_names = module.admm_subproblem_names_creator(cfg)
    stoch_scenario_names = module.stoch_scenario_names_creator(cfg)

    scenario_creator_kwargs = module.kw_creator(cfg)
    stoch_scenario_name = stoch_scenario_names[0]
    consensus_vars = module.consensus_vars_creator(
        admm_subproblem_names, stoch_scenario_name, **scenario_creator_kwargs)

    first_stage_hooks = _discover_first_stage_hooks(module)
    naming = _discover_naming_helpers(module)
    # AdmmBundler requires both callables (it does not have the
    # wrapper's None-default plumbing).  Resolve to the package
    # defaults here when the module omits them.
    combining_fn = naming["combining_names"] or default_combining_names
    split_fn = (naming["split_admm_stoch_subproblem_scenario_name"]
                or default_split_admm_stoch_subproblem_scenario_name)

    bundler = AdmmBundler(
        module=module,
        scenarios_per_bundle=cfg.scenarios_per_bundle,
        admm_subproblem_names=admm_subproblem_names,
        stoch_scenario_names=stoch_scenario_names,
        consensus_vars=consensus_vars,
        combining_fn=combining_fn,
        split_fn=split_fn,
        scenario_creator_kwargs=scenario_creator_kwargs,
        **first_stage_hooks,
    )
    bundle_names = bundler.bundle_names_creator()

    # Store on cfg as plain attributes (Pyomo Config can't handle these types)
    object.__setattr__(cfg, "_admm_variable_probability", bundler.var_prob_list)
    object.__setattr__(cfg, "_admm_scenario_names", bundle_names)
    object.__setattr__(cfg, "_admm_nodenames", None)  # bundles flatten to 2-stage

    return (bundler.scenario_creator, {},
            bundle_names, None)
