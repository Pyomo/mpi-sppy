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
from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper


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


def setup_stoch_admm(module, cfg, n_cylinders):
    """Create Stoch_AdmmWrapper for stochastic ADMM.

    Modifies cfg by attaching variable_probability.

    Returns:
        tuple: (scenario_creator, scenario_creator_kwargs,
                all_scenario_names, all_nodenames)
    """
    admm_subproblem_names = module.admm_subproblem_names_creator(cfg)
    stoch_scenario_names = module.stoch_scenario_names_creator(cfg)
    all_names = module.admm_stoch_subproblem_scenario_names_creator(
        admm_subproblem_names, stoch_scenario_names)

    scenario_creator_kwargs = module.kw_creator(cfg)
    stoch_scenario_name = stoch_scenario_names[0]
    consensus_vars = module.consensus_vars_creator(
        admm_subproblem_names, stoch_scenario_name, **scenario_creator_kwargs)

    admm = Stoch_AdmmWrapper(
        options={},
        all_admm_stoch_subproblem_scenario_names=all_names,
        split_admm_stoch_subproblem_scenario_name=module.split_admm_stoch_subproblem_scenario_name,
        admm_subproblem_names=admm_subproblem_names,
        stoch_scenario_names=stoch_scenario_names,
        scenario_creator=module.scenario_creator,
        consensus_vars=consensus_vars,
        n_cylinders=n_cylinders,
        mpicomm=MPI.COMM_WORLD,
        scenario_creator_kwargs=scenario_creator_kwargs,
        BFs=None,
    )

    # Store on cfg as plain attributes (Pyomo Config can't handle these types)
    object.__setattr__(cfg, "_admm_variable_probability", admm.var_prob_list)
    object.__setattr__(cfg, "_admm_scenario_names", all_names)
    object.__setattr__(cfg, "_admm_nodenames", admm.all_nodenames)

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

    bundler = AdmmBundler(
        module=module,
        scenarios_per_bundle=cfg.scenarios_per_bundle,
        admm_subproblem_names=admm_subproblem_names,
        stoch_scenario_names=stoch_scenario_names,
        consensus_vars=consensus_vars,
        combining_fn=module.combining_names,
        split_fn=module.split_admm_stoch_subproblem_scenario_name,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    bundle_names = bundler.bundle_names_creator()

    # Store on cfg as plain attributes (Pyomo Config can't handle these types)
    object.__setattr__(cfg, "_admm_variable_probability", bundler.var_prob_list)
    object.__setattr__(cfg, "_admm_scenario_names", bundle_names)
    object.__setattr__(cfg, "_admm_nodenames", None)  # bundles flatten to 2-stage

    return (bundler.scenario_creator, {},
            bundle_names, None)
