###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Scenario/bundle pickling and LP/MPS file writing for generic_cylinders."""

import importlib
import os

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy import global_toc, MPI

from mpisppy.generic.parsing import name_lists


# ----------------------------------------------------------------------------
# Pre-pickle preprocessing pipeline
# ----------------------------------------------------------------------------
#
# When any of cfg.presolve_before_pickle, cfg.pre_pickle_function, or
# cfg.iter0_before_pickle is set, the pickling writers run a pipeline on each
# scenario / bundle before dill_pickle:
#
#     SPPresolve  ->  user callback  ->  iteration-0 solve  ->  dill_pickle
#
# See doc/src/pickling.rst for the user-facing documentation.

_PICKLE_METADATA_VERSION = 1


def _any_pre_pickle_stage_enabled(cfg):
    return (cfg.get("presolve_before_pickle")
            or cfg.get("pre_pickle_function") is not None
            or cfg.get("iter0_before_pickle"))


def _resolve_dotted_callable(dotted):
    """Resolve a dotted name like ``pkg.mod.fn`` to a callable."""
    if "." not in dotted:
        raise ValueError(
            f"--pre-pickle-function expects a dotted name like "
            f"'package.module.function', got: {dotted!r}"
        )
    module_name, attr_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, attr_name, None)
    if fn is None or not callable(fn):
        raise ValueError(
            f"--pre-pickle-function {dotted!r} did not resolve to a callable"
        )
    return fn


def _build_presolve_options(cfg):
    """Translate cfg OBBT-related fields into the dict SPPresolve expects."""
    opts = {}
    if cfg.get("obbt"):
        opts["obbt"] = True
        obbt_options = {}
        if cfg.get("full_obbt"):
            obbt_options["nonant_variables_only"] = False
        if cfg.get("obbt_solver"):
            obbt_options["solver_name"] = cfg.obbt_solver
        if cfg.get("obbt_solver_options"):
            obbt_options["solver_options"] = sputils.option_string_to_dict(
                cfg.obbt_solver_options)
        if obbt_options:
            opts["obbt_options"] = obbt_options
    return opts


def _build_pickle_pipeline_spbase(module, cfg, scenario_creator,
                                  scenario_creator_kwargs,
                                  scenario_denouement, all_scenario_names,
                                  all_nodenames, comm):
    """Construct an SPBase suitable for the pre-pickle pipeline.

    The SPBase distributes scenarios across ranks and gives us a
    ``local_scenarios`` dict to iterate over. We also stash a solver name
    in ``options`` so SPPresolve's optional OBBT path can find it.
    """
    from mpisppy.spbase import SPBase
    sp_options = {
        "verbose": cfg.get("verbose", False),
        "toc": cfg.get("toc", True),
    }
    # OBBT looks at options["solver_name"]; provide it if available so the
    # OBBT path does not blow up when --presolve-before-pickle --obbt is set.
    pickle_solver = cfg.get("pickle_solver_name") or cfg.get("solver_name")
    if pickle_solver is not None:
        sp_options["solver_name"] = pickle_solver
    # Bundles produce per-scenario nonant names that differ across bundles
    # (each EF prefixes them by scenario), so the same nonant-names check
    # that the cylinder driver disables via cfg.turn_off_names_check must
    # also be disabled here.
    if cfg.get("turn_off_names_check"):
        sp_options["turn_off_names_check"] = True
    sp = SPBase(
        options=sp_options,
        all_scenario_names=all_scenario_names,
        scenario_creator=scenario_creator,
        scenario_denouement=scenario_denouement,
        all_nodenames=all_nodenames,
        mpicomm=comm,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )
    return sp


def _attach_dual_suffixes(model):
    """Attach Pyomo IMPORT suffixes for duals and reduced costs."""
    if not hasattr(model, "dual"):
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    if not hasattr(model, "rc"):
        model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)


def _solve_iter0_for_pickle(sp, cfg):
    """Solve every local scenario / bundle once with its original objective.

    Hard-fail on any non-optimal termination -- producing pickles with
    silently bad state would be worse than the job stopping (resolved
    decision #5 in the design doc).
    """
    solver_name = cfg.get("pickle_solver_name") or cfg.get("solver_name")
    if not solver_name:
        raise RuntimeError(
            "--iter0-before-pickle requires either --pickle-solver-name or "
            "--solver-name to be set"
        )
    options_str = (cfg.get("pickle_solver_options")
                   or cfg.get("solver_options"))
    solver_options = (sputils.option_string_to_dict(options_str)
                      if options_str else None)

    for sname, model in sp.local_scenarios.items():
        _attach_dual_suffixes(model)
        # A fresh solver per scenario keeps persistent solver bookkeeping
        # simple and avoids any cross-scenario state leakage.
        solver = pyo.SolverFactory(solver_name)
        if solver_options:
            for k, v in solver_options.items():
                solver.options[k] = v
        if hasattr(solver, "set_instance"):
            solver.set_instance(model)
            results = solver.solve(load_solutions=True)
        else:
            results = solver.solve(model, load_solutions=True)
        tc = results.solver.termination_condition
        if tc != pyo.TerminationCondition.optimal:
            raise RuntimeError(
                f"Pickle-time iter0 solve for {sname!r} did not reach an "
                f"optimal termination condition (got {tc!r}). The pickling "
                f"job is shutting down rather than producing a pickle with "
                f"unsolved state. See doc/src/pickling.rst."
            )


def _attach_pickle_metadata(sp, cfg):
    """Stash metadata about which pre-pickle stages ran on _mpisppy_data."""
    metadata = {
        "version": _PICKLE_METADATA_VERSION,
        "presolve_before_pickle": bool(cfg.get("presolve_before_pickle")),
        "obbt": bool(cfg.get("obbt")) if cfg.get("presolve_before_pickle") else False,
        "pre_pickle_function": cfg.get("pre_pickle_function"),
        "iter0_before_pickle": bool(cfg.get("iter0_before_pickle")),
        "pickle_solver_name": cfg.get("pickle_solver_name") or cfg.get("solver_name") if cfg.get("iter0_before_pickle") else None,
    }
    for model in sp.local_scenarios.values():
        # _mpisppy_data is attached by SPBase before this runs
        model._mpisppy_data.pickle_metadata = metadata


def _run_pre_pickle_pipeline(sp, cfg):
    """Run the enabled pre-pickle stages on every model in sp.local_scenarios.

    Stages are independent and ordered: presolve -> user callback -> iter0.
    """
    if cfg.get("presolve_before_pickle"):
        from mpisppy.opt.presolve import SPPresolve
        SPPresolve(sp, _build_presolve_options(cfg)).presolve()

    if cfg.get("pre_pickle_function") is not None:
        callback = _resolve_dotted_callable(cfg.pre_pickle_function)
        for sname, model in sp.local_scenarios.items():
            callback(model, cfg)

    if cfg.get("iter0_before_pickle"):
        _solve_iter0_for_pickle(sp, cfg)

    _attach_pickle_metadata(sp, cfg)


def write_scenarios(module, cfg, scenario_creator, scenario_creator_kwargs,
                    scenario_denouement, comm):
    """Pickle scenarios to disk.

    If any pre-pickle preprocessing flag (``presolve_before_pickle``,
    ``pre_pickle_function``, ``iter0_before_pickle``) is set, the writer
    builds a transient ``SPBase`` and runs the pipeline before pickling.
    Otherwise it uses the legacy fast path that calls ``scenario_creator``
    directly per scenario.

    Args:
        module: model module
        cfg: Config object
        scenario_creator: scenario creator function
        scenario_creator_kwargs: kwargs for scenario creator
        scenario_denouement: denouement function
        comm: MPI communicator
    """
    import mpisppy.utils.pickle_bundle as pickle_bundle
    import shutil
    assert hasattr(cfg, "num_scens")
    ScenCount = cfg.num_scens

    my_rank = comm.Get_rank()
    if my_rank == 0:
        if os.path.exists(cfg.pickle_scenarios_dir):
            shutil.rmtree(cfg.pickle_scenarios_dir)
        os.makedirs(cfg.pickle_scenarios_dir)
    comm.Barrier()

    if _any_pre_pickle_stage_enabled(cfg):
        all_scenario_names = module.scenario_names_creator(ScenCount)
        sp = _build_pickle_pipeline_spbase(
            module, cfg, scenario_creator, scenario_creator_kwargs,
            scenario_denouement, all_scenario_names, None, comm)
        _run_pre_pickle_pipeline(sp, cfg)
        for sname, scen in sp.local_scenarios.items():
            fname = os.path.join(cfg.pickle_scenarios_dir, sname + ".pkl")
            pickle_bundle.dill_pickle(scen, fname)
    else:
        n_proc = comm.Get_size()
        avg = ScenCount / n_proc
        slices = [list(range(int(i * avg), int((i + 1) * avg))) for i in range(n_proc)]
        local_slice = slices[my_rank]
        my_start = local_slice[0]   # zero based
        inum = sputils.extract_num(module.scenario_names_creator(1)[0])
        local_scenario_names = module.scenario_names_creator(len(local_slice),
                                                             start=inum + my_start)
        for sname in local_scenario_names:
            scen = scenario_creator(sname, **scenario_creator_kwargs)
            fname = os.path.join(cfg.pickle_scenarios_dir, sname+".pkl")
            pickle_bundle.dill_pickle(scen, fname)
    global_toc(f"Pickled Scenarios written to {cfg.pickle_scenarios_dir}")


def read_pickled_scenario(sname, cfg):
    """Read a single pickled scenario from disk.

    Args:
        sname (str): scenario name
        cfg: Config object with unpickle_scenarios_dir

    Returns:
        the unpickled scenario model
    """
    import mpisppy.utils.pickle_bundle as pickle_bundle
    fname = os.path.join(cfg.unpickle_scenarios_dir, sname+".pkl")
    scen = pickle_bundle.dill_unpickle(fname)
    return scen


def write_bundles(module, cfg, scenario_creator, scenario_creator_kwargs, comm):
    """Pickle bundles to disk.

    If any pre-pickle preprocessing flag (``presolve_before_pickle``,
    ``pre_pickle_function``, ``iter0_before_pickle``) is set, the writer
    builds a transient ``SPBase`` over the bundles and runs the pipeline
    before pickling. Otherwise it uses the legacy fast path that calls
    ``scenario_creator`` directly per bundle.

    Args:
        module: model module
        cfg: Config object
        scenario_creator: scenario creator function (already wrapped by
            ProperBundler when proper bundles are in use)
        scenario_creator_kwargs: kwargs for scenario creator
        comm: MPI communicator
    """
    import mpisppy.utils.pickle_bundle as pickle_bundle
    import shutil
    assert hasattr(cfg, "num_scens")
    ScenCount = cfg.num_scens
    bsize = int(cfg.scenarios_per_bundle)
    numbuns = ScenCount // bsize
    n_proc = comm.Get_size()
    my_rank = comm.Get_rank()

    if numbuns < n_proc:
        raise RuntimeError(
            "More MPI ranks (%d) supplied than needed given the number of bundles (%d) "
            % (n_proc, numbuns)
        )

    # We need to know if scenarios (not bundles) are one-based.
    inum = sputils.extract_num(module.scenario_names_creator(1)[0])
    all_bundle_names = [f"Bundle_{bn*bsize+inum}_{(bn+1)*bsize-1+inum}"
                        for bn in range(numbuns)]

    if my_rank == 0:
        if os.path.exists(cfg.pickle_bundles_dir):
            shutil.rmtree(cfg.pickle_bundles_dir)
        os.makedirs(cfg.pickle_bundles_dir)
    comm.Barrier()

    if _any_pre_pickle_stage_enabled(cfg):
        # Proper bundles always become a 2-stage problem; all_nodenames=None.
        sp = _build_pickle_pipeline_spbase(
            module, cfg, scenario_creator, scenario_creator_kwargs,
            None, all_bundle_names, None, comm)
        _run_pre_pickle_pipeline(sp, cfg)
        for bname, bundle in sp.local_scenarios.items():
            fname = os.path.join(cfg.pickle_bundles_dir, bname + ".pkl")
            pickle_bundle.dill_pickle(bundle, fname)
    else:
        avg = numbuns / n_proc
        slices = [list(range(int(i * avg), int((i + 1) * avg))) for i in range(n_proc)]
        local_slice = slices[my_rank]
        local_bundle_names = [all_bundle_names[bn] for bn in local_slice]
        for bname in local_bundle_names:
            bundle = scenario_creator(bname, **scenario_creator_kwargs)
            fname = os.path.join(cfg.pickle_bundles_dir, bname+".pkl")
            pickle_bundle.dill_pickle(bundle, fname)
    global_toc(f"Bundles written to {cfg.pickle_bundles_dir}")


def write_scenario_lp_mps_files_only(module, cfg, scenario_creator,
                                     scenario_creator_kwargs,
                                     scenario_denouement,
                                     bundle_wrapper=None):
    """Construct scenarios and write per-scenario LP/MPS + nonants JSON
    (and rho.csv using either scenario rhos or cfg.default_rho) WITHOUT solving.

    Args:
        module: model module
        cfg: Config object
        scenario_creator: scenario creator function
        scenario_creator_kwargs: kwargs for scenario creator
        scenario_denouement: denouement function
        bundle_wrapper: ProperBundler or None
    """
    from mpisppy.spbase import SPBase
    from mpisppy.extensions.scenario_lp_mps_files import Scenario_lp_mps_files

    all_scenario_names, all_nodenames = name_lists(module, cfg,
                                                   bundle_wrapper=bundle_wrapper)

    # SPBase builds the scenarios and scenario-tree bookkeeping; no solves happen here.
    sp_options = {
        "verbose": cfg.verbose,
        "toc": cfg.get("toc", True),
    }

    sp = SPBase(
        options=sp_options,
        all_scenario_names=all_scenario_names,
        scenario_creator=scenario_creator,
        scenario_denouement=scenario_denouement,
        all_nodenames=all_nodenames,
        mpicomm=MPI.COMM_WORLD,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    # Make SPBase look PH-like for this extension
    sp.local_subproblems = sp.local_scenarios
    sp.options["write_lp_mps_extension_options"] = {
        "write_scenario_lp_mps_files_dir": cfg.write_scenario_lp_mps_files_dir,
        "cfg": cfg,   # IMPORTANT: pass cfg so extension can use default_rho
    }

    ext = Scenario_lp_mps_files(sp)
    ext.pre_iter0()

    if sp.cylinder_rank == 0:
        global_toc(f"Wrote scenario lp/mps/nonants to {cfg.write_scenario_lp_mps_files_dir}")
