###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Extensive form solve for generic_cylinders."""

import os

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.utils.solver_spec as solver_spec
from mpisppy import global_toc

from mpisppy.generic.parsing import name_lists


def do_EF(module, cfg, scenario_creator, scenario_creator_kwargs,
          scenario_denouement, bundle_wrapper=None):
    """Solve the extensive form.

    Args:
        module: model module
        cfg: Config object
        scenario_creator: scenario creator function
        scenario_creator_kwargs: kwargs for scenario creator
        scenario_denouement: denouement function
        bundle_wrapper: ProperBundler or None
    """
    all_scenario_names, _ = name_lists(module, cfg, bundle_wrapper=bundle_wrapper)
    ef = sputils.create_EF(
        all_scenario_names,
        module.scenario_creator,
        scenario_creator_kwargs=module.kw_creator(cfg),
    )

    sroot, solver_name, solver_options = solver_spec.solver_specification(cfg, "EF")

    solver = pyo.SolverFactory(solver_name)
    if solver_options is not None:
        # We probably could just assign the dictionary in one line...
        for option_key, option_value in solver_options.items():
            solver.options[option_key] = option_value

    solver_log_dir = cfg.get("solver_log_dir", "")
    solve_kw_args = dict()
    if solver_log_dir and len(solver_log_dir) > 0:
        os.makedirs(solver_log_dir, exist_ok=True)
        solve_kw_args['keepfiles'] = True
        log_fn = "EFsolverlog.log"
        solve_kw_args['logfile'] = os.path.join(solver_log_dir, log_fn)

    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        results = solver.solve(tee=cfg.tee_EF, **solve_kw_args)
    else:
        results = solver.solve(ef, tee=cfg.tee_EF, symbolic_solver_labels=True, **solve_kw_args)
    if not pyo.check_optimal_termination(results):
        print("Warning: non-optimal solver termination")

    global_toc(f"EF objective: {pyo.value(ef.EF_Obj)}")

    if cfg.solution_base_name is not None:
        root_writer = getattr(module, "ef_root_nonants_solution_writer", None)
        tree_writer = getattr(module, "ef_tree_solution_writer", None)

        sputils.ef_nonants_csv(ef, f'{cfg.solution_base_name}.csv')
        sputils.ef_ROOT_nonants_npy_serializer(ef, f'{cfg.solution_base_name}.npy')
        if root_writer is not None:
            sputils.write_ef_first_stage_solution(ef, f'{cfg.solution_base_name}.csv',   # might overwite
                                                  first_stage_solution_writer=root_writer)
        else:
            sputils.write_ef_first_stage_solution(ef, f'{cfg.solution_base_name}.csv')
        if tree_writer is not None:
            sputils.write_ef_tree_solution(ef, f'{cfg.solution_base_name}_soldir',
                                          scenario_tree_solution_writer=tree_writer)
        else:
            sputils.write_ef_tree_solution(ef, f'{cfg.solution_base_name}_soldir')
        global_toc("Wrote EF solution data.")

    if hasattr(module, "custom_writer"):
        module.custom_writer(ef, cfg)
