###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Extensive form solve for generic_cylinders."""

import pyomo.environ as pyo

import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
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

    Returns:
        ef (ExtensiveForm): the solved EF object
    """
    all_scenario_names, all_nodenames = name_lists(module, cfg, bundle_wrapper=bundle_wrapper)

    ef_dict = vanilla.ef_options(cfg,
                                 scenario_creator,
                                 scenario_denouement,
                                 all_scenario_names,
                                 scenario_creator_kwargs=scenario_creator_kwargs,
                                 all_nodenames=all_nodenames
                                 )

    # if the user dares, let them mess with the EFdict prior to solve
    if hasattr(module, 'ef_dict_callback'):
        module.ef_dict_callback(ef_dict, cfg)

    ef = ExtensiveForm(ef_dict["options"],
                       ef_dict["all_scenario_names"],
                       ef_dict["scenario_creator"],
                       scenario_creator_kwargs=ef_dict["scenario_creator_kwargs"],
                       extensions=ef_dict["extensions"],
                       extension_kwargs=ef_dict["extension_kwargs"],
                       all_nodenames=ef_dict["all_nodenames"]
                       )

    if ef.extensions is not None:
        ef.extobject.pre_solve()

    tee = cfg.tee_EF
    results = ef.solve_extensive_form(solver_options=ef_dict["solver_options"], tee=tee)

    if not pyo.check_optimal_termination(results):
        print("Warning: non-optimal solver termination")

    global_toc(f"EF objective: {ef.get_objective_value()}")

    if ef.extensions is not None:
        results = ef.extobject.post_solve(results)

    if cfg.solution_base_name is not None:
        root_writer = getattr(module, "ef_root_nonants_solution_writer", None)
        tree_writer = getattr(module, "ef_tree_solution_writer", None)

        sputils.ef_nonants_csv(ef.ef, f'{cfg.solution_base_name}.csv')
        sputils.ef_ROOT_nonants_npy_serializer(ef.ef, f'{cfg.solution_base_name}.npy')
        if root_writer is not None:
            ef.write_first_stage_solution(f'{cfg.solution_base_name}.csv',   # might overwite
                                          first_stage_solution_writer=root_writer)
        else:
            ef.write_first_stage_solution(f'{cfg.solution_base_name}.csv')
        if tree_writer is not None:
            ef.write_tree_solution(f'{cfg.solution_base_name}_soldir',
                                   scenario_tree_solution_writer=tree_writer)
        else:
            ef.write_tree_solution(f'{cfg.solution_base_name}_soldir')
        global_toc("Wrote EF solution data.")

    if hasattr(module, "custom_writer"):
        module.custom_writer(ef, cfg)

    return ef
