###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Auxiliary functions for the netdes example.

Kept out of ``netdes.py`` so first-time readers of the introductory
example are not confronted with helpers they do not need. Functions
here are consumed by downstream tools (e.g. findW's pin-dual algorithm)
that need a first-stage point feasible for every real scenario's
per-scenario subproblem.
"""

import numpy as np

from mpisppy.utils.xhat_helpers import lp_xbar_nonants
from netdes import scenario_creator, scenario_names_creator


def feasible_xhat_creator(
    *,
    solver_name,
    solver_options=None,
    num_scens=None,
    **scenario_creator_kwargs,
):
    """Return a candidate first-stage x for netdes that is feasible to
    fix in every real scenario's per-scenario subproblem.

    Returns ``{"ROOT": np.ndarray}`` in
    ``_mpisppy_node_list[0].nonant_vardata_list`` order, ready to pass
    to ``Xhat_Eval._fix_nonants``.

    Strategy: solve the LP relaxation of every real scenario, take the
    probability-weighted average of arc-open values (the LP-xbar), and
    apply ``np.ceil``. The recourse constraint is
    ``y[e] - u[e]*x[e] <= 0``; raising ``x[e]`` to 1 only loosens this
    bound, so opening more arcs never tightens any per-scenario
    subproblem. Ceiling the LP-xbar opens every arc that any scenario's
    LP wanted positive, producing a strict cover that is feasible
    when pinned in every real scenario.

    The small ``- 1e-9`` margin keeps integer-valued LP solutions from
    being inadvertently bumped up by floating-point dust.

    Args:
        solver_name: pyomo solver name.
        solver_options: solver options dict.
        num_scens: number of real scenarios to aggregate over. If None,
            it is read from the instance file via ``parse``.
        **scenario_creator_kwargs: forwarded to
            ``netdes.scenario_creator`` (notably ``path``).
    """
    if num_scens is None:
        from parse import parse
        path = scenario_creator_kwargs.get("path")
        if path is None:
            raise ValueError(
                "feasible_xhat_creator needs num_scens or path kwarg"
            )
        num_scens = parse(path, scenario_ix=None)["K"]
    snames = scenario_names_creator(num_scens)
    arr = lp_xbar_nonants(
        scenario_creator,
        snames,
        solver_name=solver_name,
        scenario_creator_kwargs=scenario_creator_kwargs,
        solver_options=solver_options,
    )
    return {"ROOT": np.ceil(arr - 1e-9)}
