###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Auxiliary functions for the sslp example.

Kept out of ``sslp.py`` so first-time readers of the introductory
example are not confronted with helpers they do not need. Functions
here are consumed by downstream tools (e.g. findW's pin-dual algorithm)
that need a first-stage point feasible for every real scenario's
per-scenario subproblem.
"""

import numpy as np

from mpisppy.utils.xhat_helpers import lp_xbar_nonants
from sslp import scenario_creator, scenario_names_creator


def feasible_xhat_creator(
    *,
    solver_name,
    solver_options=None,
    num_scens,
    **scenario_creator_kwargs,
):
    """Return a candidate first-stage ``FacilityOpen`` for sslp.

    Returns ``{"ROOT": np.ndarray}`` in
    ``_mpisppy_node_list[0].nonant_vardata_list`` order.

    Strategy: solve the LP relaxation of every real scenario, take the
    probability-weighted average of ``FacilityOpen`` (sslp ships
    ``_mpisppy_probability = "uniform"``), then ``np.round`` to
    integer.

    Why ``np.round`` is feasibility-preserving for sslp: more open
    facilities never tightens ``DemandConstraint`` (more capacity) or
    ``ClientConstraint`` (does not involve ``FacilityOpen``). The
    shipped model also carries a high-``Penalty`` ``Dummy`` slack, so
    any pin is technically feasible; the rounded LP-xbar is a
    meaningful low-slack candidate for the pin-dual machinery.

    Args:
        solver_name: pyomo solver name.
        solver_options: solver options dict.
        num_scens: number of scenarios to aggregate over.
        **scenario_creator_kwargs: forwarded to
            ``sslp.scenario_creator`` (e.g. ``data_dir``, ``surrogate``).
    """
    snames = scenario_names_creator(num_scens)
    arr = lp_xbar_nonants(
        scenario_creator,
        snames,
        solver_name=solver_name,
        scenario_creator_kwargs=scenario_creator_kwargs,
        solver_options=solver_options,
    )
    return {"ROOT": np.round(arr)}
