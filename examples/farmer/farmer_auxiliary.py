###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Auxiliary functions for the farmer example.

Kept out of ``farmer.py`` so first-time readers of the introductory
example are not confronted with helpers they do not need. Functions
here are consumed by downstream tools (e.g. findW's pin-dual algorithm)
that need a first-stage point feasible for every real scenario's
per-scenario subproblem.
"""

from mpisppy.utils.xhat_helpers import average_xhat_nonants
from farmer import average_scenario_creator


def feasible_xhat_creator(
    *,
    solver_name,
    solver_options=None,
    **scenario_creator_kwargs,
):
    """Return a candidate first-stage ``DEVOTED_ACRES`` for farmer that
    is feasible to fix in every real scenario's per-scenario
    subproblem.

    Returns ``{"ROOT": np.ndarray}`` in
    ``_mpisppy_node_list[0].nonant_vardata_list`` order.

    Strategy: solve the average scenario; return its first-stage values
    unrounded.

    Why no rounding is needed: ``DEVOTED_ACRES`` is bounded
    ``NonNegativeReals``, so any value the solver returns is a valid
    integer-free pin. Farmer also has relatively complete recourse via
    the buy/sell second-stage variables (``QuantityPurchased``,
    ``QuantitySubQuotaSold``, ``QuantitySuperQuotaSold``), so any
    feasible first-stage acreage allocation -- including the
    average-scenario optimum -- is feasible to pin in every real
    scenario.

    The continuous-first-stage case is the simplest one for the
    ``feasible_xhat_creator`` convention. Callers always go through this
    function rather than calling ``average_xhat_nonants`` directly so
    that switching the underlying model to one with integer first-stage
    (where rounding *is* needed) does not require changes at the call
    site.
    """
    arr = average_xhat_nonants(
        average_scenario_creator,
        solver_name=solver_name,
        scenario_creator_kwargs=scenario_creator_kwargs,
        solver_options=solver_options,
    )
    return {"ROOT": arr}
