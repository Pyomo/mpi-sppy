###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Auxiliary functions for the (multistage) aircond example.

Holds the ``feasible_xhat_creator`` for aircond, kept beside the model
(``aircond.py``) so the ``<module>_auxiliary`` discovery convention in
``cfg_vanilla._find_feasible_xhat_creator`` resolves it. This is the
multistage counterpart of ``examples/farmer/farmer_auxiliary.py``; see
doc/src/feasible_xhat.rst.
"""

import numpy as np

from mpisppy.utils.xhat_helpers import ef_xhat_nonants
from mpisppy.tests.examples.aircond import (
    scenario_creator,
    scenario_names_creator,
)


def feasible_xhat_creator(
    *,
    solver_name,
    solver_options=None,
    branching_factors=None,
    **scenario_creator_kwargs,
):
    """Return a per-node candidate ``{node_name: np.ndarray}`` for aircond
    that is feasible to fix in every real scenario's per-scenario
    subproblem. Each array holds ``[RegularProd, OvertimeProd]`` for that
    non-leaf node, in ``_mpisppy_node_list`` order.

    Strategy: solve the **expected-value tree** -- the real branching
    structure with the demand randomness pinned to its mean
    (``sigma_dev=0``) -- and read off every node's production decisions.

    Why no rounding is needed: aircond's node decisions ``RegularProd``
    and ``OvertimeProd`` are bounded ``NonNegativeReals``, and the only
    hard constraint on them is ``RegularProd <= Capacity`` (which the
    expected-value solve respects). The material-balance constraint lets
    the free ``Inventory`` variable absorb any demand imbalance as
    penalized backorder, so aircond has relatively complete recourse:
    fixing any capacity-respecting production plan leaves every real
    scenario feasible. This is the multistage analogue of farmer's
    continuous-first-stage case -- the simplest one the convention has to
    handle, with the inter-stage coupling resolved by taking all node
    values from a single feasible solve of one tree.
    """
    if branching_factors is None:
        raise RuntimeError(
            "aircond feasible_xhat_creator needs branching_factors in kwargs"
        )
    proxy_kwargs = dict(scenario_creator_kwargs)
    proxy_kwargs["branching_factors"] = branching_factors
    # Expected-value proxy: deterministic demand at every node.
    proxy_kwargs["sigma_dev"] = 0.0
    proxy_kwargs["mu_dev"] = 0.0
    # _demands_creator requires start_seed; it is irrelevant at sigma_dev=0.
    proxy_kwargs.setdefault("start_seed", 0)
    num_scens = int(np.prod(branching_factors))
    snames = scenario_names_creator(num_scens)
    return ef_xhat_nonants(
        scenario_creator,
        snames,
        solver_name=solver_name,
        scenario_creator_kwargs=proxy_kwargs,
        solver_options=solver_options,
    )
