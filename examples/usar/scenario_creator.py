###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Provides a function which creates `ConcreteModel`\s for scenarios."""
from typing import Dict, Sequence

import pyomo.environ as pyo

from abstract import abstract_model
import mpisppy.scenario_tree
import mpisppy.utils.sputils


def scenario_creator(
    name: str,
    data_dicts: Sequence[Dict],
    **assigned_to_model,
) -> pyo.ConcreteModel:
    """Prepares a particular scenario for optimization with mpi-sppy.

    Assigns any extra keyword arguments as attributes of the model.

    Args:
        name: str of the scenario's int index in `data_dicts`, e.g. "0".
        data_dicts:
            ``data_dicts[int(name)]`` is the scenario's Pyomo data dict.
        **assigned_to_model: Assigned as attributes of the scenario.

    Returns:
        A scenario instantiating the USAR `AbstractModel`.
    """
    abstract = abstract_model()

    data_dict = data_dicts[int(name)]
    concrete = abstract.create_instance(data_dict)

    for key, val in assigned_to_model.items():
        setattr(concrete, key, val)

    mpisppy.utils.sputils.attach_root_node(
        concrete, concrete.first_stage_cost, [concrete.is_active_depot])

    return concrete
