###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Linear surrogate nonanticipative variables.

A linear surrogate is an auxiliary variable ``z`` satisfying ``z == expr``,
where every variable in ``expr`` is an ordinary nonanticipative variable at
one scenario-tree node.  Agreement of the ordinary nonants therefore implies
agreement of ``z``; explicitly hedging ``z`` adds a useful aggregate direction
without changing the stochastic program.
"""

from dataclasses import dataclass
import math

import numpy as np
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.expr.visitor import (
    identify_mutable_parameters,
    identify_variables,
)
from pyomo.repn.standard_repn import generate_standard_repn


@dataclass(frozen=True)
class LinearSurrogateDefinition:
    """Request an mpi-sppy-owned linear surrogate nonant.

    Args:
        name: Name used for the generated Pyomo block.
        expression: A Pyomo expression whose variables are ordinary nonants
            belonging to the same node.
        rho_multiplier: Strength applied after deriving rho from source rhos.
    """

    name: str
    expression: object
    rho_multiplier: float = 1.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                "LinearSurrogateDefinition.name must be a nonempty string."
            )
        if not math.isfinite(self.rho_multiplier) or self.rho_multiplier <= 0:
            raise ValueError("Linear surrogate rho_multiplier must be positive and finite.")


@dataclass
class _LinearSurrogateData:
    """Scenario-local normalized data used by PH."""

    name: str
    node_name: str
    target: object
    sources: tuple
    coefficients: tuple
    rho_multiplier: float
    signature: tuple
    target_index: tuple = None
    source_indices: tuple = ()
    inverse_coefficient_squares: object = None


def _ensure_mpisppy_components(model):
    if not hasattr(model, "_mpisppy_model"):
        model._mpisppy_model = pyo.Block(
            name="For mpi-sppy Pyomo additions to the scenario model"
        )
    block = model._mpisppy_model
    if not isinstance(block, pyo.Block):
        raise ValueError(
            "A model using LinearSurrogateDefinition must have a Pyomo Block "
            "at _mpisppy_model."
        )
    if not hasattr(block, "linear_surrogate_vars"):
        block.linear_surrogate_vars = pyo.Var(
            pyo.Any, domain=pyo.Reals, dense=False
        )
        block.linear_surrogate_constraints = pyo.Constraint(
            pyo.Any
        )
    return (
        block.linear_surrogate_vars,
        block.linear_surrogate_constraints,
    )


def build_linear_surrogates(node, model, definitions):
    """Validate and build the definitions attached to one ScenarioNode."""
    ordinary_nonants = ComponentSet(
        var for var in node.nonant_vardata_list
        if var not in node.surrogate_vardatas
    )
    initially_fixed = ComponentSet(v for v in ordinary_nonants if v.fixed)
    node.initially_fixed_nonant_vardatas = initially_fixed
    if definitions is None:
        return []

    surrogate_vars, surrogate_constraints = _ensure_mpisppy_components(model)
    answer = []

    for definition in definitions:
        if not isinstance(definition, LinearSurrogateDefinition):
            raise TypeError(
                "linear_surrogate_definitions entries must be "
                "LinearSurrogateDefinition objects."
            )
        key = (node.name, definition.name)
        if key in surrogate_vars:
            raise ValueError(
                f"Duplicate linear surrogate name '{definition.name}' on node "
                f"'{node.name}'."
            )

        mutable_params = list(identify_mutable_parameters(definition.expression))
        if mutable_params:
            names = ", ".join(param.name for param in mutable_params)
            raise ValueError(
                f"Linear surrogate '{definition.name}' contains mutable "
                f"parameter(s): {names}. Coefficients and constants must not "
                "change after scenario ingestion."
            )

        expression_vars = list(identify_variables(
            definition.expression, include_fixed=True
        ))
        if not expression_vars:
            raise ValueError(
                f"Linear surrogate '{definition.name}' contains no variables."
            )
        outside = [var.name for var in expression_vars if var not in ordinary_nonants]
        if outside:
            raise ValueError(
                f"Every variable in linear surrogate '{definition.name}' must "
                f"be an ordinary nonant on node '{node.name}'. Invalid "
                f"variable(s): {', '.join(outside)}."
            )

        # With modeler-fixed variables folded, the remaining expression must
        # be affine. This is deliberately done now, before mpi-sppy can fix
        # variables algorithmically.
        repn = generate_standard_repn(definition.expression, compute_values=True)
        if not repn.is_linear():
            raise ValueError(
                f"Linear surrogate '{definition.name}' is not linear after "
                "modeler-fixed variables are folded."
            )

        sources = tuple(repn.linear_vars or ())
        coefficients = tuple(float(pyo.value(c)) for c in (repn.linear_coefs or ()))
        if not sources:
            raise ValueError(
                f"Linear surrogate '{definition.name}' has no unfixed source "
                "variables after modeler-fixed variables are folded."
            )
        if any(not math.isfinite(c) or c == 0 for c in coefficients):
            raise ValueError(
                f"Linear surrogate '{definition.name}' has a zero or non-finite "
                "source coefficient."
            )

        target = surrogate_vars[key]
        surrogate_constraints[key] = target == definition.expression
        node.nonant_vardata_list.append(target)
        node.surrogate_vardatas.add(target)

        constant = float(pyo.value(repn.constant)) if repn.constant is not None else 0.0
        signature = (
            definition.name,
            tuple(sorted(
                (var.name, coef) for var, coef in zip(sources, coefficients)
            )),
            constant,
            definition.rho_multiplier,
        )
        answer.append(_LinearSurrogateData(
            name=definition.name,
            node_name=node.name,
            target=target,
            sources=sources,
            coefficients=coefficients,
            rho_multiplier=definition.rho_multiplier,
            signature=signature,
            inverse_coefficient_squares=np.reciprocal(
                np.square(np.asarray(coefficients, dtype="d"))
            ),
        ))

    return answer


def set_derived_surrogate_rhos(ph):
    """Set every generated surrogate rho from its ordinary source rhos.

    The median is robust to a heterogeneous group. The ``a**2`` scaling makes
    the result invariant to a source's units in ``z = sum(a*x) + b``.
    """
    for scenario in ph.local_scenarios.values():
        data_items = getattr(
            scenario._mpisppy_data, "linear_surrogate_data", ()
        )
        if not data_items:
            continue
        for data in data_items:
            source_rhos = np.fromiter(
                (
                    scenario._mpisppy_model.rho[index]._value
                    for index in data.source_indices
                ),
                dtype="d",
                count=len(data.source_indices),
            )
            rho = data.rho_multiplier * float(np.median(
                source_rhos * data.inverse_coefficient_squares
            ))
            if not math.isfinite(rho) or rho <= 0:
                raise ValueError(
                    f"Derived rho for linear surrogate '{data.name}' on "
                    f"scenario '{scenario.name}' is not positive and finite: {rho}."
                )
            scenario._mpisppy_model.rho[data.target_index] = rho


def prepare_linear_surrogate_data(scenario):
    """Resolve vardata references to nonant indices once at setup."""
    data_items = getattr(scenario._mpisppy_data, "linear_surrogate_data", ())
    if not data_items:
        scenario._mpisppy_data.linear_surrogate_target_indices = frozenset()
        return
    nonant_map = ComponentMap(
        (var, ndn_i)
        for ndn_i, var in scenario._mpisppy_data.nonant_indices.items()
    )
    for data in data_items:
        data.target_index = nonant_map[data.target]
        data.source_indices = tuple(nonant_map[source] for source in data.sources)
    scenario._mpisppy_data.linear_surrogate_target_indices = frozenset(
        data.target_index for data in data_items
    )


def remap_linear_surrogate_data(source_node, ref_vars):
    """Map scenario-local metadata onto a proper bundle's reference vars."""
    if not source_node.linear_surrogate_data:
        return []
    positions = ComponentMap(
        (var, i) for i, var in enumerate(source_node.nonant_vardata_list)
    )
    answer = []
    for data in source_node.linear_surrogate_data:
        target = ref_vars[(source_node.name, positions[data.target])]
        sources = tuple(
            ref_vars[(source_node.name, positions[source])]
            for source in data.sources
        )
        answer.append(_LinearSurrogateData(
            name=data.name,
            node_name=data.node_name,
            target=target,
            sources=sources,
            coefficients=data.coefficients,
            rho_multiplier=data.rho_multiplier,
            signature=data.signature,
            inverse_coefficient_squares=data.inverse_coefficient_squares.copy(),
        ))
    return answer
