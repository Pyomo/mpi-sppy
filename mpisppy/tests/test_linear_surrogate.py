###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for linear surrogate nonanticipative variables."""

from types import SimpleNamespace

import pyomo.environ as pyo
import pytest

from mpisppy.scenario_tree import ScenarioNode
from mpisppy.opt.ph import PH
from mpisppy.utils.linear_surrogate import (
    LinearSurrogateDefinition,
    prepare_linear_surrogate_data,
    set_derived_surrogate_rhos,
)


def _model():
    model = pyo.ConcreteModel(name="scenario")
    model.x = pyo.Var([1, 2], initialize={1: 2, 2: 3})
    model.y = pyo.Var(initialize=1)
    model.obj = pyo.Objective(expr=model.y)
    return model


def _node(model, expression, **definition_kwargs):
    definition = LinearSurrogateDefinition(
        "total", expression, **definition_kwargs
    )
    return ScenarioNode(
        "ROOT", 1.0, 1, model.obj, [model.x], model,
        linear_surrogate_definitions=[definition],
    )


def test_builds_auxiliary_var_constraint_and_metadata():
    model = _model()
    node = _node(model, model.x[1] + 2 * model.x[2])

    target = model._mpisppy_linear_surrogates.total.value
    assert node.nonant_vardata_list[-1] is target
    assert target in node.surrogate_vardatas
    assert len(node.linear_surrogate_data) == 1
    data = node.linear_surrogate_data[0]
    assert data.sources == (model.x[1], model.x[2])
    assert data.coefficients == (1.0, 2.0)

    target.set_value(8)
    assert pyo.value(
        model._mpisppy_linear_surrogates.total.definition.body
    ) == pytest.approx(0)


def test_modeler_fixed_source_is_folded_only_for_rho():
    model = _model()
    model.x[1].fix(2)
    node = _node(model, 4 * model.x[1] + 3 * model.x[2] + 1)

    data = node.linear_surrogate_data[0]
    assert model.x[1] in node.initially_fixed_nonant_vardatas
    assert data.sources == (model.x[2],)
    assert data.coefficients == (3.0,)
    # The generated model equation still contains the fixed source.
    target = data.target
    target.set_value(18)
    assert pyo.value(
        model._mpisppy_linear_surrogates.total.definition.body
    ) == pytest.approx(0)


def test_rejects_variable_outside_the_node():
    model = _model()
    with pytest.raises(ValueError, match="ordinary nonant on node 'ROOT'.*y"):
        _node(model, model.x[1] + model.y)


def test_rejects_nonlinear_expression():
    model = _model()
    with pytest.raises(ValueError, match="not linear"):
        _node(model, model.x[1] * model.x[2])


def test_rejects_mutable_parameter():
    model = _model()
    model.scale = pyo.Param(initialize=2, mutable=True)
    with pytest.raises(ValueError, match="mutable parameter"):
        _node(model, model.scale * model.x[1] + model.x[2])


def test_rejects_legacy_surrogate_as_source():
    model = _model()
    definition = LinearSurrogateDefinition("total", model.x[1] + model.y)
    with pytest.raises(ValueError, match="Invalid variable.*y"):
        ScenarioNode(
            "ROOT", 1.0, 1, model.obj, [model.x], model,
            surrogate_nonant_list=[model.y],
            linear_surrogate_definitions=[definition],
        )


def test_derived_rho_uses_median_and_squared_coefficients():
    model = _model()
    node = _node(
        model, model.x[1] + 2 * model.x[2], rho_multiplier=1.5
    )
    target = node.linear_surrogate_data[0].target
    indices = {
        ("ROOT", 0): model.x[1],
        ("ROOT", 1): model.x[2],
        ("ROOT", 2): target,
    }
    model._mpisppy_data = SimpleNamespace(
        nonant_indices=indices,
        linear_surrogate_data=node.linear_surrogate_data,
        prob_coeff={"ROOT": 1.0},
    )
    model._mpisppy_model = pyo.Block(concrete=True)
    model._mpisppy_model.rho = pyo.Param(
        indices.keys(), mutable=True,
        initialize={
            ("ROOT", 0): 4,
            ("ROOT", 1): 16,
            ("ROOT", 2): 999,
        },
    )
    prepare_linear_surrogate_data(model)
    ph = SimpleNamespace(local_scenarios={"s": model})

    set_derived_surrogate_rhos(ph)

    # median(4/1**2, 16/2**2) * 1.5 == 6
    assert pyo.value(model._mpisppy_model.rho["ROOT", 2]) == pytest.approx(6)


@pytest.mark.skipif(
    not pyo.SolverFactory("highs").available(False), reason="HiGHS unavailable"
)
def test_ph_applies_derived_rho_after_user_rho_setter():
    def scenario_creator(name):
        scenario_number = int(name[-1])
        model = pyo.ConcreteModel(name=name)
        model.x = pyo.Var([1, 2], bounds=(0, 10), initialize=1)
        model.cost = pyo.Expression(
            expr=(model.x[1] - scenario_number) ** 2 + model.x[2]
        )
        model.obj = pyo.Objective(expr=model.cost)
        definition = LinearSurrogateDefinition(
            "total", model.x[1] + 2 * model.x[2]
        )
        model._mpisppy_node_list = [ScenarioNode(
            "ROOT", 1.0, 1, model.cost, [model.x], model,
            linear_surrogate_definitions=[definition],
        )]
        model._mpisppy_probability = 0.5
        return model

    def rho_setter(model):
        return [
            (id(model.x[1]), 4),
            (id(model.x[2]), 16),
        ]

    options = {
        "solver_name": "highs",
        "PHIterLimit": 1,
        "defaultPHrho": 1,
        "convthresh": -1,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "solver_options_layers": [],
        "smoothed": 0,
        "asynchronousPH": False,
        "toc": False,
    }
    ph = PH(
        options, ["s1", "s2"], scenario_creator,
        rho_setter=rho_setter,
    )

    ph.PH_Prep()
    ph.Iter0()

    target_rhos = []
    for model in ph.local_scenarios.values():
        data = model._mpisppy_data.linear_surrogate_data[0]
        target_index = next(
            ndn_i for ndn_i, var in model._mpisppy_data.nonant_indices.items()
            if var is data.target
        )
        target_rhos.append(pyo.value(model._mpisppy_model.rho[target_index]))
    assert target_rhos == pytest.approx([4, 4])


def test_scenario_ingestion_rejects_different_coefficients():
    def scenario_creator(name):
        scenario_number = int(name[-1])
        model = pyo.ConcreteModel(name=name)
        model.x = pyo.Var(bounds=(0, 10))
        model.obj = pyo.Objective(expr=model.x)
        definition = LinearSurrogateDefinition(
            "total", scenario_number * model.x
        )
        model._mpisppy_node_list = [ScenarioNode(
            "ROOT", 1.0, 1, model.obj, [model.x], model,
            linear_surrogate_definitions=[definition],
        )]
        model._mpisppy_probability = 0.5
        return model

    options = {
        "solver_name": "highs",
        "PHIterLimit": 1,
        "defaultPHrho": 1,
        "convthresh": -1,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "solver_options_layers": [],
        "smoothed": 0,
        "asynchronousPH": False,
        "toc": False,
    }
    with pytest.raises(RuntimeError, match="definitions.*differ"):
        PH(options, ["s1", "s2"], scenario_creator)
