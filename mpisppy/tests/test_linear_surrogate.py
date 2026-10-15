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

import numpy as np
import pyomo.environ as pyo
import pytest

from mpisppy.scenario_tree import ScenarioNode
from mpisppy.opt.ph import PH
from mpisppy.spopt import SPOpt
from mpisppy.utils.rho_utils import assign_rho_with_fallback
from mpisppy.utils.linear_surrogate import (
    LinearSurrogateDefinition,
    finalize_linear_surrogates,
    prepare_linear_surrogate_data,
    set_derived_surrogate_rhos,
)


def test_lagrangian_prep_derives_rho_before_solver_creation(monkeypatch):
    import mpisppy.cylinders.lagrangian_bounder as lagrangian_bounder

    calls = []
    opt = SimpleNamespace(
        PH_Prep=lambda **kw: calls.append(("prep", kw)),
        _reenable_W=lambda: calls.append("enable W"),
        _create_solvers=lambda: calls.append("solvers"),
    )
    monkeypatch.setattr(
        lagrangian_bounder,
        "set_derived_surrogate_rhos",
        lambda seen_opt: calls.append(("derive", seen_opt)),
    )
    spoke = SimpleNamespace(opt=opt)

    lagrangian_bounder._LagrangianMixin.lagrangian_prep(spoke)

    assert calls == [
        ("prep", {"attach_prox": False, "defer_attach": False}),
        "enable W",
        ("derive", opt),
        "solvers",
    ]


def test_lagrangian_waiting_path_rederives_before_weight_update(monkeypatch):
    import mpisppy.cylinders.lagrangian_bounder as lagrangian_bounder

    calls = []
    opt = SimpleNamespace(
        options={"subgradient_while_waiting": True},
        Compute_Xbar=lambda verbose: calls.append("xbar"),
        Update_W=lambda verbose: calls.append("update W"),
    )
    monkeypatch.setattr(
        lagrangian_bounder,
        "set_derived_surrogate_rhos",
        lambda seen_opt: calls.append("derive"),
    )
    spoke = SimpleNamespace(
        opt=opt,
        verbose=False,
        lagrangian=lambda **kw: calls.append("solve"),
    )

    lagrangian_bounder.LagrangianOuterBound.do_while_waiting_for_new_Ws(
        spoke
    )

    assert calls == ["xbar", "derive", "update W", "solve"]


def test_lagranger_rederives_around_weight_update(monkeypatch):
    import mpisppy.cylinders.lagranger_bounder as lagranger_bounder

    calls = []
    extension = SimpleNamespace(miditer=lambda: calls.append("miditer"))
    opt = SimpleNamespace(
        extensions=object(),
        extobject=extension,
        options={"verbose": False},
        Compute_Xbar=lambda **kw: calls.append("xbar"),
        Update_W=lambda **kw: calls.append("update W"),
    )
    monkeypatch.setattr(
        lagranger_bounder,
        "set_derived_surrogate_rhos",
        lambda seen_opt: calls.append("derive"),
    )
    spoke = SimpleNamespace(
        opt=opt,
        localnonants=np.asarray([np.nan]),
        rho_rescale_factors={3: 2.0},
        _rescale_rho=lambda factor: calls.append(("rescale", factor)),
        _write_W_and_xbar=lambda iteration: calls.append("write"),
        _lagrangian=lambda iteration: calls.append("solve"),
    )

    lagranger_bounder.LagrangerOuterBound._update_weights_and_solve(spoke, 3)

    assert calls == [
        ("rescale", 2.0),
        "xbar",
        "derive",
        "update W",
        "miditer",
        "derive",
        "write",
        "solve",
    ]


def test_lagranger_applies_iteration_zero_rescale(monkeypatch):
    import mpisppy.cylinders.lagranger_bounder as lagranger_bounder

    calls = []
    opt = SimpleNamespace(
        extensions=None,
        current_solver_options={"stale": True},
    )
    monkeypatch.setattr(
        lagranger_bounder,
        "set_derived_surrogate_rhos",
        lambda seen_opt: calls.append("derive"),
    )
    spoke = SimpleNamespace(
        opt=opt,
        rho_rescale_factors={0: 0.5, 1: 0.1},
        lagrangian_prep=lambda: calls.append("prep"),
        _jensens_enabled=lambda: False,
        _rescale_rho=lambda factor: calls.append(("rescale", factor)),
        _lagrangian=lambda iteration: calls.append(("solve", iteration)),
        got_kill_signal=lambda: True,
    )

    lagranger_bounder.LagrangerOuterBound.main(spoke)

    assert calls == [
        "prep",
        ("rescale", 0.5),
        "derive",
        ("solve", 0),
    ]


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
    node = ScenarioNode(
        "ROOT", 1.0, 1, model.obj, [model.x], model,
        linear_surrogate_definitions=[definition],
    )
    model._mpisppy_node_list = [node]
    finalize_linear_surrogates(model)
    return node


def test_optimizer_fixed_baseline_uses_all_current_nonants():
    model = _model()
    model.y.fix(1)
    node = ScenarioNode(
        "ROOT", 1.0, 1, model.obj, [model.x], model,
        surrogate_nonant_list=[model.y],
    )
    # This fixing occurs after ScenarioNode captured the normalization-time
    # state, but before optimizer construction.
    model.x[1].fix(2)
    model._mpisppy_data = SimpleNamespace(
        nonant_indices={
            ("ROOT", i): var
            for i, var in enumerate(node.nonant_vardata_list)
        },
    )
    opt = SimpleNamespace(local_scenarios={"s": model})

    SPOpt._create_fixed_nonant_cache(opt)

    assert model.y in opt._initial_fixed_varibles
    assert model.x[1] in opt._initial_fixed_varibles
    assert SPOpt._can_update_best_bound(opt)

    model.x[2].fix(3)
    assert not SPOpt._can_update_best_bound(opt)


def test_builds_auxiliary_var_constraint_and_metadata():
    model = _model()
    node = _node(model, model.x[1] + 2 * model.x[2])

    target = model._mpisppy_model.linear_surrogate_vars["ROOT", "total"]
    assert node.nonant_vardata_list[-1] is target
    assert target in node.surrogate_vardatas
    assert len(node.linear_surrogate_data) == 1
    data = node.linear_surrogate_data[0]
    assert data.sources == (model.x[1], model.x[2])
    assert data.coefficients == (1.0, 2.0)

    target.set_value(8)
    assert pyo.value(
        model._mpisppy_model.linear_surrogate_constraints["ROOT", "total"].body
    ) == pytest.approx(0)


def test_signature_ignores_expression_term_order():
    first = _model()
    second = _model()

    first_node = _node(first, first.x[1] + 2 * first.x[2])
    second_node = _node(second, 2 * second.x[2] + second.x[1])

    assert (
        first_node.linear_surrogate_data[0].signature
        == second_node.linear_surrogate_data[0].signature
    )


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
        model._mpisppy_model.linear_surrogate_constraints["ROOT", "total"].body
    ) == pytest.approx(0)


def test_source_fixed_after_node_construction_is_folded_at_ingestion():
    model = _model()
    definition = LinearSurrogateDefinition(
        "total", model.x[1] + 3 * model.x[2]
    )
    node = ScenarioNode(
        "ROOT", 1.0, 1, model.obj, [model.x], model,
        linear_surrogate_definitions=[definition],
    )
    model.x[1].fix(2)
    model._mpisppy_node_list = [node]

    finalize_linear_surrogates(model)

    data = node.linear_surrogate_data[0]
    assert data.sources == (model.x[2],)
    assert data.coefficients == (3.0,)
    assert model.x[1] in node.initially_fixed_nonant_vardatas


def test_all_sources_fixed_after_node_construction_is_rejected():
    model = _model()
    definition = LinearSurrogateDefinition(
        "total", model.x[1] + model.x[2]
    )
    node = ScenarioNode(
        "ROOT", 1.0, 1, model.obj, [model.x], model,
        linear_surrogate_definitions=[definition],
    )
    model.x.fix(2)
    model._mpisppy_node_list = [node]

    with pytest.raises(ValueError, match="no unfixed source variables"):
        finalize_linear_surrogates(model)


def test_finalization_does_not_absorb_later_mpisppy_fixing():
    model = _model()
    definition = LinearSurrogateDefinition(
        "total", model.x[1] + 2 * model.x[2]
    )
    node = ScenarioNode(
        "ROOT", 1.0, 1, model.obj, [model.x], model,
        linear_surrogate_definitions=[definition],
    )
    model._mpisppy_node_list = [node]
    finalize_linear_surrogates(model)

    model.x[1].fix(2)
    finalize_linear_surrogates(model)

    data = node.linear_surrogate_data[0]
    assert data.sources == (model.x[1], model.x[2])
    assert model.x[1] not in node.initially_fixed_nonant_vardatas


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
    node = ScenarioNode(
        "ROOT", 1.0, 1, model.obj, [model.x], model,
        surrogate_nonant_list=[model.y],
        linear_surrogate_definitions=[definition],
    )
    model._mpisppy_node_list = [node]
    with pytest.raises(ValueError, match="Invalid variable.*y"):
        finalize_linear_surrogates(model)


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

    report_state = {}
    ph.cylinder_rank = 0
    assign_rho_with_fallback(
        ph, 1.0, "TestRho", report_state,
        lambda scenario, index: (
            0.0 if index == ("ROOT", 2)
            else scenario._mpisppy_model.rho[index]._value
        ),
    )
    # Primary setters leave the derived target alone and do not report its
    # intentionally zero objective coefficient as a fallback.
    assert pyo.value(model._mpisppy_model.rho["ROOT", 2]) == pytest.approx(999)
    assert report_state["TestRho"] == 0

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

    # FWPH swaps position-equivalent QP vardatas into each node and then
    # rebuilds nonant_indices. Cached MIP indices must survive that swap.
    for model in ph.local_scenarios.values():
        model.qp_nonants = pyo.Var(range(3))
        model._mpisppy_node_list[0].nonant_vardata_list = list(
            model.qp_nonants.values()
        )
    ph._attach_nonant_indices()


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
