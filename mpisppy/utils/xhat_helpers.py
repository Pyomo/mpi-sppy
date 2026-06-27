###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Helpers for producing candidate xhat values from a deterministic proxy
of a stochastic program.

Used by ``feasible_xhat_creator`` implementations in
``examples/<model>/<model>_auxiliary.py`` and by downstream consumers
(e.g. findW's pin-dual algorithm) that need a first-stage point feasible
to fix in every real scenario's per-scenario subproblem.

The Jensen's xhat path inside mpi-sppy tolerates a per-scenario-infeasible
candidate by silently skipping it; downstream consumers that pin a
candidate and solve a per-scenario MIP cannot, so they need a stronger
contract. ``feasible_xhat_creator`` is the entry point for that
contract; ``average_xhat_nonants`` is the common-case engine that the
implementations call when they have an ``average_scenario_creator``.

The rounding/repair rule that turns the average-scenario solution into
a feasible candidate is model-specific (depends on monotonicity of
recourse feasibility in each first-stage variable) and lives in each
module's ``feasible_xhat_creator``, not here.
"""

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import mpisppy.utils.sputils as sputils


def _solve_model(model, solver_name, solver_options):
    """Solve ``model`` with the named solver; raise unless the
    termination condition is optimal or feasible. Returns the results."""
    solver = pyo.SolverFactory(solver_name)
    for k, v in (solver_options or {}).items():
        solver.options[k] = v
    if sputils.is_persistent(solver):
        solver.set_instance(model)
        results = solver.solve(tee=False)
    else:
        results = solver.solve(model, tee=False)
    tc = results.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        raise RuntimeError(
            f"solve in xhat_helpers failed with termination_condition={tc}"
        )
    return results


def _solve_and_extract_root(model, solver_name, solver_options):
    _solve_model(model, solver_name, solver_options)
    root = model._mpisppy_node_list[0]
    return np.array(
        [pyo.value(v) for v in root.nonant_vardata_list], dtype="d"
    )


def _check_two_stage(model, who):
    if not hasattr(model, "_mpisppy_node_list"):
        raise RuntimeError(
            f"{who}: scenario must have _mpisppy_node_list attached "
            "(via sputils.attach_root_node)."
        )
    if len(model._mpisppy_node_list) != 1:
        raise RuntimeError(
            f"{who} is two-stage only; got "
            f"{len(model._mpisppy_node_list)} tree nodes."
        )


def average_xhat_nonants(
    average_scenario_creator,
    *,
    solver_name,
    scenario_creator_kwargs=None,
    solver_options=None,
    relax_integrality=False,
    scenario_name="AverageScenario",
):
    """Build, optionally LP-relax, and solve the average scenario; return
    its ROOT first-stage values as a 1-D ``np.ndarray`` in
    ``_mpisppy_node_list[0].nonant_vardata_list`` order.

    Two-stage only. Returns the bare array; callers that need the
    ``Xhat_Eval._fix_nonants`` cache form wrap as ``{"ROOT": arr}``.

    Note: solving the (LP-relaxed) average scenario is **not** the same
    as the per-scenario LP-xbar. The average-scenario solution can
    underestimate which first-stage variables need to be active when the
    candidate must be feasible across every real scenario, because data
    averaging can mask scenario-specific demand patterns. For a
    feasibility-oriented candidate, prefer ``lp_xbar_nonants``.
    """
    kwargs = scenario_creator_kwargs or {}
    avg = average_scenario_creator(scenario_name, **kwargs)
    _check_two_stage(avg, "average_xhat_nonants")
    if relax_integrality:
        pyo.TransformationFactory("core.relax_integer_vars").apply_to(avg)
    return _solve_and_extract_root(avg, solver_name, solver_options)


def lp_xbar_nonants(
    scenario_creator,
    scenario_names,
    *,
    solver_name,
    scenario_creator_kwargs=None,
    solver_options=None,
):
    """For each scenario, solve its LP relaxation; return the
    probability-weighted average of ROOT nonants as a 1-D ``np.ndarray``.

    Two-stage only. Each scenario's ``_mpisppy_probability`` is used as
    its weight; the literal string ``"uniform"`` is interpreted as
    ``1/len(scenario_names)``. Weights are renormalized at the end so a
    sub-list of scenarios produces a coherent average.

    This is the per-scenario LP-xbar that downstream consumers
    (e.g. findW's pin-dual tests) round to obtain a candidate
    first-stage feasible in every real scenario's per-scenario
    subproblem. For models where opening more first-stage activity
    only loosens recourse, ceiling the LP-xbar opens every variable
    that any scenario's LP wanted positive, which makes the candidate
    a strict cover.
    """
    kwargs = scenario_creator_kwargs or {}
    # Materialize once so a generator works and we can call len() / iterate.
    scenario_names = list(scenario_names)
    n = len(scenario_names)
    if n == 0:
        raise ValueError(
            "lp_xbar_nonants: scenario_names is empty; need at least "
            "one scenario to average."
        )
    arr_sum = None
    weight_sum = 0.0
    for sname in scenario_names:
        m = scenario_creator(sname, **kwargs)
        _check_two_stage(m, "lp_xbar_nonants")
        prob = getattr(m, "_mpisppy_probability", "uniform")
        if prob == "uniform":
            prob = 1.0 / n
        pyo.TransformationFactory("core.relax_integer_vars").apply_to(m)
        vals = _solve_and_extract_root(m, solver_name, solver_options)
        contribution = prob * vals
        arr_sum = contribution if arr_sum is None else arr_sum + contribution
        weight_sum += prob
    if weight_sum <= 0:
        raise ValueError(
            "lp_xbar_nonants: scenario probabilities sum to "
            f"{weight_sum}; cannot form a weighted average."
        )
    return arr_sum / weight_sum


def ef_xhat_nonants(
    scenario_creator,
    scenario_names,
    *,
    solver_name,
    scenario_creator_kwargs=None,
    solver_options=None,
    relax_integrality=False,
):
    """Solve one extensive form over ``scenario_names`` and return the
    whole nonant tree as ``{node_name: np.ndarray}`` -- the cache form
    consumed by ``Xhat_Eval._fix_nonants``.

    This is the multistage engine for ``feasible_xhat_creator``. Unlike
    ``average_xhat_nonants`` (which returns only the ROOT array), it
    returns a candidate at *every* non-leaf node, each array in that
    node's ``nonant_vardata_list`` order.

    The supplied scenario set should define a deterministic proxy whose
    tree has the same node structure as the real problem -- e.g. the
    expected-value tree: the real branching factors with the random data
    pinned to its mean -- so every real non-leaf node has a counterpart
    here. Because all node values come from one feasible solution of one
    EF, they are jointly feasible along every path *by construction*;
    this is the inter-stage coupling the two-stage case does not have.
    Whether they remain feasible to *fix* in the real (stochastic)
    scenarios is then a property of the model (relatively complete
    recourse) or the responsibility of a model-specific repair -- see
    doc/src/feasible_xhat.rst.

    Two-stage is the degenerate special case (only ``"ROOT"`` is
    populated), so this works there too; ``average_xhat_nonants`` remains
    the lighter two-stage engine.
    """
    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs or {},
    )
    if relax_integrality:
        pyo.TransformationFactory("core.relax_integer_vars").apply_to(ef)
    _solve_model(ef, solver_name, solver_options)
    # ef.ref_vars maps (node_name, i) -> the representative nonant Var,
    # where i is the index into that node's nonant_vardata_list (see
    # sputils.create_EF / sputils.ef_nonants). Group by node and order by
    # i so each array lines up with _fix_nonants' per-node loop.
    by_node = {}
    for (ndn, i), var in ef.ref_vars.items():
        by_node.setdefault(ndn, {})[i] = pyo.value(var)
    return {
        ndn: np.array([vals[i] for i in range(len(vals))], dtype="d")
        for ndn, vals in by_node.items()
    }
