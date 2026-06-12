###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# CVaR (Conditional Value-at-Risk) risk management via the Rockafellar-Uryasev
# linearization, applied as a per-scenario model transform.
#
# The risk-averse objective is
#
#     cvar_mean_weight * E[Cost]  +  cvar_weight * CVaR_alpha(Cost)
#
# (lambda * E[Cost] + beta * CVaR).  Because Sum_s p_s = 1 and the Value-at-Risk
# variable eta is a single first-stage (non-anticipative) variable, this whole
# measure distributes over scenarios:
#
#     lambda*E[Cost] + beta*CVaR = Sum_s p_s * [ lambda*Cost_s + beta*eta
#                                                + beta/(1-alpha)*delta_s ]
#
# so it is captured exactly by adding eta to the root node, a scenario-local
# delta_s >= (Cost_s - eta), and replacing each scenario objective.  eta is then
# "just another first-stage variable," so EF, PH/APH, Lagrangian, and xhat all
# inherit risk aversion with no algorithm changes.  See doc/designs/cvar_design.md.

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils


def add_cvar(scenario, *, cvar_weight, cvar_alpha, cvar_mean_weight=1.0):
    """Mutate one scenario model in place to add Rockafellar-Uryasev CVaR terms.

    The original (risk-neutral) objective is deactivated -- but left on the model
    so it remains available for separate E[Cost] reporting -- and a new ACTIVE
    objective named ``WITH_CVAR`` is added:

        cvar_mean_weight*Cost + cvar_weight*eta + cvar_weight/(1-cvar_alpha)*delta

    This is safe because every mpi-sppy objective lookup filters on active=True,
    so only ``WITH_CVAR`` is ever used.

    Components added to ``scenario``:
        ``_mpisppy_cvar_eta``        the VaR eta (free; appended to the root node
                                     nonant lists, so it is non-anticipative)
        ``_mpisppy_cvar_excess``     the excess delta_s (>= 0)
        ``_mpisppy_cvar_excess_con`` the constraint delta_s >= Cost_s - eta
        ``WITH_CVAR``                the new active risk-averse objective

    Args:
        scenario (Pyomo ConcreteModel): a single scenario instance, already
            annotated with ``_mpisppy_node_list`` and an active objective.
        cvar_weight (float): beta >= 0, the weight on CVaR.
        cvar_alpha (float): the confidence level alpha, 0 < alpha < 1.
        cvar_mean_weight (float): lambda >= 0, the weight on E[Cost] (default
            1.0).  Pure CVaR is ``cvar_mean_weight=0, cvar_weight=1``.
    """
    if not (0.0 < cvar_alpha < 1.0):
        raise ValueError(f"cvar_alpha must satisfy 0 < alpha < 1 (got {cvar_alpha})")
    if cvar_weight < 0.0:
        raise ValueError(f"cvar_weight (beta) must be >= 0 (got {cvar_weight})")
    if cvar_mean_weight < 0.0:
        raise ValueError(
            f"cvar_mean_weight (lambda) must be >= 0 (got {cvar_mean_weight})")

    obj = sputils.find_active_objective(scenario)   # pristine user cost objective
    cost = obj.expr
    sense = obj.sense                               # minimize; maximize is TBD

    scenario._mpisppy_cvar_eta = pyo.Var()
    scenario._mpisppy_cvar_excess = pyo.Var(domain=pyo.NonNegativeReals)
    scenario._mpisppy_cvar_excess_con = pyo.Constraint(
        expr=scenario._mpisppy_cvar_excess >= cost - scenario._mpisppy_cvar_eta)

    # Keep the original risk-neutral objective (deactivated) for E[Cost]
    # reporting; the new active objective is named WITH_CVAR (not PySP's MASTER).
    obj.deactivate()
    scenario.WITH_CVAR = pyo.Objective(
        expr=cvar_mean_weight * cost
        + cvar_weight * scenario._mpisppy_cvar_eta
        + (cvar_weight / (1.0 - cvar_alpha)) * scenario._mpisppy_cvar_excess,
        sense=sense)

    # eta is a single root-node first-stage var, so append it to BOTH the root
    # node's nonant_list and its (pre-built in ScenarioNode.__init__) vardata
    # list.  That is sufficient for both decomposition and the EF NAC builder.
    root = scenario._mpisppy_node_list[0]
    root.nonant_list.append(scenario._mpisppy_cvar_eta)
    root.nonant_vardata_list.append(scenario._mpisppy_cvar_eta)


def cvar_scenario_creator(scenario_creator, *, cvar_weight, cvar_alpha,
                          cvar_mean_weight=1.0):
    """Wrap a scenario_creator so every scenario it builds gets the CVaR transform.

    The wrapper is a pure per-scenario transform (no MPI/rank bookkeeping), so it
    works unchanged for the EF solve and for every cylinder.

    Args:
        scenario_creator (callable): the underlying ``scenario_creator(sname,
            **kwargs)`` that builds a risk-neutral scenario.
        cvar_weight (float): beta, passed to :func:`add_cvar`.
        cvar_alpha (float): alpha, passed to :func:`add_cvar`.
        cvar_mean_weight (float): lambda, passed to :func:`add_cvar`.

    Returns:
        callable: a scenario_creator with the same signature that returns
        scenarios carrying the risk-averse ``WITH_CVAR`` objective.
    """
    def wrapped(sname, **kwargs):
        scenario = scenario_creator(sname, **kwargs)
        add_cvar(scenario, cvar_weight=cvar_weight, cvar_alpha=cvar_alpha,
                 cvar_mean_weight=cvar_mean_weight)
        return scenario
    return wrapped
