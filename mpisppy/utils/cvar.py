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

import math

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.opt import TerminationCondition

import mpisppy.utils.sputils as sputils
from mpisppy import MPI


def add_cvar(scenario, *, cvar_weight, cvar_alpha, cvar_mean_weight=1.0,
             eta_lb=None, eta_ub=None, auto_eta_bound=True):
    """Mutate one scenario model in place to add Rockafellar-Uryasev CVaR terms.

    The original (risk-neutral) objective is deactivated -- but left on the model
    so it remains available for separate E[Cost] reporting -- and a new ACTIVE
    objective named ``WITH_CVAR`` is added:

        cvar_mean_weight*Cost + cvar_weight*eta + cvar_weight/(1-cvar_alpha)*delta

    This is safe because every mpi-sppy objective lookup filters on active=True,
    so only ``WITH_CVAR`` is ever used.

    Both minimization (CVaR of the upper/cost tail) and maximization (CVaR of the
    lower/reward tail) objectives are supported; the excess variable's domain and
    the excess constraint's direction flip with the sense, but the new objective
    expression is identical and keeps the original sense.

    eta is otherwise a free variable.  Its optimum is the Value-at-Risk
    ``VaR_alpha(Cost)``, which is provably within the range of ``Cost`` over all
    scenarios, so ``[global min cost, global max cost]`` is a valid (non-cutting)
    bound.  Only this one scenario is visible here, so this function computes
    this scenario's cost range with FBBT and stashes it in
    ``_mpisppy_cvar_eta_cost_bounds``; the cross-scenario reduction and the
    actual bounding happen later in :func:`set_cvar_eta_bounds` (SPBase calls it
    once its scenarios and the ROOT comm exist).  A user-supplied ``eta_lb`` /
    ``eta_ub`` is applied to eta immediately and overrides the auto bound on that
    side.

    Components added to ``scenario``:
        ``_mpisppy_cvar_eta``        the VaR eta (appended to the root node
                                     nonant lists, so it is non-anticipative)
        ``_mpisppy_cvar_excess``     the excess delta_s (>= 0 for minimize,
                                     <= 0 for maximize)
        ``_mpisppy_cvar_excess_con`` the constraint delta_s >= Cost_s - eta
                                     (minimize) / delta_s <= Cost_s - eta (maximize)
        ``WITH_CVAR``                the new active risk-averse objective
        ``_mpisppy_cvar_eta_cost_bounds`` this scenario's ``(cost_lb, cost_ub)``
                                     (from FBBT; +/-inf for an unbounded side),
                                     consumed by :func:`set_cvar_eta_bounds`

    Args:
        scenario (Pyomo ConcreteModel): a single scenario instance, already
            annotated with ``_mpisppy_node_list`` and an active objective.
        cvar_weight (float): beta >= 0, the weight on CVaR.
        cvar_alpha (float): the confidence level alpha, 0 < alpha < 1.
        cvar_mean_weight (float): lambda >= 0, the weight on E[Cost] (default
            1.0).  Pure CVaR is ``cvar_mean_weight=0, cvar_weight=1``.
        eta_lb (float or None): if given, a hard lower bound for eta that
            overrides the automatic lower bound (default None).
        eta_ub (float or None): if given, a hard upper bound for eta that
            overrides the automatic upper bound (default None).
        auto_eta_bound (bool): if True (default), compute this scenario's cost
            range with FBBT so :func:`set_cvar_eta_bounds` can bound eta from the
            global cost range; if False, skip it and leave eta free except for
            any explicit ``eta_lb`` / ``eta_ub``.
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
    sense = obj.sense

    # Rockafellar-Uryasev linearization.  Minimization penalizes the upper (cost)
    # tail:  delta_s >= Cost_s - eta,  delta_s >= 0.  Maximization is risk-averse
    # on the lower (reward) tail, so it mirrors:  delta_s <= Cost_s - eta,
    # delta_s <= 0.  In BOTH cases the per-scenario objective is the same
    # expression (below), carried with the original sense; only the excess
    # variable's domain and the excess constraint's direction flip.
    scenario._mpisppy_cvar_eta = pyo.Var()
    if sense == pyo.minimize:
        scenario._mpisppy_cvar_excess = pyo.Var(domain=pyo.NonNegativeReals)
        scenario._mpisppy_cvar_excess_con = pyo.Constraint(
            expr=scenario._mpisppy_cvar_excess >= cost - scenario._mpisppy_cvar_eta)
    else:
        scenario._mpisppy_cvar_excess = pyo.Var(domain=pyo.NonPositiveReals)
        scenario._mpisppy_cvar_excess_con = pyo.Constraint(
            expr=scenario._mpisppy_cvar_excess <= cost - scenario._mpisppy_cvar_eta)

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

    # A user-supplied bound is a hard override and is applied to eta now; the
    # matching side is then left alone by set_cvar_eta_bounds.
    if eta_lb is not None:
        scenario._mpisppy_cvar_eta.setlb(eta_lb)
    if eta_ub is not None:
        scenario._mpisppy_cvar_eta.setub(eta_ub)

    # Stash this scenario's cost range for the later global reduction.  An
    # unbounded side (FBBT returns None) is recorded as +/-inf so that taking the
    # min of lower bounds / max of upper bounds across scenarios correctly leaves
    # eta free on any side where some scenario's cost tail is unbounded.
    if auto_eta_bound:
        cost_lb, cost_ub = compute_bounds_on_expr(cost)
    else:
        cost_lb, cost_ub = None, None
    scenario._mpisppy_cvar_eta_cost_bounds = (
        cost_lb if (cost_lb is not None and math.isfinite(cost_lb)) else -math.inf,
        cost_ub if (cost_ub is not None and math.isfinite(cost_ub)) else math.inf,
    )


def cvar_scenario_creator(scenario_creator, *, cvar_weight, cvar_alpha,
                          cvar_mean_weight=1.0, eta_lb=None, eta_ub=None,
                          auto_eta_bound=True):
    """Wrap a scenario_creator so every scenario it builds gets the CVaR transform.

    The wrapper is a pure per-scenario transform (no MPI/rank bookkeeping), so it
    works unchanged for the EF solve and for every cylinder.  eta's global bound
    is not set here (it needs all scenarios); SPBase does that automatically via
    :func:`set_cvar_eta_bounds` once each opt object is constructed.

    Args:
        scenario_creator (callable): the underlying ``scenario_creator(sname,
            **kwargs)`` that builds a risk-neutral scenario.
        cvar_weight (float): beta, passed to :func:`add_cvar`.
        cvar_alpha (float): alpha, passed to :func:`add_cvar`.
        cvar_mean_weight (float): lambda, passed to :func:`add_cvar`.
        eta_lb (float or None): explicit eta lower bound, passed to
            :func:`add_cvar` (default None).
        eta_ub (float or None): explicit eta upper bound, passed to
            :func:`add_cvar` (default None).
        auto_eta_bound (bool): whether to auto-bound eta, passed to
            :func:`add_cvar` (default True).

    Returns:
        callable: a scenario_creator with the same signature that returns
        scenarios carrying the risk-averse ``WITH_CVAR`` objective.
    """
    def wrapped(sname, **kwargs):
        scenario = scenario_creator(sname, **kwargs)
        add_cvar(scenario, cvar_weight=cvar_weight, cvar_alpha=cvar_alpha,
                 cvar_mean_weight=cvar_mean_weight, eta_lb=eta_lb, eta_ub=eta_ub,
                 auto_eta_bound=auto_eta_bound)
        return scenario
    return wrapped


def set_cvar_eta_bounds(local_scenarios, comm):
    """Give the CVaR Value-at-Risk variable eta a valid global bound.

    eta's optimum is ``VaR_alpha(Cost)``, which is guaranteed to lie within the
    range of ``Cost`` across *all* scenarios.  :func:`add_cvar` stashed each
    scenario's own cost range in ``_mpisppy_cvar_eta_cost_bounds``; this function
    reduces those to the global ``[min cost lb, max cost ub]`` over ``comm`` and
    sets eta's bounds from it.  Only sides the user did not already fix (eta.lb /
    eta.ub still None) and that came back finite are set, so an explicit
    ``eta_lb`` / ``eta_ub`` always wins and an unbounded cost tail leaves eta free
    on that side.  A per-scenario-local bound would be invalid -- in the EF the
    eta copies are tied by non-anticipativity, so their bounds intersect, and one
    scenario whose cost range excludes the global VaR would cut off the optimum.

    Every rank in ``comm`` must call this (it performs collective reductions).
    ``comm`` must span all scenarios; ``self.comms["ROOT"]`` is the right one,
    since eta is a root-node variable shared by every scenario.

    Args:
        local_scenarios (dict): name -> scenario model for this rank (typically
            an opt object's ``local_scenarios``).
        comm: an MPI communicator spanning every scenario.
    """
    if not local_scenarios:
        return
    first = next(iter(local_scenarios.values()))
    if not hasattr(first, "_mpisppy_cvar_eta"):
        return  # not a CVaR run; nothing to do

    local_lb = math.inf
    local_ub = -math.inf
    for s in local_scenarios.values():
        cost_lb, cost_ub = s._mpisppy_cvar_eta_cost_bounds
        local_lb = min(local_lb, cost_lb)
        local_ub = max(local_ub, cost_ub)

    global_lb = np.zeros(1, dtype='d')
    global_ub = np.zeros(1, dtype='d')
    comm.Allreduce([np.array([local_lb], dtype='d'), MPI.DOUBLE],
                   [global_lb, MPI.DOUBLE], op=MPI.MIN)
    comm.Allreduce([np.array([local_ub], dtype='d'), MPI.DOUBLE],
                   [global_ub, MPI.DOUBLE], op=MPI.MAX)
    global_lb = float(global_lb[0])
    global_ub = float(global_ub[0])

    for s in local_scenarios.values():
        eta = s._mpisppy_cvar_eta
        if eta.lb is None and math.isfinite(global_lb):
            eta.setlb(global_lb)
        if eta.ub is None and math.isfinite(global_ub):
            eta.setub(global_ub)


def _solve_dual_bound(solver, solver_name, model, maximize):
    """Solve ``model`` and return a valid *dual* (best) bound on its objective.

    For a maximize this is an UPPER bound on the objective (``>=`` the true
    optimum); for a minimize a LOWER bound (``<=`` the true optimum).  Because it
    is the dual bound it is valid at **any** mip gap -- a loose gap only makes it
    looser, never invalid -- unlike the incumbent, which sits on the wrong side
    of the optimum and could cut.  Pyomo brackets the optimum as
    ``lower_bound <= opt <= upper_bound``, so the dual bound is ``upper_bound``
    for a maximize and ``lower_bound`` for a minimize.  Returns None when the
    objective is unbounded (or no finite dual bound is reported), leaving that
    side of eta free.

    A persistent solver is re-``set_instance``'d each call so it picks up the
    current model (whose objective sense and integrality may have changed since
    the previous solve).
    """
    if "_persistent" in solver_name:
        solver.set_instance(model)
        results = solver.solve(load_solutions=False)
    else:
        results = solver.solve(model, load_solutions=False)
    tc = results.solver.termination_condition
    if tc in (TerminationCondition.unbounded,
              TerminationCondition.infeasibleOrUnbounded):
        return None
    bound = results.problem.upper_bound if maximize \
        else results.problem.lower_bound
    if bound is None or not math.isfinite(bound):
        return None
    return float(bound)


def compute_cvar_eta_bounds_by_solve(scenario_names, scenario_creator, solver_name,
                                     scenario_creator_kwargs=None, comm=None,
                                     mipgap=0.5):
    """Compute a valid global bound on the CVaR VaR variable eta by solving.

    eta must be enclosed: ``lb <= min cost`` and ``ub <= ... <= max cost``, so
    the box contains the VaR and never cuts the optimum.  The two ends are found
    differently, because they need different things:

    * The **"how cheap/good" (slack) side** -- the min cost for a minimize, the
      max reward for a maximize -- only needs a valid enclosing value, so a
      **relaxation** (LP) is enough: the LP optimum is ``<= min cost`` (resp.
      ``>= max reward``), and this side sits far from the VaR so its looseness is
      harmless.
    * The **worst-case (tail) side** -- the max cost for a minimize, the min
      reward for a maximize -- cannot be gotten from a relaxation: to bound how
      bad things can get you must solve the actual **MIP**.  We only need a valid
      bound though, not the exact worst case, so the MIP is run with a **loose
      ``mipgap``** and we read its **dual bound** (valid at any gap; see
      :func:`_solve_dual_bound`).

    A side that is genuinely unbounded comes back as None, leaving eta free
    there.  Scenarios are distributed round-robin across ``comm`` (default
    ``COMM_WORLD``), so **every rank must call this** -- it ends in an Allreduce.

    Args:
        scenario_names (list): all scenario names.
        scenario_creator (callable): the risk-neutral ``scenario_creator`` (the
            one *before* the CVaR wrap), whose active objective is the pure cost.
        solver_name (str): a Pyomo solver name (persistent variants are handled).
        scenario_creator_kwargs (dict): kwargs for ``scenario_creator``.
        comm: MPI communicator to distribute over (default ``COMM_WORLD``).
        mipgap (float): the (loose) relative mip gap for the worst-case MIP
            solves; only the dual bound is used, so a loose gap keeps it cheap.

    Returns:
        tuple: ``(lb, ub)`` as floats, or ``None`` on a side that is unbounded.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if scenario_creator_kwargs is None:
        scenario_creator_kwargs = dict()
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    solver = pyo.SolverFactory(solver_name)
    try:
        available = solver.available(exception_flag=False)
    except Exception:
        available = False
    if not available:
        raise RuntimeError(
            f"solver '{solver_name}' is not available for the CVaR eta "
            "bound solves (--cvar-eta-bound-method solve)")
    # loose gap (we only read the dual bound) and single-threaded so distributed
    # bound solves do not oversubscribe; native option names per solver
    solver.options.update(sputils.translate_solver_options(
        {"mipgap": mipgap, "threads": 1}, solver_name))
    relax = pyo.TransformationFactory("core.relax_integer_vars")

    local_lb = math.inf
    local_ub = -math.inf
    for i, sname in enumerate(scenario_names):
        if i % nranks != rank:
            continue
        model = scenario_creator(sname, **scenario_creator_kwargs)
        obj = sputils.find_active_objective(model)
        cost = obj.expr
        minimizing = (obj.sense == pyo.minimize)
        # throwaway objective we can point at min or max of the cost
        obj.deactivate()
        model._mpisppy_cvar_bound_obj = pyo.Objective(expr=cost, sense=pyo.minimize)

        # Worst-case (tail) side FIRST, as a MIP (integrality intact): the max
        # cost for a minimize, the min reward for a maximize.  Read the dual
        # bound so a loose mipgap stays valid.
        model._mpisppy_cvar_bound_obj.sense = pyo.maximize if minimizing \
            else pyo.minimize
        tail = _solve_dual_bound(solver, solver_name, model, maximize=minimizing)

        # "How cheap/good" (slack) side: a relaxation is enough, so relax
        # integrality and solve the opposite direction (its dual bound is the LP
        # optimum, which encloses the true value).
        relax.apply_to(model)
        model._mpisppy_cvar_bound_obj.sense = pyo.minimize if minimizing \
            else pyo.maximize
        slack = _solve_dual_bound(solver, solver_name, model,
                                  maximize=(not minimizing))

        if minimizing:
            cost_min, cost_max = slack, tail   # slack = min (LP), tail = max (MIP)
        else:
            cost_min, cost_max = tail, slack   # tail = min (MIP), slack = max (LP)
        local_lb = min(local_lb, cost_min) if cost_min is not None else -math.inf
        local_ub = max(local_ub, cost_max) if cost_max is not None else math.inf

    global_lb = np.zeros(1, dtype='d')
    global_ub = np.zeros(1, dtype='d')
    comm.Allreduce([np.array([local_lb], dtype='d'), MPI.DOUBLE],
                   [global_lb, MPI.DOUBLE], op=MPI.MIN)
    comm.Allreduce([np.array([local_ub], dtype='d'), MPI.DOUBLE],
                   [global_ub, MPI.DOUBLE], op=MPI.MAX)
    lb = float(global_lb[0])
    ub = float(global_ub[0])
    return (lb if math.isfinite(lb) else None,
            ub if math.isfinite(ub) else None)
