###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################


import os

import pyomo.environ as pyo
from math import log10, floor

from mpisppy.utils import sputils

#: The directory holding the ``mpisppy`` package these tests belong to.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def subprocess_env():
    """Environment for a Python process a test starts, such as an mpiexec leg.

    A child started as ``python -m mpi4py <script>`` imports ``mpisppy`` from
    wherever the interpreter finds it, which with an editable install is the
    checkout it was installed from -- not necessarily the one under test, so
    a test run in a second worktree would exercise the first one's code.
    Putting this checkout first on ``PYTHONPATH`` makes the child import the
    same ``mpisppy`` as the test.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (REPO_ROOT if not existing
                         else REPO_ROOT + os.pathsep + existing)
    return env


def limit_solver_threads(solver, solver_name, threads=1):
    """Cap thread count on a directly-constructed Pyomo solver so test
    solves do not fan out across every core. Reuses the canonical->native
    option translator so we do not hardcode per-solver key names. Safe to
    call before or after set_instance for persistent solvers (thread
    options are applied at solve time)."""
    solver.options.update(
        sputils.translate_solver_options({"threads": threads}, solver_name))


def get_solver(persistent_OK=True):
    solvers = ["cplex","gurobi","xpress"]
    if persistent_OK:
        solvers = [n+e for e in ('_persistent', '') for n in solvers]
    
    for solver_name in solvers:
        try:
            solver_available = pyo.SolverFactory(solver_name).available()
        except Exception:
            solver_available = False
        if solver_available:
            break
    
    if '_persistent' in solver_name:
        persistent_solver_name = solver_name
    else:
        persistent_solver_name = solver_name+"_persistent"
    try:
        persistent_available = pyo.SolverFactory(persistent_solver_name).available()
    except Exception:
        persistent_available = False
    
    return solver_available, solver_name, persistent_available, persistent_solver_name

def round_pos_sig(x, sig=1):
    return round(x, sig-int(floor(log10(abs(x))))-1)


#: Termination conditions that mean the probe below actually solved. Compared
#: as lower-case strings because the legacy and APPSI interfaces return
#: different enums for the same outcome.
_PROBE_SOLVED = frozenset((
    "optimal", "globallyoptimal", "locallyoptimal", "feasible",
    "convergencecriteriasatisfied",
))


def _termination_condition(results):
    """Whatever this results object calls its termination condition."""
    solver = getattr(results, "solver", None)
    condition = getattr(solver, "termination_condition", None)
    if condition is None:
        # The pyomo.contrib.solver / APPSI results objects put it at the top.
        condition = getattr(results, "termination_condition", None)
    return condition


def solver_takes_model_size(solver_name, num_vars, num_cons):
    """Can `solver_name` solve a model with this many variables and constraints?

    Returns ``(solved, why)``. ``why`` is None when it solved, and otherwise
    says what happened in the solver's own words.

    The pip-installed community editions of cplex, gurobi and xpress -- which
    is what CI has -- cap model size, and they report the cap partway through
    a solve rather than up front, so the only way to ask is to hand one over.
    The probe is a trivial bounded LP of the requested size; it says nothing
    about difficulty, only about size.

    **A solve that came back is not a solve that worked.** Some solvers raise
    on a refusal and some report it through the results object -- so this
    judges the termination condition rather than the absence of an exception.
    Nor is every refusal about size: a solver that is not installed, whose
    licence has expired, or that ran out of memory also fails to solve this
    model, and a caller that reads a bare False as "too big" prints a reason
    that is not the one it found. That is what ``why`` is for.

    Threads are capped because the probe is incidental to whatever the caller
    is really doing. No time limit is set: a limit that expired on a slow
    machine would come back as a termination condition this reads as failure,
    and a size cap the caller would then never learn about.
    """
    model = pyo.ConcreteModel()
    model.varset = pyo.RangeSet(num_vars)
    model.x = pyo.Var(model.varset, bounds=(0, 1))
    model.conset = pyo.RangeSet(num_cons)
    model.c = pyo.Constraint(
        model.conset, rule=lambda m, j: m.x[(j - 1) % num_vars + 1] <= 1)
    model.obj = pyo.Objective(expr=sum(model.x[i] for i in model.varset))
    try:
        solver = pyo.SolverFactory(solver_name)
        try:
            limit_solver_threads(solver, solver_name)
        except Exception:
            # An interface whose options this cannot translate still gets to
            # answer the question being asked.
            pass
        if sputils.is_persistent(solver):
            # The legacy persistent interface refuses solve(model).
            solver.set_instance(model)
            results = solver.solve()
        else:
            results = solver.solve(model)
    except Exception as exc:
        return False, f"{solver_name} raised {type(exc).__name__}: {exc}"
    condition = _termination_condition(results)
    if str(condition).lower() not in _PROBE_SOLVED:
        return False, (f"{solver_name} returned from the solve reporting "
                       f"termination condition '{condition}'")
    return True, None
