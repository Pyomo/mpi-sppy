###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
from pyomo.solvers.plugins.solvers.cplex_persistent import CPLEXPersistent
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.solvers.plugins.solvers.xpress_persistent import XpressPersistent

import mpisppy.utils.callbacks.termination.solver_callbacks as tc

# these are mpisppy callbacks, defined in the tc module imported immediately above.

_termination_callback_solvers_to_setters = {
    CPLEXPersistent: tc.set_cplex_callback,
    GurobiPersistent: tc.set_gurobi_callback,
    XpressPersistent: tc.set_xpress_callback,
}

def supports_termination_callback(solver_instance):
    """
    Determines if this module supports a solver instance

    Parameters
    ----------
    solver_instance : An instance of a Pyomo solver

    Returns
    -------
    bool : True if this module can set a termination callback on the solver instance

    """
    return isinstance(
        solver_instance, tuple(_termination_callback_solvers_to_setters.keys())
    )


def set_termination_callback(solver_instance, termination_callback):
    """
    Sets a termination callback on the solver object.

    Parameters
    ----------
    solver_instance : pyomo.solver.plugins.solvers.persistent_solver.PersistentSolver
        Pyomo PersistentSolver. Must be one of CPLEXPersistent, GurobiPersistent,
        or XpressPersistent. The solver object is modified with the callback.
    termination_callback : callable
        A callable which takes exactly three position arguments: run-time, best
        incumbent objective, and best objective bound. Returns True if the solver
        should stop and False otherwise. Assume to be a "hot" function within the
        MIP solver and so should not have expensive operations.
    """

    if not tc.check_user_termination_callback_signature(termination_callback):
        raise RuntimeError(
            "Provided user termination callback did not match expected signature with 3 positional arguments"
            )

    try:
        _termination_callback_solvers_to_setters[solver_instance.__class__](
            solver_instance, termination_callback
        )
    except KeyError:
        raise RuntimeError(
            f"solver {solver_instance.__class__.__name___} termination callback "
            "is not currently supported."
        )
