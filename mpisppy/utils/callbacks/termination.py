# This software is distributed under the 3-clause BSD License.
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.solvers.plugins.solvers.xpress_persistent import XpressPersistent
from pyomo.solvers.plugins.solvers.cplex_persistent import CPLEXPersistent


def set_termination_callback(solver, termination_callback):
    """
    Sets a termination callback on the solver object.

    Parameters
    ----------
    solver : pyomo.solver.plugins.solvers.persistent_solver.PersistentSolver
        Pyomo PersistentSolver. Must be one of CPLEXPersistent, GurobiPersistent,
        or XpressPersistent. The solver object is modified with the callback.
    termination_callback : callable
        A callable which takes no arguments. Returns True if the solver should stop
        and False otherwise. Assume to be a "hot" function within the MIP solver
        and so should not have expensive operations.
    """

    if isinstance(solver, CPLEXPersistent):
        _set_cplex_callback(solver, termination_callback)
    elif isinstance(solver, GurobiPersistent):
        _set_gurobi_callback(solver, termination_callback)
    elif isinstance(solver, XpressPersistent):
        _set_xpress_callback(solver, termination_callback)
    else:
        raise RuntimeError(
            f"solver {solver.__class__.__name___} termination callback "
            "is not currently supported."
        )


def _set_cplex_callback(solver, termination_callback):
    cplex = solver._cplex
    cplex_model = solver._solver_model

    class Termination(
        cplex.callbacks.MIPInfoCallback,
        cplex.callbacks.ContinuousCallback,
        cplex.callbacks.CrossoverCallback,
    ):
        _tc = termination_callback

        def __call__(self):
            if self._tc():
                self.abort()
                return

    cplex_model.register_callback(Termination)


def _set_gurobi_callback(solver, termination_callback):
    gurobi_model = solver._solver_model
    gurobi_model._terminate_function = termination_callback

    def termination_callback(gurobi_model, where):
        # 0 == GRB.Callback.POLLING
        if where == 0:
            if gurobi_model._terminate_function():
                gurobi_model.terminate()

    # This overwrites GurobiPersistent's
    # existing callback. gurobipy callbacks
    # are set by gurobi_model.solve, so we
    # need to let Pyomo do this.
    solver._callback = termination_callback


def _set_xpress_callback(solver, termination_callback):
    xpress_problem = solver._solver_model

    def cbchecktime_callback(xpress_problem, termination_callback):
        if termination_callback():
            return 1
        return 0

    xpress_problem.addcbchecktime(cbchecktime_callback, termination_callback, 0)
