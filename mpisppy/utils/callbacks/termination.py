# This software is distributed under the 3-clause BSD License.
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.solvers.plugins.solvers.xpress_persistent import XpressPersistent 
from pyomo.solvers.plugins.solvers.cplex_persistent import CPLEXPersistent 

def set_termination_callback(ob):
    '''
    Sets a termination callback. Object ob needs
    two methods: solver, which returns a Pyomo
    PersistentSolver, and solver_terminate, which
    returns True if the optimization solver should 
    terminate.

    ob.solver_terminate will be a relatively hot function,
    so it should not do complex operations as this
    will bog down the optimization solver.

    Caller should know if ob.solver() is persistent
    '''
    solver = ob.solver()

    if isinstance(solver, CPLEXPersistent):
        _set_cplex_callback(ob, solver)
    elif isinstance(solver, GurobiPersistent):
        _set_gurobi_callback(ob, solver)
    elif isinstance(solver, XpressPersistent):
        _set_xpress_callback(ob, solver)
    else:
        raise RuntimeError(f"solver {solver.__class__.__name___} termination callback "
                            "is not currently supported.")

def _set_cplex_callback(ob, solver):
    cplex = solver._cplex
    cplex_model = solver._solver_model

    class Termination(cplex.callbacks.MIPInfoCallback,
                      cplex.callbacks.ContinuousCallback,
                      cplex.callbacks.CrossoverCallback):
        _ob = ob
        def __call__(self):
            if self._ob.solver_terminate():
                self.abort()
                return

    cplex_model.register_callback(Termination)

def _set_gurobi_callback(ob, solver):
    gurobi_model = solver._solver_model
    gurobi_model._terminate_function = ob.solver_terminate

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

def _set_xpress_callback(ob, solver):
    xpress_problem = solver._solver_model

    def cbchecktime_callback(xpress_problem, ob):
        if ob.solver_terminate():
            return 1
        return 0

    xpress_problem.addcbchecktime(cbchecktime_callback, ob, 0)
