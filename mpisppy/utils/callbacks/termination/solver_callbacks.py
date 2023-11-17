# This software is distributed under the 3-clause BSD License.


def set_cplex_callback(solver, termination_callback):
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


def set_gurobi_callback(solver, termination_callback):
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


def set_xpress_callback(solver, termination_callback):
    xpress_problem = solver._solver_model

    def cbchecktime_callback(xpress_problem, termination_callback):
        if termination_callback():
            return 1
        return 0

    xpress_problem.addcbchecktime(cbchecktime_callback, termination_callback, 0)
