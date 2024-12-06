###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# NOTES: The design of the generic termination callback below is strictly MIP-centric.
#        Specifically, we assume the following 3 attributes can be extracted, and
#        are (mostly) defined in the context of MIPs. If we want to expand to LPs
#        and other non-branch-and-bound contexts, additional work is required.

def check_user_termination_callback_signature(user_termination_callback):

    import inspect
    return len(inspect.signature(user_termination_callback).parameters) == 3

def set_cplex_callback(solver, user_termination_callback):
    
    cplex = solver._cplex
    cplex_model = solver._solver_model

    class Termination(
        cplex.callbacks.MIPInfoCallback,
    ):
        _tc = user_termination_callback

        def __call__(self):
            runtime = self.get_time() - self.get_start_time()
            obj_best = self.get_incumbent_objective_value()
            obj_bound = self.get_best_objective_value()
            if self._tc(runtime, obj_best, obj_bound):
                self.abort()
                return

    cplex_model.register_callback(Termination)


def set_gurobi_callback(solver, user_termination_callback):
    
    gurobi_model = solver._solver_model
    gurobi_model._terminate_function = user_termination_callback

    # TBD - best placement? For speeed...
    from gurobipy import GRB

    def gurobi_callback(gurobi_model, where):
        if where == GRB.Callback.MIP:
            runtime = gurobi_model.cbGet(GRB.Callback.RUNTIME)
            obj_best = gurobi_model.cbGet(GRB.Callback.MIP_OBJBST)
            obj_bound = gurobi_model.cbGet(GRB.Callback.MIP_OBJBND)
            if gurobi_model._terminate_function(runtime, obj_best, obj_bound):
                gurobi_model.terminate()

    # This overwrites GurobiPersistent's
    # existing callback. gurobipy callbacks
    # are set by gurobi_model.solve, so we
    # need to let Pyomo do this.
    solver._callback = gurobi_callback


def set_xpress_callback(solver, user_termination_callback):
    
    xpress_problem = solver._solver_model

    def cbchecktime_callback(xpress_problem, termination_callback):
        runtime = xpress_problem.attributes.time
        obj_best = xpress_problem.attributes.mipbestobjval
        obj_bound = xpress_problem.attributes.bestbound
        if termination_callback(runtime, obj_best, obj_bound):
            return 1
        return 0

    # per the Xpress documentation, this callback is invoked every time the Optimizer
    # checks if the time limit has been reached. So broader than what is presently
    # needed for our present MIP-based use cases.
    xpress_problem.addcbchecktime(cbchecktime_callback, user_termination_callback, 0)
