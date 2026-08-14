###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import pytest

import mpisppy.utils.callbacks.termination.termination_callbacks as termination_callbacks_module
from mpisppy.utils.callbacks.termination.termination_callbacks import (
    set_termination_callback,
    supports_termination_callback,
)
from mpisppy.utils.callbacks.termination.tests.markshare2 import model
from pyomo.environ import SolverFactory

import time

class _TestTermination:
    def __init__(self):
        self.model = model
        self._solver = SolverFactory(self._solver_name)
        self._solver.set_instance(model)

        self.time_start = time.time()

    def solver_terminate(self, runtime, best_obj, best_bound):
        if runtime > 2:
            return True
        else:
            return False

    def solve(self):
        assert supports_termination_callback(self._solver)
        set_termination_callback(self._solver, self.solver_terminate)
        self._set_time_limit()
        self._solver.solve(tee=True)

    def solve_without_loading(self):
        assert supports_termination_callback(self._solver)
        set_termination_callback(self._solver, self.solver_terminate)
        self._set_time_limit()
        return self._solver.solve(tee=False, load_solutions=False)


class CPLEXTermination(_TestTermination):
    
    _solver_name = "cplex_persistent"

    def _set_time_limit(self):
        self._solver.options["timelimit"] = 20

class GurobiTermination(_TestTermination):
    
    _solver_name = "gurobi_persistent"

    def _set_time_limit(self):
        self._solver.options["timelimit"] = 20


class XpressTermination(_TestTermination):
    
    _solver_name = "xpress_persistent"

    def _set_time_limit(self):
        self._solver.options["maxtime"] = 20


@pytest.mark.skipif(
    not SolverFactory("cplex_persistent").available(exception_flag=False),
    reason="cplex_persistent not available",
)
def test_cplex_termination_callback():
    
    st = time.time()
    cplextest = CPLEXTermination()
    cplextest.solve()
    end = time.time()
    assert end - st < 5

    
@pytest.mark.skipif(
    not SolverFactory("gurobi_persistent").available(exception_flag=False),
    reason="gurobi_persistent not available",
)
def test_gurobi_termination_callback():
    
    st = time.time()
    gurobitest = GurobiTermination()
    gurobitest.solve()
    end = time.time()
    assert end - st < 5


@pytest.mark.skipif(
    not SolverFactory("gurobi_persistent").available(exception_flag=False),
    reason="gurobi_persistent not available",
)
def test_gurobi_termination_callback_status():

    gurobitest = GurobiTermination()
    results = gurobitest.solve_without_loading()
    assert str(results.solver.status) == "aborted"
    assert str(results.solver.termination_condition) == "userInterrupt"
    assert str(results.solution[0].status) == "stoppedByLimit"

    
@pytest.mark.skipif(
    not SolverFactory("xpress_persistent").available(exception_flag=False),
    reason="xpress_persistent not available",
)
def test_xpress_termination_callback():
    
    st = time.time()
    xpresstest = XpressTermination()
    xpresstest.solve()
    end = time.time()
    assert end - st < 5


@pytest.mark.skipif(
    not SolverFactory("xpress_persistent").available(exception_flag=False),
    reason="xpress_persistent not available",
)
def test_xpress_termination_callback_status():

    xpresstest = XpressTermination()
    results = xpresstest.solve_without_loading()
    assert str(results.solver.status) == "warning"
    assert str(results.solver.termination_condition) == "other"
    assert str(results.solution[0].status) == "feasible"

    
def test_unsupported():
    
    cbc = SolverFactory("cbc")

    assert not supports_termination_callback("xpress_persistent")
    assert not supports_termination_callback(cbc)


def test_subclass_dispatch(monkeypatch):

    calls = []

    class BaseSolver:
        pass

    class ChildSolver(BaseSolver):
        pass

    monkeypatch.setattr(
        termination_callbacks_module,
        "_termination_callback_solvers_to_setters",
        {BaseSolver: lambda solver, cb: calls.append((solver, cb))},
    )

    solver = ChildSolver()

    def termination_callback(runtime, best_obj, best_bound):
        return False

    assert supports_termination_callback(solver)
    set_termination_callback(solver, termination_callback)
    assert calls == [(solver, termination_callback)]


def test_unsupported_error_message(monkeypatch):

    class UnsupportedSolver:
        pass

    monkeypatch.setattr(
        termination_callbacks_module,
        "_termination_callback_solvers_to_setters",
        {},
    )

    def termination_callback(runtime, best_obj, best_bound):
        return False

    with pytest.raises(
        RuntimeError,
        match="solver UnsupportedSolver termination callback is not currently supported",
    ):
        set_termination_callback(UnsupportedSolver(), termination_callback)
