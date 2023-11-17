from mpisppy.utils.callbacks.termination import (
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

    def solver_terminate(self):
        t_now = time.time()
        if t_now - self.time_start > 2:
            return True
        else:
            return False

    def solve(self):
        assert supports_termination_callback(self._solver)
        set_termination_callback(self._solver, self.solver_terminate)
        self._set_time_limit()
        self._solver.solve(tee=True)


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


def test_cplex_termination_callback():
    st = time.time()
    cplextest = CPLEXTermination()
    try:
        cplextest.solve()
    except ValueError:
        pass
    end = time.time()
    assert end - st < 5


def test_gurobi_termination_callback():
    st = time.time()
    gurobitest = GurobiTermination()
    gurobitest.solve()
    end = time.time()
    assert end - st < 5


def test_xpress_termination_callback():
    st = time.time()
    xpresstest = XpressTermination()
    xpresstest.solve()
    end = time.time()
    assert end - st < 5


def test_unsupported():
    cbc = SolverFactory("cbc")

    assert not supports_termination_callback("xpress_persistent")
    assert not supports_termination_callback(cbc)
