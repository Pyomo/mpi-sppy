from termination import set_termination_callback

from markshare2 import model 

from pyomo.environ import SolverFactory

import time

class _TestTermination:
    def __init__(self):
        self.model = model
        self._solver = SolverFactory(self._solver_name)
        self._solver.set_instance(model)

        self.time_start = time.time()

    def _get_solver_name(self):
        pass

    def solver(self):
        return self._solver

    def solver_terminate(self):
        t_now = time.time()
        if t_now - self.time_start > 5:
            return True
        else:
            return False

    def solve(self):
        set_termination_callback(self)
        self._set_time_limit()
        self._solver.solve(tee=True)

class TestCPLEXTermination(_TestTermination):
    _solver_name = 'cplex_persistent'
    def _set_time_limit(self):
        self._solver.options['timelimit'] = 20

class TestXpressTermination(_TestTermination):
    _solver_name = 'xpress_persistent'
    def _set_time_limit(self):
        self._solver.options['maxtime'] = 20

cplextest = TestCPLEXTermination()
try:
    cplextest.solve()
except ValueError:
    pass

xpresstest = TestXpressTermination()
xpresstest.solve()
