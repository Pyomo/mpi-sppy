###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2026, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import time

import pytest
from pyomo.environ import SolverFactory

import mpisppy.extensions.timed_mipgap as timed_mipgap_module
from mpisppy.extensions.timed_mipgap import TimedMIPGapCB
from mpisppy.utils.callbacks.termination.tests.markshare2 import model as markshare2_model


class _MockSolver:
    def __init__(self, has_instance=True):
        self._has_instance = has_instance

    def has_instance(self):
        return self._has_instance


class _MockScenario:
    def __init__(self, solver_plugin):
        self._solver_plugin = solver_plugin


class _MockPH:
    def __init__(self, timecurve, solver_plugin):
        self.options = {"timed_mipgap": {"timecurve": timecurve}}
        self.local_scenarios = {"Scenario1": _MockScenario(solver_plugin)}


def _make_ph(timecurve="0.02:100  0.05:200", solver_plugin=None):
    if solver_plugin is None:
        solver_plugin = _MockSolver()
    return _MockPH(timecurve, solver_plugin)


class _TimedMIPGapIntegration:
    _solver_name = None
    _time_limit_option = None

    def __init__(self):
        self.model = markshare2_model.clone()
        self._solver = SolverFactory(self._solver_name)
        self._solver.set_instance(self.model)
        self._ph = _make_ph(timecurve="1e20:2", solver_plugin=self._solver)
        self._ext = TimedMIPGapCB(self._ph)

    def _set_time_limit(self):
        self._solver.options[self._time_limit_option] = 20

    def solve(self):
        self._ext.iter0_post_solver_creation()
        self._set_time_limit()
        return self._solver.solve(tee=False, load_solutions=False)


class _TimedMIPGapCPLEX(_TimedMIPGapIntegration):
    _solver_name = "cplex_persistent"
    _time_limit_option = "timelimit"


class _TimedMIPGapGurobi(_TimedMIPGapIntegration):
    _solver_name = "gurobi_persistent"
    _time_limit_option = "timelimit"


class _TimedMIPGapXpress(_TimedMIPGapIntegration):
    _solver_name = "xpress_persistent"
    _time_limit_option = "maxtime"


def test_timecurve_parsing_tolerates_whitespace():
    ext = TimedMIPGapCB(_make_ph(timecurve="0.02:100   0.05:200"))

    assert ext.timecurve == {0.02: 100.0, 0.05: 200.0}


def test_timecurve_parsing_rejects_bad_entries():
    with pytest.raises(RuntimeError, match='format "gap:time"'):
        TimedMIPGapCB(_make_ph(timecurve="0.02:100 badentry"))


def test_timecurve_parsing_rejects_duplicate_gaps():
    with pytest.raises(RuntimeError, match="duplicate gap entry"):
        TimedMIPGapCB(_make_ph(timecurve="0.02:100 0.02:200"))


def test_timecurve_parsing_rejects_nonmonotone_curve():
    with pytest.raises(RuntimeError, match="strictly increasing"):
        TimedMIPGapCB(_make_ph(timecurve="0.02:100 0.01:200"))

    with pytest.raises(RuntimeError, match="strictly increasing"):
        TimedMIPGapCB(_make_ph(timecurve="0.02:100 0.05:100"))


def test_timecurve_parsing_rejects_empty_curve():
    with pytest.raises(RuntimeError, match="must not be empty"):
        TimedMIPGapCB(_make_ph(timecurve="   "))


def test_timecurve_parsing_rejects_nonfinite_entries():
    with pytest.raises(RuntimeError, match="finite gap and time values"):
        TimedMIPGapCB(_make_ph(timecurve="inf:2"))

    with pytest.raises(RuntimeError, match="finite gap and time values"):
        TimedMIPGapCB(_make_ph(timecurve="0.1:nan"))


def test_compute_relative_gap_guards_missing_and_zero_values():
    assert TimedMIPGapCB._compute_relative_gap(None, 1.0) is None
    assert TimedMIPGapCB._compute_relative_gap(1.0, None) is None
    assert TimedMIPGapCB._compute_relative_gap(float("inf"), 1.0) is None
    assert TimedMIPGapCB._compute_relative_gap(1.0, float("nan")) is None
    assert TimedMIPGapCB._compute_relative_gap(0.0, 0.0) == pytest.approx(0.0)
    assert TimedMIPGapCB._compute_relative_gap(0.0, 2.0) == pytest.approx(1.0)


def test_should_terminate_uses_runtime_and_relative_gap():
    ext = TimedMIPGapCB(_make_ph(timecurve="0.02:100 0.05:200"))

    assert not ext._should_terminate(50.0, 100.0, 99.0)
    assert ext._should_terminate(101.0, 100.0, 99.0)
    assert not ext._should_terminate(101.0, 100.0, 97.0)
    assert ext._should_terminate(201.0, 100.0, 96.0)
    assert not ext._should_terminate(101.0, None, 99.0)
    assert not ext._should_terminate(float("inf"), 100.0, 99.0)
    assert not ext._should_terminate(101.0, float("inf"), 99.0)
    assert ext._should_terminate(101.0, 0.0, 0.0)


def test_iter0_post_solver_creation_registers_generic_callback(monkeypatch):
    calls = []
    solver_plugin = _MockSolver()
    ext = TimedMIPGapCB(_make_ph(solver_plugin=solver_plugin))

    monkeypatch.setattr(timed_mipgap_module.sputils, "is_persistent", lambda solver: True)
    monkeypatch.setattr(timed_mipgap_module, "supports_termination_callback", lambda solver: True)
    monkeypatch.setattr(
        timed_mipgap_module,
        "set_termination_callback",
        lambda solver, cb: calls.append((solver, cb)),
    )

    ext.iter0_post_solver_creation()

    assert len(calls) == 1
    assert calls[0][0] is solver_plugin
    callback = calls[0][1]
    assert callback(101.0, 100.0, 99.0)
    assert not callback(101.0, 100.0, 97.0)
    assert not callback(101.0, None, 99.0)


@pytest.mark.skipif(
    not SolverFactory("cplex_persistent").available(exception_flag=False),
    reason="cplex_persistent not available",
)
def test_timed_mipgap_cplex_integration():
    st = time.time()
    test = _TimedMIPGapCPLEX()
    results = test.solve()
    end = time.time()

    assert end - st < 10
    assert str(results.solver.status) == "aborted"
    assert str(results.solver.termination_condition) == "userInterrupt"
    assert str(results.solution[0].status) == "stoppedByLimit"


@pytest.mark.skipif(
    not SolverFactory("gurobi_persistent").available(exception_flag=False),
    reason="gurobi_persistent not available",
)
def test_timed_mipgap_gurobi_integration():
    st = time.time()
    test = _TimedMIPGapGurobi()
    results = test.solve()
    end = time.time()

    assert end - st < 10
    assert str(results.solver.status) == "aborted"
    assert str(results.solver.termination_condition) == "userInterrupt"
    assert str(results.solution[0].status) == "stoppedByLimit"


@pytest.mark.skipif(
    not SolverFactory("xpress_persistent").available(exception_flag=False),
    reason="xpress_persistent not available",
)
def test_timed_mipgap_xpress_integration():
    st = time.time()
    test = _TimedMIPGapXpress()
    results = test.solve()
    end = time.time()

    assert end - st < 10
    assert str(results.solver.status) == "warning"
    assert str(results.solver.termination_condition) == "other"
    assert str(results.solution[0].status) == "feasible"
