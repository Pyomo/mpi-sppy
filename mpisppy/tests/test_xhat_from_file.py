###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unit tests for ``XhatInnerBoundBase._try_file_xhat``.

Covers: off-by-default, missing file, multi-stage reject, length
mismatch, happy path (finite ``Eobj`` → ``update_if_improving``),
and infeasibility-style result (non-finite ``Eobj`` → no update, nonants
restored).

The tests stub out ``self.opt`` and the spoke's bound-update method so
that no MPI, solver, or full cylinder wiring is needed.
"""

import math
import os
import tempfile
import unittest

import numpy as np
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy.cylinders.xhatbase import XhatInnerBoundBase
from mpisppy.spbase import SPBase


def _binary_scenario(sname, **_):
    m = pyo.ConcreteModel(name=sname)
    m.x = pyo.Var([0, 1, 2], domain=pyo.Binary)
    m.fsc = pyo.Expression(expr=m.x[0] + m.x[1] + m.x[2])
    m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])
    sputils.attach_root_node(m, m.fsc, [m.x])
    m._mpisppy_probability = "uniform"
    return m


def _make_sp(num=2, options=None):
    opts = {"toc": False, "verbose": False}
    if options is not None:
        opts.update(options)
    return SPBase(
        options=opts,
        all_scenario_names=[f"scen{i}" for i in range(num)],
        scenario_creator=_binary_scenario,
    )


class _FakeXhatEval:
    """Just enough ``Xhat_Eval`` surface for ``_try_file_xhat``."""

    def __init__(self, sp, evaluate_result):
        self.local_scenarios = sp.local_scenarios
        self.options = sp.options
        self.multistage = sp.multistage
        self._evaluate_result = evaluate_result
        self.evaluate_calls = []
        self.restore_calls = 0

    def evaluate(self, nonant_cache):
        self.evaluate_calls.append(nonant_cache)
        return self._evaluate_result

    def _restore_nonants(self):
        self.restore_calls += 1


def _make_helper(sp, evaluate_result=None):
    """Build an ``XhatInnerBoundBase``-shaped object without going through
    its __init__ (which needs the whole spoke stack)."""
    h = XhatInnerBoundBase.__new__(XhatInnerBoundBase)
    h.opt = _FakeXhatEval(sp, evaluate_result)
    h.cylinder_rank = 0
    h.updates = []

    def _fake_update(eobj):
        h.updates.append(eobj)
        return True
    h.update_if_improving = _fake_update
    return h


def _write_npy(values):
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(path, np.array(values, dtype=float))
    return path


class TestFileXhatDisabled(unittest.TestCase):

    def test_off_by_default_is_noop(self):
        sp = _make_sp()
        helper = _make_helper(sp, evaluate_result=10.0)
        helper._try_file_xhat()
        self.assertEqual(helper.opt.evaluate_calls, [])
        self.assertEqual(helper.updates, [])

    def test_empty_string_path_is_noop(self):
        sp = _make_sp(options={"xhat_from_file": ""})
        helper = _make_helper(sp, evaluate_result=10.0)
        helper._try_file_xhat()
        self.assertEqual(helper.opt.evaluate_calls, [])


class TestFileXhatHardFails(unittest.TestCase):

    def test_missing_file_raises(self):
        sp = _make_sp(options={"xhat_from_file": "/tmp/does_not_exist_xhat.npy"})
        helper = _make_helper(sp, evaluate_result=10.0)
        with self.assertRaises(RuntimeError) as cm:
            helper._try_file_xhat()
        self.assertIn("does not exist", str(cm.exception))

    def test_length_mismatch_raises(self):
        # Problem has 3 binary nonants; file has 2.
        path = _write_npy([1.0, 0.0])
        try:
            sp = _make_sp(options={"xhat_from_file": path})
            helper = _make_helper(sp, evaluate_result=10.0)
            with self.assertRaises(RuntimeError) as cm:
                helper._try_file_xhat()
            msg = str(cm.exception)
            self.assertIn("length", msg)
            self.assertIn("2", msg)
            self.assertIn("3", msg)
        finally:
            os.remove(path)

    def test_multistage_raises(self):
        path = _write_npy([1.0, 0.0, 1.0])
        try:
            sp = _make_sp(options={"xhat_from_file": path})
            helper = _make_helper(sp, evaluate_result=10.0)
            helper.opt.multistage = True  # force the multi-stage path
            with self.assertRaises(RuntimeError) as cm:
                helper._try_file_xhat()
            self.assertIn("two-stage", str(cm.exception))
        finally:
            os.remove(path)


class TestFileXhatHappyPath(unittest.TestCase):

    def test_finite_eobj_updates_and_restores(self):
        path = _write_npy([1.0, 0.0, 1.0])
        try:
            sp = _make_sp(options={"xhat_from_file": path})
            helper = _make_helper(sp, evaluate_result=42.0)
            helper._try_file_xhat()
            # evaluate was called with {'ROOT': [1,0,1]}
            self.assertEqual(len(helper.opt.evaluate_calls), 1)
            nc = helper.opt.evaluate_calls[0]
            self.assertIn("ROOT", nc)
            np.testing.assert_array_equal(nc["ROOT"], [1.0, 0.0, 1.0])
            # update_if_improving received Eobj
            self.assertEqual(helper.updates, [42.0])
            # nonants were restored for the subsequent main loop
            self.assertEqual(helper.opt.restore_calls, 1)
        finally:
            os.remove(path)

    def test_infeasible_like_eobj_does_not_update_but_still_restores(self):
        """When evaluate reports a non-finite objective (some scenario
        was infeasible), the helper must NOT call update_if_improving —
        but it must still restore nonants."""
        path = _write_npy([1.0, 0.0, 1.0])
        try:
            sp = _make_sp(options={"xhat_from_file": path})
            helper = _make_helper(sp, evaluate_result=float("inf"))
            helper._try_file_xhat()
            self.assertEqual(len(helper.opt.evaluate_calls), 1)
            self.assertEqual(helper.updates, [])
            self.assertEqual(helper.opt.restore_calls, 1)
        finally:
            os.remove(path)

    def test_none_eobj_does_not_update(self):
        """Evaluate returning None is the 'no expected value' case —
        skip the update, still restore."""
        path = _write_npy([1.0, 0.0, 1.0])
        try:
            sp = _make_sp(options={"xhat_from_file": path})
            helper = _make_helper(sp, evaluate_result=None)
            helper._try_file_xhat()
            self.assertEqual(helper.updates, [])
            self.assertEqual(helper.opt.restore_calls, 1)
        finally:
            os.remove(path)


class TestMathIsfinite(unittest.TestCase):
    """Sanity check that math.isfinite is the predicate we intend:
    inf is not finite, large real is finite."""

    def test_finite_positive(self):
        self.assertTrue(math.isfinite(1e18))

    def test_infinity_not_finite(self):
        self.assertFalse(math.isfinite(float("inf")))

    def test_nan_not_finite(self):
        self.assertFalse(math.isfinite(float("nan")))


if __name__ == "__main__":
    unittest.main()
