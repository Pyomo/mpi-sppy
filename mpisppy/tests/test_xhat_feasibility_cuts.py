###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unit tests for the xhat feasibility-cut plumbing.

Covers:
- Binary-only startup check in ``XhatFeasibilityCutExtension.setup_hub``
  (positive + negative cases).
- Cut installation via the hub extension: basic, zero-count no-op,
  all-zero-row skipped, and bundle-safe install against a proper bundle.
- The no-good-cut encoding produced by ``XhatBase._maybe_emit_feasibility_cut``
  exercised through a fake ``spcomm`` — verifies both the coefficient
  math and the trailing count slot.
- Extension ``__init__`` rejects being attached when
  ``cfg.xhat_feasibility_cuts_count`` is zero.

End-to-end PH+spoke runs are deliberately out of scope for this file;
those require MPI and the full cylinder machinery.
"""

import unittest

import numpy as np
import pyomo.environ as pyo

import mpisppy.extensions.xhatbase as xhatbase_extension
import mpisppy.scenario_tree as stree
import mpisppy.utils.sputils as sputils
from mpisppy.cylinders.spwindow import Field
from mpisppy.extensions.xhat_feasibility_cut_extension import (
    XhatFeasibilityCutExtension,
)
from mpisppy.spbase import SPBase


# ---------------------------------------------------------------------------
# Minimal scenario creators (pure Pyomo, no solver required).
# ---------------------------------------------------------------------------


def _binary_two_stage_scenario(sname, **kwargs):
    m = pyo.ConcreteModel(name=sname)
    m.x = pyo.Var([0, 1, 2], domain=pyo.Binary)
    m.fsc = pyo.Expression(expr=m.x[0] + m.x[1] + m.x[2])
    m.obj = pyo.Objective(expr=2 * m.x[0] + 3 * m.x[1] + m.x[2])
    sputils.attach_root_node(m, m.fsc, [m.x])
    m._mpisppy_probability = "uniform"
    return m


def _continuous_two_stage_scenario(sname, **kwargs):
    m = pyo.ConcreteModel(name=sname)
    m.x = pyo.Var([0, 1, 2], bounds=(0, 1))  # continuous 0..1
    m.fsc = pyo.Expression(expr=m.x[0] + m.x[1] + m.x[2])
    m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])
    sputils.attach_root_node(m, m.fsc, [m.x])
    m._mpisppy_probability = "uniform"
    return m


def _integer_not_binary_scenario(sname, **kwargs):
    """Integer vars with bounds (0, 5) — not treatable as binary."""
    m = pyo.ConcreteModel(name=sname)
    m.x = pyo.Var([0, 1], domain=pyo.Integers, bounds=(0, 5))
    m.fsc = pyo.Expression(expr=m.x[0] + m.x[1])
    m.obj = pyo.Objective(expr=m.x[0] + m.x[1])
    sputils.attach_root_node(m, m.fsc, [m.x])
    m._mpisppy_probability = "uniform"
    return m


def _integer_0_1_bounded_scenario(sname, **kwargs):
    """Integer vars with bounds (0, 1) — the extension must accept these."""
    m = pyo.ConcreteModel(name=sname)
    m.x = pyo.Var([0, 1, 2], domain=pyo.Integers, bounds=(0, 1))
    m.fsc = pyo.Expression(expr=m.x[0] + m.x[1] + m.x[2])
    m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])
    sputils.attach_root_node(m, m.fsc, [m.x])
    m._mpisppy_probability = "uniform"
    return m


def _make_sp(creator, num=2, cuts_count=3):
    """Build a tiny SPBase around one of the scenario creators above."""
    return SPBase(
        options={
            "toc": False,
            "verbose": False,
            "xhat_feasibility_cuts_count": cuts_count,
        },
        all_scenario_names=[f"scen{i}" for i in range(num)],
        scenario_creator=creator,
    )


class _FakeSolverPlugin:
    """Stands in for a Pyomo SolverFactory instance.

    ``sputils.is_persistent`` checks ``isinstance(p, PersistentSolver)``;
    this stub is neither, so the extension's persistent-solver branch
    is correctly skipped.
    """
    pass


def _attach_fake_solver_plugin(sp):
    for s in sp.local_scenarios.values():
        s._solver_plugin = _FakeSolverPlugin()


class _Opt:
    """Minimal ``opt`` surface that ``XhatFeasibilityCutExtension`` uses."""

    def __init__(self, sp, spcomm=None):
        self.local_scenarios = sp.local_scenarios
        self.options = sp.options
        self.spcomm = spcomm
        self.multistage = sp.multistage


# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------


class TestStartupCheck(unittest.TestCase):

    def test_passes_on_binary_nonants(self):
        sp = _make_sp(_binary_two_stage_scenario)
        ext = XhatFeasibilityCutExtension(_Opt(sp))
        ext.setup_hub()  # must not raise
        # ConstraintList should be attached on every scenario.
        for s in sp.local_scenarios.values():
            self.assertTrue(hasattr(s._mpisppy_model, "xhat_feasibility_cuts"))

    def test_passes_on_integer_0_1_bounded(self):
        """Integer vars with bounds (0, 1) are semantically binary."""
        sp = _make_sp(_integer_0_1_bounded_scenario)
        ext = XhatFeasibilityCutExtension(_Opt(sp))
        ext.setup_hub()  # must not raise

    def test_raises_on_continuous_nonant(self):
        sp = _make_sp(_continuous_two_stage_scenario)
        ext = XhatFeasibilityCutExtension(_Opt(sp))
        with self.assertRaises(RuntimeError) as cm:
            ext.setup_hub()
        msg = str(cm.exception)
        self.assertIn("binary", msg)
        self.assertIn("xhat-feasibility-cuts-count", msg)

    def test_raises_on_integer_not_0_1(self):
        sp = _make_sp(_integer_not_binary_scenario)
        ext = XhatFeasibilityCutExtension(_Opt(sp))
        with self.assertRaises(RuntimeError):
            ext.setup_hub()

    def test_init_rejects_zero_cuts_count(self):
        sp = _make_sp(_binary_two_stage_scenario, cuts_count=0)
        with self.assertRaises(RuntimeError) as cm:
            XhatFeasibilityCutExtension(_Opt(sp))
        self.assertIn("xhat_feasibility_cuts_count", str(cm.exception))


# ---------------------------------------------------------------------------
# Cut installation
# ---------------------------------------------------------------------------


class TestCutInstallation(unittest.TestCase):

    def _build(self, num=2, cuts_count=3):
        sp = _make_sp(_binary_two_stage_scenario, num=num, cuts_count=cuts_count)
        _attach_fake_solver_plugin(sp)
        ext = XhatFeasibilityCutExtension(_Opt(sp))
        ext.setup_hub()
        return sp, ext

    def _row_len(self, sp):
        any_s = next(iter(sp.local_scenarios.values()))
        return 1 + len(any_s._mpisppy_data.nonant_indices)

    def _buf_for(self, sp, cuts_count=3):
        return np.zeros(cuts_count * self._row_len(sp) + 1)

    def test_installs_one_cut_on_every_scenario(self):
        sp, ext = self._build(num=3)
        buf = self._buf_for(sp)
        # Install a single cut: x[0] + x[1] + x[2] >= 1, packed as
        # rhs_constant=-1, coefs=[1, 1, 1].
        buf[0] = -1.0
        buf[1:4] = [1.0, 1.0, 1.0]
        buf[-1] = 1.0  # count
        ext._install_cuts(buf)
        for s in sp.local_scenarios.values():
            self.assertEqual(len(s._mpisppy_model.xhat_feasibility_cuts), 1)

    def test_zero_count_is_noop(self):
        sp, ext = self._build()
        buf = self._buf_for(sp)  # count stays 0
        ext._install_cuts(buf)
        for s in sp.local_scenarios.values():
            self.assertEqual(len(s._mpisppy_model.xhat_feasibility_cuts), 0)

    def test_all_zero_row_is_skipped_even_if_counted(self):
        """Count header claims 2 cuts but both rows are all zeros."""
        sp, ext = self._build()
        buf = self._buf_for(sp)
        buf[-1] = 2.0
        ext._install_cuts(buf)
        for s in sp.local_scenarios.values():
            self.assertEqual(len(s._mpisppy_model.xhat_feasibility_cuts), 0)

    def test_two_successive_install_calls_accumulate(self):
        sp, ext = self._build(num=2)
        buf = self._buf_for(sp)
        buf[0] = -1.0
        buf[1:4] = [1.0, 1.0, 1.0]
        buf[-1] = 1.0
        ext._install_cuts(buf)
        ext._install_cuts(buf)
        for s in sp.local_scenarios.values():
            self.assertEqual(len(s._mpisppy_model.xhat_feasibility_cuts), 2)


# ---------------------------------------------------------------------------
# No-good cut encoding (spoke side)
# ---------------------------------------------------------------------------


class _FakeSendArray:
    def __init__(self, length):
        self._arr = np.zeros(length)

    def value_array(self):
        return self._arr


class _FakeSpcomm:
    """Just enough of SPCommunicator for ``_maybe_emit_feasibility_cut``."""

    def __init__(self, buffer_length):
        self.send_buffers = {
            Field.XHAT_FEASIBILITY_CUT: _FakeSendArray(buffer_length),
        }
        self.sent = []  # (buf_snapshot, field) for every put_send_buffer call

    def is_send_field_registered(self, field):
        return field in self.send_buffers

    def put_send_buffer(self, send_array, field):
        # Snapshot the buffer contents to allow assertions after clearing.
        self.sent.append((send_array.value_array().copy(), field))


class _FakeOpt:
    """Stand-in for Xhat_Eval in emission tests."""

    def __init__(self, sp, cuts_count=3):
        self.local_scenarios = sp.local_scenarios
        self.options = dict(sp.options)
        self.options["xhat_feasibility_cuts_count"] = cuts_count
        # buffer layout matches FieldLengths' formula
        any_s = next(iter(sp.local_scenarios.values()))
        nonant_len = len(any_s._mpisppy_data.nonant_indices)
        self.spcomm = _FakeSpcomm(cuts_count * (nonant_len + 1) + 1)
        # Satisfy attributes XhatBase.__init__ expects.
        self.cylinder_rank = 0
        self.n_proc = 1
        self.scenario_names_to_rank = {"ROOT": {
            sname: 0 for sname in sp.local_scenarios
        }}


def _make_xhatbase(sp, cuts_count=3):
    """Construct an XhatBase extension with a fake opt for emission tests.

    We bypass ``__init__`` (which wants a real opt) and set attributes
    directly.
    """
    ext = xhatbase_extension.XhatBase.__new__(xhatbase_extension.XhatBase)
    ext.opt = _FakeOpt(sp, cuts_count=cuts_count)
    ext.cylinder_rank = ext.opt.cylinder_rank
    ext.n_proc = ext.opt.n_proc
    ext.verbose = False
    ext.scenario_name_to_rank = ext.opt.scenario_names_to_rank
    return ext


def _set_xhat(sp, xhat_values):
    """Write xhat values into every local scenario's nonant vars."""
    for s in sp.local_scenarios.values():
        for v, val in zip(s._mpisppy_data.nonant_indices.values(), xhat_values):
            v.set_value(val)


class TestNoGoodCutEncoding(unittest.TestCase):

    def _expected_encoding(self, xhat):
        """Reference encoding: constant = sum(xhat) - 1; coef_i = 1 - 2 xhat_i."""
        xhat_ints = [1 if v > 0.5 else 0 for v in xhat]
        rhs_constant = sum(xhat_ints) - 1
        coefs = [1 - 2 * xi for xi in xhat_ints]
        return rhs_constant, coefs

    def _emit_and_get_row(self, sp, xhat_values, cuts_count=3):
        ext = _make_xhatbase(sp, cuts_count=cuts_count)
        _set_xhat(sp, xhat_values)
        ext._maybe_emit_feasibility_cut()
        self.assertEqual(len(ext.opt.spcomm.sent), 1,
                         "expected exactly one put_send_buffer call")
        buf, field = ext.opt.spcomm.sent[0]
        self.assertEqual(field, Field.XHAT_FEASIBILITY_CUT)
        nonant_len = len(xhat_values)
        row_len = 1 + nonant_len
        self.assertEqual(buf.shape[0], cuts_count * row_len + 1)
        # Count slot must be 1.
        self.assertEqual(buf[-1], 1.0)
        # Rows 1..cuts_count-1 must be zero.
        for k in range(1, cuts_count):
            self.assertTrue(
                np.all(buf[k * row_len:(k + 1) * row_len] == 0.0),
                f"row {k} should be zero-padded",
            )
        return buf[:row_len]

    def test_encoding_all_zero_xhat(self):
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        row = self._emit_and_get_row(sp, [0, 0, 0])
        exp_const, exp_coefs = self._expected_encoding([0, 0, 0])
        self.assertEqual(row[0], exp_const)
        np.testing.assert_array_equal(row[1:], exp_coefs)

    def test_encoding_all_one_xhat(self):
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        row = self._emit_and_get_row(sp, [1, 1, 1])
        exp_const, exp_coefs = self._expected_encoding([1, 1, 1])
        self.assertEqual(row[0], exp_const)
        np.testing.assert_array_equal(row[1:], exp_coefs)

    def test_encoding_mixed_xhat(self):
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        row = self._emit_and_get_row(sp, [1, 0, 1])
        exp_const, exp_coefs = self._expected_encoding([1, 0, 1])
        self.assertEqual(row[0], exp_const)
        np.testing.assert_array_equal(row[1:], exp_coefs)

    def test_cut_excludes_its_own_xhat(self):
        """Sanity: the emitted cut evaluated at xhat must violate the
        inequality constant + sum coef_i x_i >= 0."""
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        xhat = [1, 0, 1]
        row = self._emit_and_get_row(sp, xhat)
        lhs = row[0] + sum(c * x for c, x in zip(row[1:], xhat))
        self.assertLess(lhs, 0.0,
                        "no-good cut should violate at its own xhat")

    def test_emit_is_noop_when_feature_off(self):
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        ext = _make_xhatbase(sp, cuts_count=0)
        _set_xhat(sp, [1, 0, 1])
        ext._maybe_emit_feasibility_cut()
        self.assertEqual(ext.opt.spcomm.sent, [])


# ---------------------------------------------------------------------------
# Multi-node nonants in the startup check
# ---------------------------------------------------------------------------


def _multi_node_binary_scenario(sname, **kwargs):
    """A hand-built scenario with nonants at two different nodes.

    We do NOT wrap this in SPBase (which would insist on proper
    branching-factor plumbing). Instead we construct just the
    attributes ``XhatFeasibilityCutExtension._assert_all_nonants_binary``
    needs: ``_mpisppy_data.nonant_indices`` keyed by ``(node_name, i)``.
    """
    m = pyo.ConcreteModel(name=sname)
    m.x_root = pyo.Var([0, 1], domain=pyo.Binary)
    m.x_stage2 = pyo.Var([0], domain=pyo.Binary)
    m.obj = pyo.Objective(expr=m.x_root[0] + m.x_root[1] + m.x_stage2[0])
    m._mpisppy_node_list = [
        stree.ScenarioNode("ROOT", cond_prob=1.0, stage=1,
                           cost_expression=m.x_root[0],
                           nonant_list=[m.x_root], scen_model=m),
        stree.ScenarioNode("ROOT_0", cond_prob=1.0, stage=2,
                           cost_expression=m.x_stage2[0],
                           nonant_list=[m.x_stage2],
                           parent_name="ROOT", scen_model=m),
    ]
    m._mpisppy_probability = 1.0
    return m


class _MultiNodeStub:
    """Fake opt wrapping a multi-node scenario without SPBase."""

    def __init__(self, scenario, options):
        # Mimic the attributes _assert_all_nonants_binary needs.
        scenario._mpisppy_data = _FakeData()
        scenario._mpisppy_data.nonant_indices = {}
        for node in scenario._mpisppy_node_list:
            for i, v in enumerate(node.nonant_vardata_list):
                scenario._mpisppy_data.nonant_indices[(node.name, i)] = v
        self.local_scenarios = {"s0": scenario}
        self.options = options
        self.spcomm = None
        self.multistage = True


class _FakeData:
    pass


class TestMultiNodeStartupCheck(unittest.TestCase):

    def test_passes_with_binary_nonants_in_all_nodes(self):
        scen = _multi_node_binary_scenario("s0")
        opt = _MultiNodeStub(scen, {"xhat_feasibility_cuts_count": 1})
        ext = XhatFeasibilityCutExtension(opt)
        # setup_hub also attaches the ConstraintList; for this lightweight
        # stub we just run the scan directly.
        ext._assert_all_nonants_binary()

    def test_raises_on_non_binary_in_non_root_node(self):
        scen = _multi_node_binary_scenario("s0")
        # Mutate the stage-2 var into a continuous one in place.
        del scen.x_stage2
        scen.x_stage2 = pyo.Var([0], bounds=(0, 1))  # continuous
        # Rebuild the node list with the new var.
        scen._mpisppy_node_list[1] = stree.ScenarioNode(
            "ROOT_0", cond_prob=1.0, stage=2,
            cost_expression=scen.x_stage2[0],
            nonant_list=[scen.x_stage2],
            parent_name="ROOT", scen_model=scen,
        )
        opt = _MultiNodeStub(scen, {"xhat_feasibility_cuts_count": 1})
        ext = XhatFeasibilityCutExtension(opt)
        with self.assertRaises(RuntimeError) as cm:
            ext._assert_all_nonants_binary()
        self.assertIn("ROOT_0", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
