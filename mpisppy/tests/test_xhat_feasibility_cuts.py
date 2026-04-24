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

import mpisppy.cylinders.xhatbase as cylinder_xhatbase  # noqa: F401 — ensures class-level send_fields registration is covered
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


# ---------------------------------------------------------------------------
# Extension plumbing: register / sync / persistent-solver branch
# ---------------------------------------------------------------------------


class _SpcommStub:
    """Lightweight ``spcomm`` stand-in for register_receive_fields /
    sync_with_spokes coverage."""

    def __init__(self, fields_to_ranks=None, recv_buffer=None):
        self.fields_to_ranks = fields_to_ranks or {}
        self._recv_buffer_by_rank = {}
        if recv_buffer is not None:
            # Seed whichever rank shows up.
            ranks = self.fields_to_ranks.get(Field.XHAT_FEASIBILITY_CUT, [0])
            for r in ranks:
                self._recv_buffer_by_rank[r] = recv_buffer

    def register_recv_field(self, field, rank):
        return self._recv_buffer_by_rank.setdefault(rank, _RecvBufferStub())


class _RecvBufferStub:
    def __init__(self, arr=None, is_new_result=True):
        self._arr = arr if arr is not None else np.zeros(1)
        self._is_new = is_new_result

    def is_new(self):
        return self._is_new

    def array(self):
        return self._arr


class _PersistentSolverStub:
    """Stub that passes ``sputils.is_persistent`` via isinstance.

    ``sputils.is_persistent`` checks ``isinstance(p, PersistentSolver)``,
    so we inherit from the real abstract base.
    """
    pass


class TestExtensionRegistrationAndSync(unittest.TestCase):
    def _make_ext(self, cuts_count=3):
        sp = _make_sp(_binary_two_stage_scenario, num=2, cuts_count=cuts_count)
        _attach_fake_solver_plugin(sp)
        ext = XhatFeasibilityCutExtension(_Opt(sp))
        ext.setup_hub()
        return sp, ext

    def test_register_send_fields_is_noop(self):
        _, ext = self._make_ext()
        # Should return without raising; exercise line 116.
        ext.register_send_fields()

    def test_register_receive_fields_no_emitting_spoke(self):
        """If no rank advertises the cut field, recv_buffer stays None."""
        sp, ext = self._make_ext()
        ext.opt.spcomm = _SpcommStub(fields_to_ranks={})
        ext.register_receive_fields()
        self.assertIsNone(ext._recv_buffer)

    def test_register_receive_fields_with_one_emitter(self):
        sp, ext = self._make_ext()
        recv = _RecvBufferStub()
        ext.opt.spcomm = _SpcommStub(
            fields_to_ranks={Field.XHAT_FEASIBILITY_CUT: [2]},
            recv_buffer=recv,
        )
        ext.register_receive_fields()
        self.assertIs(ext._recv_buffer, recv)

    def test_register_receive_fields_multiple_emitters_asserts(self):
        sp, ext = self._make_ext()
        ext.opt.spcomm = _SpcommStub(
            fields_to_ranks={Field.XHAT_FEASIBILITY_CUT: [1, 2]},
        )
        with self.assertRaises(AssertionError):
            ext.register_receive_fields()

    def test_sync_with_spokes_no_buffer(self):
        sp, ext = self._make_ext()
        ext._recv_buffer = None
        # No side effects, no raise.
        ext.sync_with_spokes()
        for s in sp.local_scenarios.values():
            self.assertEqual(len(s._mpisppy_model.xhat_feasibility_cuts), 0)

    def test_sync_with_spokes_not_new_is_noop(self):
        sp, ext = self._make_ext()
        # A buffer that claims "not new" should be ignored.
        nonant_len = 3
        buf = np.zeros(3 * (nonant_len + 1) + 1)
        buf[0] = -1.0
        buf[1:4] = [1.0, 1.0, 1.0]
        buf[-1] = 1.0
        ext._recv_buffer = _RecvBufferStub(arr=buf, is_new_result=False)
        ext.sync_with_spokes()
        for s in sp.local_scenarios.values():
            self.assertEqual(len(s._mpisppy_model.xhat_feasibility_cuts), 0)

    def test_sync_with_spokes_new_installs_cut(self):
        sp, ext = self._make_ext()
        nonant_len = 3
        buf = np.zeros(3 * (nonant_len + 1) + 1)
        buf[0] = -1.0
        buf[1:4] = [1.0, 1.0, 1.0]
        buf[-1] = 1.0
        ext._recv_buffer = _RecvBufferStub(arr=buf, is_new_result=True)
        ext.sync_with_spokes()
        for s in sp.local_scenarios.values():
            self.assertEqual(len(s._mpisppy_model.xhat_feasibility_cuts), 1)


class TestCutInstallPersistentSolverBranch(unittest.TestCase):
    """Exercise the persistent-solver branch in ``_install_cuts``."""

    def test_persistent_solver_gets_add_constraint(self):
        # Build a PersistentSolver subclass so sputils.is_persistent says True.
        from pyomo.solvers.plugins.solvers.persistent_solver import (
            PersistentSolver,
        )

        class _PersistentPlugin(PersistentSolver):
            def __init__(self):
                # Bypass PersistentSolver.__init__ which wants a config; we
                # only need an isinstance-compatible object with add_constraint.
                self.added = []

            def add_constraint(self, con):
                self.added.append(con)

            # Abstracts we don't actually exercise in this test.
            def _presolve(self, *a, **kw): pass
            def _postsolve(self, *a, **kw): pass
            def _warm_start(self, *a, **kw): pass
            def _apply_solver(self, *a, **kw): pass
            def _get_dual_values(self, *a, **kw): return {}
            def _load_vars(self, *a, **kw): pass
            def _get_expr_from_pyomo_repn(self, *a, **kw): pass

        sp = _make_sp(_binary_two_stage_scenario, num=2, cuts_count=3)
        for s in sp.local_scenarios.values():
            s._solver_plugin = _PersistentPlugin()
        ext = XhatFeasibilityCutExtension(_Opt(sp))
        ext.setup_hub()

        nonant_len = 3
        buf = np.zeros(3 * (nonant_len + 1) + 1)
        buf[0] = -1.0
        buf[1:4] = [1.0, 1.0, 1.0]
        buf[-1] = 1.0
        ext._install_cuts(buf)

        # Each local scenario's persistent plugin should have been told
        # about the new constraint.
        for s in sp.local_scenarios.values():
            self.assertEqual(len(s._solver_plugin.added), 1)


# ---------------------------------------------------------------------------
# _maybe_emit_feasibility_cut: early-return branches and buffer-size check
# ---------------------------------------------------------------------------


class TestEmitEarlyReturns(unittest.TestCase):
    def _ext(self, sp, cuts_count=3, spcomm=None):
        ext = xhatbase_extension.XhatBase.__new__(xhatbase_extension.XhatBase)
        ext.opt = _FakeOpt(sp, cuts_count=cuts_count)
        if spcomm is not None:
            ext.opt.spcomm = spcomm
        ext.cylinder_rank = 0
        ext.n_proc = 1
        ext.verbose = False
        ext.scenario_name_to_rank = ext.opt.scenario_names_to_rank
        return ext

    def test_spcomm_is_none_silently_returns(self):
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        ext = self._ext(sp, cuts_count=3, spcomm="replace")
        ext.opt.spcomm = None   # override the _FakeSpcomm installed by _FakeOpt
        _set_xhat(sp, [1, 0, 1])
        # Must not raise.
        ext._maybe_emit_feasibility_cut()

    def test_send_field_not_registered_silently_returns(self):
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        ext = self._ext(sp, cuts_count=3)
        # Remove the pre-registered buffer so is_send_field_registered is False.
        ext.opt.spcomm.send_buffers.pop(Field.XHAT_FEASIBILITY_CUT)
        _set_xhat(sp, [1, 0, 1])
        ext._maybe_emit_feasibility_cut()
        self.assertEqual(ext.opt.spcomm.sent, [])

    def test_buffer_size_mismatch_raises(self):
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        # cuts_count=3 in options → emitter expects 3*(3+1)+1 = 13.
        # Replace the buffer with a wrong size.
        ext = self._ext(sp, cuts_count=3)
        ext.opt.spcomm.send_buffers[Field.XHAT_FEASIBILITY_CUT] = \
            _FakeSendArray(5)  # wrong size
        _set_xhat(sp, [1, 0, 1])
        with self.assertRaises(RuntimeError) as cm:
            ext._maybe_emit_feasibility_cut()
        self.assertIn("buffer has length 5", str(cm.exception))

    def test_missing_nonant_value_silently_returns(self):
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        ext = self._ext(sp, cuts_count=3)
        # Clear one nonant's value to hit the `xv is None` guard.
        for s in sp.local_scenarios.values():
            first_v = next(iter(s._mpisppy_data.nonant_indices.values()))
            first_v.set_value(None, skip_validation=True)
            break
        ext._maybe_emit_feasibility_cut()
        self.assertEqual(ext.opt.spcomm.sent, [])


# ---------------------------------------------------------------------------
# Plumbing tests: config registration, cfg_vanilla helpers, FieldLengths,
# and the _try_one exception wrapper.
# ---------------------------------------------------------------------------


class TestConfigArgRegistration(unittest.TestCase):
    def test_xhat_feasibility_cut_args_registers_with_default_zero(self):
        from mpisppy.utils import config as cfgmod
        cfg = cfgmod.Config()
        cfg.xhat_feasibility_cut_args()
        self.assertIn("xhat_feasibility_cuts_count", cfg)
        self.assertEqual(cfg.get("xhat_feasibility_cuts_count"), 0)

    def test_generic_parsing_registers_the_flag(self):
        """Covers the cfg.xhat_feasibility_cut_args() call inside
        mpisppy.generic.parsing.parse_args."""
        import sys
        import types
        import mpisppy.generic.parsing as parsing

        stub = types.ModuleType("__stub_model_for_parse_args__")
        def inparser_adder(cfg):
            cfg.add_to_config("num_scens", "stub", int, default=1)
        stub.inparser_adder = inparser_adder

        saved_argv = sys.argv
        sys.argv = ["prog"]   # no positional flags; defaults only
        try:
            cfg = parsing.parse_args(stub)
        finally:
            sys.argv = saved_argv

        # The flag made it into the Config — which is the signal that
        # parse_args actually ran cfg.xhat_feasibility_cut_args().
        self.assertIn("xhat_feasibility_cuts_count", cfg)


class TestCfgVanillaPlumbing(unittest.TestCase):
    def _cfg(self, xhat_cap=0):
        """Build a minimal Config with whatever fields shared_options needs."""
        from mpisppy.utils import config as cfgmod
        cfg = cfgmod.Config()
        cfg.popular_args()
        cfg.two_sided_args()
        cfg.ph_args()
        cfg.aph_args()
        cfg.xhat_feasibility_cut_args()
        if xhat_cap:
            cfg.xhat_feasibility_cuts_count = xhat_cap
        cfg.solver_name = "gurobi"
        return cfg

    def test_shared_options_propagates_cap(self):
        from mpisppy.utils import cfg_vanilla as vanilla
        cfg = self._cfg(xhat_cap=4)
        shopts = vanilla.shared_options(cfg)
        self.assertEqual(shopts["xhat_feasibility_cuts_count"], 4)

    def test_shared_options_defaults_to_zero_when_off(self):
        from mpisppy.utils import cfg_vanilla as vanilla
        cfg = self._cfg(xhat_cap=0)
        shopts = vanilla.shared_options(cfg)
        self.assertEqual(shopts["xhat_feasibility_cuts_count"], 0)

    def test_add_xhat_feasibility_cuts_noop_when_off(self):
        from mpisppy.utils import cfg_vanilla as vanilla
        cfg = self._cfg(xhat_cap=0)
        hub_dict = {"opt_kwargs": {"options": {}}}
        result = vanilla.add_xhat_feasibility_cuts(hub_dict, cfg)
        # Same dict back; no extensions attached; no option injected.
        self.assertIs(result, hub_dict)
        self.assertNotIn("extensions", hub_dict["opt_kwargs"])
        self.assertNotIn("xhat_feasibility_cuts_count",
                         hub_dict["opt_kwargs"]["options"])

    def test_add_xhat_feasibility_cuts_attaches_extension_when_on(self):
        from mpisppy.utils import cfg_vanilla as vanilla
        from mpisppy.extensions.xhat_feasibility_cut_extension import (
            XhatFeasibilityCutExtension,
        )
        cfg = self._cfg(xhat_cap=7)
        hub_dict = {"opt_kwargs": {"options": {}}}
        vanilla.add_xhat_feasibility_cuts(hub_dict, cfg)
        # extension_adder installs either a direct extension or a
        # MultiExtension; confirm XhatFeasibilityCutExtension made it in.
        exts = hub_dict["opt_kwargs"].get("extensions")
        self.assertIsNotNone(exts)
        ext_classes = (
            [exts] if not isinstance(exts, type) or exts.__name__ != "MultiExtension"
            else hub_dict["opt_kwargs"]["extension_kwargs"]["ext_classes"]
        )
        # extension_adder produces either a single class or a MultiExtension
        # with ext_classes. Handle both.
        mx_classes = hub_dict["opt_kwargs"].get("extension_kwargs", {}).get(
            "ext_classes", [])
        self.assertTrue(
            XhatFeasibilityCutExtension in ext_classes
            or XhatFeasibilityCutExtension in mx_classes,
            f"Expected XhatFeasibilityCutExtension in hub extensions; got {exts!r}",
        )
        self.assertEqual(
            hub_dict["opt_kwargs"]["options"]["xhat_feasibility_cuts_count"],
            7,
        )


class TestFieldLengthsPicksUpCap(unittest.TestCase):
    """Covers the FieldLengths setter line that reads
    ``opt.options['xhat_feasibility_cuts_count']``."""

    def test_field_length_reflects_cap(self):
        from mpisppy.cylinders.spwindow import FieldLengths, Field
        sp = _make_sp(_binary_two_stage_scenario, num=2, cuts_count=5)
        # FieldLengths wants opt.nonant_length; SPBase sets that up.
        fl = FieldLengths(sp)
        # row_len = 1 + 3 nonants; cap=5 → 5*4 + 1 = 21.
        self.assertEqual(fl[Field.XHAT_FEASIBILITY_CUT], 5 * 4 + 1)

    def test_field_length_off_is_one(self):
        from mpisppy.cylinders.spwindow import FieldLengths, Field
        sp = _make_sp(_binary_two_stage_scenario, num=2, cuts_count=0)
        fl = FieldLengths(sp)
        # With cap=0 the buffer collapses to just the trailing count slot.
        self.assertEqual(fl[Field.XHAT_FEASIBILITY_CUT], 1)


class TestGenericExtensionsWiring(unittest.TestCase):
    """Covers the one-line gate in mpisppy/generic/extensions.py that
    defers to cfg_vanilla.add_xhat_feasibility_cuts when the flag is
    positive. We minimally construct the cfg the function reads."""

    def _make_configured_cfg(self, cuts_count):
        from mpisppy.utils import config as cfgmod
        cfg = cfgmod.Config()
        # configure_extensions reads these; register them as no-ops.
        cfg.popular_args()
        cfg.ph_args()
        cfg.aph_args()
        cfg.two_sided_args()
        cfg.fixer_args()
        cfg.relaxed_ph_fixer_args()
        cfg.integer_relax_then_enforce_args()
        cfg.gapper_args()
        cfg.gapper_args(name="lagrangian")
        cfg.ph_primal_args()
        cfg.ph_dual_args()
        cfg.relaxed_ph_args()
        cfg.gradient_args()
        cfg.dynamic_rho_args()
        cfg.reduced_costs_args()
        cfg.sep_rho_args()
        cfg.coeff_rho_args()
        cfg.sensi_rho_args()
        cfg.reduced_costs_rho_args()
        cfg.norm_rho_args()
        cfg.primal_dual_rho_args()
        cfg.wxbar_read_write_args()
        cfg.tracking_args()
        cfg.converger_args()
        cfg.xhat_feasibility_cut_args()
        cfg.add_to_config("user_defined_extensions",
                          description="list of user-defined extensions",
                          domain=list, default=None)
        cfg.add_to_config("write_scenario_lp_mps_files_dir",
                          description="not used here",
                          domain=str, default=None)
        cfg.solver_name = "gurobi"
        cfg.xhat_feasibility_cuts_count = cuts_count
        return cfg

    def _fake_hub_dict(self):
        return {"opt_kwargs": {"options": {}}}

    def test_configure_extensions_attaches_when_on(self):
        from mpisppy.generic.extensions import configure_extensions
        from mpisppy.extensions.xhat_feasibility_cut_extension import (
            XhatFeasibilityCutExtension,
        )
        cfg = self._make_configured_cfg(cuts_count=3)
        hub = self._fake_hub_dict()
        configure_extensions(hub, module=None, cfg=cfg)
        exts = hub["opt_kwargs"].get("extensions")
        mx_classes = hub["opt_kwargs"].get("extension_kwargs", {}).get(
            "ext_classes", [])
        self.assertTrue(
            exts is XhatFeasibilityCutExtension
            or XhatFeasibilityCutExtension in mx_classes,
            f"Expected XhatFeasibilityCutExtension in hub; got exts={exts!r}, "
            f"ext_classes={mx_classes!r}",
        )

    def test_configure_extensions_skipped_when_off(self):
        from mpisppy.generic.extensions import configure_extensions
        from mpisppy.extensions.xhat_feasibility_cut_extension import (
            XhatFeasibilityCutExtension,
        )
        cfg = self._make_configured_cfg(cuts_count=0)
        hub = self._fake_hub_dict()
        configure_extensions(hub, module=None, cfg=cfg)
        exts = hub["opt_kwargs"].get("extensions")
        mx_classes = hub["opt_kwargs"].get("extension_kwargs", {}).get(
            "ext_classes", [])
        self.assertFalse(
            exts is XhatFeasibilityCutExtension
            or XhatFeasibilityCutExtension in mx_classes,
            "XhatFeasibilityCutExtension should NOT be attached when off",
        )


class TestTryOneExceptionWrapper(unittest.TestCase):
    """Covers the ``try / except`` wrapper around
    ``_maybe_emit_feasibility_cut`` inside ``XhatBase._try_one``.

    We don't drive ``_try_one`` end-to-end (that would need the whole
    Xhat_Eval machinery); we exercise the tiny wrapper by calling the
    same snippet directly. The wrapper is intentionally simple: log on
    rank 0 and swallow the exception so the xhatter keeps going.
    """

    def test_try_one_reaches_the_emit_wrapper_on_infeasibility(self):
        """Drive ``_try_one`` just far enough to hit the emit wrapper.

        Stubs ``self.opt`` with the minimum set of methods ``_try_one``
        touches on the two-stage infeasibility path, and replaces
        ``_maybe_emit_feasibility_cut`` with one that raises. The
        wrapper must log-and-swallow (no re-raise) so the xhatter
        keeps going.
        """
        import mpisppy.extensions.xhatbase as xb
        sp = _make_sp(_binary_two_stage_scenario, num=1)
        ext = xb.XhatBase.__new__(xb.XhatBase)
        ext.cylinder_rank = 0
        ext.n_proc = 1
        ext.verbose = False
        ext.scenario_name_to_rank = {"ROOT": {"scen0": 0}}

        class _StubOpt:
            def __init__(self, sp):
                self.local_scenarios = sp.local_scenarios
                self.options = {}
                self._saved = 0
                self._restored = 0
                # inject a nonant_cache on the one local scenario to
                # satisfy the two-stage branch in _try_one.
                for s in sp.local_scenarios.values():
                    s._mpisppy_data.nonant_cache = np.zeros(
                        len(s._mpisppy_data.nonant_indices))
            def _save_nonants(self): self._saved += 1
            def _restore_nonants(self): self._restored += 1
            def _fix_nonants(self, xhats): pass
            def solve_loop(self, **kw): pass
            def no_incumbent_prob(self): return 1.0  # force infeasibility
            def Eobjective(self, **kw): return 0.0
            def update_best_solution_if_improving(self, obj): return False

        class _StubComm:
            def bcast(self, data, root=0): return data
        ext.opt = _StubOpt(sp)
        ext.comms = {"ROOT": _StubComm()}

        # Monkey-patch the emit helper to raise so the wrapper's except
        # branch fires.
        def _boom(self):
            raise RuntimeError("synthetic emit failure")
        ext._maybe_emit_feasibility_cut = _boom.__get__(ext, xb.XhatBase)

        # Should NOT raise: _try_one catches and logs.
        result = ext._try_one({"ROOT": "scen0"},
                              solver_options=None,
                              verbose=False,
                              restore_nonants=True)
        self.assertIsNone(result)
        # nonants were restored despite the exception.
        self.assertEqual(ext.opt._restored, 1)


if __name__ == "__main__":
    unittest.main()
