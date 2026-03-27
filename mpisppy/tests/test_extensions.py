###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy/extensions/extension.py.

Covers the four extension base classes:
  * Extension      – base class for PH/SPOpt extensions
  * MultiExtension – container that delegates to multiple Extension objects
  * EFExtension    – base class for EF extensions
  * EFMultiExtension – container for multiple EFExtension objects

No solver or MPI is required for these tests.
"""

import unittest

from mpisppy.extensions.extension import (
    EFExtension,
    EFMultiExtension,
    Extension,
    MultiExtension,
)


class _MockOpt:
    """Minimal mock of an SPOpt/PH object, sufficient to construct extensions."""
    pass


class _MockEF:
    """Minimal mock of an EF object, sufficient to construct EFExtensions."""
    pass


# ---------------------------------------------------------------------------
# Extension base class
# ---------------------------------------------------------------------------

class TestExtensionInit(unittest.TestCase):
    """Tests for Extension.__init__() and attribute storage."""

    def test_stores_opt(self):
        opt = _MockOpt()
        ext = Extension(opt)
        self.assertIs(ext.opt, opt)


class TestExtensionMethodReturnValues(unittest.TestCase):
    """Every no-op method of Extension should return None (no side-effects)."""

    def setUp(self):
        self.ext = Extension(_MockOpt())

    def test_setup_hub_returns_none(self):
        self.assertIsNone(self.ext.setup_hub())

    def test_register_send_fields_returns_none(self):
        self.assertIsNone(self.ext.register_send_fields())

    def test_register_receive_fields_returns_none(self):
        self.assertIsNone(self.ext.register_receive_fields())

    def test_sync_with_spokes_returns_none(self):
        self.assertIsNone(self.ext.sync_with_spokes())

    def test_pre_solve_returns_none(self):
        self.assertIsNone(self.ext.pre_solve(subproblem=None))

    def test_post_solve_returns_results(self):
        sentinel = object()
        result = self.ext.post_solve(subproblem=None, results=sentinel)
        self.assertIs(result, sentinel)

    def test_pre_solve_loop_returns_none(self):
        self.assertIsNone(self.ext.pre_solve_loop())

    def test_post_solve_loop_returns_none(self):
        self.assertIsNone(self.ext.post_solve_loop())

    def test_pre_iter0_returns_none(self):
        self.assertIsNone(self.ext.pre_iter0())

    def test_iter0_post_solver_creation_returns_none(self):
        self.assertIsNone(self.ext.iter0_post_solver_creation())

    def test_post_iter0_returns_none(self):
        self.assertIsNone(self.ext.post_iter0())

    def test_post_iter0_after_sync_returns_none(self):
        self.assertIsNone(self.ext.post_iter0_after_sync())

    def test_miditer_returns_none(self):
        self.assertIsNone(self.ext.miditer())

    def test_enditer_returns_none(self):
        self.assertIsNone(self.ext.enditer())

    def test_enditer_after_sync_returns_none(self):
        self.assertIsNone(self.ext.enditer_after_sync())

    def test_post_everything_returns_none(self):
        self.assertIsNone(self.ext.post_everything())


# ---------------------------------------------------------------------------
# MultiExtension
# ---------------------------------------------------------------------------

class _TrackingExtension(Extension):
    """Extension that records which methods were called."""

    def __init__(self, opt):
        super().__init__(opt)
        self.called = []

    def setup_hub(self):
        self.called.append("setup_hub")

    def register_send_fields(self):
        self.called.append("register_send_fields")

    def register_receive_fields(self):
        self.called.append("register_receive_fields")

    def sync_with_spokes(self):
        self.called.append("sync_with_spokes")

    def pre_solve(self, subproblem):
        self.called.append("pre_solve")

    def post_solve(self, subproblem, results):
        self.called.append("post_solve")
        return results

    def pre_solve_loop(self):
        self.called.append("pre_solve_loop")

    def post_solve_loop(self):
        self.called.append("post_solve_loop")

    def pre_iter0(self):
        self.called.append("pre_iter0")

    def iter0_post_solver_creation(self):
        self.called.append("iter0_post_solver_creation")

    def post_iter0(self):
        self.called.append("post_iter0")

    def post_iter0_after_sync(self):
        self.called.append("post_iter0_after_sync")

    def miditer(self):
        self.called.append("miditer")

    def enditer(self):
        self.called.append("enditer")

    def enditer_after_sync(self):
        self.called.append("enditer_after_sync")

    def post_everything(self):
        self.called.append("post_everything")


class TestMultiExtensionInit(unittest.TestCase):
    """Tests for MultiExtension construction."""

    def test_creates_one_extension(self):
        ph = _MockOpt()
        multi = MultiExtension(ph, [_TrackingExtension])
        self.assertIn("_TrackingExtension", multi.extdict)

    def test_creates_multiple_extensions(self):
        ph = _MockOpt()

        class Ext1(Extension):
            pass

        class Ext2(Extension):
            pass

        multi = MultiExtension(ph, [Ext1, Ext2])
        self.assertIn("Ext1", multi.extdict)
        self.assertIn("Ext2", multi.extdict)

    def test_stores_opt_via_parent(self):
        ph = _MockOpt()
        multi = MultiExtension(ph, [])
        self.assertIs(multi.opt, ph)

    def test_empty_ext_list(self):
        ph = _MockOpt()
        multi = MultiExtension(ph, [])
        self.assertEqual(len(multi.extdict), 0)


class TestMultiExtensionDelegation(unittest.TestCase):
    """MultiExtension must call every method on every contained extension."""

    def _make_multi(self, n=2):
        ph = _MockOpt()
        exts = [_TrackingExtension(ph) for _ in range(n)]
        multi = MultiExtension.__new__(MultiExtension)
        multi.opt = ph
        multi.extdict = {f"ext{i}": exts[i] for i in range(n)}
        return multi, exts

    def test_setup_hub_delegated(self):
        multi, exts = self._make_multi()
        multi.setup_hub()
        for ext in exts:
            self.assertIn("setup_hub", ext.called)

    def test_register_send_fields_delegated(self):
        multi, exts = self._make_multi()
        multi.register_send_fields()
        for ext in exts:
            self.assertIn("register_send_fields", ext.called)

    def test_register_receive_fields_delegated(self):
        multi, exts = self._make_multi()
        multi.register_receive_fields()
        for ext in exts:
            self.assertIn("register_receive_fields", ext.called)

    def test_sync_with_spokes_delegated(self):
        multi, exts = self._make_multi()
        multi.sync_with_spokes()
        for ext in exts:
            self.assertIn("sync_with_spokes", ext.called)

    def test_pre_solve_delegated(self):
        multi, exts = self._make_multi()
        multi.pre_solve(None)
        for ext in exts:
            self.assertIn("pre_solve", ext.called)

    def test_post_solve_passes_results_through(self):
        multi, exts = self._make_multi(1)
        sentinel = object()
        result = multi.post_solve(None, sentinel)
        self.assertIs(result, sentinel)
        self.assertIn("post_solve", exts[0].called)

    def test_pre_solve_loop_delegated(self):
        multi, exts = self._make_multi()
        multi.pre_solve_loop()
        for ext in exts:
            self.assertIn("pre_solve_loop", ext.called)

    def test_post_solve_loop_delegated(self):
        multi, exts = self._make_multi()
        multi.post_solve_loop()
        for ext in exts:
            self.assertIn("post_solve_loop", ext.called)

    def test_pre_iter0_delegated(self):
        multi, exts = self._make_multi()
        multi.pre_iter0()
        for ext in exts:
            self.assertIn("pre_iter0", ext.called)

    def test_iter0_post_solver_creation_delegated(self):
        multi, exts = self._make_multi()
        multi.iter0_post_solver_creation()
        for ext in exts:
            self.assertIn("iter0_post_solver_creation", ext.called)

    def test_post_iter0_delegated(self):
        multi, exts = self._make_multi()
        multi.post_iter0()
        for ext in exts:
            self.assertIn("post_iter0", ext.called)

    def test_post_iter0_after_sync_delegated(self):
        multi, exts = self._make_multi()
        multi.post_iter0_after_sync()
        for ext in exts:
            self.assertIn("post_iter0_after_sync", ext.called)

    def test_miditer_delegated(self):
        multi, exts = self._make_multi()
        multi.miditer()
        for ext in exts:
            self.assertIn("miditer", ext.called)

    def test_enditer_delegated(self):
        multi, exts = self._make_multi()
        multi.enditer()
        for ext in exts:
            self.assertIn("enditer", ext.called)

    def test_enditer_after_sync_delegated(self):
        multi, exts = self._make_multi()
        multi.enditer_after_sync()
        for ext in exts:
            self.assertIn("enditer_after_sync", ext.called)

    def test_post_everything_delegated(self):
        multi, exts = self._make_multi()
        multi.post_everything()
        for ext in exts:
            self.assertIn("post_everything", ext.called)


# ---------------------------------------------------------------------------
# EFExtension base class
# ---------------------------------------------------------------------------

class TestEFExtensionInit(unittest.TestCase):
    """Tests for EFExtension.__init__()."""

    def test_stores_ef(self):
        ef = _MockEF()
        ext = EFExtension(ef)
        self.assertIs(ext.ef, ef)


class TestEFExtensionMethods(unittest.TestCase):
    """EFExtension no-op methods should return the expected values."""

    def setUp(self):
        self.ext = EFExtension(_MockEF())

    def test_pre_solve_returns_none(self):
        self.assertIsNone(self.ext.pre_solve())

    def test_post_solve_returns_results(self):
        sentinel = object()
        result = self.ext.post_solve(sentinel)
        self.assertIs(result, sentinel)

    def test_get_objective_value_returns_unchanged(self):
        self.assertAlmostEqual(self.ext.get_objective_value(42.5), 42.5)


# ---------------------------------------------------------------------------
# EFMultiExtension
# ---------------------------------------------------------------------------

class _TrackingEFExtension(EFExtension):
    """EFExtension that records calls and modifies objective value."""

    def __init__(self, ef, delta=0.0):
        super().__init__(ef)
        self.called = []
        self.delta = delta

    def pre_solve(self):
        self.called.append("pre_solve")

    def post_solve(self, results):
        self.called.append("post_solve")
        return results

    def get_objective_value(self, obj_val, **kwargs):
        self.called.append("get_objective_value")
        return obj_val + self.delta


class TestEFMultiExtensionInit(unittest.TestCase):
    """Tests for EFMultiExtension construction."""

    def test_creates_one_extension(self):
        ef = _MockEF()
        multi = EFMultiExtension(ef, [_TrackingEFExtension])
        self.assertIn("_TrackingEFExtension", multi.extdict)

    def test_empty_ext_list(self):
        ef = _MockEF()
        multi = EFMultiExtension(ef, [])
        self.assertEqual(len(multi.extdict), 0)

    def test_stores_ef_via_parent(self):
        ef = _MockEF()
        multi = EFMultiExtension(ef, [])
        self.assertIs(multi.ef, ef)


class TestEFMultiExtensionDelegation(unittest.TestCase):
    """EFMultiExtension must call every method on every contained extension."""

    def _make_multi(self, n=2, deltas=None):
        ef = _MockEF()
        if deltas is None:
            deltas = [0.0] * n
        exts = [_TrackingEFExtension(ef, d) for d in deltas]
        multi = EFMultiExtension.__new__(EFMultiExtension)
        multi.ef = ef
        multi.extdict = {f"ext{i}": exts[i] for i in range(n)}
        return multi, exts

    def test_pre_solve_delegated(self):
        multi, exts = self._make_multi()
        multi.pre_solve()
        for ext in exts:
            self.assertIn("pre_solve", ext.called)

    def test_post_solve_delegated(self):
        multi, exts = self._make_multi()
        sentinel = object()
        result = multi.post_solve(sentinel)
        self.assertIs(result, sentinel)
        for ext in exts:
            self.assertIn("post_solve", ext.called)

    def test_get_objective_value_chained(self):
        # Each extension adds its delta; total should be cumulative
        multi, exts = self._make_multi(n=2, deltas=[1.0, 2.0])
        result = multi.get_objective_value(10.0)
        self.assertAlmostEqual(result, 13.0)
        for ext in exts:
            self.assertIn("get_objective_value", ext.called)


if __name__ == "__main__":
    unittest.main()
