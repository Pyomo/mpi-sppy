###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy.debug_utils.buffer_inspect."""

import unittest

import numpy as np

from mpisppy.cylinders.spcommunicator import RecvArray, SendArray
from mpisppy.cylinders.spwindow import Field
from mpisppy.debug_utils import InspectContext, Report, inspect_buffer


class _FakeSP:
    """Duck-typed SPBase substitute for tests that exercise ctx.spbase."""
    def __init__(self, nonant_length):
        self.nonant_length = nonant_length


# Helper to publish a SendArray cleanly: set values then bump the id.
def _publish(buf, values):
    for i, v in enumerate(values):
        buf[i] = v
    buf._next_write_id()


# ---- generic checks ---------------------------------------------------------


class TestGenericChecks(unittest.TestCase):

    def test_fresh_send_buffer_passes(self):
        buf = SendArray(5)
        rep = inspect_buffer(buf, Field.NONANT,
                             ctx=InspectContext(nonant_count=5), send=True)
        self.assertTrue(rep.ok, msg=str(rep))

    def test_fresh_recv_buffer_passes(self):
        buf = RecvArray(5)
        rep = inspect_buffer(buf, Field.NONANT,
                             ctx=InspectContext(nonant_count=5), send=False)
        self.assertTrue(rep.ok, msg=str(rep))

    def test_padding_overrun_detected(self):
        buf = RecvArray(2)
        buf._full_array[5] = 42.0   # write into padding region
        rep = inspect_buffer(buf, Field.NONANT, send=False)
        self.assertFalse(rep.ok)
        self.assertTrue(any("padding region modified" in f for f in rep.findings))

    def test_write_id_must_be_integer_valued(self):
        buf = RecvArray(1)
        buf._array[-1] = 3.5
        rep = inspect_buffer(buf, Field.NONANT, send=False)
        self.assertFalse(rep.ok)
        self.assertTrue(any("not integer-valued" in f for f in rep.findings))

    def test_write_id_must_be_finite(self):
        buf = RecvArray(1)
        buf._array[-1] = np.inf
        rep = inspect_buffer(buf, Field.NONANT, send=False)
        self.assertFalse(rep.ok)
        self.assertTrue(any("non-finite" in f for f in rep.findings))

    def test_write_id_must_be_non_negative(self):
        buf = RecvArray(1)
        buf._array[-1] = -2.0
        rep = inspect_buffer(buf, Field.NONANT, send=False)
        self.assertFalse(rep.ok)
        self.assertTrue(any("negative" in f for f in rep.findings))

    def test_send_id_mismatch_detected(self):
        buf = SendArray(3)
        _publish(buf, [1.0, 2.0, 3.0])
        # Tamper with the trailing slot post-publish
        buf._array[-1] = 99.0
        rep = inspect_buffer(buf, Field.NONANT, send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("!= buf.id()" in f for f in rep.findings))

    def test_recv_write_id_regression_detected(self):
        buf = RecvArray(1)
        buf._id = 5
        buf._array[0] = 0.0
        buf._array[-1] = 2.0
        rep = inspect_buffer(buf, Field.SHUTDOWN, send=False)
        self.assertFalse(rep.ok)
        self.assertTrue(any("went backwards" in f for f in rep.findings))

    def test_recv_last_write_id_baseline(self):
        buf = RecvArray(1)
        buf._array[0] = 0.0
        buf._array[-1] = 2.0
        ctx = InspectContext(last_write_id=10)
        rep = inspect_buffer(buf, Field.SHUTDOWN, ctx=ctx, send=False)
        self.assertFalse(rep.ok)
        self.assertTrue(any("ctx.last_write_id" in f for f in rep.findings))

    def test_expected_write_id_mismatch(self):
        buf = SendArray(1)
        _publish(buf, [0.0])
        ctx = InspectContext(expected_write_id=99)
        rep = inspect_buffer(buf, Field.SHUTDOWN, ctx=ctx, send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("expected_write_id" in f for f in rep.findings))

    def test_nan_in_data_after_publish_is_finding(self):
        buf = SendArray(3)
        buf._next_write_id()    # publish without setting values: data stays NaN
        rep = inspect_buffer(buf, Field.NONANT,
                             ctx=InspectContext(nonant_count=3), send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("contains NaN" in f for f in rep.findings))

    def test_inf_in_data_is_finding(self):
        buf = SendArray(3)
        buf[0] = 1.0
        buf[1] = np.inf
        buf[2] = 3.0
        buf._next_write_id()
        rep = inspect_buffer(buf, Field.NONANT,
                             ctx=InspectContext(nonant_count=3), send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("contains inf" in f for f in rep.findings))


# ---- per-field checks -------------------------------------------------------


class TestShutdownChecks(unittest.TestCase):

    def test_legit_shutdown_passes(self):
        buf = SendArray(1)
        _publish(buf, [1.0])
        rep = inspect_buffer(buf, Field.SHUTDOWN, send=True)
        self.assertTrue(rep.ok, msg=str(rep))

    def test_stomped_shutdown_caught(self):
        # data=1.0 but write_id stayed at 0: the suspected stomp signature
        buf = RecvArray(1)
        buf._array[0] = 1.0
        rep = inspect_buffer(buf, Field.SHUTDOWN, send=False)
        self.assertFalse(rep.ok)
        self.assertTrue(any("write_id==0" in f for f in rep.findings))

    def test_shutdown_value_not_in_set(self):
        buf = SendArray(1)
        buf[0] = 0.5
        buf._next_write_id()
        rep = inspect_buffer(buf, Field.SHUTDOWN, send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("expected 0.0 or 1.0" in f for f in rep.findings))

    def test_initial_state_passes(self):
        # data=NaN, write_id=0: canonical initial state; allowed
        buf = RecvArray(1)
        rep = inspect_buffer(buf, Field.SHUTDOWN, send=False)
        self.assertTrue(rep.ok, msg=str(rep))


class TestNonantChecks(unittest.TestCase):

    def test_length_mismatch_via_explicit_count(self):
        buf = SendArray(5)
        _publish(buf, [0.0] * 5)
        rep = inspect_buffer(buf, Field.NONANT,
                             ctx=InspectContext(nonant_count=7), send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("data length 5" in f for f in rep.findings))

    def test_length_mismatch_via_spbase_fallback(self):
        buf = SendArray(5)
        _publish(buf, [0.0] * 5)
        ctx = InspectContext(spbase=_FakeSP(nonant_length=7))
        rep = inspect_buffer(buf, Field.NONANT, ctx=ctx, send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("data length 5" in f for f in rep.findings))

    def test_explicit_count_takes_precedence_over_spbase(self):
        buf = SendArray(5)
        _publish(buf, [0.0] * 5)
        ctx = InspectContext(nonant_count=5, spbase=_FakeSP(nonant_length=7))
        rep = inspect_buffer(buf, Field.NONANT, ctx=ctx, send=True)
        self.assertTrue(rep.ok, msg=str(rep))

    def test_out_of_bounds_componentwise(self):
        buf = SendArray(4)
        _publish(buf, [0.5, 7.0, 2.0, -1.0])
        lo = np.array([0.0, 0.0, 0.0, 0.0])
        hi = np.array([5.0, 5.0, 5.0, 5.0])
        rep = inspect_buffer(buf, Field.NONANT,
                             ctx=InspectContext(nonant_count=4,
                                                nonant_lower=lo,
                                                nonant_upper=hi),
                             send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("below lower bound" in f for f in rep.findings))
        self.assertTrue(any("above upper bound" in f for f in rep.findings))

    def test_unpublished_nonant_skips_bounds_check(self):
        # Fresh SendArray: write_id=0, data=NaN. Bounds compare must NOT fire.
        buf = SendArray(3)
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([1.0, 1.0, 1.0])
        rep = inspect_buffer(buf, Field.NONANT,
                             ctx=InspectContext(nonant_count=3,
                                                nonant_lower=lo,
                                                nonant_upper=hi),
                             send=True)
        self.assertTrue(rep.ok, msg=str(rep))


class TestBoundsBufferChecks(unittest.TestCase):

    def test_lower_above_upper(self):
        buf = SendArray(3)
        _publish(buf, [0.0, 2.0, 5.0])   # lowers
        upper = np.array([1.0, 1.0, 6.0])
        rep = inspect_buffer(buf, Field.NONANT_LOWER_BOUNDS,
                             ctx=InspectContext(nonant_count=3,
                                                nonant_upper=upper),
                             send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("lower > upper" in f for f in rep.findings))

    def test_upper_below_lower(self):
        buf = SendArray(3)
        _publish(buf, [10.0, 0.5, 6.0])   # uppers
        lower = np.array([0.0, 1.0, 5.0])
        rep = inspect_buffer(buf, Field.NONANT_UPPER_BOUNDS,
                             ctx=InspectContext(nonant_count=3,
                                                nonant_lower=lower),
                             send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("upper < lower" in f for f in rep.findings))


class TestObjectiveChecks(unittest.TestCase):

    def test_inner_bound_length_1_ok(self):
        buf = SendArray(1)
        _publish(buf, [42.5])
        rep = inspect_buffer(buf, Field.OBJECTIVE_INNER_BOUND, send=True)
        self.assertTrue(rep.ok, msg=str(rep))

    def test_inner_bound_wrong_length_caught(self):
        buf = SendArray(3)
        _publish(buf, [1.0, 2.0, 3.0])
        rep = inspect_buffer(buf, Field.OBJECTIVE_INNER_BOUND, send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("data length 3 != 1" in f for f in rep.findings))


class TestBestXhatChecks(unittest.TestCase):

    def test_length_too_short(self):
        buf = SendArray(2)
        _publish(buf, [0.0, 1.0])
        rep = inspect_buffer(buf, Field.BEST_XHAT,
                             ctx=InspectContext(nonant_count=5), send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("BEST_XHAT data length 2" in f for f in rep.findings))

    def test_nonant_prefix_out_of_bounds(self):
        buf = SendArray(6)
        # First 3 are nonants, last 3 are per-scenario costs
        _publish(buf, [0.5, 10.0, 2.0, 100.0, 200.0, 300.0])
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([5.0, 5.0, 5.0])
        rep = inspect_buffer(buf, Field.BEST_XHAT,
                             ctx=InspectContext(nonant_count=3,
                                                nonant_lower=lo,
                                                nonant_upper=hi),
                             send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("BEST_XHAT nonant portion above upper bound" in f
                            for f in rep.findings))


# ---- report and verbose -----------------------------------------------------


class TestReportAndVerbose(unittest.TestCase):

    def test_verbose_populates_dump(self):
        buf = SendArray(1)
        _publish(buf, [1.0])
        rep = inspect_buffer(buf, Field.SHUTDOWN, send=True, verbose=True)
        self.assertIsNotNone(rep.dump)
        self.assertIn("SHUTDOWN", rep.dump)
        self.assertIn("logical_len", rep.dump)
        self.assertIn("padded_len", rep.dump)

    def test_non_verbose_dump_is_none(self):
        buf = SendArray(1)
        _publish(buf, [1.0])
        rep = inspect_buffer(buf, Field.SHUTDOWN, send=True, verbose=False)
        self.assertIsNone(rep.dump)

    def test_str_round_trip(self):
        buf = RecvArray(1)
        buf._array[0] = 1.0
        rep = inspect_buffer(buf, Field.SHUTDOWN, send=False, verbose=True)
        s = str(rep)
        self.assertIn("FAIL", s)
        self.assertIn("SHUTDOWN", s)

    def test_report_severity_ladders_to_error(self):
        r = Report()
        r.add("warn 1", severity="warn")
        self.assertEqual(r.severity, "warn")
        r.add("err 1", severity="error")
        self.assertEqual(r.severity, "error")
        # subsequent warn doesn't downgrade
        r.add("warn 2", severity="warn")
        self.assertEqual(r.severity, "error")


# ---- config flag wiring -----------------------------------------------------


class TestConfigFlagWiring(unittest.TestCase):
    """The CLI flag must register in Config and propagate to shared_options."""

    def test_default_false(self):
        from mpisppy.utils.config import Config
        cfg = Config()
        cfg.popular_args()
        self.assertFalse(cfg.inspect_buffers_on_shutdown)

    def test_propagated_to_shared_options(self):
        from mpisppy.utils.config import Config
        import mpisppy.utils.cfg_vanilla as vanilla
        cfg = Config()
        cfg.popular_args()
        cfg.solver_name = "gurobi"
        cfg.inspect_buffers_on_shutdown = True
        opts = vanilla.shared_options(cfg, is_hub=False)
        self.assertTrue(opts.get("inspect_buffers_on_shutdown"))

    def test_default_propagates_as_false(self):
        from mpisppy.utils.config import Config
        import mpisppy.utils.cfg_vanilla as vanilla
        cfg = Config()
        cfg.popular_args()
        cfg.solver_name = "gurobi"
        opts = vanilla.shared_options(cfg, is_hub=True)
        self.assertFalse(opts.get("inspect_buffers_on_shutdown"))


if __name__ == "__main__":
    unittest.main()
