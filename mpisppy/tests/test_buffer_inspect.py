###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy.debug_utils.buffer_inspect."""

import types
import unittest
import warnings
from types import SimpleNamespace

import numpy as np

from mpisppy.cylinders.spcommunicator import RecvArray, SendArray
from mpisppy.cylinders.spoke import Spoke
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
        self.assertTrue(any("only 1.0 is ever published" in f for f in rep.findings))

    def test_shutdown_zero_is_rejected(self):
        # No producer ever writes 0.0 -- it would only appear via a stomp,
        # an RMA race, or a producer bug. Must be flagged.
        buf = SendArray(1)
        buf[0] = 0.0
        buf._next_write_id()
        rep = inspect_buffer(buf, Field.SHUTDOWN, send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("data[0]=0.0" in f for f in rep.findings))

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

    def test_multi_scenario_buffer_passes(self):
        # Publisher with K scenarios publishes a NONANT buffer of length
        # nonant_count * K. The checker must accept any positive multiple.
        buf = SendArray(24)  # e.g., 4 scenarios * 6 nonants
        _publish(buf, [0.0] * 24)
        rep = inspect_buffer(buf, Field.NONANT,
                             ctx=InspectContext(nonant_count=6), send=True)
        self.assertTrue(rep.ok, msg=str(rep))

    def test_non_multiple_length_caught(self):
        buf = SendArray(10)  # 10 is not a multiple of 6
        _publish(buf, [0.0] * 10)
        rep = inspect_buffer(buf, Field.NONANT,
                             ctx=InspectContext(nonant_count=6), send=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("not a positive multiple" in f for f in rep.findings))

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


# ---- integration with Spoke.got_kill_signal --------------------------------


def _make_spoke_stub(shutdown_buf, *, inspect_on=True,
                     extra_recv=None, send=None, nonant_length=None):
    """Build a duck-typed stub sufficient to drive Spoke.got_kill_signal.

    extra_recv: optional dict mapping (Field, origin) -> RecvArray that the
        sweep will visit in addition to the SHUTDOWN entry.
    send: optional dict mapping Field -> SendArray for the send-side sweep.
    nonant_length: if set, exposed via stub.opt.nonant_length so checkers
        that fall back to spbase pick it up.
    """
    stub = SimpleNamespace()
    recv = {(Field.SHUTDOWN, 0): shutdown_buf}
    if extra_recv:
        recv.update(extra_recv)
    stub.receive_buffers = recv
    stub.send_buffers = dict(send) if send else {}
    stub._make_key = lambda field, origin: (field, origin)
    stub._split_key = lambda key: key
    # The real method copies from the MPI window into the buffer; for the
    # stub the buffer is already populated, so this is a no-op.
    stub.get_receive_buffer = lambda buf, field, origin, synchronize=True: True
    stub.opt = SimpleNamespace(
        options={"inspect_buffers_on_shutdown": inspect_on},
    )
    if nonant_length is not None:
        stub.opt.nonant_length = nonant_length
    stub.cylinder_rank = 0
    stub.strata_rank = 1
    stub.global_rank = 1
    stub.allreduce_or = lambda v: v
    # Bind the sweep helpers from the real class onto the stub so that
    # got_kill_signal can call self._inspect_buffers_on_shutdown(...).
    stub._inspect_buffers_on_shutdown = types.MethodType(
        Spoke._inspect_buffers_on_shutdown, stub)
    stub._warn_if_buffer_bad = types.MethodType(
        Spoke._warn_if_buffer_bad, stub)
    return stub


class TestSpokeGotKillSignalWarning(unittest.TestCase):
    """The print -> warnings.warn switch in Spoke.got_kill_signal."""

    def test_stomped_shutdown_emits_runtime_warning(self):
        # data=1.0 but write_id stayed at 0: the suspected stomp signature
        buf = RecvArray(1)
        buf._array[0] = 1.0
        stub = _make_spoke_stub(buf, inspect_on=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            fired = Spoke.got_kill_signal(stub)
        self.assertTrue(fired)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime_warnings), 1, msg=[str(w.message) for w in caught])
        self.assertIn("buffer_inspect", str(runtime_warnings[0].message))

    def test_legit_shutdown_emits_no_warning(self):
        # Properly published shutdown signal: data=1.0 with write_id>=1
        buf = RecvArray(1)
        buf._array[0] = 1.0
        buf._array[-1] = 1.0      # write_id slot
        buf._id = 1
        stub = _make_spoke_stub(buf, inspect_on=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fired = Spoke.got_kill_signal(stub)
        self.assertTrue(fired)

    def test_flag_off_suppresses_warning_on_stomped_buffer(self):
        # Inspector must not run when the flag is off, even with a bad buffer.
        buf = RecvArray(1)
        buf._array[0] = 1.0
        stub = _make_spoke_stub(buf, inspect_on=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fired = Spoke.got_kill_signal(stub)
        self.assertTrue(fired)

    def test_sweep_inspects_every_buffer(self):
        # SHUTDOWN legit + healthy NONANT recv + healthy NONANT send: no warnings.
        # Then add a stomped OBJECTIVE_INNER_BOUND recv: exactly one warning,
        # naming that field.
        good_shutdown = RecvArray(1)
        good_shutdown._array[0] = 1.0
        good_shutdown._array[-1] = 1.0
        good_shutdown._id = 1

        good_nonant_recv = RecvArray(3)
        for i, v in enumerate([0.0, 1.0, 2.0]):
            good_nonant_recv._array[i] = v
        good_nonant_recv._array[-1] = 1.0
        good_nonant_recv._id = 1

        good_nonant_send = SendArray(3)
        _publish(good_nonant_send, [0.0, 1.0, 2.0])

        bad_inner = RecvArray(1)
        bad_inner._array[0] = 0.5
        bad_inner._full_array[3] = 7.0  # write into padding region

        extra_recv = {
            (Field.NONANT, 1): good_nonant_recv,
            (Field.OBJECTIVE_INNER_BOUND, 1): bad_inner,
        }
        send = {Field.NONANT: good_nonant_send}
        stub = _make_spoke_stub(
            good_shutdown,
            inspect_on=True,
            extra_recv=extra_recv,
            send=send,
            nonant_length=3,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            fired = Spoke.got_kill_signal(stub)
        self.assertTrue(fired)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(
            len(runtime_warnings), 1,
            msg=[str(w.message) for w in runtime_warnings],
        )
        msg = str(runtime_warnings[0].message)
        self.assertIn("OBJECTIVE_INNER_BOUND", msg)
        self.assertIn("(recv)", msg)

    def test_sweep_healthy_run_emits_no_warnings(self):
        # Multiple healthy buffers (the false-positive regression guard).
        good_shutdown = RecvArray(1)
        good_shutdown._array[0] = 1.0
        good_shutdown._array[-1] = 1.0
        good_shutdown._id = 1

        good_nonant_recv = RecvArray(3)
        for i, v in enumerate([0.0, 1.0, 2.0]):
            good_nonant_recv._array[i] = v
        good_nonant_recv._array[-1] = 1.0
        good_nonant_recv._id = 1

        good_inner_recv = RecvArray(1)
        good_inner_recv._array[0] = 12.5
        good_inner_recv._array[-1] = 1.0
        good_inner_recv._id = 1

        good_nonant_send = SendArray(3)
        _publish(good_nonant_send, [0.0, 1.0, 2.0])

        good_outer_send = SendArray(1)
        _publish(good_outer_send, [10.0])

        stub = _make_spoke_stub(
            good_shutdown,
            inspect_on=True,
            extra_recv={
                (Field.NONANT, 1): good_nonant_recv,
                (Field.OBJECTIVE_INNER_BOUND, 1): good_inner_recv,
            },
            send={
                Field.NONANT: good_nonant_send,
                Field.OBJECTIVE_OUTER_BOUND: good_outer_send,
            },
            nonant_length=3,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fired = Spoke.got_kill_signal(stub)
        self.assertTrue(fired)


if __name__ == "__main__":
    unittest.main()
