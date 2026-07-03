###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the W-oscillation detection and interruption extension.

The pure detector logic (zero-crossing counting, the W-vector signature, and the
recurrence tracker) and the config validators (detection and interruption) are
tested directly with no MPI; end-to-end tests confirm that
``--detect-W-oscillations`` wires the extension in and writes the CSV (PR1) and
that ``--interrupt-W-oscillations`` drives the W-damping and slam action
layers through a real PH run (PR2).
"""
import contextlib
import csv
import io
import json
import os
import sys
import runpy
import tempfile
import unittest
from unittest.mock import patch

from mpisppy.extensions import w_oscillation as wosc
from mpisppy.tests.utils import get_solver

solver_available, solver_name, _, _ = get_solver()

_MASK64 = (1 << 64) - 1


class TestZeroCrossings(unittest.TestCase):
    def test_count_sign_changes_basic(self):
        self.assertEqual(wosc.count_sign_changes([1, -1, 1, -1], tol=1e-6), 3)
        self.assertEqual(wosc.count_sign_changes([1, 2, 3, 4], tol=1e-6), 0)
        # near-zero entries are skipped, not treated as a crossing
        self.assertEqual(wosc.count_sign_changes([1, 0.0, 1], tol=1e-6), 0)
        self.assertEqual(wosc.count_sign_changes([1, 1e-9, -1], tol=1e-6), 1)

    def test_oscillating_is_flagged(self):
        res = wosc.zero_crossings_detect([1, -1, 1, -1, 1, -1])
        self.assertTrue(res["flagged"])
        self.assertGreaterEqual(res["w_crossings"], 2)

    def test_converging_not_flagged(self):
        # monotone, damping series: no sign changes and a small damping ratio
        res = wosc.zero_crossings_detect([10, 5, 2, 1, 0.5, 0.2, 0.1])
        self.assertFalse(res["flagged"])
        self.assertEqual(res["w_crossings"], 0)
        self.assertEqual(res["diff_crossings"], 0)
        self.assertLess(res["diffs_ratio"], 0.2)

    def test_w_crossing_threshold_boundary(self):
        traj = [1, -1, 1]  # exactly 2 W sign changes
        self.assertTrue(wosc.zero_crossings_detect(
            traj, thresh_w_crossings=2, thresh_diff_crossings=99,
            thresh_diffs_ratio=99)["flagged"])
        self.assertFalse(wosc.zero_crossings_detect(
            traj, thresh_w_crossings=3, thresh_diff_crossings=99,
            thresh_diffs_ratio=99)["flagged"])

    def test_window_truncation(self):
        # only the last 2 points are kept -> no crossing seen
        traj = [1, -1, 1, 1]
        res = wosc.zero_crossings_detect(traj, window=2, thresh_w_crossings=1,
                                         thresh_diff_crossings=99,
                                         thresh_diffs_ratio=99)
        self.assertEqual(res["w_crossings"], 0)


class TestSignature(unittest.TestCase):
    def test_deterministic(self):
        a = wosc.signature_term(3, 1.2345, 1e-6)
        b = wosc.signature_term(3, 1.2345, 1e-6)
        self.assertEqual(a, b)
        self.assertNotEqual(a, wosc.signature_term(4, 1.2345, 1e-6))

    def test_quantization(self):
        # values within a quantum hash identically; across a quantum, differ
        self.assertEqual(wosc.signature_term(1, 1.0, 1e-3),
                         wosc.signature_term(1, 1.0 + 1e-9, 1e-3))
        self.assertNotEqual(wosc.signature_term(1, 1.0, 1e-3),
                            wosc.signature_term(1, 1.01, 1e-3))

    def test_distribution_independent_sum(self):
        # full sum == sum of partial sums (mod 2^64), and order-independent
        scen = [(0, 1.23), (1, -4.5), (2, 0.7), (3, 9.1)]
        q = 1e-6
        full = sum(wosc.signature_term(si, wv, q) for si, wv in scen) & _MASK64
        g1 = sum(wosc.signature_term(si, wv, q) for si, wv in scen[:1]) & _MASK64
        g2 = sum(wosc.signature_term(si, wv, q) for si, wv in scen[1:]) & _MASK64
        self.assertEqual((g1 + g2) & _MASK64, full)
        rev = sum(wosc.signature_term(si, wv, q)
                  for si, wv in reversed(scen)) & _MASK64
        self.assertEqual(rev, full)


class TestRecurrenceTracker(unittest.TestCase):
    def test_two_cycle_flagged(self):
        rt = wosc.RecurrenceTracker(window=10, min_period=2)
        out = [rt.push(s) for s in [10, 20, 10, 20]]
        self.assertEqual(out[0], (False, 0))
        self.assertEqual(out[1], (False, 0))
        self.assertEqual(out[2], (True, 2))  # 10 recurs, 20 differed in between

    def test_constant_not_flagged(self):
        rt = wosc.RecurrenceTracker(window=10, min_period=2)
        out = [rt.push(7) for _ in range(5)]
        self.assertTrue(all(not f for f, _ in out))  # convergence != cycle

    def test_monotone_not_flagged(self):
        rt = wosc.RecurrenceTracker(window=10, min_period=2)
        out = [rt.push(s) for s in [1, 2, 3, 4, 5]]
        self.assertTrue(all(not f for f, _ in out))

    def test_period_three(self):
        rt = wosc.RecurrenceTracker(window=10, min_period=2)
        out = [rt.push(s) for s in [1, 2, 3, 1, 2, 3]]
        # first recurrence is at the 4th push (1 repeats at distance 3)
        self.assertEqual(out[3], (True, 3))


class TestConfigValidation(unittest.TestCase):
    def test_requires_output_csv(self):
        with self.assertRaises(ValueError):
            wosc.validate_detect_config({"methods": {"zero_crossings": {}}})

    def test_requires_methods(self):
        with self.assertRaises(ValueError):
            wosc.validate_detect_config({"output_csv": "x.csv"})

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            wosc.validate_detect_config(
                {"output_csv": "x.csv", "methods": {"bogus": {}}})

    def test_bad_report_mode(self):
        with self.assertRaises(ValueError):
            wosc.validate_detect_config(
                {"output_csv": "x.csv", "report_mode": "sometimes",
                 "methods": {"zero_crossings": {}}})

    def test_defaults_filled(self):
        cfg = wosc.validate_detect_config(
            {"output_csv": "x.csv", "methods": {"zero_crossings": {"tol": 1e-3}}})
        zc = cfg["methods"]["zero_crossings"]
        self.assertEqual(zc["tol"], 1e-3)                 # override honored
        self.assertEqual(zc["thresh_w_crossings"], 2)     # default filled
        self.assertEqual(cfg["report_mode"], "on_detect")
        self.assertEqual(cfg["check_every"], 1)


@unittest.skipIf(not solver_available, "no MIP solver available")
class TestEndToEnd(unittest.TestCase):
    """--detect-W-oscillations attaches the extension and writes the CSV."""

    def test_detect_flag_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = os.path.join(tmp, "wosc.csv")
            ctrl = os.path.join(tmp, "detect.json")
            with open(ctrl, "w") as f:
                json.dump({
                    "output_csv": out_csv,
                    "warmup_iters": 2,
                    "report_mode": "every_check",
                    "min_scenarios_to_report": 1,
                    "methods": {
                        "zero_crossings": {"thresh_w_crossings": 1,
                                           "thresh_diff_crossings": 1,
                                           "thresh_diffs_ratio": 0.0},
                        "w_hash_recurrence": {"window": 6, "min_period": 2},
                    },
                }, f)

            argv = [
                "generic_cylinders",
                "--module-name", "mpisppy.tests.examples.farmer",
                "--num-scens", "3",
                "--solver-name", solver_name,
                "--max-solver-threads", "1",
                "--max-iterations", "8",
                "--default-rho", "1",
                "--detect-W-oscillations", ctrl,
            ]
            with patch.object(sys, "argv", argv):
                runpy.run_module("mpisppy.generic_cylinders",
                                 run_name="__main__")

            self.assertTrue(os.path.exists(out_csv),
                            "detection CSV not written — extension not wired in")
            with open(out_csv) as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], wosc._AGG_COLUMNS)
            for row in rows[1:]:
                self.assertEqual(len(row), len(wosc._AGG_COLUMNS))
                self.assertIn(row[3], wosc.VALID_METHODS)
                # full variable name, not an index tuple
                self.assertIn("DevotedAcreage", row[2])


class TestWDamped(unittest.TestCase):
    def test_rescales_increment(self):
        # Update_W applied W = w0 + rho*xdiff; damping to factor keeps only
        # factor of that increment.
        rho, xdiff, w0 = 2.0, 0.5, 10.0
        w_after_update = w0 + rho * xdiff                  # 11.0
        # factor=0.5 -> half the step -> w0 + 0.5*rho*xdiff = 10.5
        self.assertAlmostEqual(
            wosc.w_damped(w_after_update, rho, xdiff, 0.5), 10.5)

    def test_factor_zero_cancels_step(self):
        rho, xdiff, w0 = 3.0, -0.4, 1.0
        w_after_update = w0 + rho * xdiff
        # factor=0 fully undoes the increment -> back to w0
        self.assertAlmostEqual(
            wosc.w_damped(w_after_update, rho, xdiff, 0.0), w0)

    def test_zero_xdiff_is_inert(self):
        # at a fixed point x == xbar -> no change regardless of factor
        self.assertEqual(wosc.w_damped(5.0, 2.0, 0.0, 0.5), 5.0)


class TestInterruptConfigValidation(unittest.TestCase):
    def test_action_required_and_known(self):
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({})
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({"action": "bogus"})

    def test_factor_must_be_in_range(self):
        # 1.0 is a no-op and >1 / <0 are nonsense: all rejected.
        for bad in (1.0, 1.5, -0.5):
            with self.assertRaises(ValueError):
                wosc.validate_interrupt_config(
                    {"action": "w_damping", "w_damping": {"factor": bad}})
        # 0.0 is allowed (fully cancels the dual step that iteration).
        cfg = wosc.validate_interrupt_config(
            {"action": "w_damping", "w_damping": {"factor": 0.0}})
        self.assertEqual(cfg["w_damping"]["factor"], 0.0)

    def test_slam_requires_directives_file(self):
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({"action": "slam"})
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({"action": "slam", "slam": {}})

    def test_slam_cooldown_defaults_and_validation(self):
        # default fills in; explicit value is kept; < 1 is rejected.
        cfg = wosc.validate_interrupt_config(
            {"action": "slam", "slam": {"directives_file": "d.csv"}})
        self.assertEqual(cfg["slam"]["iters_between_slams"], 3)
        cfg = wosc.validate_interrupt_config(
            {"action": "slam",
             "slam": {"directives_file": "d.csv", "iters_between_slams": 10}})
        self.assertEqual(cfg["slam"]["iters_between_slams"], 10)
        for bad in (0, -2):
            with self.assertRaises(ValueError):
                wosc.validate_interrupt_config(
                    {"action": "slam",
                     "slam": {"directives_file": "d.csv",
                              "iters_between_slams": bad}})

    def test_slam_due_cooldown(self):
        # No slam yet: always due (start_iter alone gates the first slam).
        self.assertTrue(wosc.slam_due(5, None, 3))
        # Slammed at 7 with cooldown 3: not due at 8, 9; due again at 10.
        self.assertFalse(wosc.slam_due(8, 7, 3))
        self.assertFalse(wosc.slam_due(9, 7, 3))
        self.assertTrue(wosc.slam_due(10, 7, 3))
        # Cooldown 1 reproduces the every-iteration behavior.
        self.assertTrue(wosc.slam_due(8, 7, 1))

    def test_both_requires_both_sections(self):
        # 'both' needs a valid slam directives file even with w_damping defaults
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({"action": "both"})
        cfg = wosc.validate_interrupt_config(
            {"action": "both", "slam": {"directives_file": "d.csv"}})
        self.assertIn("w_damping", cfg)
        self.assertIn("slam", cfg)

    def test_trigger_defaults_and_validation(self):
        cfg = wosc.validate_interrupt_config({"action": "w_damping"})
        self.assertEqual(cfg["trigger"]["start_iter"], 5)
        # the global inter-action cadence knob was dropped (the slam-specific
        # cooldown lives in the slam block as iters_between_slams)
        self.assertNotIn("iters_between_actions", cfg["trigger"])
        self.assertEqual(cfg["w_damping"]["factor"], 0.5)
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config(
                {"action": "w_damping",
                 "trigger": {"min_scenarios_flagged": 0}})

    def test_detect_block_parsed(self):
        cfg = wosc.validate_interrupt_config({
            "action": "w_damping",
            "detect": {"output_csv": "x.csv",
                       "methods": {"zero_crossings": {}}},
        })
        self.assertIn("detect", cfg)
        self.assertEqual(cfg["detect"]["output_csv"], "x.csv")

    def test_default_detect_config(self):
        d = wosc.default_detect_config()
        self.assertEqual(d["output_csv"], "w_oscillations.csv")
        self.assertIn("zero_crossings", d["methods"])
        self.assertIn("w_hash_recurrence", d["methods"])

    def test_non_object_subblock_rejected(self):
        # a sub-block supplied as something other than a JSON object must give a
        # clear ValueError (naming the field), not a low-level TypeError from
        # the subsequent dict.update()
        for action, key, block in (
            ("w_damping", "trigger", [1, 2]),
            ("w_damping", "w_damping", "0.5"),
            ("slam", "slam", ["file.txt"]),
        ):
            with self.assertRaises(ValueError) as ctx:
                wosc.validate_interrupt_config({"action": action, key: block})
            self.assertIn(key, str(ctx.exception))


@unittest.skipIf(not solver_available, "no MIP solver available")
class TestEndToEndInterrupt(unittest.TestCase):
    """--interrupt-W-oscillations drives the W-damping and slam actions.

    Uses only the interrupt flag (no --detect-W-oscillations) with an inline
    ``detect`` block, so it also exercises the "interruption implies detection"
    path.  Thresholds are set low so the detector flags the acreage nonants
    early; the slam directive targets a single crop and fixes it to its lower
    bound (0 acres), which is always feasible for farmer."""

    def test_interrupt_flag_damps_w_and_slams(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = os.path.join(tmp, "wosc.csv")
            slam_csv = os.path.join(tmp, "slam.csv")
            # farmer's crop names carry a group suffix (SUGAR_BEETS0, ...), so
            # match with a wildcard ('[' / ']' are literal in slam globs).
            with open(slam_csv, "w") as f:
                f.write("name,can_slam,directions,priority\n")
                f.write("DevotedAcreage[SUGAR_BEETS*],1,lb,1\n")
            ctrl = os.path.join(tmp, "interrupt.json")
            with open(ctrl, "w") as f:
                json.dump({
                    "action": "both",
                    "trigger": {"min_scenarios_flagged": 1, "start_iter": 3},
                    "w_damping": {"factor": 0.5},
                    "slam": {"directives_file": slam_csv,
                             "iters_between_slams": 5},
                    "detect": {
                        "output_csv": out_csv,
                        "warmup_iters": 2,
                        "report_mode": "every_check",
                        "methods": {
                            "zero_crossings": {"thresh_w_crossings": 1,
                                               "thresh_diff_crossings": 1,
                                               "thresh_diffs_ratio": 0.0},
                        },
                    },
                }, f)

            argv = [
                "generic_cylinders",
                "--module-name", "mpisppy.tests.examples.farmer",
                "--num-scens", "3",
                "--solver-name", solver_name,
                "--max-solver-threads", "1",
                "--max-iterations", "8",
                "--default-rho", "1",
                "--verbose",
                "--interrupt-W-oscillations", ctrl,
            ]
            buf = io.StringIO()
            with patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(buf):
                runpy.run_module("mpisppy.generic_cylinders",
                                 run_name="__main__")
            out = buf.getvalue()

            # Detection still happened (implied by interruption): CSV written.
            self.assertTrue(os.path.exists(out_csv),
                            "detection CSV not written under interrupt mode")
            with open(out_csv) as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], wosc._AGG_COLUMNS)
            self.assertTrue(any("DevotedAcreage" in r[2] for r in rows[1:]))

            # Both action layers fired: the Slammer slammed the targeted crop
            # and the monitor damped W on the flagged nonants.
            self.assertIn("Slammer: slammed", out)
            self.assertIn("DevotedAcreage[SUGAR_BEETS", out)
            self.assertIn("damped W on", out)
            # slam fixes at most one nonant per slam event
            self.assertIn("slammed 1 nonant(s)", out)
            # ... and the cooldown then suppresses further slams while damping
            # continues (iters_between_slams=5 > the remaining iterations, so
            # the run ends inside the cooldown).
            self.assertIn("slam cooling down", out)
            self.assertEqual(out.count("slammed 1 nonant(s)"), 1)

    def test_interrupt_without_request_writes_no_report(self):
        """A pure --interrupt-W-oscillations run (no --detect flag, no detect
        block) runs the detection engine to drive the actions but writes **no**
        cycling report CSV; the report is opt-in."""
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = os.path.join(tmp, "interrupt.json")
            with open(ctrl, "w") as f:
                json.dump({
                    "action": "w_damping",
                    "trigger": {"start_iter": 3},
                    "w_damping": {"factor": 0.5},
                }, f)

            argv = [
                "generic_cylinders",
                "--module-name", "mpisppy.tests.examples.farmer",
                "--num-scens", "3",
                "--solver-name", solver_name,
                "--max-solver-threads", "1",
                "--max-iterations", "6",
                "--default-rho", "1",
                "--verbose",
                "--interrupt-W-oscillations", ctrl,
            ]
            # The default detector's output_csv is a bare filename, so run from
            # the temp dir and confirm no such file is created there.
            cwd = os.getcwd()
            buf = io.StringIO()
            try:
                os.chdir(tmp)
                with patch.object(sys, "argv", argv), \
                        contextlib.redirect_stdout(buf):
                    runpy.run_module("mpisppy.generic_cylinders",
                                     run_name="__main__")
            finally:
                os.chdir(cwd)
            out = buf.getvalue()

            self.assertFalse(
                os.path.exists(os.path.join(tmp, "w_oscillations.csv")),
                "report CSV was written without an explicit detection request")
            # The opt-in report being off is reported in the verbose summary.
            self.assertIn("report disabled", out)


if __name__ == "__main__":
    unittest.main()
