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
that ``--interrupt-W-oscillations`` drives the rho-reduction and slam action
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


class TestReducedRho(unittest.TestCase):
    def test_multiplies(self):
        self.assertEqual(wosc.reduced_rho(1.0, 0.5, 1e-3), 0.5)
        self.assertAlmostEqual(wosc.reduced_rho(0.1, 0.1, 1e-3), 0.01)

    def test_floors_at_min_rho(self):
        # below the floor after scaling -> clamped to min_rho (stays > 0)
        self.assertEqual(wosc.reduced_rho(1e-3, 0.5, 1e-3), 1e-3)
        self.assertEqual(wosc.reduced_rho(1e-6, 0.5, 1e-3), 1e-3)


class TestInterruptConfigValidation(unittest.TestCase):
    def test_action_required_and_known(self):
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({})
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({"action": "bogus"})

    def test_factor_must_reduce(self):
        for bad in (0.0, 1.0, 1.5, -0.5):
            with self.assertRaises(ValueError):
                wosc.validate_interrupt_config(
                    {"action": "rho_reduction",
                     "rho_reduction": {"factor": bad, "min_rho": 1e-3}})

    def test_min_rho_must_be_positive(self):
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config(
                {"action": "rho_reduction",
                 "rho_reduction": {"factor": 0.5, "min_rho": 0.0}})

    def test_slam_requires_directives_file(self):
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({"action": "slam"})
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({"action": "slam", "slam": {}})

    def test_both_requires_both_sections(self):
        # 'both' needs a valid slam directives file even with rho defaults
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config({"action": "both"})
        cfg = wosc.validate_interrupt_config(
            {"action": "both", "slam": {"directives_file": "d.csv"}})
        self.assertIn("rho_reduction", cfg)
        self.assertIn("slam", cfg)

    def test_trigger_defaults_and_validation(self):
        cfg = wosc.validate_interrupt_config({"action": "rho_reduction"})
        self.assertEqual(cfg["trigger"]["start_iter"], 5)
        self.assertEqual(cfg["trigger"]["iters_between_actions"], 3)
        self.assertEqual(cfg["rho_reduction"]["factor"], 0.5)
        with self.assertRaises(ValueError):
            wosc.validate_interrupt_config(
                {"action": "rho_reduction",
                 "trigger": {"iters_between_actions": 0}})

    def test_detect_block_parsed(self):
        cfg = wosc.validate_interrupt_config({
            "action": "rho_reduction",
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


@unittest.skipIf(not solver_available, "no MIP solver available")
class TestEndToEndInterrupt(unittest.TestCase):
    """--interrupt-W-oscillations drives the rho-reduction and slam actions.

    Uses only the interrupt flag (no --detect-W-oscillations) with an inline
    ``detect`` block, so it also exercises the "interruption implies detection"
    path.  Thresholds are set low so the detector flags the acreage nonants
    early; the slam directive targets a single crop and fixes it to its lower
    bound (0 acres), which is always feasible for farmer."""

    def test_interrupt_flag_reduces_rho_and_slams(self):
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
                    "trigger": {"min_scenarios_flagged": 1, "start_iter": 3,
                                "iters_between_actions": 1},
                    "rho_reduction": {"factor": 0.5, "min_rho": 1e-3},
                    "slam": {"directives_file": slam_csv},
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
            # and the monitor reduced rho on the flagged nonants.
            self.assertIn("Slammer: slammed", out)
            self.assertIn("DevotedAcreage[SUGAR_BEETS", out)
            self.assertIn("reduced rho on", out)


if __name__ == "__main__":
    unittest.main()
