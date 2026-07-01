###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the W-oscillation detection extension (PR1).

The pure detector logic (zero-crossing counting, the W-vector signature, and the
recurrence tracker) is tested directly with no MPI; an end-to-end test confirms
the ``--detect-W-oscillations`` flag wires the extension in and writes the CSV.
"""
import csv
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


if __name__ == "__main__":
    unittest.main()
