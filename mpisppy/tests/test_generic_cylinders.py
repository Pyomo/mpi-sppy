###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for generic_cylinders.py entry point."""
import csv
import glob
import io
import os
import sys
import runpy
import tempfile
import unittest
from unittest.mock import patch

from mpisppy.tests.utils import get_solver

solver_available, solver_name, _, _ = get_solver()


class TestGenericCylindersUsage(unittest.TestCase):
    """Test that generic_cylinders prints usage and exits when called with no args."""

    def test_no_args_prints_usage(self):
        captured = io.StringIO()
        with patch.object(sys, "argv", ["generic_cylinders"]), \
             patch("sys.stdout", captured):
            with self.assertRaises(SystemExit):
                runpy.run_module("mpisppy.generic_cylinders", run_name="__main__")
        output = captured.getvalue()
        self.assertIn("--module-name", output)
        self.assertIn("--smps-dir", output)
        self.assertIn("--mps-files-directory", output)


class TestGenericCylindersWtracker(unittest.TestCase):
    """End-to-end test that --wtracker actually attaches the extension and records W."""

    @unittest.skipIf(not solver_available,
                     "no MIP solver available")
    def test_wtracker_flag_records_W(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "wt")
            argv = [
                "generic_cylinders",
                "--module-name", "mpisppy.tests.examples.farmer",
                "--num-scens", "3",
                "--solver-name", solver_name,
                "--max-iterations", "6",
                "--default-rho", "1",
                "--wtracker",
                "--wtracker-file-prefix", prefix,
                "--wtracker-wlen", "3",
                "--wtracker-reportlen", "5",
            ]
            with patch.object(sys, "argv", argv):
                runpy.run_module("mpisppy.generic_cylinders",
                                 run_name="__main__")

            stdev_files = glob.glob(prefix + "_stdev_*.csv")
            self.assertTrue(stdev_files,
                            "wtracker did not write a stdev CSV — "
                            "extension was probably not wired in")

            with open(stdev_files[0]) as f:
                rows = list(csv.reader(f))
            # header + at least one (varname, scenname) row
            self.assertGreaterEqual(len(rows), 2,
                                    "wtracker stdev CSV has no data rows")
            self.assertEqual(rows[0], ["", "mean", "stdev"])
            # Each data row: index string, mean (float), stdev (non-negative float)
            for row in rows[1:]:
                idx, mean_s, stdev_s = row
                self.assertTrue(idx.startswith("('"),
                                f"unexpected index format: {idx!r}")
                mean = float(mean_s)
                stdev = float(stdev_s)
                self.assertTrue(mean == mean,  # not NaN
                                f"mean is NaN for {idx}")
                self.assertGreaterEqual(stdev, 0.0,
                                        f"negative stdev for {idx}")
            # Sanity: farmer has 3 nonants; with 3 scens we have 9 traces,
            # reportlen=5 caps rows at 5
            self.assertLessEqual(len(rows) - 1, 5)


if __name__ == "__main__":
    unittest.main()
