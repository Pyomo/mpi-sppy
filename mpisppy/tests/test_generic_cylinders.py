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
from types import SimpleNamespace
from unittest.mock import patch

from mpisppy.generic.decomp import _get_rho_setter
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
                "--max-solver-threads", "1",
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


def _make_rho_cfg(**overrides):
    base = dict(
        default_rho=None,
        sep_rho=False,
        coeff_rho=False,
        sensi_rho=False,
        cg_hub=False,
        dualcg_hub=False,
        ph_xfeas_spoke=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _NoRhoSetterModule:
    pass


class TestGetRhoSetter(unittest.TestCase):
    """The default_rho-required check should fire only for PH-based cylinders."""

    def test_ph_hub_no_rho_raises(self):
        cfg = _make_rho_cfg()
        with self.assertRaises(RuntimeError):
            _get_rho_setter(_NoRhoSetterModule(), cfg)

    def test_cg_hub_alone_does_not_require_rho(self):
        cfg = _make_rho_cfg(cg_hub=True)
        self.assertIsNone(_get_rho_setter(_NoRhoSetterModule(), cfg))
        self.assertIsNone(cfg.default_rho)

    def test_dualcg_hub_alone_does_not_require_rho(self):
        cfg = _make_rho_cfg(dualcg_hub=True)
        self.assertIsNone(_get_rho_setter(_NoRhoSetterModule(), cfg))

    def test_cg_hub_with_ph_spoke_requires_rho(self):
        cfg = _make_rho_cfg(cg_hub=True, ph_xfeas_spoke=True)
        with self.assertRaises(RuntimeError):
            _get_rho_setter(_NoRhoSetterModule(), cfg)

    def test_module_rho_setter_satisfies_check(self):
        class M:
            @staticmethod
            def _rho_setter():
                pass
        cfg = _make_rho_cfg()
        self.assertIsNotNone(_get_rho_setter(M(), cfg))

    def test_sep_rho_sets_default(self):
        cfg = _make_rho_cfg(sep_rho=True)
        _get_rho_setter(_NoRhoSetterModule(), cfg)
        self.assertEqual(cfg.default_rho, 1)


class TestGenericCylindersCVaR(unittest.TestCase):
    """--cvar wires the risk-management transform into the generic driver."""

    @unittest.skipIf(not solver_available, "no MIP solver available")
    def test_cvar_ef_matches_direct_build(self):
        import re
        import pyomo.environ as pyo
        import mpisppy.utils.sputils as sputils
        import mpisppy.utils.cvar as cvar
        import mpisppy.tests.examples.farmer as farmer

        weight, alpha = 1.5, 0.75

        # reference EF-CVaR objective, built directly with the same parameters
        names = farmer.scenario_names_creator(3)
        creator = cvar.cvar_scenario_creator(
            farmer.scenario_creator, cvar_weight=weight, cvar_alpha=alpha)
        ef = sputils.create_EF(names, creator,
                               scenario_creator_kwargs={"num_scens": 3},
                               suppress_warnings=True)
        solver = pyo.SolverFactory(solver_name)
        if "persistent" in solver_name:
            solver.set_instance(ef)
        solver.solve(ef)
        ref_obj = pyo.value(ef.EF_Obj)

        # the same solve through the CLI driver
        argv = [
            "generic_cylinders",
            "--module-name", "mpisppy.tests.examples.farmer",
            "--num-scens", "3",
            "--EF",
            "--EF-solver-name", solver_name,
            "--cvar",
            "--cvar-weight", str(weight),
            "--cvar-alpha", str(alpha),
        ]
        captured = io.StringIO()
        with patch.object(sys, "argv", argv), patch("sys.stdout", captured):
            runpy.run_module("mpisppy.generic_cylinders", run_name="__main__")
        out = captured.getvalue()

        match = re.search(r"EF objective:\s*([-\d.eE+]+)", out)
        self.assertIsNotNone(match, f"no EF objective in output:\n{out}")
        self.assertAlmostEqual(float(match.group(1)), ref_obj, places=2)


if __name__ == "__main__":
    unittest.main()
