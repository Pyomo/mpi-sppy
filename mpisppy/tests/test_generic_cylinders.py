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
import json
import os
import runpy
import sys
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
        lshaped_hub=False,
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


_USER_EXT_SOURCE = """
import abc
import json
import os
from mpisppy.extensions.extension import Extension
from mpisppy.extensions.xhatclosest import XhatClosest  # imported: not loaded


class _Helper:
    pass


class _AbstractBase(Extension, abc.ABC):
    @abc.abstractmethod
    def which(self):
        pass

    def post_everything(self):
        opts = self.opt.options.get(__name__)
        out = os.path.join(os.path.dirname(__file__),
                           f"{__name__}.{self.which()}.out")
        with open(out, "w") as f:
            json.dump({"options": opts,
                       "iterations": self.opt._PHIter}, f)


class RecordingExt(_AbstractBase):
    def which(self):
        return "first"


class SecondExt(_AbstractBase):
    def which(self):
        return "second"
"""


class TestUserDefinedExtensions(unittest.TestCase):
    """--user-defined-extensions loads the Extension classes a module defines
    (issue #895)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        # a fresh module name per test, so an import from an earlier test
        # is never reused
        self.modname = f"user_ext_{self.id().rsplit('.', 1)[-1]}"
        self._write_module(self.modname, _USER_EXT_SOURCE)
        self._saved_path = list(sys.path)
        self._saved_modules = set(sys.modules)

    def tearDown(self):
        sys.path[:] = self._saved_path
        for name in set(sys.modules) - self._saved_modules:
            if name.startswith(self.modname):
                del sys.modules[name]
        self._tmp.cleanup()

    def _write_module(self, relname, source):
        path = os.path.join(self.tmp, relname.replace(".", os.sep) + ".py")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(source)

    def _write_json(self, relname, options):
        path = os.path.join(self.tmp, relname.replace(".", os.sep) + ".json")
        with open(path, "w") as f:
            json.dump(options, f)

    def _load(self, name):
        from mpisppy.generic.extensions import load_user_extension_module
        return load_user_extension_module(name)

    def test_loads_defined_concrete_classes_only(self):
        sys.path.insert(0, self.tmp)
        mod_name, classes, options = self._load(self.modname)
        self.assertEqual(mod_name, self.modname)
        self.assertEqual([c.__name__ for c in classes],
                         ["RecordingExt", "SecondExt"])
        self.assertIsNone(options)

    def test_json_next_to_module(self):
        sys.path.insert(0, self.tmp)
        self._write_json(self.modname, {"answer": 42})
        cwd = os.getcwd()
        # the json is found by the module's location, not the cwd
        with tempfile.TemporaryDirectory() as other:
            os.chdir(other)
            try:
                _, _, options = self._load(self.modname)
            finally:
                os.chdir(cwd)
        self.assertEqual(options, {"answer": 42})

    def test_path_to_module(self):
        # tmp is not on sys.path; the directory part of the name puts it there
        self._write_json(self.modname, {"answer": 7})
        mod_name, classes, options = self._load(
            os.path.join(self.tmp, self.modname))
        self.assertEqual(mod_name, self.modname)
        self.assertEqual(len(classes), 2)
        self.assertEqual(options, {"answer": 7})

    def test_dotted_name_in_package(self):
        sys.path.insert(0, self.tmp)
        pkg = self.modname + "_pkg"
        self._write_module(pkg + ".impl", _USER_EXT_SOURCE)
        self._write_module(pkg + ".__init__",
                           "from .impl import RecordingExt  # noqa\n")
        self._write_json(pkg + ".impl", {"answer": 3})
        mod_name, classes, options = self._load(pkg + ".impl")
        self.assertEqual(mod_name, pkg + ".impl")
        self.assertEqual(len(classes), 2)
        self.assertEqual(options, {"answer": 3})
        # the package only re-exports the class, so it defines none
        with self.assertRaisesRegex(RuntimeError, "Could not find"):
            self._load(pkg)

    def test_no_extension_class_raises(self):
        sys.path.insert(0, self.tmp)
        name = self.modname + "_empty"
        self._write_module(
            name, "from mpisppy.extensions.extension import Extension\n")
        with self.assertRaisesRegex(RuntimeError, "Could not find"):
            self._load(name)

    def _run_farmer(self, user_exts):
        argv = [
            "generic_cylinders",
            "--module-name", "mpisppy.tests.examples.farmer",
            "--num-scens", "3",
            "--solver-name", solver_name,
            "--max-solver-threads", "1",
            "--max-iterations", "2",
            "--default-rho", "1",
            "--user-defined-extensions", user_exts,
        ]
        with patch.object(sys, "argv", argv):
            runpy.run_module("mpisppy.generic_cylinders",
                             run_name="__main__")

    @unittest.skipIf(not solver_available,
                     "no MIP solver available")
    def test_generic_cylinders_runs_user_extensions(self):
        sys.path.insert(0, self.tmp)
        self._write_json(self.modname, {"answer": 42})
        self._run_farmer(self.modname)
        for which in ("first", "second"):
            out = os.path.join(self.tmp, f"{self.modname}.{which}.out")
            self.assertTrue(os.path.exists(out),
                            f"the {which} user extension's post_everything "
                            "did not run")
            with open(out) as f:
                record = json.load(f)
            self.assertEqual(record["options"], {"answer": 42})
            self.assertEqual(record["iterations"], 2)

    @unittest.skipIf(not solver_available,
                     "no MIP solver available")
    def test_same_class_name_in_two_modules_is_refused(self):
        sys.path.insert(0, self.tmp)
        other = self.modname + "_other"
        self._write_module(other, _USER_EXT_SOURCE)
        with self.assertRaisesRegex(RuntimeError,
                                    "Two extension classes are named"):
            self._run_farmer(f"{self.modname} {other}")


class TestGenericCylindersCVaR(unittest.TestCase):
    """--cvar wires the risk-management transform into the generic driver."""

    @unittest.skipIf(not solver_available, "no MIP solver available")
    def test_cvar_ef_matches_direct_build(self):
        import re

        import pyomo.environ as pyo

        import mpisppy.tests.examples.farmer as farmer
        import mpisppy.utils.cvar as cvar
        import mpisppy.utils.sputils as sputils

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
