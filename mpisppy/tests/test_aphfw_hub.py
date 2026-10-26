###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the optional C++ FW-APH hub (--aphfw-hub).

The hub needs the compiled aphfw_cpp module, which a normal install does not
build, so most of this file checks that mpi-sppy is unaffected when it is
missing. The end-to-end run, and the MPI integration tests (the
aphfw_*_driver.py scripts, each run under mpiexec), are skipped unless
aphfw_cpp can be imported.
"""
import importlib.util
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mpisppy.utils.config as config
from mpisppy.generic.parsing import add_driver_args
from mpisppy.tests.utils import get_solver, subprocess_env

solver_available, solver_name, _, _ = get_solver()
aphfw_cpp_available = importlib.util.find_spec("aphfw_cpp") is not None
mpiexec_available = shutil.which("mpiexec") is not None


def _cfg(**values):
    cfg = config.Config()
    add_driver_args(cfg)
    for name, value in values.items():
        cfg[name] = value
    return cfg


class TestAphfwOptional(unittest.TestCase):
    """Without the C++ build, nothing but --aphfw-hub notices."""

    def test_import_does_not_load_aphfw_cpp(self):
        code = ("import sys; import mpisppy.opt.aphfw, mpisppy.generic.hub, "
                "mpisppy.cylinders.hub, mpisppy.utils.cfg_vanilla; "
                "print('aphfw_cpp' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], check=True,
                             capture_output=True, text=True,
                             env=dict(os.environ))
        self.assertEqual(out.stdout.strip().splitlines()[-1], "False")

    def test_flag_is_registered(self):
        cfg = _cfg()
        self.assertIn("aphfw_hub", cfg)
        self.assertFalse(cfg.aphfw_hub)
        self.assertIn("aphfw_hub", config.Config.HUBS)

    def test_missing_build_gives_build_instructions(self):
        from mpisppy.opt.aphfw import _import_aphfw_cpp
        # A None entry makes the import fail whether or not it is built.
        with patch.dict(sys.modules, {"aphfw_cpp": None}):
            with self.assertRaisesRegex(RuntimeError, "python -m mpisppy.aphfw_cpp.build"):
                _import_aphfw_cpp()

    def test_build_help_without_sources(self):
        # A pip-installed mpi-sppy has no aphfw_cpp sources to build from.
        from mpisppy.opt.aphfw import _build_help
        msg = _build_help(src_dir=os.path.join(os.sep, "no", "such", "dir"))
        self.assertIn("clone of the mpi-sppy repository", msg)
        self.assertNotIn("--solver-home", msg)

    def test_one_hub_only(self):
        cfg = _cfg(aphfw_hub=True, fwph_hub=True)
        with self.assertRaisesRegex(ValueError, "Only one hub"):
            cfg.checker()

    def test_rejects_integer_relax_then_enforce(self):
        from mpisppy.utils.cfg_vanilla import _check_aphfw_compat
        with self.assertRaisesRegex(RuntimeError, "integer-relax-then-enforce"):
            _check_aphfw_compat(_cfg(aphfw_hub=True,
                                     integer_relax_then_enforce=True))


@unittest.skipIf(not aphfw_cpp_available, "aphfw_cpp is not built")
@unittest.skipIf(not solver_available, "no MIP solver available")
class TestAphfwHubRun(unittest.TestCase):

    def test_farmer_hub_only(self):
        argv = [
            "generic_cylinders",
            "--module-name", "mpisppy.tests.examples.farmer",
            "--num-scens", "3",
            "--solver-name", solver_name,
            "--aphfw-hub",
            "--default-rho", "1",
            "--aphfw-max-cycles", "5",
        ]
        with patch.object(sys, "argv", argv):
            runpy.run_module("mpisppy.generic_cylinders", run_name="__main__")

    def test_solver_options_reach_cpp(self):
        # The C++ solves read mipGap, timeLimit and threads from JSON files the
        # hub writes; capture what it writes for each.
        import json

        import mpisppy.opt.aphfw as aphfw_module
        written = {}
        real_dump = json.dump

        def capture(obj, f, *args, **kwargs):
            written[os.path.basename(f.name)] = dict(obj)
            return real_dump(obj, f, *args, **kwargs)

        # sizes, not farmer: the mipgap options exist only for a model whose
        # inparser_adder calls cfg.mip_options().
        argv = [
            "generic_cylinders",
            "--module-name", "mpisppy.tests.examples.sizes.sizes",
            "--num-scens", "3",
            "--solver-name", solver_name,
            "--aphfw-hub",
            "--default-rho", "1",
            "--aphfw-max-cycles", "2",
            "--iter0-mipgap", "0.02",
            "--iterk-mipgap", "0.003",
            "--max-solver-threads", "2",
            "--aphfw-solve-time-limit", "77",
            "--solver-options", "mipgap=0.5",
        ]
        with patch.object(sys, "argv", argv), \
                patch.object(aphfw_module.json, "dump", capture), \
                self.assertLogs(aphfw_module.logger, "WARNING") as logs:
            runpy.run_module("mpisppy.generic_cylinders", run_name="__main__")
        self.assertEqual(written["iter0.json"],
                         {"timeLimit": 77.0, "threads": 2, "mipGap": 0.02})
        self.assertEqual(written["grad.json"],
                         {"timeLimit": 77.0, "threads": 2, "mipGap": 0.003})
        self.assertEqual(written["nl.json"], {"timeLimit": 77.0, "threads": 2})
        self.assertIn("--solver-options does not reach", "\n".join(logs.output))


# The MPI integration tests, ported from aph-fw's tests/: each is a driver
# script with a fixed rank count, run as its own mpiexec job.
_DRIVERS = {
    "aphfw_hub_only_driver": 3,
    "aphfw_wheelspinner_hub_only_driver": 3,
    "aphfw_wheelspinner_with_spokes_driver": 6,
    "aphfw_export_feed_rho_driver": 3,
    "aphfw_rho_from_python_driver": 3,
    "aphfw_omicron_in_opt_driver": 3,
    "aphfw_dynamic_rho_driver": 3,
    "aphfw_final_lagrangian_driver": 3,
}


@unittest.skipUnless(aphfw_cpp_available, "aphfw_cpp is not built")
@unittest.skipUnless(mpiexec_available, "mpiexec is not available")
class TestAphfwDrivers(unittest.TestCase):

    def _run_driver(self, name, np):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              name + ".py")
        with tempfile.TemporaryDirectory() as tmpdir:  # for solver logs
            result = subprocess.run(
                ["mpiexec", "-np", str(np), sys.executable, "-m", "mpi4py",
                 script],
                capture_output=True, text=True, timeout=1800, check=False,
                env=subprocess_env(), cwd=tmpdir)
        tail = (result.stdout + result.stderr)[-3000:]
        self.assertEqual(result.returncode, 0, f"{name} failed:\n{tail}")
        self.assertIn("PASS", result.stdout, f"{name} did not pass:\n{tail}")


for _name, _np in _DRIVERS.items():
    def _test(self, name=_name, np=_np):
        self._run_driver(name, np)
    setattr(TestAphfwDrivers, "test_" + _name.removesuffix("_driver"), _test)


if __name__ == "__main__":
    unittest.main()
