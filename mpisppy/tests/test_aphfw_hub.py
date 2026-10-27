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

The Pyomo solver (for whatever is solved on the Python side) is the one
get_solver() finds, unless $MPISPPY_APHFW_TEST_SOLVER names one; the driver
scripts use that variable too, else Pyomo's gurobi.  A solver named there that
is not available is an error, not a skip.
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

import numpy as np
import pyomo.environ as pyo

import mpisppy.utils.config as config
from mpisppy.generic.parsing import add_driver_args
from mpisppy.opt.aphfw import APHFWOpt
from mpisppy.tests.utils import (
    APHFW_TEST_SOLVER_VAR,
    aphfw_test_solver,
    get_solver,
    subprocess_env,
)

if os.environ.get(APHFW_TEST_SOLVER_VAR):
    solver_name = aphfw_test_solver()
    solver_available = pyo.SolverFactory(solver_name).available(exception_flag=False)
    if not solver_available:
        raise RuntimeError(f"{APHFW_TEST_SOLVER_VAR}={solver_name}, but Pyomo "
                           "cannot use that solver here")
else:
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


class _FakeCpp:
    """Stands in for aphfw_cpp.APHFWLibrary in the Python-side tests."""

    def __init__(self, n, w=0.0):
        self.n, self.w, self.bounds = n, w, None

    def set_all_local_nonant_bounds(self, lb, ub):
        self.bounds = (lb.copy(), ub.copy())

    def compute_final_lagrangian_bound(self):
        return -100.0

    def get_all_local_w(self, buf):
        buf[:] = self.w

    def _fill(self, buf):
        buf[:] = 1.0

    get_all_local_prox_x = get_all_local_xbar = get_all_local_rho = _fill
    get_all_local_nonants = _fill

    def get_current_cycle(self):
        return 1


def _fake_opt(cfg=None, nonants=()):
    """An object with what the APHFWOpt methods under test read."""
    from types import SimpleNamespace

    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(len(nonants)))
    for i, (lb, ub) in enumerate(nonants):
        m.x[i].setlb(lb)
        m.x[i].setub(ub)
    idx = [("ROOT", i) for i in range(len(nonants))]
    m._mpisppy_data = SimpleNamespace(
        nonant_indices={ndn_i: m.x[ndn_i[1]] for ndn_i in idx})
    m._mpisppy_model = SimpleNamespace(
        **{name: pyo.Param(idx, mutable=True, initialize=0.0)
           for name in ("W", "xbars", "rho")})
    for name in ("W", "xbars", "rho"):
        m.add_component("_" + name, getattr(m._mpisppy_model, name))
    opt = SimpleNamespace(options={"cfg": cfg, "PHIterLimit": 5},
                          local_scenarios={"s": m},
                          _local_nonant_length=len(nonants),
                          _scenario_dir="scen")
    opt._cpp = _FakeCpp(len(nonants))
    from mpisppy import MPI
    opt.mpicomm = MPI.COMM_WORLD
    for name in ("_get_option", "_bound_is_valid"):
        setattr(opt, name, getattr(APHFWOpt, name).__get__(opt))
    return opt, m


class TestAphfwPythonSide(unittest.TestCase):
    """The hub's Python side, with the C++ library stubbed out."""

    def test_rejects_variable_probability(self):
        from mpisppy.utils.cfg_vanilla import aphfw_hub
        with self.assertRaisesRegex(RuntimeError, "variable_probability"):
            aphfw_hub(_cfg(aphfw_hub=True), None, None, ["s"],
                      variable_probability=lambda s: None)

    def test_zero_float_options(self):
        # 0 for an option whose C++ default is 0 is left to that default; for
        # one the C++ requires positive it is an error, not the default.
        cfg = _cfg(aphfw_hub=True, aphfw_fw_tau=0.0,
                   aphfw_iter0_w_multiplier=0.0)
        opt, _ = _fake_opt(cfg)
        d = APHFWOpt._make_cpp_config(opt)
        self.assertNotIn("FW-tau", d)
        self.assertNotIn("iter0-w-multiplier", d)
        cfg.aphfw_primal_term_thresh = 0.0
        with self.assertRaisesRegex(ValueError, "--aphfw-primal-term-thresh"):
            APHFWOpt._make_cpp_config(opt)

    def test_max_iterations_sets_cycle_limit(self):
        opt, _ = _fake_opt(_cfg(aphfw_hub=True))
        self.assertEqual(APHFWOpt._make_cpp_config(opt)["aph-max-cycles"], 5)
        opt.options["PHIterLimit"] = 0
        with self.assertRaisesRegex(ValueError, "--max-iterations"):
            APHFWOpt._make_cpp_config(opt)

    def test_fixed_nonants_pushed_as_bounds(self):
        opt, m = _fake_opt(nonants=[(0, 10), (None, None), (0, 10)])
        m.x[2].fix(4.0)
        APHFWOpt._push_bounds_to_cpp(opt)
        lb, ub = opt._cpp.bounds
        np.testing.assert_array_equal(lb, [0, -np.inf, 4.0])
        np.testing.assert_array_equal(ub, [10, np.inf, 4.0])

    def test_sync_keeps_fixed_values(self):
        # The C++ prox x of a fixed nonant must not overwrite the value it is
        # fixed at, which _push_bounds_to_cpp then sends as both bounds.
        opt, m = _fake_opt(nonants=[(0, 10), (0, 10)])
        m.x[1].fix(4.0)
        APHFWOpt._sync_cpp_to_pyomo(opt)
        self.assertEqual((m.x[0].value, m.x[1].value), (1.0, 4.0))

    def test_write_back_keeps_fixed_values(self):
        opt, m = _fake_opt(nonants=[(0, 10), (0, 10)])
        m.x[1].fix(4.0)
        APHFWOpt._write_back_to_pyomo(opt)
        self.assertEqual((m.x[0].value, m.x[1].value), (1.0, 4.0))

    def test_final_lagrangian_not_published_with_fixed_nonants(self):
        from types import SimpleNamespace
        published = []
        opt, _ = _fake_opt(nonants=[(0, 1)])
        opt.spcomm = SimpleNamespace(publish_outer_bound=published.append)
        opt.best_bound_obj_val = None
        opt._can_update_best_bound = lambda: False
        self.assertIsNone(APHFWOpt._do_final_lagrangian(opt))
        self.assertEqual((published, opt.best_bound_obj_val), ([], None))
        opt._can_update_best_bound = lambda: True
        self.assertEqual(APHFWOpt._do_final_lagrangian(opt), -100.0)
        self.assertEqual(published, [-100.0])

    def test_rejects_converger(self):
        from mpisppy.generic.hub import build_hub_dict
        with self.assertRaisesRegex(RuntimeError, "converger"):
            build_hub_dict(_cfg(aphfw_hub=True), None, None, None, None,
                           ph_converger=object)

    def test_scenario_dir_must_be_shared(self):
        from mpisppy import MPI
        from mpisppy.opt.aphfw import _check_dir_shared
        with tempfile.TemporaryDirectory() as tmpdir:
            _check_dir_shared(MPI.COMM_WORLD, tmpdir)
            with patch("os.path.isdir", return_value=False), \
                    self.assertRaisesRegex(RuntimeError, "--aphfw-scenario-dir"):
                _check_dir_shared(MPI.COMM_WORLD, tmpdir)


@unittest.skipIf(not aphfw_cpp_available, "aphfw_cpp is not built")
@unittest.skipIf(not solver_available, "no MIP solver available")
class TestAphfwHubRun(unittest.TestCase):

    def _farmer_argv(self, *extra):
        return ["generic_cylinders",
                "--module-name", "mpisppy.tests.examples.farmer",
                "--num-scens", "3",
                "--solver-name", solver_name,
                "--aphfw-hub",
                "--default-rho", "1",
                *extra]

    def test_scenario_dir_used_and_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = self._farmer_argv("--max-iterations", "2",
                                     "--aphfw-scenario-dir", tmpdir)
            written = []
            real_init = APHFWOpt._make_cpp_config

            def record(self):
                written.append(self._scenario_dir)
                return real_init(self)

            with patch.object(sys, "argv", argv), \
                    patch.object(APHFWOpt, "_make_cpp_config", record):
                runpy.run_module("mpisppy.generic_cylinders", run_name="__main__")
            self.assertTrue(written[0].startswith(tmpdir), written)
            # the hub finalizes the C++ side, which removes the files
            self.assertEqual(os.listdir(tmpdir), [])

    def test_extension_hooks(self):
        # The hub calls every PH extension hook, so that, e.g., the reduced
        # costs fixer gets its per-iteration target and WTracker its Ws.
        from mpisppy.extensions.extension import Extension
        from mpisppy.spin_the_wheel import WheelSpinner
        from mpisppy.tests.examples.farmer import (
            scenario_creator,
            scenario_denouement,
        )
        from mpisppy.utils.cfg_vanilla import aphfw_hub

        called = {}
        hooks = ("pre_iter0", "iter0_post_solver_creation", "post_iter0",
                 "post_iter0_after_sync", "miditer", "enditer",
                 "enditer_after_sync", "post_everything")

        class Recorder(Extension):
            pass

        for hook in hooks:
            setattr(Recorder, hook, lambda self, hook=hook:
                    called.setdefault(hook, self.opt._PHIter))

        # A fixing made before iteration 0 must be in the C++ for its
        # startup solves: read the C++ bounds after the first cycle.
        def fix_first(self):
            called.setdefault("iter0_post_solver_creation", self.opt._PHIter)
            s = next(iter(self.opt.local_scenarios.values()))
            next(iter(s._mpisppy_data.nonant_indices.values())).fix(123.0)

        def bounds_after_cycle_1(self):
            called.setdefault("post_iter0", self.opt._PHIter)
            n = self.opt._local_nonant_length
            lb, ub = np.empty(n), np.empty(n)
            self.opt._cpp.get_all_local_nonant_bounds(lb, ub)
            called["first_bounds"] = (lb[0], ub[0])

        Recorder.iter0_post_solver_creation = fix_first
        Recorder.post_iter0 = bounds_after_cycle_1

        cfg = _cfg(aphfw_hub=True, solver_name=solver_name, default_rho=1.0,
                   max_iterations=3)
        hub_dict = aphfw_hub(cfg, scenario_creator, scenario_denouement,
                             [f"scen{i}" for i in range(3)],
                             scenario_creator_kwargs={"num_scens": 3},
                             ph_extensions=Recorder)
        WheelSpinner(hub_dict, []).spin()
        self.assertEqual(called.pop("first_bounds"), (123.0, 123.0))
        self.assertEqual(set(called), set(hooks))
        # the iteration-0 hooks see iteration 0, as under PH
        for hook in ("pre_iter0", "iter0_post_solver_creation", "post_iter0",
                     "post_iter0_after_sync"):
            self.assertEqual(called[hook], 0, hook)

    def test_refuses_maximization(self):
        from mpisppy.tests.examples.farmer import (
            scenario_creator,
            scenario_denouement,
        )
        from mpisppy.utils.cfg_vanilla import aphfw_hub
        cfg = _cfg(aphfw_hub=True, solver_name=solver_name, default_rho=1.0)
        hub_dict = aphfw_hub(cfg, scenario_creator, scenario_denouement,
                             [f"scen{i}" for i in range(3)],
                             scenario_creator_kwargs={"num_scens": 3,
                                                      "sense": pyo.maximize})
        opt = APHFWOpt(**hub_dict["opt_kwargs"])
        with self.assertRaisesRegex(RuntimeError, "only minimization"):
            opt.aphfw_main()

    def test_initial_outer_bound_reaches_hub(self):
        from mpisppy.utils.cfg_vanilla import aphfw_hub
        cfg = _cfg(aphfw_hub=True)
        cfg.quick_assign("initial_outer_bound", float, -1234.5)
        hub_dict = aphfw_hub(cfg, None, None, ["s"])
        self.assertEqual(
            hub_dict["hub_kwargs"]["options"]["initial_outer_bound"], -1234.5)

    def test_false_flag_reaches_cpp(self):
        # check-omicron-invariant defaults to true in C++; the bindings used to
        # drop a false bool, which left it on.  verbosity 1 echoes the options.
        argv = [sys.executable, "-m", "mpisppy.generic_cylinders",
                *self._farmer_argv("--max-iterations", "1",
                                   "--aphfw-verbosity", "1",
                                   "--disable-aphfw-check-omicron-invariant")[1:]]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(argv, capture_output=True, text=True,
                                    timeout=300, check=False,
                                    env=subprocess_env(), cwd=tmpdir)
        out = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, out[-3000:])
        self.assertIn("check-omicron-invariant: false", out)

    def test_farmer_hub_only(self):
        argv = [
            "generic_cylinders",
            "--module-name", "mpisppy.tests.examples.farmer",
            "--num-scens", "3",
            "--solver-name", solver_name,
            "--aphfw-hub",
            "--default-rho", "1",
            "--max-iterations", "5",
        ]
        with patch.object(sys, "argv", argv):
            runpy.run_module("mpisppy.generic_cylinders", run_name="__main__")

    def test_max_iterations_caps_cycles(self):
        # The hub stops at --max-iterations cycles, as other hubs stop at
        # that many iterations.  Farmer needs about 70 to converge.  Run it as
        # a subprocess so that a hang fails the test instead of stalling it.
        argv = [sys.executable, "-m", "mpisppy.generic_cylinders",
                *self._farmer_argv("--max-iterations", "7")[1:]]
        with tempfile.TemporaryDirectory() as tmpdir:  # for solver logs
            try:
                result = subprocess.run(
                    argv, capture_output=True, text=True, timeout=300,
                    check=False, env=subprocess_env(), cwd=tmpdir)
            except subprocess.TimeoutExpired:
                self.fail("the hub did not finish")
        out = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, out[-3000:])
        # the iteration count on the last line of the termination table
        lines = out.split("Statistics at termination")[-1].splitlines()
        last_iter = int(lines[2].split("]")[1].split()[0])
        self.assertEqual(last_iter, 7, out[-3000:])

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
            "--max-iterations", "2",
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

    def _run_driver(self, name, np, args=()):
        """Run one driver under mpiexec; return its stdout."""
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              name + ".py")
        with tempfile.TemporaryDirectory() as tmpdir:  # for solver logs
            result = subprocess.run(
                ["mpiexec", "-np", str(np), sys.executable, "-m", "mpi4py",
                 script, *args],
                capture_output=True, text=True, timeout=1800, check=False,
                env=subprocess_env(), cwd=tmpdir)
        tail = (result.stdout + result.stderr)[-3000:]
        self.assertEqual(result.returncode, 0, f"{name} failed:\n{tail}")
        self.assertIn("PASS", result.stdout, f"{name} did not pass:\n{tail}")
        return result.stdout

    def test_final_lagrangian_flag(self):
        # --aphfw-final-lagrangian runs the termination Lagrangian, and its
        # bound only competes with the published one, so it cannot lower it.
        def best_outer(out):
            line = [ln for ln in out.splitlines()
                    if ln.startswith("FINAL_LAGRANGIAN_RESULT")][-1]
            return float(line.split("best_outer=")[1])
        off = self._run_driver("aphfw_final_lagrangian_driver", 3)
        on = self._run_driver("aphfw_final_lagrangian_driver", 3,
                              ["--final-lagrangian"])
        self.assertNotIn("Termination Lagrangian bound is", off)
        self.assertIn("Termination Lagrangian bound is", on)
        self.assertGreaterEqual(best_outer(on), best_outer(off) - 1e-6)


    def test_reduced_costs_fixer(self):
        # The fixer needs the hub's iteration hooks for its targets and found
        # no _solver_plugin on the scenarios (AttributeError) once it fixed.
        args = ["--module-name", "mpisppy.tests.examples.sizes.sizes",
                "--num-scens", "3", "--solver-name", solver_name,
                "--aphfw-hub", "--default-rho", "1", "--max-iterations", "3",
                "--lagrangian", "--reduced-costs", "--rc-fixer",
                "--rc-fix-fraction-pre-iter0", "0.5", "--rc-verbose"]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["mpiexec", "-np", "3", sys.executable, "-m", "mpi4py",
                 "-m", "mpisppy.generic_cylinders", *args],
                capture_output=True, text=True, timeout=600, check=False,
                env=subprocess_env(), cwd=tmpdir)
        out = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, out[-3000:])
        self.assertIn("Total unique vars fixed by heuristic", out)


for _name, _np in _DRIVERS.items():
    def _test(self, name=_name, np=_np):
        self._run_driver(name, np)
    setattr(TestAphfwDrivers, "test_" + _name.removesuffix("_driver"), _test)


if __name__ == "__main__":
    unittest.main()
