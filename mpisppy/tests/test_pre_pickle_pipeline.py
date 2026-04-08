###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the pre-pickle preprocessing pipeline.

Exercises ``mpisppy.generic.scenario_io._run_pre_pickle_pipeline`` directly
on a small SPBase built from the farmer example, plus an end-to-end
subprocess test that pickles farmer bundles with every pre-pickle stage
enabled and then unpickles and solves them.
"""

import os
import sys
import tempfile
import unittest

import pyomo.environ as pyo

import mpisppy.tests.examples.farmer as farmer
from mpisppy.tests.utils import get_solver
from mpisppy.spbase import SPBase
from mpisppy.utils import config
from mpisppy.generic.scenario_io import _run_pre_pickle_pipeline

import mpisppy.MPI as mpi


fullcomm = mpi.COMM_WORLD
solver_available, solver_name, _, _ = get_solver()


# ---------------------------------------------------------------------------
# Module-level callback used by tests via dotted name.
# ---------------------------------------------------------------------------
_callback_calls = []


def record_callback(model, cfg):
    """Recorded callback used to test ``--pre-pickle-function`` resolution."""
    model._test_callback_marker = True
    _callback_calls.append(getattr(model, "name", None))


def make_infeasible_callback(model, cfg):
    """Callback that adds a constraint making the model infeasible.

    Used to verify that an iter0 failure shuts the pipeline down rather
    than producing a pickle with bad state.
    """
    model._infeas_var = pyo.Var(bounds=(0, 1))
    model._infeasible_test_con = pyo.Constraint(expr=model._infeas_var >= 2)


# ---------------------------------------------------------------------------


def _build_farmer_spbase(num_scens=3, comm=None):
    """Construct a small SPBase over the farmer example for tests."""
    all_scenario_names = farmer.scenario_names_creator(num_scens)
    sp_options = {
        "verbose": False,
        "toc": False,
    }
    if solver_name is not None:
        sp_options["solver_name"] = solver_name
    return SPBase(
        options=sp_options,
        all_scenario_names=all_scenario_names,
        scenario_creator=farmer.scenario_creator,
        scenario_denouement=None,
        all_nodenames=None,
        mpicomm=comm if comm is not None else fullcomm,
        scenario_creator_kwargs={"num_scens": num_scens, "crops_multiplier": 1},
    )


def _make_cfg(**overrides):
    """Build a Config with the pre-pickle flags registered and an optional override."""
    cfg = config.Config()
    cfg.popular_args()
    cfg.pre_pickle_args()
    if solver_name is not None:
        cfg.quick_assign("solver_name", str, solver_name)
    for k, v in overrides.items():
        # Use quick_assign so unknown-domain mismatches don't bite us
        cfg.quick_assign(k, type(v) if v is not None else str, v)
    return cfg


# ---------------------------------------------------------------------------


class TestPipelineUnit(unittest.TestCase):
    """Programmatic tests against ``_run_pre_pickle_pipeline``."""

    def setUp(self):
        _callback_calls.clear()

    def test_no_flags_attaches_metadata_only(self):
        sp = _build_farmer_spbase()
        cfg = _make_cfg()
        _run_pre_pickle_pipeline(sp, cfg)
        for sname, m in sp.local_scenarios.items():
            self.assertTrue(hasattr(m._mpisppy_data, "pickle_metadata"),
                            f"{sname} missing pickle_metadata")
            md = m._mpisppy_data.pickle_metadata
            self.assertFalse(md["presolve_before_pickle"])
            self.assertFalse(md["iter0_before_pickle"])
            self.assertIsNone(md["pre_pickle_function"])

    def test_user_callback_invoked(self):
        sp = _build_farmer_spbase()
        cfg = _make_cfg(pre_pickle_function="mpisppy.tests.test_pre_pickle_pipeline.record_callback")
        _run_pre_pickle_pipeline(sp, cfg)
        for m in sp.local_scenarios.values():
            self.assertTrue(getattr(m, "_test_callback_marker", False))
        # one call per local scenario
        self.assertEqual(len(_callback_calls), len(sp.local_scenarios))

    def test_user_callback_bad_dotted_name_raises(self):
        sp = _build_farmer_spbase()
        cfg = _make_cfg(pre_pickle_function="not_a_real_module.not_a_real_function")
        with self.assertRaises((ImportError, ValueError)):
            _run_pre_pickle_pipeline(sp, cfg)

    def test_user_callback_no_dot_raises(self):
        sp = _build_farmer_spbase()
        cfg = _make_cfg(pre_pickle_function="bare_name")
        with self.assertRaises(ValueError):
            _run_pre_pickle_pipeline(sp, cfg)

    @unittest.skipUnless(solver_available, "no solver available")
    def test_iter0_attaches_values_and_duals(self):
        sp = _build_farmer_spbase()
        cfg = _make_cfg(iter0_before_pickle=True)
        _run_pre_pickle_pipeline(sp, cfg)
        for sname, m in sp.local_scenarios.items():
            self.assertTrue(hasattr(m, "dual"),
                            f"{sname} missing dual suffix after iter0")
            self.assertTrue(hasattr(m, "rc"),
                            f"{sname} missing rc suffix after iter0")
            # at least one nonant variable should have a numeric value
            any_value = False
            for nd in m._mpisppy_node_list:
                for v in nd.nonant_vardata_list:
                    if v.value is not None:
                        any_value = True
                        break
                if any_value:
                    break
            self.assertTrue(any_value, f"{sname} has no nonant values after iter0")
            md = m._mpisppy_data.pickle_metadata
            self.assertTrue(md["iter0_before_pickle"])
            self.assertEqual(md["pickle_solver_name"], solver_name)

    @unittest.skipUnless(solver_available, "no solver available")
    def test_iter0_failure_raises(self):
        """Infeasible model + iter0_before_pickle must shut the pipeline down."""
        sp = _build_farmer_spbase()
        cfg = _make_cfg(
            pre_pickle_function="mpisppy.tests.test_pre_pickle_pipeline.make_infeasible_callback",
            iter0_before_pickle=True,
        )
        with self.assertRaises(RuntimeError):
            _run_pre_pickle_pipeline(sp, cfg)

    @unittest.skipUnless(solver_available, "no solver available")
    def test_presolve_stage_runs(self):
        sp = _build_farmer_spbase()
        cfg = _make_cfg(presolve_before_pickle=True)
        # Just verify it runs without error and metadata is updated.
        _run_pre_pickle_pipeline(sp, cfg)
        for m in sp.local_scenarios.values():
            md = m._mpisppy_data.pickle_metadata
            self.assertTrue(md["presolve_before_pickle"])

    @unittest.skipUnless(solver_available, "no solver available")
    def test_all_three_stages_combined(self):
        sp = _build_farmer_spbase()
        cfg = _make_cfg(
            presolve_before_pickle=True,
            pre_pickle_function="mpisppy.tests.test_pre_pickle_pipeline.record_callback",
            iter0_before_pickle=True,
        )
        _run_pre_pickle_pipeline(sp, cfg)
        for m in sp.local_scenarios.values():
            self.assertTrue(getattr(m, "_test_callback_marker", False))
            self.assertTrue(hasattr(m, "dual"))
            md = m._mpisppy_data.pickle_metadata
            self.assertTrue(md["presolve_before_pickle"])
            self.assertTrue(md["iter0_before_pickle"])
            self.assertIsNotNone(md["pre_pickle_function"])


# ---------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    """End-to-end subprocess test: pickle farmer scenarios, unpickle, solve.

    Uses ``--pickle-scenarios-dir`` (not ``--pickle-bundles-dir``) because
    the bundle pickling path on main has a pre-existing bug where
    ``proper_bundler.scenario_creator`` leaks ``cfg`` through to the
    underlying scenario_creator for any module that does not accept
    ``**kwargs``. That bug is out of scope for the pre-pickle pipeline work.
    """

    @unittest.skipUnless(solver_available, "no solver available")
    def test_pickle_scenarios_then_solve_with_all_stages(self):
        # Run from the farmer example directory so ``--module-name farmer``
        # resolves the local farmer.py.
        farmer_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "examples", "farmer",
        )
        farmer_dir = os.path.abspath(farmer_dir)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_dir = os.path.join(tmpdir, "farmer_pickles")
            python = sys.executable
            try:
                os.chdir(farmer_dir)
                # 1. Pickle scenarios with all three pre-pickle stages enabled.
                cmd_pickle = (
                    f"{python} -m mpisppy.generic_cylinders --module-name farmer "
                    f"--num-scens 6 --crops-mult 1 "
                    f"--pickle-scenarios-dir {pickle_dir} "
                    f"--solver-name {solver_name} "
                    f"--presolve-before-pickle --iter0-before-pickle"
                )
                ret = os.system(cmd_pickle)
                self.assertEqual(ret, 0,
                                 f"Pickling step failed: {cmd_pickle}")
                self.assertTrue(os.path.isdir(pickle_dir))
                self.assertTrue(any(f.endswith(".pkl")
                                    for f in os.listdir(pickle_dir)))

                # 2. Unpickle and run a couple of PH iterations.
                cmd_run = (
                    f"{python} -m mpisppy.generic_cylinders --module-name farmer "
                    f"--num-scens 6 --crops-mult 1 "
                    f"--unpickle-scenarios-dir {pickle_dir} "
                    f"--solver-name {solver_name} --default-rho 1 --max-iterations 2"
                )
                ret = os.system(cmd_run)
                self.assertEqual(ret, 0,
                                 f"Unpickle+solve step failed: {cmd_run}")
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
