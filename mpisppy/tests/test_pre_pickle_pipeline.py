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


# Parse --python-args (extra args inserted after "python" in subcommands so
# the dedicated coverage CI job can capture coverage from the subprocesses
# spawned by TestEndToEnd). This mirrors the shim in test_pickle_bundle.py.
python_args = ""
_remaining = []
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i].startswith("--python-args="):
        python_args = sys.argv[_i].split("=", 1)[1]
    elif sys.argv[_i] == "--python-args" and _i + 1 < len(sys.argv):
        _i += 1
        python_args = sys.argv[_i]
    else:
        _remaining.append(sys.argv[_i])
    _i += 1
sys.argv = [sys.argv[0]] + _remaining


fullcomm = mpi.COMM_WORLD
solver_available, solver_name, _, _ = get_solver()


# ---------------------------------------------------------------------------
# Module-level callback used by tests via dotted name. We assert against
# a per-model marker (not a module-level list) so the test still works
# when the file is run as __main__ (the dotted name then resolves to a
# *different* module object from the runner's __main__, and any global
# state in __main__ would not be observed).
# ---------------------------------------------------------------------------


def record_callback(model, cfg):
    """Recorded callback used to test ``--pre-pickle-function`` resolution."""
    model._test_callback_marker = True


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
        # The callback marks each model it sees; we verify at least one
        # local scenario was marked (rank-aware: one might have zero).
        marked = sum(1 for m in sp.local_scenarios.values()
                     if getattr(m, "_test_callback_marker", False))
        self.assertEqual(marked, len(sp.local_scenarios))

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
    """End-to-end subprocess tests: pickle farmer, unpickle, solve."""

    def _farmer_dir(self):
        return os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "examples", "farmer",
        ))

    @unittest.skipUnless(solver_available, "no solver available")
    def test_pickle_scenarios_then_solve_with_all_stages(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_dir = os.path.join(tmpdir, "farmer_pickles")
            python = sys.executable
            try:
                os.chdir(self._farmer_dir())
                cmd_pickle = (
                    f"{python} {python_args} -m mpisppy.generic_cylinders "
                    f"--module-name farmer "
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

                cmd_run = (
                    f"{python} {python_args} -m mpisppy.generic_cylinders "
                    f"--module-name farmer "
                    f"--num-scens 6 --crops-mult 1 "
                    f"--unpickle-scenarios-dir {pickle_dir} "
                    f"--solver-name {solver_name} --default-rho 1 --max-iterations 2"
                )
                ret = os.system(cmd_run)
                self.assertEqual(ret, 0,
                                 f"Unpickle+solve step failed: {cmd_run}")
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(solver_available, "no solver available")
    def test_iter0_from_pickle_skips_solve_loop(self):
        """Round-trip with --iter0-from-pickle: pickle, then unpickle and run.

        Asserts that the second run completes (so PHBase.Iter0 successfully
        consumed the pickled iter0 solution) and that the "Skipping" toc
        message appears in stdout.
        """
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_dir = os.path.join(tmpdir, "farmer_pickles_iter0")
            run_log = os.path.join(tmpdir, "run.log")
            python = sys.executable
            try:
                os.chdir(self._farmer_dir())
                cmd_pickle = (
                    f"{python} -m mpisppy.generic_cylinders --module-name farmer "
                    f"--num-scens 6 --crops-mult 1 "
                    f"--pickle-scenarios-dir {pickle_dir} "
                    f"--solver-name {solver_name} --iter0-before-pickle"
                )
                ret = os.system(cmd_pickle)
                self.assertEqual(ret, 0,
                                 f"Pickling step failed: {cmd_pickle}")

                cmd_run = (
                    f"{python} -m mpisppy.generic_cylinders --module-name farmer "
                    f"--num-scens 6 --crops-mult 1 "
                    f"--unpickle-scenarios-dir {pickle_dir} "
                    f"--solver-name {solver_name} --default-rho 1 "
                    f"--max-iterations 2 --iter0-from-pickle "
                    f"> {run_log} 2>&1"
                )
                ret = os.system(cmd_run)
                self.assertEqual(ret, 0,
                                 f"Unpickle+iter0_from_pickle step failed: "
                                 f"{cmd_run}")
                with open(run_log) as f:
                    log = f.read()
                self.assertIn("Skipping PHBase.Iter0 solve loop", log,
                              "Did not see iter0 skip message in run output")
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(solver_available, "no solver available")
    def test_iter0_from_pickle_without_pickled_iter0_fails(self):
        """If the pickle has no iter0 solution, --iter0-from-pickle must fail."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_dir = os.path.join(tmpdir, "farmer_pickles_no_iter0")
            run_log = os.path.join(tmpdir, "run.log")
            python = sys.executable
            try:
                os.chdir(self._farmer_dir())
                # Pickle WITHOUT --iter0-before-pickle
                cmd_pickle = (
                    f"{python} -m mpisppy.generic_cylinders --module-name farmer "
                    f"--num-scens 6 --crops-mult 1 "
                    f"--pickle-scenarios-dir {pickle_dir} "
                    f"--solver-name {solver_name}"
                )
                ret = os.system(cmd_pickle)
                self.assertEqual(ret, 0,
                                 f"Pickling step failed: {cmd_pickle}")

                cmd_run = (
                    f"{python} -m mpisppy.generic_cylinders --module-name farmer "
                    f"--num-scens 6 --crops-mult 1 "
                    f"--unpickle-scenarios-dir {pickle_dir} "
                    f"--solver-name {solver_name} --default-rho 1 "
                    f"--max-iterations 2 --iter0-from-pickle "
                    f"> {run_log} 2>&1"
                )
                ret = os.system(cmd_run)
                self.assertNotEqual(
                    ret, 0,
                    "--iter0-from-pickle should have hard-failed because the "
                    "pickle has no iter0 solution"
                )
                with open(run_log) as f:
                    log = f.read()
                self.assertIn("do not carry an iter0 solution from pickle time", log)
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(solver_available, "no solver available")
    def test_pickle_bundles_then_solve_with_all_stages(self):
        """Same coverage as above but exercising the proper-bundle path."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_dir = os.path.join(tmpdir, "farmer_bundles")
            python = sys.executable
            try:
                os.chdir(self._farmer_dir())
                cmd_pickle = (
                    f"{python} {python_args} -m mpisppy.generic_cylinders "
                    f"--module-name farmer "
                    f"--num-scens 6 --crops-mult 1 "
                    f"--pickle-bundles-dir {pickle_dir} --scenarios-per-bundle 3 "
                    f"--solver-name {solver_name} "
                    f"--presolve-before-pickle --iter0-before-pickle"
                )
                ret = os.system(cmd_pickle)
                self.assertEqual(ret, 0,
                                 f"Pickling step failed: {cmd_pickle}")
                self.assertTrue(os.path.isdir(pickle_dir))
                self.assertTrue(any(f.endswith(".pkl")
                                    for f in os.listdir(pickle_dir)))

                cmd_run = (
                    f"{python} {python_args} -m mpisppy.generic_cylinders "
                    f"--module-name farmer "
                    f"--num-scens 6 --crops-mult 1 "
                    f"--unpickle-bundles-dir {pickle_dir} --scenarios-per-bundle 3 "
                    f"--solver-name {solver_name} --default-rho 1 --max-iterations 2"
                )
                ret = os.system(cmd_run)
                self.assertEqual(ret, 0,
                                 f"Unpickle+solve step failed: {cmd_run}")
            finally:
                os.chdir(cwd)


# ---------------------------------------------------------------------------


class TestADMMBundlePipeline(unittest.TestCase):
    """Resolved decision #8: pre-pickle pipeline must not break ADMM models.

    Builds a small ``AdmmBundler`` over the stoch_distr test example,
    constructs an ``SPBase`` whose scenarios are ADMM bundles (each with
    consensus constraints, augmented scenario tree, and a per-scenario
    variable_probability list), and runs ``_run_pre_pickle_pipeline``
    with all stages enabled. The test passes if the pipeline completes
    and the iter0 solve produces values on the bundle without tripping
    over consensus structure.
    """

    @unittest.skipUnless(solver_available, "no solver available")
    def test_pipeline_runs_on_admm_bundles(self):
        # Imports inside the test so the module loads even if stoch_distr
        # is missing on a given environment.
        import mpisppy.tests.examples.stoch_distr.stoch_distr as stoch_distr
        from mpisppy.utils.admm_bundler import AdmmBundler

        admm_cfg = config.Config()
        stoch_distr.inparser_adder(admm_cfg)
        admm_cfg.num_stoch_scens = 4
        admm_cfg.num_admm_subproblems = 2

        admm_subproblem_names = stoch_distr.admm_subproblem_names_creator(
            admm_cfg.num_admm_subproblems)
        stoch_scenario_names = stoch_distr.stoch_scenario_names_creator(
            admm_cfg.num_stoch_scens)
        scenario_creator_kwargs = stoch_distr.kw_creator(admm_cfg)
        consensus_vars = stoch_distr.consensus_vars_creator(
            admm_subproblem_names, stoch_scenario_names[0],
            **scenario_creator_kwargs)

        bundler = AdmmBundler(
            module=stoch_distr,
            scenarios_per_bundle=admm_cfg.num_stoch_scens,  # full bundling
            admm_subproblem_names=admm_subproblem_names,
            stoch_scenario_names=stoch_scenario_names,
            consensus_vars=consensus_vars,
            combining_fn=stoch_distr.combining_names,
            split_fn=stoch_distr.split_admm_stoch_subproblem_scenario_name,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        bundle_names = bundler.bundle_names_creator()
        self.assertGreater(len(bundle_names), 0)

        # SPBase wants a scenario_creator with signature fn(name, **kwargs);
        # AdmmBundler.scenario_creator already has that.
        sp_options = {
            "verbose": False,
            "toc": False,
            "solver_name": solver_name,
            # Bundles produce per-scenario nonant names that differ across
            # bundles; ADMM also requires per-bundle variable_probability,
            # so disable both spbase consistency checks for this test.
            "turn_off_names_check": True,
            "do_not_check_variable_probabilities": True,
        }
        sp = SPBase(
            options=sp_options,
            all_scenario_names=bundle_names,
            scenario_creator=bundler.scenario_creator,
            scenario_denouement=None,
            all_nodenames=None,
            mpicomm=fullcomm,
            scenario_creator_kwargs={},
            variable_probability=bundler.var_prob_list,
        )

        # Now run the pre-pickle pipeline with all three stages enabled.
        cfg = _make_cfg(
            presolve_before_pickle=True,
            pre_pickle_function="mpisppy.tests.test_pre_pickle_pipeline.record_callback",
            iter0_before_pickle=True,
        )
        _run_pre_pickle_pipeline(sp, cfg)

        # Each local bundle should now have:
        # - the user callback marker
        # - dual + rc suffixes
        # - pickle metadata reflecting all three stages
        # - at least one nonant variable with a numeric value
        for bname, bundle in sp.local_scenarios.items():
            self.assertTrue(getattr(bundle, "_test_callback_marker", False),
                            f"{bname} missing user callback marker")
            self.assertTrue(hasattr(bundle, "dual"))
            self.assertTrue(hasattr(bundle, "rc"))
            md = bundle._mpisppy_data.pickle_metadata
            self.assertTrue(md["presolve_before_pickle"])
            self.assertTrue(md["iter0_before_pickle"])
            self.assertIsNotNone(md["pre_pickle_function"])

            any_value = False
            for nd in bundle._mpisppy_node_list:
                for v in nd.nonant_vardata_list:
                    if v.value is not None:
                        any_value = True
                        break
                if any_value:
                    break
            self.assertTrue(any_value,
                            f"{bname} has no nonant values after iter0")


if __name__ == "__main__":
    unittest.main()
