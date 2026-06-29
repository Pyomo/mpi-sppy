###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""CI tests for the out-of-the-box (OOTB) interpreter wiring.

Solver-free: the OOTB decision/apply path and the base-tier probe only need to
*build* a scenario (no solve), so the whole gather_facts -> recommend ->
apply_decision -> configure flow runs in CI on the farmer example. The
environment rank count is supplied via --inspect-only N so the EF and
decomposition branches are both exercised deterministically without launching
mpiexec. (The solver-dependent run/measurement tiers are exercised on demand /
locally; see test_ootb_validate / test_ootb_calibrate.)
"""

import os
import sys
import unittest

import mpisppy.utils.config as config
from mpisppy.generic import parsing
from mpisppy.generic import out_of_the_box as ootb


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


_FARMER_DIR = os.path.join(_repo_root(), "examples", "farmer")


def _farmer_module():
    if _FARMER_DIR not in sys.path:
        sys.path.insert(0, _FARMER_DIR)
    import farmer
    return farmer


def _farmer_cfg(num_scens=6, **overrides):
    module = _farmer_module()
    cfg = config.Config()
    parsing.add_driver_args(cfg, module)
    cfg.num_scens = num_scens
    cfg.module_name = "farmer"
    for k, v in overrides.items():
        cfg[k] = v
    return cfg, module


class TestConfigureNoSolver(unittest.TestCase):
    """Drive configure() (probe + recommend + apply) without solving."""

    def setUp(self):
        self._argv = sys.argv
        sys.argv = ["prog", "--module-name", "farmer", "--num-scens", "6"]

    def tearDown(self):
        sys.argv = self._argv

    def test_base_few_ranks_picks_ef(self):
        cfg, module = _farmer_cfg(out_of_the_box="", inspect_only="1")
        state = ootb.configure(module, cfg)
        self.assertTrue(state.decision.run_ef)
        self.assertEqual(state.decision.ef_reason, "min_ranks")
        self.assertTrue(cfg.EF)                       # apply_decision set it
        # the probe populated a size profile (farmer is continuous)
        self.assertIsNotNone(state.facts.vars_cont)
        self.assertEqual(state.facts.vars_int, 0)

    def test_minus_tier_decomposes_by_count(self):
        cfg, module = _farmer_cfg(out_of_the_box_minus="", inspect_only="6")
        state = ootb.configure(module, cfg)
        self.assertFalse(state.decision.run_ef)
        self.assertTrue(cfg.lagrangian and cfg.xhatshuffle)
        self.assertIsNone(state.facts.vars_cont)      # minus: no probe
        # minus cannot bundle
        self.assertIsNone(cfg.get("scenarios_per_bundle"))

    def test_base_forced_decomposition_bundles(self):
        # user --lagrangian forces decomposition even though the problem is small
        sys.argv = sys.argv + ["--lagrangian"]
        cfg, module = _farmer_cfg(num_scens=60, out_of_the_box="",
                                  inspect_only="3", lagrangian=True)
        state = ootb.configure(module, cfg)
        self.assertFalse(state.decision.run_ef)
        spb = cfg.get("scenarios_per_bundle")
        self.assertIsNotNone(spb)                     # 60 scens -> bundles
        self.assertEqual(60 % int(spb), 0)

    def test_report_suggestions_runs(self):
        cfg, module = _farmer_cfg(out_of_the_box="", inspect_only="1")
        state = ootb.configure(module, cfg)
        ootb.report_suggestions(state)                # config-time suggestions
        self.assertIsInstance(state.decision.suggestions, list)


class TestInspectStandalone(unittest.TestCase):
    def test_verify_instantiation_and_standalone(self):
        cfg, module = _farmer_cfg()
        profile = ootb.verify_instantiation(module, cfg)
        self.assertEqual(set(profile),
                         {"vars_int", "vars_cont", "nonants_total", "nonants_int"})
        self.assertGreater(profile["vars_cont"], 0)
        ootb.inspect_only_standalone(module, cfg)     # prints, returns None


class TestApplyDecision(unittest.TestCase):
    def test_apply_sets_cfg_values(self):
        cfg, _ = _farmer_cfg()
        d = ootb.Decision()
        d.args = [ootb.ChosenArg("--lagrangian", None, "x"),
                  ootb.ChosenArg("--solver-name", "gurobi", "x"),
                  ootb.ChosenArg("--scenarios-per-bundle", "10", "x"),
                  ootb.ChosenArg("--rel-gap", "0.01", "x")]
        ootb.apply_decision(d, cfg)
        self.assertTrue(cfg.lagrangian)
        self.assertEqual(cfg.solver_name, "gurobi")
        self.assertEqual(cfg.scenarios_per_bundle, 10)
        self.assertAlmostEqual(cfg.rel_gap, 0.01)

    def test_ef_solver_name_carried_over(self):
        cfg, _ = _farmer_cfg()
        d = ootb.Decision(run_ef=True)
        d.args = [ootb.ChosenArg("--solver-name", "cplex", "x")]
        ootb.apply_decision(d, cfg)
        self.assertTrue(cfg.EF)
        self.assertEqual(cfg.EF_solver_name, "cplex")   # belt-and-suspenders


class TestCommandLineAndFlags(unittest.TestCase):
    def test_command_line_two_stage(self):
        facts = ootb.Facts("farmer", 3, set(), 6)
        d = ootb.Decision()
        d.args = [ootb.ChosenArg("--lagrangian", None, "x"),
                  ootb.ChosenArg("--solver-name", "gurobi", "x")]
        cl = d.command_line(facts)
        self.assertIn("--module-name farmer", cl)
        self.assertIn("--num-scens 6", cl)
        self.assertIn("--lagrangian", cl)
        self.assertIn("-np 3", cl)

    def test_command_line_multistage(self):
        facts = ootb.Facts("aircond", 3, set(), 6, multistage=True,
                           branching_factors=[3, 2])
        cl = ootb.Decision().command_line(facts)
        self.assertIn("--branching-factors 3 2", cl)
        self.assertNotIn("--num-scens", cl)

    def test_requested_and_effort_and_policy(self):
        cfg, _ = _farmer_cfg(out_of_the_box="")
        self.assertTrue(ootb.requested(cfg))
        effort, path = ootb.effort_and_policy(cfg)
        self.assertEqual(effort, "base")
        self.assertIsNone(path)
        cfg2, _ = _farmer_cfg()
        self.assertFalse(ootb.requested(cfg2))
        self.assertEqual(ootb.effort_and_policy(cfg2), (None, None))

    def test_two_tiers_is_an_error(self):
        cfg, _ = _farmer_cfg(out_of_the_box="", out_of_the_box_minus="")
        with self.assertRaises(RuntimeError):
            ootb.effort_and_policy(cfg)

    def test_user_flags_from_argv(self):
        saved = sys.argv
        try:
            sys.argv = ["p", "--lagrangian", "--max-iterations=50", "-q"]
            flags = ootb._user_flags()
            self.assertIn("--lagrangian", flags)
            self.assertIn("--max-iterations", flags)   # =value stripped
            self.assertNotIn("-q", flags)              # single dash ignored
        finally:
            sys.argv = saved


class TestSuggestionGenerators(unittest.TestCase):
    """Exercise each computed suggestion generator."""

    def setUp(self):
        self.policy = ootb.load_policy()

    def _msgs(self, d, facts, outcome=None):
        return ootb.make_suggestions(d, facts, self.policy, outcome)

    def test_few_ranks_and_minus(self):
        facts = ootb.Facts("m", 1, set(), 3, effort="minus")
        d = ootb.Decision(run_ef=True, ef_reason="min_ranks")
        self.assertTrue(any("only 1 MPI" in m for m in self._msgs(d, facts)))

    def test_no_persistent_and_more_ranks(self):
        facts = ootb.Facts("m", 6, set(), 10, effort="base")
        d = ootb.Decision(run_ef=False, chosen_solver="gurobi", num_cylinders=3)
        msgs = self._msgs(d, facts)
        self.assertTrue(any("persistent" in m for m in msgs))
        self.assertTrue(any("cylinders configured" in m for m in msgs))

    def test_linearized_prox(self):
        facts = ootb.Facts("m", 6, set(), 10, effort="base")
        d = ootb.Decision(run_ef=False, chosen_solver="cbc", num_cylinders=3)
        self.assertTrue(any("linearized" in m for m in self._msgs(d, facts)))

    def test_minus_no_bundling(self):
        facts = ootb.Facts("m", 6, set(), 100, effort="minus")
        d = ootb.Decision(run_ef=False, chosen_solver="gurobi", num_cylinders=3)
        self.assertTrue(any("minus" in m for m in self._msgs(d, facts)))

    def test_outcome_based(self):
        facts = ootb.Facts("m", 6, set(), 10, effort="base")
        d = ootb.Decision(run_ef=False, chosen_solver="gurobi", num_cylinders=3)
        outcome = {"converged": False, "iterations": 100, "rel_gap": 0.2}
        self.assertTrue(any("iterations" in m for m in self._msgs(d, facts, outcome)))

    def test_disabled_generator_skipped(self):
        facts = ootb.Facts("m", 1, set(), 3, effort="base")
        d = ootb.Decision(run_ef=True, ef_reason="min_ranks")
        pol = ootb.load_policy()
        pol["suggestions"]["disabled"] = ["_sg_ran_ef_few_ranks"]
        msgs = ootb.make_suggestions(d, facts, pol)
        self.assertFalse(any("only 1 MPI" in m for m in msgs))


if __name__ == "__main__":
    unittest.main()
