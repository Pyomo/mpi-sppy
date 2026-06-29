###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""CI gate for the out-of-the-box (OOTB) policy-file validator.

Runs only the solver-free, fast layers -- layer 1 (static schema) and the
synthetic-facts subset of layer 2 (pure recommend() decisions on hand-built
Facts) -- on the shipped policy file(s). Example instantiation, anything needing
a solver, and all of layer 3 run nightly / on demand / locally, NOT here. See
doc/designs/out_of_the_box_design.md sec. 8.
"""

import copy
import os
import unittest

from mpisppy.generic import ootb_validate as val
from mpisppy.generic import out_of_the_box as ootb


def _shipped_policy_files():
    d = ootb._policies_dir()
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.startswith("ootb_policy_") and f.endswith(".json")]


class TestStaticAndSyntheticGate(unittest.TestCase):
    """Every shipped policy must pass layers 1 + 2-synthetic cleanly."""

    def test_every_shipped_policy_passes(self):
        files = _shipped_policy_files()
        self.assertTrue(files, "no shipped OOTB policy files found")
        for path in files:
            with self.subTest(policy=os.path.basename(path)):
                report = val.run_validation(path)   # layers 1 + 2-synthetic only
                fails = [c for c in report["checks"] if not c["ok"]]
                self.assertTrue(
                    report["ok"],
                    msg="validation failures:\n" + "\n".join(
                        f"  [{c['layer']}] {c['name']}: {c['detail']}" for c in fails))

    def test_default_policy_static_layer(self):
        policy = ootb.load_policy()
        fails = [c for c in val.validate_static(policy) if not c.ok]
        self.assertEqual(
            [], fails,
            msg="\n".join(f"{c.name}: {c.detail}" for c in fails))

    def test_default_policy_decision_layer(self):
        policy = ootb.load_policy()
        fails = [c for c in val.validate_decisions_synthetic(policy) if not c.ok]
        self.assertEqual(
            [], fails,
            msg="\n".join(f"{c.name}: {c.detail}" for c in fails))


class TestValidFlags(unittest.TestCase):
    def test_known_flags_are_valid(self):
        flags = val.valid_flags()
        for f in ("--lagrangian", "--xhatshuffle", "--fwph", "--subgradient",
                  "--reduced-costs", "--xhatxbar", "--solver-name",
                  "--EF-solver-name", "--scenarios-per-bundle", "--grad-rho",
                  "--default-rho", "--rel-gap", "--max-iterations",
                  "--dynamic-rho-primal-crit", "--linearize-proximal-terms"):
            self.assertIn(f, flags)

    def test_decomposition_flags_are_real(self):
        flags = val.valid_flags()
        bad = sorted(f for f in ootb.DECOMPOSITION_FLAGS if f not in flags)
        self.assertEqual([], bad, msg=f"unknown DECOMPOSITION_FLAGS: {bad}")


class TestValidatorCatchesBadPolicies(unittest.TestCase):
    """The validator must actually FAIL on malformed policies (not just pass)."""

    def setUp(self):
        self.policy = copy.deepcopy(ootb.load_policy())

    def _fails(self, checks):
        return [c for c in checks if not c.ok]

    def test_missing_top_level_key(self):
        del self.policy["solver"]
        self.assertTrue(self._fails(val.validate_static(self.policy)))

    def test_bogus_option_flag(self):
        self.policy["option_categories"]["termination"]["flag"] = "--not-a-real-flag"
        self.assertTrue(self._fails(val.validate_static(self.policy)))

    def test_spoke_rung_not_a_real_flag(self):
        self.policy["spoke_ladder"]["rungs"][0]["flag"] = "--bogus-spoke"
        self.assertTrue(self._fails(val.validate_static(self.policy)))

    def test_cold_start_guess_names_unknown_key(self):
        # regression: a _cold_start_guess entry that is prose, not a real key.
        # (bundle_sizing stays hand-authored even after effort calibration.)
        self.policy["bundle_sizing"]["_cold_start_guess"].append("not_a_key")
        self.assertTrue(self._fails(val.validate_static(self.policy)))

    def test_rho_setter_must_list_all_setters(self):
        self.policy["option_categories"]["rho_setter"]["superseded_by"] = ["--grad-rho"]
        self.assertTrue(self._fails(val.validate_static(self.policy)))

    def test_decision_layer_detects_broken_ef_gate(self):
        # If the rank floor is absurdly high, OOTB would never decompose; the
        # "large problem -> decompose" decision check must then fail.
        self.policy["ef_fallback"]["min_ranks_for_decomposition"] = 10**9
        self.assertTrue(self._fails(val.validate_decisions_synthetic(self.policy)))


class TestExamplesLayer(unittest.TestCase):
    """The decision-on-real-models layer builds scenarios (no solve), so it runs
    here; the farmer checks must pass."""

    def test_validate_decisions_examples_farmer_passes(self):
        checks = val.validate_decisions_examples(ootb.load_policy())
        self.assertTrue(checks)
        farmer = [c for c in checks if "farmer" in c.name]
        self.assertTrue(farmer)
        self.assertTrue(all(c.ok for c in farmer),
                        msg="\n".join(f"{c.name}: {c.detail}"
                                      for c in farmer if not c.ok))


class TestHelpers(unittest.TestCase):
    def test_scen_cli(self):
        self.assertEqual(val._scen_cli({"scens": {"num_scens": 6}}),
                         ["--num-scens", "6"])
        self.assertEqual(
            val._scen_cli({"scens": {"branching_factors": [3, 2]}}),
            ["--branching-factors", "3 2"])     # one space-joined token

    def test_child_env_scrubs_mpi_vars(self):
        os.environ["OMPI_TESTVAR"] = "1"
        try:
            self.assertNotIn("OMPI_TESTVAR", val._child_env())
            self.assertIn("PATH", val._child_env())
        finally:
            del os.environ["OMPI_TESTVAR"]

    def test_example_models_paths_exist(self):
        for spec in val.example_models():
            self.assertTrue(os.path.isdir(spec["dir"]), spec["dir"])

    def test_parse_decompose_converged(self):
        out = ("[ 0.8] Terminating based on inter-cylinder relative gap  0.9%\n"
               "[ 0.8] Statistics at termination\n"
               "[ 0.8] Iter.  Best Bound  Best Incumbent  Rel. Gap  Abs. Gap\n"
               "[ 0.8]   99   -124.0   -123.0   0.910%   1.1\n")
        converged, iters, gap = val._parse_decompose(out)
        self.assertTrue(converged)
        self.assertEqual(iters, 99)
        self.assertAlmostEqual(gap, 0.0091, places=4)

    def test_parse_decompose_maxed_out(self):
        out = ("[ 9.0] Statistics at termination\n"
               "[ 9.0]  100   -10.0   -5.0   50.000%   5.0\n")
        converged, iters, gap = val._parse_decompose(out)
        self.assertFalse(converged)
        self.assertEqual(iters, 100)


class TestReportAndMain(unittest.TestCase):
    def test_format_report_covers_branches(self):
        report = {
            "policy_file": "p.json", "policy_version": "2026-06-28", "ok": False,
            "checks": [{"layer": "static", "name": "a", "ok": True, "detail": ""},
                       {"layer": "decision", "name": "b", "ok": False,
                        "detail": "boom"}],
            "runs": [{"example": "farmer", "mode": "EF", "returncode": 0,
                      "walltime": 0.5, "objective": -1.0, "rel_gap": None,
                      "iterations": None, "flagged": False, "flag_reason": ""},
                     {"example": "sizes", "mode": "decompose", "returncode": 0,
                      "walltime": 1.0, "objective": None, "rel_gap": 0.5,
                      "iterations": 100, "flagged": True,
                      "flag_reason": "maxed out"}],
        }
        text = val.format_report(report)
        self.assertIn("CHECK FAILURE", text)
        self.assertIn("FLAGGED RUNS", text)
        self.assertIn("OVERALL: FAIL", text)

    def test_main_passes_on_default_policy(self):
        self.assertEqual(val.main([]), 0)              # layers 1+2 on default

    def test_main_fails_on_missing_policy(self):
        self.assertEqual(val.main(["/no/such/policy.json"]), 1)


if __name__ == "__main__":
    unittest.main()
