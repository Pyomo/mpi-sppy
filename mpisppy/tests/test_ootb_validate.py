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
        self.policy["effort_scaling"]["_cold_start_guess"].append("not_a_key")
        self.assertTrue(self._fails(val.validate_static(self.policy)))

    def test_rho_setter_must_list_all_setters(self):
        self.policy["option_categories"]["rho_setter"]["superseded_by"] = ["--grad-rho"]
        self.assertTrue(self._fails(val.validate_static(self.policy)))

    def test_decision_layer_detects_broken_ef_gate(self):
        # If the rank floor is absurdly high, OOTB would never decompose; the
        # "large problem -> decompose" decision check must then fail.
        self.policy["ef_fallback"]["min_ranks_for_decomposition"] = 10**9
        self.assertTrue(self._fails(val.validate_decisions_synthetic(self.policy)))


if __name__ == "__main__":
    unittest.main()
