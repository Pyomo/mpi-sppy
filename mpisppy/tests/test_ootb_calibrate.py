###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""CI test for the OOTB effort-calibration tool's PURE parts.

Only the fitting / policy-assembly logic is exercised here -- it is solver-free.
The measurement (timed example solves) needs a solver and is run on demand /
locally, not in CI. See doc/designs/out_of_the_box_design.md sec. 9.
"""

import copy
import json
import unittest

from mpisppy.generic import ootb_calibrate as cal
from mpisppy.generic import ootb_validate as val
from mpisppy.generic import out_of_the_box as ootb


def _synthetic_points(cont_coeff, int_weight, exponent, int_nonant_coeff):
    """Points generated from a known effort model (zero measurement noise)."""
    pts = []
    for (vc, vi, ni) in [(10, 0, 0), (5, 3, 2), (0, 8, 5), (20, 1, 1)]:
        for spb in (1, 2, 4, 8):
            cont = vc * spb
            nint = vi * spb
            seconds = (cont_coeff * cont + int_weight * (nint ** exponent)
                       + int_nonant_coeff * ni)
            pts.append({"vars_cont": vc, "vars_int": vi, "nonants_int": ni,
                        "spb": spb, "seconds": seconds})
    return pts


class TestFit(unittest.TestCase):
    def test_recovers_known_coefficients(self):
        truth = dict(cont_coeff=0.01, int_weight=0.5, exponent=2.0,
                     int_nonant_coeff=0.2)
        fit = cal.fit_effort_model(_synthetic_points(**truth))
        self.assertAlmostEqual(fit["cont_coeff"], 0.01, places=5)
        self.assertAlmostEqual(fit["int_weight"], 0.5, places=5)
        self.assertEqual(fit["int_exponent"], 2.0)
        self.assertAlmostEqual(fit["int_nonant_coeff"], 0.2, places=5)
        self.assertGreater(fit["r2"], 0.999)

    def test_requires_minimum_points(self):
        with self.assertRaises(ValueError):
            cal.fit_effort_model([{"vars_cont": 1, "vars_int": 0,
                                   "nonants_int": 0, "spb": 1, "seconds": 1.0}])

    def test_round_sig_preserves_tiny_values(self):
        # decimal rounding would zero this; significant-figure rounding keeps it.
        self.assertAlmostEqual(cal._round_sig(1.5618e-09), 1.5618e-09)
        self.assertEqual(cal._round_sig(0.0), 0.0)
        self.assertAlmostEqual(cal._round_sig(123456.789, 3), 123000.0)


class TestCalibratedPolicy(unittest.TestCase):
    def setUp(self):
        self.base = ootb.load_policy()
        self.fit = cal.fit_effort_model(_synthetic_points(
            cont_coeff=0.01, int_weight=0.5, exponent=2.0, int_nonant_coeff=0.2))

    def test_effort_scaling_replaced(self):
        pol = cal.calibrated_policy(self.base, self.fit, [], "gurobi", "2026-07-01")
        es = pol["effort_scaling"]
        self.assertAlmostEqual(es["cont_coeff"], 0.01, places=5)
        self.assertEqual(es["int_exponent"], 2.0)
        self.assertNotIn("_cold_start_guess", es)   # numbers are now data-tuned
        self.assertEqual(pol["policy_version"], "2026-07-01")

    def test_ef_budget_is_seconds_like(self):
        # the coefficients are fitted in seconds, so the budget IS the target
        # and reads as seconds rather than an opaque huge number.
        pol = cal.calibrated_policy(self.base, self.fit, [], "gurobi", "2026-07-01")
        self.assertEqual(pol["ef_fallback"]["ef_effort_budget"],
                         pol["ef_fallback"]["ef_target_seconds"])

    def test_ef_budget_stays_a_cold_start_guess(self):
        # Calibration fixes the UNITS of ef_effort_budget, not its magnitude:
        # it is ef_target_seconds (itself a guess) over a scale that is 1 by
        # construction. Listing the source but not the derived number would
        # read as though the derivation measured something.
        pol = cal.calibrated_policy(self.base, self.fit, [], "gurobi", "2026-07-01")
        guesses = pol["ef_fallback"]["_cold_start_guess"]
        self.assertIn("ef_effort_budget", guesses)
        self.assertIn("ef_target_seconds", guesses)

    def test_refuses_a_policy_with_no_ef_target_seconds(self):
        # Without a target there is nothing to convert, and ootb_validate
        # already rejects such a policy (ef_target_seconds must be > 0). Emitting
        # a file whose _comment and _cold_start_guess describe a conversion that
        # never happened would be worse than refusing.
        base = copy.deepcopy(self.base)
        base["ef_fallback"].pop("ef_target_seconds", None)
        with self.assertRaises(ValueError):
            cal.calibrated_policy(base, self.fit, [], "gurobi", "2026-07-01")

    def test_refuses_a_budget_that_rounds_to_zero(self):
        # A target below 0.5 rounds to 0, and the EF gate tests
        # `whole <= budget`, so EF-when-small would be silently off.
        base = copy.deepcopy(self.base)
        base["ef_fallback"]["ef_target_seconds"] = 0.4
        with self.assertRaises(ValueError):
            cal.calibrated_policy(base, self.fit, [], "gurobi", "2026-07-01")

    def test_stray_guess_entries_keep_their_input_order(self):
        # Orphan entries are preserved (so ootb_validate still flags them), and
        # must come back in input order -- iterating a set would make the
        # emitted file depend on PYTHONHASHSEED.
        base = copy.deepcopy(self.base)
        strays = ["zz_renamed", "aa_renamed", "mm_renamed"]
        base["ef_fallback"]["_cold_start_guess"] = (
            list(base["ef_fallback"]["_cold_start_guess"]) + strays)
        pol = cal.calibrated_policy(base, self.fit, [], "gurobi", "2026-07-01")
        got = pol["ef_fallback"]["_cold_start_guess"]
        self.assertEqual([g for g in got if g in strays], strays)

    def test_refuses_a_non_finite_or_over_large_target(self):
        # Symmetric with the scale. json accepts the bare token Infinity and an
        # arbitrarily long integer literal; inf > 0 is true, and a huge int
        # raises OverflowError inside the division (and inside math.isfinite,
        # which is why the guard converts under try rather than asking).
        for bad in (float("inf"), float("nan"), 10 ** 400, "120", 0, -5):
            with self.subTest(target=bad):
                base = copy.deepcopy(self.base)
                base["ef_fallback"]["ef_target_seconds"] = bad
                with self.assertRaises(ValueError):
                    cal.calibrated_policy(base, self.fit, [], "gurobi",
                                          "2026-07-01")

    def test_seconds_per_effort_unit_is_gone(self):
        # Retired: nothing read it, it was 1 by construction of the fit (the
        # coefficients are fitted in seconds), and carrying it invited prose
        # that claimed a conversion the file never performed.
        self.assertNotIn("seconds_per_effort_unit", self.fit)
        pol = cal.calibrated_policy(copy.deepcopy(self.base), self.fit,
                                    [], "gurobi", "2026-07-01")
        self.assertNotIn("seconds_per_effort_unit", pol["effort_scaling"])
        self.assertNotIn("seconds_per_effort_unit", json.dumps(pol))

    def test_shipped_policy_is_reproducible_by_the_calibrator(self):
        # The shipped file must be a fixed point of calibrated_policy for the
        # fit it records, so the prose in the artifact a user reads cannot
        # drift away from the code that writes it.
        shipped = ootb.load_policy()
        es, calib = shipped["effort_scaling"], shipped["effort_scaling"]["_calibration"]
        fit = {k: es[k] for k in ("cont_coeff", "int_weight", "int_exponent",
                                  "int_nonant_coeff")}
        fit.update(r2=calib["r2"], n_points=calib["n_points"])
        rebuilt = cal.calibrated_policy(copy.deepcopy(shipped), fit,
                                        [], calib["solver"], calib["date"])
        # Compare the WHOLE policy, not a hand-picked key list: the claim is
        # that nothing the calibrator writes can drift from the shipped file,
        # and a hand-picked list would miss the effort_scaling prose.
        self.assertEqual(json.dumps(shipped, sort_keys=True, indent=1),
                         json.dumps(rebuilt, sort_keys=True, indent=1))

    def test_calibrated_policy_passes_static_validation(self):
        # end-to-end: a fitted policy must still be well-formed.
        pol = cal.calibrated_policy(self.base, self.fit, [], "gurobi", "2026-07-01")
        fails = [c for c in val.validate_static(pol) if not c.ok]
        self.assertEqual([], fails,
                         msg="\n".join(f"{c.name}: {c.detail}" for c in fails))
        fails = [c for c in val.validate_decisions_synthetic(pol) if not c.ok]
        self.assertEqual([], fails,
                         msg="\n".join(f"{c.name}: {c.detail}" for c in fails))


class TestCalibrationSpecs(unittest.TestCase):
    def test_specs_use_larger_counts(self):
        specs = cal._calibration_specs()
        self.assertEqual({s["name"] for s in specs},
                         {"farmer", "sizes", "aircond"})
        farmer = next(s for s in specs if s["name"] == "farmer")
        self.assertEqual(farmer["scens"], {"num_scens": 16})
        self.assertEqual(cal._design_columns(
            {"vars_cont": 2, "vars_int": 3, "nonants_int": 4, "spb": 5}, 2.0),
            [10, (15) ** 2.0, 4])


if __name__ == "__main__":
    unittest.main()
