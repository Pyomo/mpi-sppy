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
        self.assertEqual(fit["seconds_per_effort_unit"], 1.0)

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
        self.assertIn("seconds_per_effort_unit", es)
        self.assertNotIn("_cold_start_guess", es)   # numbers are now data-tuned
        self.assertEqual(pol["policy_version"], "2026-07-01")

    def test_ef_budget_is_seconds_like(self):
        # budget = target seconds / seconds_per_effort_unit (~1), so it reads as
        # seconds rather than an opaque huge number.
        pol = cal.calibrated_policy(self.base, self.fit, [], "gurobi", "2026-07-01")
        self.assertEqual(pol["ef_fallback"]["ef_effort_budget"],
                         pol["ef_fallback"]["ef_target_seconds"])

    def test_calibrated_policy_passes_static_validation(self):
        # end-to-end: a fitted policy must still be well-formed.
        pol = cal.calibrated_policy(self.base, self.fit, [], "gurobi", "2026-07-01")
        fails = [c for c in val.validate_static(pol) if not c.ok]
        self.assertEqual([], fails,
                         msg="\n".join(f"{c.name}: {c.detail}" for c in fails))
        fails = [c for c in val.validate_decisions_synthetic(pol) if not c.ok]
        self.assertEqual([], fails,
                         msg="\n".join(f"{c.name}: {c.detail}" for c in fails))


if __name__ == "__main__":
    unittest.main()
