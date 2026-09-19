###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the pure-logic helpers in mpisppy/confidence_intervals/ciutils.py.

These exercise the deterministic math and validation helpers that do not need a
solver or MPI: prime factorization, branching-factor construction/scaling,
node-list ordering validation, and numeric gap correction.
"""

import unittest

import pyomo.environ as pyo

import mpisppy.confidence_intervals.ciutils as ciutils


def _product_of_factors(factor_dict):
    """Reconstruct n from the {prime: multiplicity} dict that _prime_factors returns."""
    prod = 1
    for factor, mult in factor_dict.items():
        prod *= factor ** mult
    return prod


class _FakeNode:
    """Stand-in for scenario_tree.ScenarioNode exposing only what is_sorted reads."""
    def __init__(self, name, stage, parent_name):
        self.name = name
        self.stage = stage
        self.parent_name = parent_name


class Test_prime_factors(unittest.TestCase):
    def test_zero_is_special_cased(self):
        self.assertEqual(ciutils._prime_factors(0), {0: 1})

    def test_one_has_no_prime_factors(self):
        self.assertEqual(ciutils._prime_factors(1), {})

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            ciutils._prime_factors(-6)

    def test_prime_input(self):
        # a prime factors to itself with multiplicity one
        self.assertEqual(ciutils._prime_factors(7), {7: 1})

    def test_prime_power(self):
        # 8 == 2**3
        self.assertEqual(ciutils._prime_factors(8), {2: 3})

    def test_composite_multiplicities_and_product(self):
        # 360 = 2^3 * 3^2 * 5
        factors = ciutils._prime_factors(360)
        # multiplicities are the values; keys may be float due to internal /=
        self.assertEqual(sum(factors.values()), 6)
        self.assertEqual(_product_of_factors(factors), 360)

    def test_product_invariant_over_a_range(self):
        for n in range(2, 50):
            factors = ciutils._prime_factors(n)
            self.assertEqual(_product_of_factors(factors), n)


class Test_branching_factors_from_numscens(unittest.TestCase):
    def test_two_stage_returns_none(self):
        self.assertIsNone(ciutils.branching_factors_from_numscens(100, 2))

    def test_exact_factorization(self):
        # 8 == 2*2*2 over 3 branching factors (4 stages)
        bf = ciutils.branching_factors_from_numscens(8, 4)
        self.assertEqual(len(bf), 3)
        self.assertEqual(int(self._prod(bf)), 8)

    def test_count_matches_num_stages_minus_one(self):
        bf = ciutils.branching_factors_from_numscens(27, 3)
        self.assertEqual(len(bf), 2)
        self.assertEqual(int(self._prod(bf)), 27)

    def test_increments_numscens_when_not_factorable(self):
        # 7 is prime and cannot be split into 2 factors >= 2, so the routine
        # bumps numscens up to the next workable count (8) and never returns
        # fewer leaves than requested.
        bf = ciutils.branching_factors_from_numscens(7, 3)
        self.assertEqual(len(bf), 2)
        self.assertGreaterEqual(self._prod(bf), 7)

    @staticmethod
    def _prod(factors):
        p = 1
        for f in factors:
            p *= f
        return p


class Test_scalable_branching_factors(unittest.TestCase):
    def test_docstring_example(self):
        # documented behavior: 233 scaled like [5,3,2] -> [10,6,4] (240 leaves)
        self.assertEqual(
            ciutils.scalable_branching_factors(233, [5, 3, 2]),
            [10, 6, 4],
        )

    def test_too_few_scens_falls_back_to_twos(self):
        # numscens below 2**(numstages-1) -> all branching factors are 2
        self.assertEqual(
            ciutils.scalable_branching_factors(4, [5, 3, 2]),
            [2, 2, 2],
        )

    def test_result_covers_requested_scens(self):
        numscens = 1000
        ref = [4, 3, 2]
        bf = ciutils.scalable_branching_factors(numscens, ref)
        self.assertEqual(len(bf), len(ref))
        prod = 1
        for f in bf:
            prod *= f
        self.assertGreaterEqual(prod, numscens)
        # every branching factor is an integer value and at least 1
        self.assertTrue(all(f == int(f) and f >= 1 for f in bf))


class Test_is_sorted(unittest.TestCase):
    def test_well_constructed_list_passes(self):
        nodes = [
            _FakeNode("ROOT", 1, None),
            _FakeNode("ROOT_0", 2, "ROOT"),
            _FakeNode("ROOT_0_1", 3, "ROOT_0"),
        ]
        # no exception == well constructed
        ciutils.is_sorted(nodes)

    def test_wrong_stage_raises(self):
        nodes = [
            _FakeNode("ROOT", 1, None),
            _FakeNode("ROOT_0", 3, "ROOT"),  # stage should be 2
        ]
        with self.assertRaises(RuntimeError):
            ciutils.is_sorted(nodes)

    def test_wrong_parent_raises(self):
        nodes = [
            _FakeNode("ROOT", 1, None),
            _FakeNode("ROOT_0", 2, "NOT_ROOT"),  # parent should be ROOT
        ]
        with self.assertRaises(RuntimeError):
            ciutils.is_sorted(nodes)


class Test_correcting_numeric(unittest.TestCase):
    def test_minimize_clips_small_negative_to_zero(self):
        # tiny negative gap inside tolerance is numerical noise -> clip to 0
        G = ciutils.correcting_numeric(
            -1e-9, cfg={}, relative_error=True, threshold=1e-4, objfct=100.0
        )
        self.assertEqual(G, 0.0)

    def test_minimize_passes_through_positive(self):
        G = ciutils.correcting_numeric(
            5.0, cfg={}, relative_error=True, threshold=1e-4, objfct=100.0
        )
        self.assertEqual(G, 5.0)

    def test_minimize_keeps_large_wrong_sign_gap(self):
        # a genuinely wrong-sign gap (beyond tolerance) is returned unchanged
        G = ciutils.correcting_numeric(
            -50.0, cfg={}, relative_error=False, threshold=1e-2, objfct=100.0
        )
        self.assertEqual(G, -50.0)

    def test_maximize_clips_small_positive_to_zero(self):
        # sense passed explicitly (the path gap_estimators uses)
        G = ciutils.correcting_numeric(
            1e-9, cfg={}, relative_error=True, threshold=1e-4, objfct=100.0,
            sense=pyo.maximize,
        )
        self.assertEqual(G, 0.0)

    def test_maximize_keeps_large_wrong_sign_gap(self):
        G = ciutils.correcting_numeric(
            50.0, cfg={}, relative_error=False, threshold=1e-2, objfct=100.0,
            sense=pyo.maximize,
        )
        self.assertEqual(G, 50.0)

    def test_maximize_sense_from_cfg_key(self):
        # When sense is not passed, fall back to the cfg entry written by
        # pyomo_opt_sense(); the key is "pyomo_opt_sense". (A small positive
        # gap is numerical noise for a maximize problem and clips to 0.)
        cfg = {"pyomo_opt_sense": pyo.maximize}
        G = ciutils.correcting_numeric(
            1e-9, cfg=cfg, relative_error=True, threshold=1e-4, objfct=100.0
        )
        self.assertEqual(G, 0.0)

    def test_missing_objfct_raises(self):
        with self.assertRaises(RuntimeError):
            ciutils.correcting_numeric(
                1.0, cfg={}, relative_error=False, threshold=1e-2, objfct=None
            )


if __name__ == "__main__":
    unittest.main()
