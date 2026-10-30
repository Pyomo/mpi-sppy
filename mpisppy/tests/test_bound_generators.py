###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Regression tests for bound-vector construction with unbounded variables."""

import types
import unittest

import numpy as np

from mpisppy.cylinders.reduced_costs_spoke import (
    _lb_generator as _rc_lb_generator,
    _ub_generator as _rc_ub_generator,
)
from mpisppy.opt.presolve import (
    _INF as _PRESOLVE_INF,
    _lb_generator as _presolve_lb_generator,
    _ub_generator as _presolve_ub_generator,
)


class TestBoundGenerators(unittest.TestCase):

    def test_unbounded_entries_do_not_shift_later_bounds(self):
        variables = [
            types.SimpleNamespace(lb=None, ub=None),
            types.SimpleNamespace(lb=-2.0, ub=2.0),
            types.SimpleNamespace(lb=None, ub=None),
            types.SimpleNamespace(lb=4.0, ub=8.0),
            types.SimpleNamespace(lb=None, ub=None),
        ]
        for name, lb_generator, ub_generator, infinity in (
            ("reduced costs", _rc_lb_generator, _rc_ub_generator, np.inf),
            ("presolve", _presolve_lb_generator, _presolve_ub_generator,
             _PRESOLVE_INF),
        ):
            expected_lbs = np.array(
                [-infinity, -2.0, -infinity, 4.0, -infinity]
            )
            expected_ubs = np.array(
                [infinity, 2.0, infinity, 8.0, infinity]
            )
            with self.subTest(path=name, bound="lower"):
                actual = np.fromiter(
                    lb_generator(variables), dtype=float, count=len(variables)
                )
                np.testing.assert_array_equal(actual, expected_lbs)
            with self.subTest(path=name, bound="upper"):
                actual = np.fromiter(
                    ub_generator(variables), dtype=float, count=len(variables)
                )
                np.testing.assert_array_equal(actual, expected_ubs)


if __name__ == "__main__":
    unittest.main()
