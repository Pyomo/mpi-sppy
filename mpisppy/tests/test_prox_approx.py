###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy/utils/prox_approx.py.

The proximal-term linearization manager builds an outer (piecewise-linear)
approximation of y = x**2.  These tests exercise the deterministic geometry and
the tolerance-aware cut bookkeeping directly, with persistent_solver=None, so no
solver is required.  A small stand-in scenario mirrors the structure phbase
attaches (scenario._mpisppy_model.{xsqvar,xsqvar_cuts,xbars,rho,W} and
scenario._mpisppy_data.nonant_cost_coeffs).
"""

import types
import unittest

import pyomo.environ as pyo

from mpisppy.utils.prox_approx import (
    ProxApproxManager,
    ProxApproxManagerContinuous,
    ProxApproxManagerDiscrete,
    _compute_mb,
    _f_ls,
    _df_ls,
)

NDN_I = ("ROOT", 0)


def _make_scenario(integer=False, lb=-100, ub=100, xbar=0.0, rho=1.0, W=0.0, cost=0.0):
    """Build the minimal structure a ProxApproxManager reads from a scenario."""
    # the x variable being linearized
    xmodel = pyo.ConcreteModel()
    domain = pyo.Integers if integer else pyo.Reals
    xmodel.x = pyo.Var(within=domain, bounds=(lb, ub))

    # the block phbase calls scenario._mpisppy_model
    mm = pyo.ConcreteModel()
    mm.xsqvar = pyo.Var([NDN_I], bounds=(0, None), initialize=0.0)
    mm.xsqvar_cuts = pyo.Constraint([NDN_I], pyo.Integers)
    mm.xbars = pyo.Param([NDN_I], mutable=True, initialize=xbar)
    mm.rho = pyo.Param([NDN_I], mutable=True, initialize=rho)
    mm.W = pyo.Param([NDN_I], mutable=True, initialize=W)

    scenario = types.SimpleNamespace(
        _mpisppy_model=mm,
        _mpisppy_data=types.SimpleNamespace(nonant_cost_coeffs={NDN_I: cost}),
    )
    return scenario, xmodel.x


def _continuous(**kw):
    scenario, xvar = _make_scenario(integer=False, **kw)
    return ProxApproxManagerContinuous(scenario, xvar, NDN_I)


def _discrete(**kw):
    scenario, xvar = _make_scenario(integer=True, **kw)
    return ProxApproxManagerDiscrete(scenario, xvar, NDN_I)


class Test_pure_helpers(unittest.TestCase):
    def test_compute_mb_known_value(self):
        # secant of y=x^2 over [3,4]: slope 2*3+1=7, intercept -3*4=-12
        self.assertEqual(_compute_mb(3), (7, -12))

    def test_compute_mb_is_secant_through_lattice_points(self):
        # the line m*x+b must pass through (n, n^2) and (n+1, (n+1)^2)
        for n in range(-5, 6):
            m, b = _compute_mb(n)
            self.assertEqual(m * n + b, n ** 2)
            self.assertEqual(m * (n + 1) + b, (n + 1) ** 2)

    def test_f_ls_zero_on_parabola(self):
        # residual is (x-xpnt, x^2-ypnt); zero when the point is on y=x^2
        self.assertEqual(_f_ls([5.0], 5.0, 25.0), (0.0, 0.0))

    def test_f_ls_offset(self):
        self.assertEqual(_f_ls([2.0], 1.0, 10.0), (1.0, 4.0 - 10.0))

    def test_df_ls_jacobian(self):
        # derivative of the residual wrt the projection variable
        self.assertEqual(_df_ls([3.0], 1.0, 1.0), ((3.0,), (6.0,)))


class Test_factory_dispatch(unittest.TestCase):
    def test_continuous_var_gets_continuous_manager(self):
        scenario, xvar = _make_scenario(integer=False)
        mgr = ProxApproxManager(scenario, xvar, NDN_I)
        self.assertIsInstance(mgr, ProxApproxManagerContinuous)

    def test_integer_var_gets_discrete_manager(self):
        scenario, xvar = _make_scenario(integer=True)
        mgr = ProxApproxManager(scenario, xvar, NDN_I)
        self.assertIsInstance(mgr, ProxApproxManagerDiscrete)


class Test_check_and_add_value(unittest.TestCase):
    """The tolerance-aware sorted-insert bookkeeping (no model side effects)."""

    def setUp(self):
        # cut_values is seeded with [0.0] by __init__
        self.mgr = _continuous(lb=-1000, ub=1000)
        self.assertEqual(list(self.mgr.cut_values), [0.0])

    def test_append_when_far_enough(self):
        self.assertTrue(self.mgr.check_and_add_value(5.0, 0.1))
        self.assertEqual(list(self.mgr.cut_values), [0.0, 5.0])

    def test_reject_append_within_tolerance(self):
        self.mgr.check_and_add_value(5.0, 0.1)
        # 5.05 is within 0.1 of the current max (5.0) -> rejected
        self.assertFalse(self.mgr.check_and_add_value(5.05, 0.1))
        self.assertEqual(list(self.mgr.cut_values), [0.0, 5.0])

    def test_insert_at_front(self):
        self.assertTrue(self.mgr.check_and_add_value(-3.0, 0.1))
        self.assertEqual(list(self.mgr.cut_values), [-3.0, 0.0])

    def test_reject_front_within_tolerance(self):
        self.mgr.check_and_add_value(-3.0, 0.1)
        # -3.05 is within 0.1 of the current min (-3.0) -> rejected
        self.assertFalse(self.mgr.check_and_add_value(-3.05, 0.1))
        self.assertEqual(list(self.mgr.cut_values), [-3.0, 0.0])

    def test_insert_in_middle(self):
        self.mgr.check_and_add_value(5.0, 0.1)
        self.assertTrue(self.mgr.check_and_add_value(2.5, 0.1))
        self.assertEqual(list(self.mgr.cut_values), [0.0, 2.5, 5.0])

    def test_reject_middle_close_to_right_neighbor(self):
        self.mgr.check_and_add_value(5.0, 0.1)
        # 4.95 is within 0.1 of its right neighbor (5.0) -> rejected
        self.assertFalse(self.mgr.check_and_add_value(4.95, 0.1))
        self.assertEqual(list(self.mgr.cut_values), [0.0, 5.0])

    def test_reject_middle_close_to_left_neighbor(self):
        self.mgr.check_and_add_value(5.0, 0.1)
        # 0.05 is within 0.1 of its left neighbor (0.0) -> rejected
        self.assertFalse(self.mgr.check_and_add_value(0.05, 0.1))
        self.assertEqual(list(self.mgr.cut_values), [0.0, 5.0])

    def test_values_stay_sorted(self):
        for v in (5.0, -3.0, 2.5, 8.0, -1.0):
            self.mgr.check_and_add_value(v, 0.1)
        vals = list(self.mgr.cut_values)
        self.assertEqual(vals, sorted(vals))


class Test_continuous_add_cut(unittest.TestCase):
    def test_adds_tangent_cut(self):
        mgr = _continuous(lb=-100, ub=100)
        added = mgr.add_cut(5.0, 1e-6, None)
        self.assertEqual(added, 1)
        self.assertEqual(mgr.cut_index, 1)
        self.assertEqual(len(mgr.cuts), 1)
        # the cut is the tangent of y=x^2 at a=5: xsqvar - 2*a*x + a^2 >= 0,
        # which is tight (==0) at (x=a, xsqvar=a^2)
        con = list(mgr.cuts.values())[0]
        mgr.xvar.value = 5.0
        mgr.xvarsqrd.value = 25.0
        self.assertEqual(pyo.value(con.lower), 0)
        self.assertFalse(con.has_ub())
        self.assertAlmostEqual(pyo.value(con.body), 0.0)

    def test_tangent_is_a_lower_bound_off_the_point(self):
        mgr = _continuous(lb=-100, ub=100)
        mgr.add_cut(5.0, 1e-6, None)
        con = list(mgr.cuts.values())[0]
        # off the tangent point the true parabola sits above the cut line,
        # so the cut body (xsqvar - line) is positive when xsqvar = x^2
        mgr.xvar.value = 7.0
        mgr.xvarsqrd.value = 49.0
        self.assertGreater(pyo.value(con.body), 0.0)

    def test_value_clamped_to_bounds(self):
        mgr = _continuous(lb=0, ub=10)
        mgr.add_cut(50.0, 1e-6, None)  # 50 is outside [0,10] -> clamped to 10
        self.assertIn(10.0, list(mgr.cut_values))
        self.assertNotIn(50.0, list(mgr.cut_values))

    def test_duplicate_cut_rejected(self):
        mgr = _continuous(lb=-100, ub=100)
        self.assertEqual(mgr.add_cut(5.0, 0.1, None), 1)
        # same point within tolerance -> no new cut
        self.assertEqual(mgr.add_cut(5.0, 0.1, None), 0)
        self.assertEqual(mgr.cut_index, 1)
        self.assertEqual(len(mgr.cuts), 1)


class Test_discrete_add_cut(unittest.TestCase):
    def test_interior_point_adds_two_secant_cuts(self):
        mgr = _discrete(lb=-100, ub=100)
        added = mgr.add_cut(3.0, 1.0, None)
        self.assertEqual(added, 2)
        # one cut to the right (indexed by val+1) and one to the left (by val)
        self.assertIn((*NDN_I, 4), mgr.cuts)
        self.assertIn((*NDN_I, 3), mgr.cuts)

    def test_rounds_to_nearest_integer(self):
        mgr = _discrete(lb=-100, ub=100)
        mgr.add_cut(2.7, 1.0, None)  # rounds to 3
        self.assertIn((*NDN_I, 4), mgr.cuts)
        self.assertIn((*NDN_I, 3), mgr.cuts)

    def test_repeat_adds_no_new_cuts(self):
        mgr = _discrete(lb=-100, ub=100)
        mgr.add_cut(3.0, 1.0, None)
        self.assertEqual(mgr.add_cut(3.0, 1.0, None), 0)

    def test_at_upper_bound_only_left_cut(self):
        mgr = _discrete(lb=0, ub=3)
        added = mgr.add_cut(3.0, 1.0, None)  # val==ub -> no right cut
        self.assertEqual(added, 1)
        self.assertIn((*NDN_I, 3), mgr.cuts)
        self.assertNotIn((*NDN_I, 4), mgr.cuts)

    def test_at_lower_bound_only_right_cut(self):
        mgr = _discrete(lb=3, ub=10)
        added = mgr.add_cut(3.0, 1.0, None)  # val==lb -> no left cut
        self.assertEqual(added, 1)
        self.assertIn((*NDN_I, 4), mgr.cuts)
        self.assertNotIn((*NDN_I, 3), mgr.cuts)

    def test_secant_cut_is_tight_at_both_endpoints(self):
        mgr = _discrete(lb=-100, ub=100)
        mgr.add_cut(3.0, 1.0, None)
        # the right cut is the secant over (3,4): tight at x=3 and x=4
        con = mgr.cuts[(*NDN_I, 4)]
        self.assertEqual(pyo.value(con.lower), 0)
        self.assertFalse(con.has_ub())
        for xval in (3, 4):
            mgr.xvar.value = xval
            mgr.xvarsqrd.value = xval ** 2
            self.assertAlmostEqual(pyo.value(con.body), 0.0)


if __name__ == "__main__":
    unittest.main()
