###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the variable-fixing heuristic in
mpisppy/extensions/reduced_costs_fixer.py.

ReducedCostsFixer.reduced_costs_fixing() fixes non-anticipative variables to a
bound when their (expected) reduced cost is large and consistent with that
bound, and unfixes them when the reduced cost decays or xbar drifts away.  The
spoke communication is bypassed here: reduced costs are passed in directly so
the fix/unfix decision logic can be checked deterministically without MPI or a
solver.

Note: pre_iter0() classifies a variable that was fixed when the run started
as modeler-fixed (to be left alone forever); _build takes that baseline from
the specs' 'modeler_fixed' flags. To exercise the unfix paths we fix the
"heuristic-fixed" variables only after pre_iter0(), exactly as the real
iteration loop would.
"""

import types
import unittest

import numpy as np
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet

from mpisppy.extensions.reduced_costs_fixer import ReducedCostsFixer

NDN = "ROOT"


class _Val:
    """Stand-in for an xbar Param data object (read via .value)."""
    def __init__(self, value):
        self.value = value


def _make_opt(minimizing=True, fix_fraction=1.0, zero_rc_tol=1e-9, bound_tol=1e-6):
    rc_options = {
        "verbose": False,
        "debug": False,
        "zero_rc_tol": zero_rc_tol,
        "fix_fraction_target_pre_iter0": 0.0,
        "fix_fraction_target_iter0": fix_fraction,
        "fix_fraction_target_iterK": fix_fraction,
        "rc_bound_tol": bound_tol,
    }
    return types.SimpleNamespace(
        is_minimizing=minimizing,
        cylinder_rank=0,
        nonant_length=0,
        options={"verbose": False, "rc_options": rc_options},
        local_scenarios={},
    )


def _build(specs, minimizing=True, fix_fraction=1.0, zero_rc_tol=1e-9, bound_tol=1e-6):
    """Build (ext, vars) from a list of nonant specs.

    Each spec is a dict with keys: lb, ub, value, xbar, and optional flags
    'modeler_fixed', 'heuristic_fixed', 'surrogate', 'integer'.
    """
    n = len(specs)

    def _bounds(m, i):
        return (specs[i].get("lb", 0.0), specs[i].get("ub", 10.0))

    def _domain(m, i):
        return pyo.Integers if specs[i].get("integer") else pyo.Reals

    sm = pyo.ConcreteModel()
    sm.v = pyo.Var(range(n), bounds=_bounds, domain=_domain)

    nonant_indices = {}
    xbars = {}
    surrogates = ComponentSet()  # id-based; tolerates unhashable VarData
    for i, sp in enumerate(specs):
        var = sm.v[i]
        var._value = sp["value"]
        if sp.get("modeler_fixed"):
            var.fix(sp["value"])
        nonant_indices[(NDN, i)] = var
        xbars[(NDN, i)] = _Val(sp["xbar"])
        if sp.get("surrogate"):
            surrogates.add(var)

    scenario = types.SimpleNamespace(
        _solver_plugin=None,  # is_persistent(None) is False -> no update_var calls
        _mpisppy_data=types.SimpleNamespace(
            nonant_indices=nonant_indices,
            all_surrogate_nonants=surrogates,
        ),
        _mpisppy_model=types.SimpleNamespace(xbars=xbars),
    )

    opt = _make_opt(minimizing, fix_fraction, zero_rc_tol, bound_tol)
    opt.local_scenarios = {"Scen1": scenario}
    opt.nonant_length = n
    # Mirror SPOpt's record of which nonants were fixed when the run started.
    # The fixer asks this rather than reading xvar.fixed, because on a run
    # resumed from a checkpoint every mid-run fixing is already applied and
    # xvar.fixed would file the extension's own earlier fixings as the
    # modeler's -- never to be unfixed again. Captured here, before any
    # heuristic fixing below, which is when SPOpt captures it.
    initially_fixed = ComponentSet(
        var for var in nonant_indices.values() if var.fixed)
    opt._initial_fixed_varibles = initially_fixed
    opt.was_initially_fixed = initially_fixed.__contains__

    ext = ReducedCostsFixer(opt)
    ext.pre_iter0()
    # simulate variables the heuristic fixed on a previous iteration
    for i, sp in enumerate(specs):
        if sp.get("heuristic_fixed"):
            sm.v[i].fix(sp["value"])
    # simulate an iterK call
    ext.fix_fraction_target = fix_fraction
    return ext, [sm.v[i] for i in range(n)]


class Test_fix_decisions_minimize(unittest.TestCase):
    def test_fix_to_lb_when_positive_rc_at_lower_bound(self):
        ext, v = _build([{"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0}])
        ext.reduced_costs_fixing(np.array([5.0]))
        self.assertTrue(v[0].fixed)
        self.assertEqual(v[0].value, 0.0)  # fixed at lb

    def test_fix_to_ub_when_negative_rc_at_upper_bound(self):
        ext, v = _build([{"lb": 0, "ub": 10, "value": 10.0, "xbar": 10.0}])
        ext.reduced_costs_fixing(np.array([-5.0]))
        self.assertTrue(v[0].fixed)
        self.assertEqual(v[0].value, 10.0)  # fixed at ub

    def test_interior_xbar_not_fixed(self):
        # large rc but xbar is far from either bound -> not fixed
        ext, v = _build([{"lb": 0, "ub": 10, "value": 5.0, "xbar": 5.0}])
        ext.reduced_costs_fixing(np.array([5.0]))
        self.assertFalse(v[0].fixed)

    def test_rc_below_zero_tol_not_fixed(self):
        ext, v = _build(
            [{"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0}],
            zero_rc_tol=1e-6,
        )
        # |rc| below zero_rc_tol -> treated as zero -> not fixed
        ext.reduced_costs_fixing(np.array([1e-9]))
        self.assertFalse(v[0].fixed)


class Test_target_quantile(unittest.TestCase):
    def test_low_fraction_fixes_only_largest_rc(self):
        # fix_fraction 0 -> target is the max |rc|, so only the biggest is fixed
        specs = [
            {"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0},
            {"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0},
        ]
        ext, v = _build(specs, fix_fraction=0.0)
        ext.reduced_costs_fixing(np.array([3.0, 9.0]))
        self.assertFalse(v[0].fixed)  # rc 3 < target 9
        self.assertTrue(v[1].fixed)   # rc 9 >= target 9

    def test_full_fraction_fixes_all_qualifying(self):
        # fix_fraction 1 -> target is the min nonzero |rc|, so both qualify
        specs = [
            {"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0},
            {"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0},
        ]
        ext, v = _build(specs, fix_fraction=1.0)
        ext.reduced_costs_fixing(np.array([3.0, 9.0]))
        self.assertTrue(v[0].fixed)
        self.assertTrue(v[1].fixed)


class Test_unfix_paths(unittest.TestCase):
    def test_unfix_when_rc_drops_below_target(self):
        # a heuristic-fixed var whose rc is now ~0 gets released
        ext, v = _build(
            [{"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0, "heuristic_fixed": True}],
            zero_rc_tol=1e-6,
        )
        self.assertTrue(v[0].fixed)
        ext.reduced_costs_fixing(np.array([1e-9]))
        self.assertFalse(v[0].fixed)

    def test_unfix_when_xbar_drifts_from_fixed_value(self):
        # rc is still large, but xbar has moved away from the fixed value
        ext, v = _build(
            [{"lb": 0, "ub": 10, "value": 0.0, "xbar": 3.0, "heuristic_fixed": True}],
            bound_tol=1e-6,
        )
        self.assertTrue(v[0].fixed)
        ext.reduced_costs_fixing(np.array([5.0]))
        self.assertFalse(v[0].fixed)

    def test_nan_rc_unfixes_a_fixed_var(self):
        # var0 was heuristic-fixed; its rc is now nan (not converged in LP-LR).
        # var1 carries a real rc so the all-nan early-return does not fire.
        specs = [
            {"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0, "heuristic_fixed": True},
            {"lb": 0, "ub": 10, "value": 5.0, "xbar": 5.0},
        ]
        ext, v = _build(specs)
        self.assertTrue(v[0].fixed)
        ext.reduced_costs_fixing(np.array([np.nan, 5.0]))
        self.assertFalse(v[0].fixed)  # nan -> released

    def test_nan_rc_leaves_unfixed_var_alone(self):
        # one real rc keeps the early all-nan guard from firing
        specs = [
            {"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0},
            {"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0},
        ]
        ext, v = _build(specs)
        ext.reduced_costs_fixing(np.array([np.nan, 5.0]))
        self.assertFalse(v[0].fixed)  # nan -> untouched
        self.assertTrue(v[1].fixed)   # real positive rc at lb -> fixed


class Test_skips_and_guards(unittest.TestCase):
    def test_modeler_fixed_var_is_left_alone(self):
        # fixed before pre_iter0 -> recorded as modeler-fixed -> never touched
        ext, v = _build(
            [{"lb": 0, "ub": 10, "value": 5.0, "xbar": 5.0, "modeler_fixed": True}]
        )
        self.assertIn((NDN, 0), ext._modeler_fixed_nonants)
        ext.reduced_costs_fixing(np.array([5.0]))
        self.assertTrue(v[0].fixed)
        self.assertEqual(v[0].value, 5.0)  # not moved to a bound

    def test_surrogate_nonant_is_skipped(self):
        # qualifies on rc/bound, but is a surrogate -> skipped, stays unfixed
        ext, v = _build(
            [{"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0, "surrogate": True}]
        )
        ext.reduced_costs_fixing(np.array([5.0]))
        self.assertFalse(v[0].fixed)

    def test_all_nan_reduced_costs_is_a_noop(self):
        ext, v = _build(
            [{"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0, "heuristic_fixed": True}]
        )
        self.assertTrue(v[0].fixed)
        # all reduced costs nan -> returns immediately, so the fixed var that
        # would otherwise be released stays fixed (no unfixing happens)
        ext.reduced_costs_fixing(np.array([np.nan]))
        self.assertTrue(v[0].fixed)


class Test_fixings_from_before_a_resume(unittest.TestCase):
    """A resumed run arrives with this extension's mid-run fixings applied.

    pre_iter0 must classify a nonant by whether it was fixed when the run
    started, not by whether it is fixed now; otherwise every fixing made
    before the stop is filed as the modeler's and never unfixed again. The
    resume is simulated by fixing a variable after the baseline was taken and
    running pre_iter0 again, which is the state a resumed run's pre_iter0
    sees.
    """

    def test_reduced_costs_fixer(self):
        ext, v = _build([
            {"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0, "integer": True},
            {"lb": 0, "ub": 10, "value": 3.0, "xbar": 3.0, "integer": True,
             "modeler_fixed": True},
        ])
        v[0].fix(0.0)       # fixed by this extension before the stop
        ext.pre_iter0()
        self.assertNotIn((NDN, 0), ext._modeler_fixed_nonants,
                         msg="its own earlier fixing was filed as the "
                             "modeler's")
        self.assertIn((NDN, 0), ext._integer_nonants,
                      msg="its own earlier fixing dropped out of the "
                          "denominator of the fix-fraction target")
        self.assertIn((NDN, 1), ext._modeler_fixed_nonants)

    def test_relaxed_ph_fixer(self):
        from mpisppy.extensions.relaxed_ph_fixer import RelaxedPHFixer
        rc_ext, v = _build([
            {"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0},
            {"lb": 0, "ub": 10, "value": 3.0, "xbar": 3.0,
             "modeler_fixed": True},
        ])
        ext = RelaxedPHFixer(rc_ext.opt)
        v[0].fix(0.0)       # fixed by this extension before the stop
        ext.pre_iter0()
        self.assertNotIn((NDN, 0), ext._modeler_fixed_nonants,
                         msg="its own earlier fixing was filed as the "
                             "modeler's")
        self.assertIn((NDN, 1), ext._modeler_fixed_nonants)


class Test_fix_decisions_maximize(unittest.TestCase):
    def test_fix_to_lb_when_negative_rc_at_lower_bound(self):
        ext, v = _build(
            [{"lb": 0, "ub": 10, "value": 0.0, "xbar": 0.0}], minimizing=False
        )
        ext.reduced_costs_fixing(np.array([-5.0]))
        self.assertTrue(v[0].fixed)
        self.assertEqual(v[0].value, 0.0)

    def test_fix_to_ub_when_positive_rc_at_upper_bound(self):
        ext, v = _build(
            [{"lb": 0, "ub": 10, "value": 10.0, "xbar": 10.0}], minimizing=False
        )
        ext.reduced_costs_fixing(np.array([5.0]))
        self.assertTrue(v[0].fixed)
        self.assertEqual(v[0].value, 10.0)


if __name__ == "__main__":
    unittest.main()
