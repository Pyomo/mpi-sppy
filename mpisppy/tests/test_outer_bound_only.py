###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Targeted tests for the outer_bound_only solve path added to
SPOpt.solve_one / solve_loop (and threaded through PHBase.solve_loop).

The bound-only path is what the Lagrangian outer-bound spokes use to get a
dual bound without paying to load a solution back into the Pyomo models, so
these tests drive solve_loop the way lagrangian_prep does: PH_Prep, reenable
W, create solvers, then call solve_loop directly.
"""

import inspect
import unittest

import numpy as np

import mpisppy.opt.ph
import mpisppy.phbase
import mpisppy.spopt
import mpisppy.tests.examples.farmer as farmer
from mpisppy.tests.utils import get_solver

solver_available, solver_name, *_ = get_solver()

SCENARIO_NAMES = ["Scenario1", "Scenario2", "Scenario3"]


def _make_ph():
    """Construct a farmer PH object. This does NOT solve (and does not need a
    working solver), so every subproblem still holds its 'not computed yet'
    None bounds from _set_initial_bounds."""
    options = {
        # any string is fine here: construction never instantiates the solver
        "solver_name": solver_name or "gurobi",
        "PHIterLimit": 0,
        "defaultPHrho": 1.0,
        "convthresh": 0.0,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "asynchronousPH": False,
    }
    return mpisppy.opt.ph.PH(
        options,
        SCENARIO_NAMES,
        farmer.scenario_creator,
        farmer.scenario_denouement,
        scenario_creator_kwargs={"crops_multiplier": 1},
    )


def _make_prepped_ph():
    """Build a farmer PH object and take it through the same prep the
    Lagrangian spoke does, so solve_loop can be called directly."""
    ph = _make_ph()
    # Mirror _LagrangianMixin.lagrangian_prep: W attached to the objective
    # now (no prox), solvers created, ready for a direct solve_loop.
    ph.PH_Prep(attach_prox=False, defer_attach=False)
    ph._reenable_W()
    ph._create_solvers()
    return ph


class TestOuterBoundOnlySignature(unittest.TestCase):
    """No solver needed: these just pin down the calling convention."""

    def test_outer_bound_only_is_keyword_only(self):
        # Copilot review: outer_bound_only had been inserted before warmstart,
        # which shifts warmstart's positional slot. It is keyword-only now, so
        # a positional argument can never land on it by accident.
        for func in (
            mpisppy.spopt.SPOpt.solve_one,
            mpisppy.spopt.SPOpt.solve_loop,
            mpisppy.phbase.PHBase.solve_loop,
        ):
            params = inspect.signature(func).parameters
            self.assertEqual(
                params["outer_bound_only"].kind,
                inspect.Parameter.KEYWORD_ONLY,
                msg=f"{func.__qualname__} outer_bound_only must be keyword-only",
            )
            # warmstart must remain positionally reachable (its pre-PR slot).
            self.assertEqual(
                params["warmstart"].kind,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                msg=f"{func.__qualname__} warmstart must stay positional",
            )


class TestNotComputedBounds(unittest.TestCase):
    """No solver needed: a subproblem before its first solve (or after a
    bound-only solve that produced no bound) holds None -- 'not computed' --
    and Ebound must propagate that rather than crash or invent a number.

    This is the path that matters for a Lagrangian outer-bound run in which a
    subproblem times out with no dual bound to show for it.
    """

    def test_initial_bounds_are_none(self):
        ph = _make_ph()
        for s in ph.local_scenarios.values():
            self.assertIsNone(s._mpisppy_data.outer_bound)
            self.assertIsNone(s._mpisppy_data.inner_bound)

    def test_ebound_is_none_when_no_bound_computed(self):
        # Every subproblem is still at its None bound (nothing solved yet), so
        # the expected outer bound is not available: Ebound returns None, which
        # is what lets the bound spoke decline to send.
        ph = _make_ph()
        self.assertIsNone(ph.Ebound())

    def test_ebound_is_none_if_any_scenario_missing(self):
        # One missing bound is enough to spoil the expectation.
        ph = _make_ph()
        for i, s in enumerate(ph.local_scenarios.values()):
            s._mpisppy_data.outer_bound = None if i == 0 else 10.0 * (i + 1)
        self.assertIsNone(ph.Ebound())

    def test_ebound_is_the_weighted_sum_when_all_present(self):
        # Once every subproblem has a real bound, Ebound is the probability-
        # weighted sum (guarding against a false "missing" short-circuit).
        ph = _make_ph()
        expected = 0.0
        for i, s in enumerate(ph.local_scenarios.values()):
            ob = 10.0 * (i + 1)
            s._mpisppy_data.outer_bound = ob
            expected += s._mpisppy_probability * ob
        self.assertAlmostEqual(ph.Ebound(), expected)


@unittest.skipIf(not solver_available, "no solver found")
class TestOuterBoundOnly(unittest.TestCase):

    def test_bound_only_populates_bound_and_skips_solution(self):
        # The whole point: an outer bound is produced, and no solution is
        # loaded (solution_available stays False so a later staleness check
        # or PRIOR_SOLUTION warmstart won't mistake stale Vars for a solve).
        ph = _make_prepped_ph()
        ph.solve_loop(
            solver_options={"threads": 1},
            need_solution=False,
            outer_bound_only=True,
            gripe=True,
        )
        for s in ph.local_scenarios.values():
            self.assertFalse(s._mpisppy_data.solution_available)
            self.assertTrue(np.isfinite(s._mpisppy_data.outer_bound))

    def test_ebound_finite_after_successful_bound_only_solve(self):
        # The happy-path complement to the None tests: when the bound-only
        # solves all report a bound, Ebound is a real number (no false
        # 'missing' short-circuit), so the spoke will send it.
        ph = _make_prepped_ph()
        ph.solve_loop(
            solver_options={"threads": 1},
            need_solution=False,
            outer_bound_only=True,
            gripe=True,
        )
        bound = ph.Ebound()
        self.assertIsNotNone(bound)
        self.assertTrue(np.isfinite(bound))

    def test_bound_only_with_need_solution_raises(self):
        # outer_bound_only loads no solution, so asking for one is a
        # contradiction that must fail loudly (and not with an assert that
        # python -O would strip).
        ph = _make_prepped_ph()
        with self.assertRaises(ValueError):
            ph.solve_loop(
                solver_options={"threads": 1},
                need_solution=True,
                outer_bound_only=True,
            )

    def test_normal_solve_still_loads_solution(self):
        # Guard the default path: without outer_bound_only, solutions load and
        # both bounds are populated.
        ph = _make_prepped_ph()
        ph.solve_loop(solver_options={"threads": 1})
        for s in ph.local_scenarios.values():
            self.assertTrue(s._mpisppy_data.solution_available)
            self.assertTrue(np.isfinite(s._mpisppy_data.outer_bound))
            self.assertTrue(np.isfinite(s._mpisppy_data.inner_bound))


if __name__ == "__main__":
    unittest.main()
