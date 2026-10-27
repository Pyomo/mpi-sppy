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
from unittest import mock

import numpy as np
from pyomo.opt import (
    SolutionStatus,
    SolverResults,
    SolverStatus,
    TerminationCondition,
)

import mpisppy.opt.ph
import mpisppy.phbase
import mpisppy.spopt
import mpisppy.tests.examples.farmer as farmer
from mpisppy.extensions.extension import (
    Extension,
    INFER_BOUND,
    SubproblemSolveOutcome,
)
from mpisppy.tests.utils import get_solver

solver_available, solver_name, *_ = get_solver()

SCENARIO_NAMES = ["Scenario1", "Scenario2", "Scenario3"]


def _make_ph(extensions=None):
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
        extensions=extensions,
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



class _NoBoundPlugin:
    """A solver plugin that comes back with no usable bound.

    Only what solve_one touches: an options mapping and a solve() returning a
    results object whose termination condition is one of the outcomes
    no_outer_bound_results screens out. Not a persistent solver, so solve_one
    takes the ordinary path.
    """

    def __init__(self, termination_condition):
        self.options = {}
        self._tc = termination_condition

    def solve(self, s, **kwargs):
        results = SolverResults()
        results.solver.status = SolverStatus.warning
        results.solver.termination_condition = self._tc
        return results


class _ExplicitBoundExtension(Extension):
    def solve_subproblem(
            self, subproblem, default_solve, *, outer_bound_only=False):
        results = SolverResults()
        results.solver.status = SolverStatus.ok
        results.solver.termination_condition = TerminationCondition.optimal
        return SubproblemSolveOutcome(
            results,
            solution_available=False,
            outer_bound=-123.0,
            inner_bound=None,
        )


class _InferNonoptimalBoundsExtension(Extension):
    def solve_subproblem(
            self, subproblem, default_solve, *, outer_bound_only=False):
        results = SolverResults()
        results.solver.status = SolverStatus.warning
        results.solver.termination_condition = (
            TerminationCondition.maxTimeLimit)
        results.problem.lower_bound = -321.0
        results.problem.upper_bound = 654.0
        return SubproblemSolveOutcome(
            results,
            solution_available=False,
            outer_bound=INFER_BOUND,
            inner_bound=INFER_BOUND,
        )


class _InferDefaultNonoptimalBoundsExtension(Extension):
    def solve_subproblem(
            self, subproblem, default_solve, *, outer_bound_only=False):
        default_outcome = default_solve()
        self.default_outcome = default_outcome
        return SubproblemSolveOutcome(
            default_outcome.results,
            solution_available=default_outcome.solution_available,
            outer_bound=INFER_BOUND,
            inner_bound=INFER_BOUND,
        )


class _DirectSolutionExtension(Extension):
    def solve_subproblem(
            self, subproblem, default_solve, *, outer_bound_only=False):
        for variable in subproblem._mpisppy_data.nonant_indices.values():
            variable.set_value(7.0)
        return SubproblemSolveOutcome(
            None,
            solution_available=True,
            outer_bound=-111.0,
            inner_bound=222.0,
        )


class TestExplicitSolveOutcome(unittest.TestCase):
    def test_extension_can_report_a_bound_without_a_solution(self):
        ph = _make_ph(extensions=_ExplicitBoundExtension)
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        scenario._solver_plugin = _NoBoundPlugin(
            TerminationCondition.infeasible)

        ph.solve_one(
            None,
            name,
            scenario,
            gripe=True,
            need_solution=False,
        )

        self.assertFalse(scenario._mpisppy_data.solution_available)
        self.assertEqual(scenario._mpisppy_data.outer_bound, -123.0)
        self.assertIsNone(scenario._mpisppy_data.inner_bound)

    def test_extension_infers_bounds_from_nonoptimal_results(self):
        ph = _make_ph(extensions=_InferNonoptimalBoundsExtension)
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        scenario._solver_plugin = _NoBoundPlugin(
            TerminationCondition.infeasible)

        ph.solve_one(
            None,
            name,
            scenario,
            gripe=True,
            need_solution=False,
        )

        self.assertFalse(scenario._mpisppy_data.solution_available)
        self.assertEqual(scenario._mpisppy_data.outer_bound, -321.0)
        self.assertEqual(scenario._mpisppy_data.inner_bound, 654.0)

    def test_extension_can_infer_bounds_from_failed_default_solve(self):
        ph = _make_ph(extensions=_InferDefaultNonoptimalBoundsExtension)
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        plugin = _NoBoundPlugin(TerminationCondition.maxTimeLimit)
        scenario._solver_plugin = plugin

        original_solve = plugin.solve

        def solve_with_bounds(subproblem, **kwargs):
            results = original_solve(subproblem, **kwargs)
            results.problem.lower_bound = -321.0
            results.problem.upper_bound = 654.0
            return results

        plugin.solve = solve_with_bounds
        ph.solve_one(
            None,
            name,
            scenario,
            gripe=True,
            need_solution=False,
        )

        self.assertFalse(scenario._mpisppy_data.solution_available)
        self.assertEqual(scenario._mpisppy_data.outer_bound, -321.0)
        self.assertEqual(scenario._mpisppy_data.inner_bound, 654.0)
        self.assertIsNone(ph.extobject.default_outcome.outer_bound)
        self.assertIsNone(ph.extobject.default_outcome.inner_bound)

    def test_extension_can_load_solution_and_report_bounds_without_results(self):
        ph = _make_ph(extensions=_DirectSolutionExtension)
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        plugin = _CountingPlugin()
        scenario._solver_plugin = plugin

        ph.solve_one(None, name, scenario, gripe=True, need_solution=True)

        self.assertEqual(plugin.solve_count, 0)
        self.assertTrue(scenario._mpisppy_data.solution_available)
        self.assertEqual(scenario._mpisppy_data.outer_bound, -111.0)
        self.assertEqual(scenario._mpisppy_data.inner_bound, 222.0)
        self.assertTrue(all(
            variable.value == 7.0
            for variable in scenario._mpisppy_data.nonant_indices.values()
        ))


class _NoDefaultSolveExtension(Extension):
    def solve_subproblem(
            self, subproblem, default_solve, *, outer_bound_only=False):
        return SubproblemSolveOutcome(
            None,
            solution_available=False,
            outer_bound=None,
            inner_bound=None,
        )


class _CountingPlugin:
    def __init__(self):
        self.options = {}
        self.solve_count = 0

    def solve(self, scenario, **kwargs):
        self.solve_count += 1
        results = SolverResults()
        results.solver.status = SolverStatus.ok
        results.solver.termination_condition = TerminationCondition.optimal
        results.problem.lower_bound = -10.0
        results.problem.upper_bound = 10.0
        results.solution.add().status = SolutionStatus.feasible
        return results


class _FailThenSucceedPlugin(_CountingPlugin):
    def solve(self, scenario, **kwargs):
        if self.solve_count == 0:
            self.solve_count += 1
            raise RuntimeError("first solve failed")
        return super().solve(scenario, **kwargs)


class _InvalidOutcomeExtension(Extension):
    def __init__(self, ph):
        super().__init__(ph)
        self.solve_count = 0

    def solve_subproblem(
            self, subproblem, default_solve, *, outer_bound_only=False):
        self.solve_count += 1
        return None


class _RetryDefaultSolveExtension(Extension):
    def solve_subproblem(
            self, subproblem, default_solve, *, outer_bound_only=False):
        try:
            default_solve()
        except RuntimeError:
            pass
        return default_solve()


class TestSolveReplacementContract(unittest.TestCase):
    def test_gripe_does_not_bypass_replacement_extension(self):
        ph = _make_ph(extensions=_NoDefaultSolveExtension)
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        plugin = _CountingPlugin()
        scenario._solver_plugin = plugin

        ph.solve_one(
            None,
            name,
            scenario,
            gripe=True,
            need_solution=False,
        )

        self.assertEqual(plugin.solve_count, 0)
        self.assertFalse(scenario._mpisppy_data.solution_available)
        self.assertIsNone(scenario._mpisppy_data.outer_bound)

    def test_load_exception_preserves_original_solve_one_behavior(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                ph = _make_ph()
                name = list(ph.local_scenarios)[0]
                scenario = ph.local_scenarios[name]
                plugin = _CountingPlugin()
                scenario._solver_plugin = plugin
                scenario._mpisppy_data.solution_available = True
                scenario._mpisppy_data.termination_condition = "prior"
                error = RuntimeError("solution load failed")

                if persistent:
                    plugin.load_vars = mock.Mock(side_effect=error)
                else:
                    scenario.solutions.load_from = mock.Mock(side_effect=error)

                with mock.patch(
                        "mpisppy.spopt.sputils.is_persistent",
                        return_value=persistent):
                    with self.assertRaisesRegex(
                            RuntimeError, "solution load failed"):
                        ph.solve_one(
                            None,
                            name,
                            scenario,
                            gripe=True,
                            need_solution=True,
                            update_objective=False,
                        )

                self.assertEqual(plugin.solve_count, 1)
                self.assertTrue(scenario._mpisppy_data.solution_available)
                self.assertEqual(
                    scenario._mpisppy_data.termination_condition,
                    str(TerminationCondition.optimal),
                )

    def test_load_exception_is_tolerated_when_solution_is_not_needed(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                ph = _make_ph()
                name = list(ph.local_scenarios)[0]
                scenario = ph.local_scenarios[name]
                plugin = _CountingPlugin()
                scenario._solver_plugin = plugin
                error = RuntimeError("solution load failed")
                termination_conditions_seen_by_load = []

                def fail_load(*args):
                    termination_conditions_seen_by_load.append(
                        scenario._mpisppy_data.termination_condition)
                    raise error

                if persistent:
                    plugin.load_vars = mock.Mock(side_effect=fail_load)
                else:
                    scenario.solutions.load_from = mock.Mock(
                        side_effect=fail_load)

                with mock.patch(
                        "mpisppy.spopt.sputils.is_persistent",
                        return_value=persistent):
                    ph.solve_one(
                        None,
                        name,
                        scenario,
                        gripe=True,
                        need_solution=False,
                        update_objective=False,
                    )

                self.assertEqual(plugin.solve_count, 1)
                self.assertFalse(scenario._mpisppy_data.solution_available)
                self.assertEqual(scenario._mpisppy_data.outer_bound, -10.0)
                self.assertEqual(scenario._mpisppy_data.inner_bound, 10.0)
                self.assertEqual(
                    termination_conditions_seen_by_load,
                    [str(TerminationCondition.optimal)],
                )

    def test_solver_exception_is_not_gated_by_need_solution(self):
        ph = _make_ph()
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        plugin = _FailThenSucceedPlugin()
        scenario._solver_plugin = plugin

        with self.assertRaisesRegex(RuntimeError, "first solve failed"):
            ph.solve_one(
                None,
                name,
                scenario,
                gripe=False,
                need_solution=False,
                update_objective=False,
            )

        self.assertEqual(plugin.solve_count, 1)

    def test_gripe_diagnostic_retry_does_not_load_solution(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                ph = _make_ph()
                name = list(ph.local_scenarios)[0]
                scenario = ph.local_scenarios[name]
                plugin = _FailThenSucceedPlugin()
                scenario._solver_plugin = plugin

                if persistent:
                    plugin.load_vars = mock.Mock()
                    load_solution = plugin.load_vars
                else:
                    scenario.solutions.load_from = mock.Mock()
                    load_solution = scenario.solutions.load_from

                with mock.patch(
                        "mpisppy.spopt.sputils.is_persistent",
                        return_value=persistent):
                    with self.assertRaisesRegex(
                            RuntimeError, "first solve failed"):
                        ph.solve_one(
                            None,
                            name,
                            scenario,
                            gripe=True,
                            need_solution=True,
                            update_objective=False,
                        )

                self.assertEqual(plugin.solve_count, 2)
                load_solution.assert_not_called()

    def test_extension_can_retry_default_solve_after_solver_failure(self):
        ph = _make_ph(extensions=_RetryDefaultSolveExtension)
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        plugin = _FailThenSucceedPlugin()
        scenario._solver_plugin = plugin

        scenario.solutions.load_from = mock.Mock()
        with mock.patch.object(ph, "_check_staleness") as check_staleness:
            ph.solve_one(
                None,
                name,
                scenario,
                gripe=True,
                need_solution=True,
                update_objective=False,
            )

        self.assertEqual(plugin.solve_count, 2)
        scenario.solutions.load_from.assert_called_once()
        check_staleness.assert_called_once_with(scenario)
        self.assertTrue(scenario._mpisppy_data.solution_available)
        self.assertEqual(scenario._mpisppy_data.outer_bound, -10.0)
        self.assertEqual(scenario._mpisppy_data.inner_bound, 10.0)

    def test_solver_log_context_failure_remains_a_solver_failure(self):
        ph = _make_ph()
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        plugin = _CountingPlugin()
        scenario._solver_plugin = plugin
        scenario._mpisppy_data.termination_condition = "prior"

        log_context = mock.MagicMock()
        log_context.__enter__.side_effect = RuntimeError(
            "log context failed")
        with mock.patch(
                "mpisppy.spopt.sputils.solver_log_stream",
                return_value=log_context):
            with self.assertRaisesRegex(RuntimeError, "log context failed"):
                ph.solve_one(
                    None,
                    name,
                    scenario,
                    gripe=False,
                    need_solution=True,
                    update_objective=False,
                )

        self.assertEqual(plugin.solve_count, 0)
        self.assertIsNone(scenario._mpisppy_data.termination_condition)

    def test_result_classification_exception_is_not_retried(self):
        ph = _make_ph()
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        plugin = _CountingPlugin()
        scenario._solver_plugin = plugin
        classification_calls = 0

        def classify_results(results):
            nonlocal classification_calls
            classification_calls += 1
            self.assertEqual(
                scenario._mpisppy_data.termination_condition,
                str(TerminationCondition.optimal),
            )
            if classification_calls == 1:
                raise RuntimeError("classification failed")
            return True

        with mock.patch(
                "mpisppy.spopt.sputils.not_good_enough_results",
                side_effect=classify_results):
            with self.assertRaisesRegex(
                    RuntimeError, "classification failed"):
                ph.solve_one(
                    None,
                    name,
                    scenario,
                    gripe=True,
                    need_solution=True,
                    update_objective=False,
                )

        self.assertEqual(plugin.solve_count, 1)
        self.assertEqual(classification_calls, 1)
        self.assertEqual(
            scenario._mpisppy_data.termination_condition,
            str(TerminationCondition.optimal),
        )

    def test_invalid_extension_outcome_is_not_retried(self):
        ph = _make_ph(extensions=_InvalidOutcomeExtension)
        name = list(ph.local_scenarios)[0]
        scenario = ph.local_scenarios[name]
        scenario._solver_plugin = _CountingPlugin()

        with self.assertRaisesRegex(
                TypeError,
                "solve_subproblem must return SubproblemSolveOutcome"):
            ph.solve_one(
                None,
                name,
                scenario,
                gripe=True,
                need_solution=True,
                update_objective=False,
            )

        self.assertEqual(ph.extobject.solve_count, 1)


class TestStaleBoundNotRetained(unittest.TestCase):
    """A solve that produces no bound must clear the subproblem's bound.

    Keeping the previous value looks harmless -- each stale value really was a
    valid outer bound for the weights it was computed with -- but Ebound sums
    p_s * outer_bound_s across scenarios, and a Lagrangian bound is valid as a
    *sum* only when every scenario uses weights from a single generation
    satisfying sum_s p_s W_s = 0:

        sum_s p_s L_s(W'_s) <= OPT + (sum_s p_s W'_s)^T xbar*

    Mixing one scenario's stale bound with the others' fresh ones leaves that
    trailing term in place, so the result is not an outer bound at all. Nothing
    downstream can detect it: Ebound's missing-bound check looks for None, and a
    retained number is not None. The hub then latches the best outer bound it
    has seen, so a single too-good value is never corrected by later honest
    ones.

    No solver needed: the failing solve is stubbed.
    """

    def _ph_with_prior_bounds(self, value=-100.0):
        """A PH object whose subproblems already carry bounds from an earlier
        iteration's weights."""
        ph = _make_ph()
        for s in ph.local_scenarios.values():
            s._mpisppy_data.outer_bound = value
        return ph

    def test_bound_only_infeasible_clears_the_bound(self):
        ph = self._ph_with_prior_bounds()
        name = list(ph.local_scenarios)[0]
        s = ph.local_scenarios[name]
        s._solver_plugin = _NoBoundPlugin(TerminationCondition.infeasible)
        ph.solve_one(None, name, s, gripe=False, need_solution=False,
                     outer_bound_only=True)
        self.assertIsNone(s._mpisppy_data.outer_bound)

    def test_bound_only_unbounded_clears_the_bound(self):
        # An unbounded subproblem has Lagrangian value -inf, so a retained
        # finite number over-claims regardless of any weight mixing.
        ph = self._ph_with_prior_bounds()
        name = list(ph.local_scenarios)[0]
        s = ph.local_scenarios[name]
        s._solver_plugin = _NoBoundPlugin(TerminationCondition.unbounded)
        ph.solve_one(None, name, s, gripe=False, need_solution=False,
                     outer_bound_only=True)
        self.assertIsNone(s._mpisppy_data.outer_bound)

    def test_ebound_declines_rather_than_mixing_generations(self):
        # The end-to-end shape of the bug: one scenario keeps a bound from the
        # previous weights while the other two get fresh ones. Ebound must
        # refuse, not return a finite number that is not a bound.
        ph = self._ph_with_prior_bounds()
        names = list(ph.local_scenarios)
        stale = ph.local_scenarios[names[0]]
        stale._solver_plugin = _NoBoundPlugin(TerminationCondition.infeasible)
        ph.solve_one(None, names[0], stale, gripe=False, need_solution=False,
                     outer_bound_only=True)
        for n in names[1:]:
            ph.local_scenarios[n]._mpisppy_data.outer_bound = -50.0
        self.assertIsNone(ph.Ebound())

    def test_failed_ordinary_solve_also_clears_the_bound(self):
        # The same exposure exists off the outer_bound_only path: the
        # not_good_enough_results branch used to leave outer_bound untouched.
        ph = self._ph_with_prior_bounds()
        name = list(ph.local_scenarios)[0]
        s = ph.local_scenarios[name]
        s._solver_plugin = _NoBoundPlugin(TerminationCondition.infeasible)
        ph.solve_one(None, name, s, gripe=False, need_solution=False)
        self.assertIsNone(s._mpisppy_data.outer_bound)
        self.assertFalse(s._mpisppy_data.solution_available)


class _StubGuest:
    """An agnostic guest whose solve fails, and nothing else.

    This is what pyomo_guest.py, ampl_guest.py and gams_guest.py all do when a
    solve fails: mark the solve as having produced no solution and return.
    None of them touches outer_bound (two clear inner_bound), so a bound left
    over from an earlier solve is the guest's whole failure behavior. Passing
    scenarios in `succeed_for` models the success branch instead, where a guest
    records the bound its solve produced.
    """

    def __init__(self, succeed_for=(), bound=None):
        self._succeed_for = [id(s) for s in succeed_for]
        self._bound = bound
        self.calls = 0

    def callout_agnostic(self, kws):
        self.calls += 1
        s = kws["s"]
        if id(s) in self._succeed_for:
            s._mpisppy_data.solution_available = True
            s._mpisppy_data.outer_bound = self._bound
        else:
            s._mpisppy_data.solution_available = False


class TestAgnosticFailedSolveClearsBound(unittest.TestCase):
    """The same invariant on the agnostic path.

    solve_one dispatches to a guest callout instead of a Pyomo solver when the
    object carries an Ag, and the guest returns from a failed solve without
    touching outer_bound. The clearing therefore has to happen before the
    dispatch, or an AMPL/GAMS/Pyomo-guest run with a lagrangian spoke
    (agnostic_cylinders.py builds one) sums a bound from an earlier generation
    of weights with fresh ones and reports the total as an outer bound.

    No solver needed: the guest is stubbed, and the agnostic path never reaches
    a solver plugin.
    """

    def _agnostic_ph(self, guest, value=-100.0, extensions=None):
        """A PH object carrying bounds from an earlier iteration, wired to
        solve through `guest`."""
        ph = _make_ph(extensions=extensions)
        for s in ph.local_scenarios.values():
            s._mpisppy_data.outer_bound = value
            # solve_one asks whether the plugin is persistent before it looks
            # at Ag; on this path nothing else touches it.
            s._solver_plugin = None
        ph.Ag = guest
        return ph

    def test_failed_guest_solve_clears_the_bound(self):
        guest = _StubGuest()
        ph = self._agnostic_ph(guest)
        name = list(ph.local_scenarios)[0]
        s = ph.local_scenarios[name]
        ph.solve_one(None, name, s, gripe=False, need_solution=False)
        self.assertEqual(guest.calls, 1)
        self.assertIsNone(s._mpisppy_data.outer_bound)
        self.assertFalse(s._mpisppy_data.solution_available)

    def test_failed_guest_solve_makes_ebound_none(self):
        # The exposure that matters: one guest solve fails while the others
        # report fresh bounds. Ebound must decline rather than mix generations.
        guest = _StubGuest()
        ph = self._agnostic_ph(guest)
        names = list(ph.local_scenarios)
        failed = ph.local_scenarios[names[0]]
        ph.solve_one(None, names[0], failed, gripe=False, need_solution=False)
        for n in names[1:]:
            ph.local_scenarios[n]._mpisppy_data.outer_bound = -50.0
        self.assertIsNone(ph.Ebound())

    def test_successful_guest_solve_keeps_its_bound(self):
        # The clear must not cost a guest the bound it did compute.
        ph = self._agnostic_ph(_StubGuest())
        ph.Ag = _StubGuest(succeed_for=ph.local_scenarios.values(), bound=-42.0)
        for name, s in ph.local_scenarios.items():
            ph.solve_one(None, name, s, gripe=False, need_solution=False)
            self.assertEqual(s._mpisppy_data.outer_bound, -42.0)
        self.assertAlmostEqual(ph.Ebound(), -42.0)

    def test_solve_subproblem_override_is_rejected_with_agnostic(self):
        guest = _StubGuest()
        ph = self._agnostic_ph(
            guest,
            extensions=_NoDefaultSolveExtension,
        )
        name = list(ph.local_scenarios)[0]

        with self.assertRaisesRegex(
                RuntimeError,
                "solve_subproblem overrides are not supported with Agnostic"):
            ph.solve_one(
                None,
                name,
                ph.local_scenarios[name],
                gripe=False,
                need_solution=False,
            )

        self.assertEqual(guest.calls, 0)


if __name__ == "__main__":
    unittest.main()
