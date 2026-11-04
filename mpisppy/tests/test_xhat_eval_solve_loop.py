###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Solver-free regression tests for Xhat_Eval solve-loop callbacks."""

import types
import unittest

from mpisppy.cylinders._preloop_xhat_mixin import _PreLoopXhatMixin
from mpisppy.extensions.extension import Extension, MultiExtension
from mpisppy.extensions.xhatbase import (
    XhatBase,
    _validate_stage2ef_extension_hooks,
)
from mpisppy.utils.xhat_eval import Xhat_Eval


class _RecordingExtension(Extension):

    def pre_solve_loop(self):
        self.opt.events.append("pre_solve_loop")

    def post_solve_loop(self):
        self.opt.events.append("post_solve_loop")


class _RecordingExtensionA(_RecordingExtension):
    pass


class _RecordingExtensionB(_RecordingExtension):
    pass


class _NoopExtension(Extension):
    pass


class _AllSolverCallbacksExtension(Extension):

    def pre_solve(self, subproblem):
        pass

    def solve_subproblem(self, subproblem, subproblem_solve, *,
                         outer_bound_only=False):
        pass

    def post_solve(self, subproblem, results):
        pass

    def pre_solve_loop(self):
        pass

    def post_solve_loop(self):
        pass


class _Scenario:

    def __init__(self, nonant_cache):
        self._mpisppy_data = types.SimpleNamespace(nonant_cache=nonant_cache)


class _Comm:
    size = 1

    @staticmethod
    def bcast(value, root):
        return value


class _XhatEvalHarness(Xhat_Eval):
    """The production solve_loop with its solver-facing edges replaced."""

    def __init__(self, *, infeasible=False, improving=True,
                 extension_classes=(_RecordingExtension,)):
        # Deliberately skip SPOpt construction: these tests exercise lifecycle
        # placement and candidate routing, not Pyomo or MPI setup.
        self.events = []
        self.solve_kwargs = []
        self._subproblems_solvers_created = True
        self.local_scenarios = {
            "scen0": _Scenario([1.0]),
            "scen1": _Scenario([2.0]),
        }
        self.options = {
            "verbose": False,
            "solver_options": None,
            "xhat_feasibility_cuts_count": 0,
        }
        self.verbose = False
        self.cylinder_rank = 0
        self.global_rank = 0
        self.n_proc = 1
        self.scenario_names_to_rank = {"ROOT": {"scen0": 0}}
        self._spcomm = None
        self._infeasible = infeasible
        self._improving = improving

        if extension_classes is None:
            self.extensions = None
            self.extobject = None
        elif len(extension_classes) == 1:
            self.extensions = extension_classes[0]
            self.extobject = extension_classes[0](self)
        else:
            self.extensions = MultiExtension
            self.extobject = MultiExtension(self, extension_classes)

    def solve_one(self, solver_options, k, s, **kwargs):
        self.events.append(f"solve:{k}")
        self.solve_kwargs.append(kwargs)
        if kwargs.get("compute_val_at_nonant"):
            self.objs_dict[k] = float(len(self.objs_dict) + 1)
        return 0.0

    def _save_nonants(self):
        self.events.append("save_nonants")

    def _fix_nonants(self, cache):
        self.events.append("fix_nonants")

    def no_incumbent_prob(self):
        return 1.0 if self._infeasible else 0.0

    def write_iis_on_xhatter_infeasible(self, *args, **kwargs):
        self.events.append("infeasible")

    def Eobjective(self, verbose=False, fct=None):
        self.events.append("objective")
        return 10.0

    def update_best_solution_if_improving(self, obj):
        self.events.append("cache")
        return self._improving

    def _restore_nonants(self, *args, **kwargs):
        self.events.append("restore_nonants")


def _try_ordinary_candidate(opt):
    xhatter = XhatBase(opt)
    xhatter.comms = {"ROOT": _Comm()}
    return xhatter._try_one({"ROOT": "scen0"}, restore_nonants=True)


class TestXhatEvalSolveLoopLifecycle(unittest.TestCase):

    def test_callbacks_once_and_around_all_scenario_solves(self):
        opt = _XhatEvalHarness()

        opt.solve_loop()

        self.assertEqual(
            opt.events,
            ["pre_solve_loop", "solve:scen0", "solve:scen1",
             "post_solve_loop"],
        )

    def test_callbacks_are_gated_when_extensions_are_disabled(self):
        opt = _XhatEvalHarness(extension_classes=None)

        opt.solve_loop()

        self.assertEqual(opt.events, ["solve:scen0", "solve:scen1"])

    def test_compute_val_at_nonant_behavior_is_preserved(self):
        opt = _XhatEvalHarness()
        opt.objs_dict = {"stale": -1.0}

        opt.solve_loop(compute_val_at_nonant=True)

        self.assertEqual(opt.objs_dict, {"scen0": 1.0, "scen1": 2.0})
        self.assertTrue(all(call["compute_val_at_nonant"]
                            for call in opt.solve_kwargs))

    def test_multiextension_dispatches_both_callbacks(self):
        opt = _XhatEvalHarness(
            extension_classes=(_RecordingExtensionA, _RecordingExtensionB))

        opt.solve_loop()

        self.assertEqual(opt.events.count("pre_solve_loop"), 2)
        self.assertEqual(opt.events.count("post_solve_loop"), 2)

    def test_ordinary_candidate_full_order_when_improving(self):
        opt = _XhatEvalHarness(improving=True)

        self.assertEqual(_try_ordinary_candidate(opt), 10.0)

        self.assertEqual(
            opt.events,
            ["save_nonants", "fix_nonants", "pre_solve_loop",
             "solve:scen0", "solve:scen1", "post_solve_loop",
             "objective", "cache", "restore_nonants"],
        )

    def test_callbacks_run_for_non_improving_candidate(self):
        opt = _XhatEvalHarness(improving=False)

        self.assertEqual(_try_ordinary_candidate(opt), 10.0)

        self.assertEqual(opt.events.count("pre_solve_loop"), 1)
        self.assertEqual(opt.events.count("post_solve_loop"), 1)
        self.assertLess(opt.events.index("post_solve_loop"),
                        opt.events.index("cache"))

    def test_callbacks_run_for_infeasible_candidate(self):
        opt = _XhatEvalHarness(infeasible=True)

        self.assertIsNone(_try_ordinary_candidate(opt))

        self.assertEqual(
            opt.events,
            ["save_nonants", "fix_nonants", "pre_solve_loop",
             "solve:scen0", "solve:scen1", "post_solve_loop",
             "infeasible", "restore_nonants"],
        )

    def test_evaluate_and_preloop_candidates_use_the_same_lifecycle(self):
        # Xhat_Eval.evaluate is the file-fed candidate's evaluation route.
        file_opt = _XhatEvalHarness()
        self.assertEqual(file_opt.evaluate({"ROOT": [1.0]}), 10.0)
        self.assertEqual(file_opt.events.count("pre_solve_loop"), 1)
        self.assertEqual(file_opt.events.count("post_solve_loop"), 1)

        # Jensen and feasible-Xhat candidates both use _evaluate_xhat.
        for candidate_name in ("Jensen", "feasible-Xhat"):
            with self.subTest(candidate=candidate_name):
                opt = _XhatEvalHarness()
                spoke = _PreLoopXhatMixin()
                spoke.opt = opt
                self.assertEqual(spoke._evaluate_xhat({"ROOT": [1.0]}), 10.0)
                self.assertEqual(opt.events.count("pre_solve_loop"), 1)
                self.assertEqual(opt.events.count("post_solve_loop"), 1)

    def test_callbacks_are_once_per_candidate(self):
        opt = _XhatEvalHarness()

        _try_ordinary_candidate(opt)
        _try_ordinary_candidate(opt)

        self.assertEqual(opt.events.count("pre_solve_loop"), 2)
        self.assertEqual(opt.events.count("post_solve_loop"), 2)

    def test_stage2ef_rejects_solver_callbacks_before_candidate_mutation(self):
        opt = _XhatEvalHarness(
            extension_classes=(_AllSolverCallbacksExtension,))
        xhatter = XhatBase(opt)
        xhatter.comms = {"ROOT": _Comm()}

        with self.assertRaisesRegex(
                RuntimeError,
                r"_AllSolverCallbacksExtension: pre_solve, solve_subproblem, "
                r"post_solve, pre_solve_loop, post_solve_loop"):
            xhatter._try_one(
                {"ROOT": "scen0"},
                stage2_ef_solver_name="unused",
                branching_factors=[1, 1],
            )

        self.assertEqual(opt.events, [])

    def test_stage2ef_rejects_solver_callbacks_during_xhatter_setup(self):
        opt = _XhatEvalHarness(
            extension_classes=(_AllSolverCallbacksExtension,))
        opt.options["stage2_ef_solver_name"] = "unused"

        with self.assertRaisesRegex(RuntimeError,
                                    "direct stage-2 EF xhat route"):
            XhatBase(opt)

    def test_stage2ef_guard_inspects_multiextension_leaves(self):
        opt = _XhatEvalHarness(
            extension_classes=(_RecordingExtensionA,
                               _AllSolverCallbacksExtension))

        with self.assertRaises(RuntimeError) as exc:
            _validate_stage2ef_extension_hooks(opt)

        message = str(exc.exception)
        self.assertIn("_RecordingExtensionA: pre_solve_loop, post_solve_loop",
                      message)
        self.assertIn("_AllSolverCallbacksExtension: pre_solve", message)
        self.assertNotIn("MultiExtension:", message)

    def test_stage2ef_guard_allows_extensions_without_solver_callbacks(self):
        opt = _XhatEvalHarness(extension_classes=(_NoopExtension,))

        _validate_stage2ef_extension_hooks(opt)


if __name__ == "__main__":
    unittest.main()
