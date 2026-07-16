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


def _make_prepped_ph():
    """Build a farmer PH object and take it through the same prep the
    Lagrangian spoke does, so solve_loop can be called directly."""
    options = {
        "solver_name": solver_name,
        "PHIterLimit": 0,
        "defaultPHrho": 1.0,
        "convthresh": 0.0,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "asynchronousPH": False,
    }
    scenario_names = ["Scenario1", "Scenario2", "Scenario3"]
    ph = mpisppy.opt.ph.PH(
        options,
        scenario_names,
        farmer.scenario_creator,
        farmer.scenario_denouement,
        scenario_creator_kwargs={"crops_multiplier": 1},
    )
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
