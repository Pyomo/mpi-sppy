###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""End-to-end equivalence test for proximal-term linearization.

``mpisppy/utils/prox_approx.py`` builds a piecewise-linear outer approximation
of the PH proximal term ``(x - xbar)**2`` so that an LP/MIP solver can be used
in place of a QP solver.  The unit tests in ``test_prox_approx.py`` check the
cut geometry and bookkeeping directly (with ``persistent_solver=None``); they
never solve a subproblem, so they cannot tell whether the approximation
actually steers PH to the right place.

This test drives the real PH engine on a small stochastic LP (farmer-3) and
verifies the *consequence*: PH run with ``linearize_proximal_terms=True`` must
converge to the same first-stage solution as PH run with the exact quadratic
prox term -- and both must match the extensive-form (EF) optimum, so a shared
but wrong fixed point cannot pass.  A solver is required (the exact-prox run
needs a QP-capable solver); ``get_solver`` only ever returns cplex/gurobi/
xpress, all of which handle the quadratic prox.
"""

import unittest

import pyomo.environ as pyo

import mpisppy.opt.ph
import mpisppy.utils.sputils as sputils
from mpisppy.tests.examples import farmer
from mpisppy.tests.utils import get_solver, limit_solver_threads

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()

SCENARIO_NAMES = ["scen0", "scen1", "scen2"]
SCENARIO_KWARGS = {"num_scens": 3, "crops_multiplier": 1}

# Per-crop agreement is ~0.01 acres at 100 PH iterations; 1.0 acre (0.2% of the
# 500-acre total) leaves a large margin against solver/version differences while
# still being a meaningful equivalence check.
ACRE_TOL = 1.0


def _ph_options(linearize):
    opts = {
        "solver_name": solver_name,
        "PHIterLimit": 100,
        "defaultPHrho": 1.0,
        # tiny threshold so both runs take the full iteration budget and
        # converge tightly (the assertions are on the solution, not on
        # whether convergence was declared)
        "convthresh": 1e-8,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        # keep test solves single-threaded on shared CI runners
        "solver_options_layers": [sputils.solver_options_layer("default", {"threads": 1})],
        "linearize_proximal_terms": linearize,
    }
    if linearize:
        # tight tolerance so the piecewise-linear approximation tracks the
        # quadratic prox closely
        opts["proximal_linearization_tolerance"] = 1e-4
    return opts


def _run_ph_first_stage(linearize):
    """Run PH to (near) convergence; return the ROOT consensus xbar keyed by
    crop, and the reported objective."""
    ph = mpisppy.opt.ph.PH(
        _ph_options(linearize),
        SCENARIO_NAMES,
        farmer.scenario_creator,
        None,
        scenario_creator_kwargs=SCENARIO_KWARGS,
    )
    _, obj, _ = ph.ph_main()
    # Two-stage: every scenario shares the single ROOT-node consensus, so the
    # xbar read off any one local scenario is the first-stage solution.
    scenario = next(iter(ph.local_scenarios.values()))
    xbar = {
        xvar.index(): scenario._mpisppy_model.xbars[ndn_i].value
        for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items()
    }
    return xbar, obj


def _ef_first_stage():
    """Ground truth: the extensive-form first-stage solution, keyed by crop."""
    ef = sputils.create_EF(
        SCENARIO_NAMES, farmer.scenario_creator,
        scenario_creator_kwargs=SCENARIO_KWARGS,
    )
    solver = pyo.SolverFactory(solver_name)
    limit_solver_threads(solver, solver_name)
    if "_persistent" in solver_name:
        # persistent solvers need the instance attached before solve()
        solver.set_instance(ef)
    solver.solve(ef)
    rep = getattr(ef, ef._ef_scenario_names[0])
    xbar = {crop: pyo.value(rep.DevotedAcreage[crop]) for crop in rep.DevotedAcreage}
    return xbar, pyo.value(ef.EF_Obj)


@unittest.skipUnless(solver_available, "no solver available")
class TestProxApproxEndToEnd(unittest.TestCase):
    """PH with linearized prox must match PH with the exact quadratic prox."""

    @classmethod
    def setUpClass(cls):
        cls.ef_sol, cls.ef_obj = _ef_first_stage()
        cls.quad_sol, cls.quad_obj = _run_ph_first_stage(linearize=False)
        cls.lin_sol, cls.lin_obj = _run_ph_first_stage(linearize=True)

    def test_linearized_matches_quadratic_first_stage(self):
        # the core check: linearizing the prox term does not move the answer
        self.assertEqual(set(self.quad_sol), set(self.lin_sol))
        for crop, quad in self.quad_sol.items():
            self.assertAlmostEqual(
                quad, self.lin_sol[crop], delta=ACRE_TOL,
                msg=f"{crop}: quadratic={quad} linearized={self.lin_sol[crop]}",
            )

    def test_both_match_extensive_form(self):
        # guard against both runs sharing the same wrong fixed point
        for crop, ef in self.ef_sol.items():
            self.assertAlmostEqual(
                self.quad_sol[crop], ef, delta=ACRE_TOL,
                msg=f"{crop}: quadratic={self.quad_sol[crop]} ef={ef}",
            )
            self.assertAlmostEqual(
                self.lin_sol[crop], ef, delta=ACRE_TOL,
                msg=f"{crop}: linearized={self.lin_sol[crop]} ef={ef}",
            )

    def test_objectives_agree(self):
        scale = abs(self.ef_obj)
        # linearized vs quadratic PH objective
        self.assertAlmostEqual(self.quad_obj, self.lin_obj, delta=scale * 1e-4)
        # both PH objectives recover the EF optimum at convergence
        self.assertAlmostEqual(self.quad_obj, self.ef_obj, delta=scale * 1e-3)
        self.assertAlmostEqual(self.lin_obj, self.ef_obj, delta=scale * 1e-3)


if __name__ == "__main__":
    unittest.main()
