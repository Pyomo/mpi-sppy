###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Phase 1 tests for CVaR risk management (mpisppy/utils/cvar.py):
#   - structural checks of the per-scenario transform (no solver needed)
#   - EF-CVaR vs a closed-form CVaR on a tiny deterministic instance
#   - pure CVaR (cvar_mean_weight == 0)
#   - cvar_weight == 0 reproduces the risk-neutral EF (regression guard)

import unittest
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.utils.cvar as cvar
import mpisppy.tests.examples.farmer as farmer
from mpisppy.tests.utils import get_solver, limit_solver_threads

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()


# ---------------------------------------------------------------------------
# A tiny deterministic instance with a known closed-form CVaR.
#
# Each scenario's cost is a constant (the single variable is fixed to 0), so the
# EF's only freedom is the shared VaR eta and the per-scenario excess delta.  The
# EF therefore computes CVaR exactly and we can compare to the closed form.
#
# Costs {10, 20, 30, 40}, uniform probability, alpha = 0.6:
#   E[Cost]      = 25
#   VaR (eta*)   = 30                 (the 0.6-quantile is unique here)
#   CVaR_0.6     = (1/(1-0.6))*(1/4)*(40-30) + 30 = 36.25
# ---------------------------------------------------------------------------
TINY_COSTS = {"s0": 10.0, "s1": 20.0, "s2": 30.0, "s3": 40.0}
TINY_ALPHA = 0.6
TINY_EXPECTED_COST = 25.0
TINY_VAR = 30.0
TINY_CVAR = 36.25


def tiny_scenario_creator(sname, **kwargs):
    model = pyo.ConcreteModel(name=sname)
    model.x = pyo.Var(bounds=(0.0, 0.0))            # forced to 0; a real nonant
    cost_expr = TINY_COSTS[sname] + model.x
    model.cost = pyo.Objective(expr=cost_expr, sense=pyo.minimize)
    model._mpisppy_probability = 1.0 / len(TINY_COSTS)
    sputils.attach_root_node(model, cost_expr, [model.x])
    return model


def _solve(model):
    solver = pyo.SolverFactory(solver_name)
    limit_solver_threads(solver, solver_name)
    if "_persistent" in solver_name:
        solver.set_instance(model)
    results = solver.solve(model, tee=False)
    pyo.assert_optimal_termination(results)
    return results


def _farmer_original_costs(ef, names):
    """(probability, realized original cost) per farmer scenario at the solution.

    Uses each scenario's original risk-neutral objective ``Total_Cost_Objective``,
    which the CVaR transform leaves on the model (deactivated) for exactly this
    kind of E[Cost] reporting.  Costs follow the minimize convention.
    """
    out = []
    for sname in names:
        s = getattr(ef, sname)
        out.append((s._mpisppy_probability, pyo.value(s.Total_Cost_Objective.expr)))
    return out


def _farmer_expected_cost(ef, names):
    return sum(p * c for p, c in _farmer_original_costs(ef, names))


def _farmer_worst_cost(ef, names):
    # worst == highest cost (minimize convention)
    return max(c for _, c in _farmer_original_costs(ef, names))


class StructureTests(unittest.TestCase):
    """The transform's structure; these do not need a solver."""

    def _farmer_scenario(self):
        return farmer.scenario_creator("scen0", num_scens=3)

    def test_with_cvar_is_the_only_active_objective(self):
        s = self._farmer_scenario()
        original = sputils.find_active_objective(s)
        self.assertIsNot(original, None)
        cvar.add_cvar(s, cvar_weight=0.5, cvar_alpha=0.9)
        active = sputils.find_active_objective(s)   # raises if not exactly one
        self.assertIs(active, s.WITH_CVAR)
        self.assertTrue(s.WITH_CVAR.active)
        self.assertFalse(original.active)           # original kept but deactivated

    def test_components_added(self):
        s = self._farmer_scenario()
        cvar.add_cvar(s, cvar_weight=0.5, cvar_alpha=0.9)
        self.assertTrue(hasattr(s, "_mpisppy_cvar_eta"))
        self.assertTrue(hasattr(s, "_mpisppy_cvar_excess"))
        self.assertTrue(hasattr(s, "_mpisppy_cvar_excess_con"))
        # excess delta_s must be >= 0
        self.assertIs(s._mpisppy_cvar_excess.domain, pyo.NonNegativeReals)

    def test_eta_appended_to_root_nonants(self):
        s = self._farmer_scenario()
        root = s._mpisppy_node_list[0]
        n_list_before = len(root.nonant_list)
        n_vardata_before = len(root.nonant_vardata_list)
        cvar.add_cvar(s, cvar_weight=0.5, cvar_alpha=0.9)
        self.assertEqual(len(root.nonant_list), n_list_before + 1)
        self.assertEqual(len(root.nonant_vardata_list), n_vardata_before + 1)
        self.assertIs(root.nonant_list[-1], s._mpisppy_cvar_eta)
        self.assertIs(root.nonant_vardata_list[-1], s._mpisppy_cvar_eta)

    def test_wrapper_applies_transform(self):
        wrapped = cvar.cvar_scenario_creator(
            farmer.scenario_creator, cvar_weight=0.5, cvar_alpha=0.9)
        s = wrapped("scen0", num_scens=3)
        self.assertTrue(hasattr(s, "WITH_CVAR"))
        self.assertIs(sputils.find_active_objective(s), s.WITH_CVAR)

    def test_bad_alpha_raises(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                cvar.add_cvar(self._farmer_scenario(),
                              cvar_weight=1.0, cvar_alpha=bad)

    def test_negative_weights_raise(self):
        with self.assertRaises(ValueError):
            cvar.add_cvar(self._farmer_scenario(),
                          cvar_weight=-1.0, cvar_alpha=0.9)
        with self.assertRaises(ValueError):
            cvar.add_cvar(self._farmer_scenario(), cvar_weight=1.0,
                          cvar_alpha=0.9, cvar_mean_weight=-1.0)


@unittest.skipIf(not solver_available, "no solver is available")
class EFClosedFormTests(unittest.TestCase):
    """EF-CVaR on the tiny instance vs the closed-form CVaR."""

    def _eta_value(self, ef):
        # NAC has tied all scenario etas together; read the first one.
        s0 = getattr(ef, "s0")
        return pyo.value(s0._mpisppy_cvar_eta)

    def test_ef_cvar_matches_closed_form(self):
        # lambda = 1, beta = 1  =>  E[Cost] + CVaR
        ef = sputils.create_EF(
            list(TINY_COSTS.keys()),
            cvar.cvar_scenario_creator(
                tiny_scenario_creator, cvar_weight=1.0, cvar_alpha=TINY_ALPHA),
            suppress_warnings=True)
        _solve(ef)
        self.assertAlmostEqual(pyo.value(ef.EF_Obj),
                               TINY_EXPECTED_COST + TINY_CVAR, places=4)
        self.assertAlmostEqual(self._eta_value(ef), TINY_VAR, places=4)

    def test_pure_cvar(self):
        # lambda = 0, beta = 1  =>  CVaR only
        ef = sputils.create_EF(
            list(TINY_COSTS.keys()),
            cvar.cvar_scenario_creator(
                tiny_scenario_creator, cvar_weight=1.0, cvar_alpha=TINY_ALPHA,
                cvar_mean_weight=0.0),
            suppress_warnings=True)
        _solve(ef)
        self.assertAlmostEqual(pyo.value(ef.EF_Obj), TINY_CVAR, places=4)
        self.assertAlmostEqual(self._eta_value(ef), TINY_VAR, places=4)

    def test_weight_zero_is_risk_neutral(self):
        # beta = 0  =>  plain E[Cost]
        ef = sputils.create_EF(
            list(TINY_COSTS.keys()),
            cvar.cvar_scenario_creator(
                tiny_scenario_creator, cvar_weight=0.0, cvar_alpha=TINY_ALPHA),
            suppress_warnings=True)
        _solve(ef)
        self.assertAlmostEqual(pyo.value(ef.EF_Obj), TINY_EXPECTED_COST, places=4)


@unittest.skipIf(not solver_available, "no solver is available")
class FarmerRegressionTests(unittest.TestCase):
    """cvar_weight == 0 must reproduce the risk-neutral EF on a real model."""

    def test_beta_zero_reproduces_plain_ef(self):
        names = farmer.scenario_names_creator(3)
        kwargs = {"num_scens": 3}

        ef_plain = sputils.create_EF(
            names, farmer.scenario_creator,
            scenario_creator_kwargs=kwargs, suppress_warnings=True)
        _solve(ef_plain)

        ef_cvar = sputils.create_EF(
            names,
            cvar.cvar_scenario_creator(
                farmer.scenario_creator, cvar_weight=0.0, cvar_alpha=0.9),
            scenario_creator_kwargs=kwargs, suppress_warnings=True)
        _solve(ef_cvar)

        self.assertAlmostEqual(pyo.value(ef_plain.EF_Obj),
                               pyo.value(ef_cvar.EF_Obj), places=3)


@unittest.skipIf(not solver_available, "no solver is available")
class RiskReturnTradeoffTests(unittest.TestCase):
    """Risk aversion should trade expected cost for a better tail.

    The risk-neutral solution is *the* minimizer of E[Cost], so the CVaR-optimal
    plan can never have a lower E[Cost]; a good example makes it strictly higher
    (worse) while improving the worst case.  On the classic 3-scenario farmer the
    CVaR plan moves E[Cost] from about -108390 to -103313 while improving the
    worst-case cost from about -48820 to -57640.  A zero gap would mean the
    example does not exercise risk aversion and we would need a better one.
    """

    NUM_SCENS = 3            # classic Birge-Louveaux farmer (deterministic)
    CVAR_ALPHA = 0.8
    CVAR_WEIGHT = 5.0

    def _solved_ef(self, scenario_creator):
        names = farmer.scenario_names_creator(self.NUM_SCENS)
        ef = sputils.create_EF(
            names, scenario_creator,
            scenario_creator_kwargs={"num_scens": self.NUM_SCENS},
            suppress_warnings=True)
        _solve(ef)
        return ef, names

    def test_cvar_worsens_expectation_and_improves_tail(self):
        ef_rn, names = self._solved_ef(farmer.scenario_creator)
        ef_cvar, _ = self._solved_ef(
            cvar.cvar_scenario_creator(
                farmer.scenario_creator,
                cvar_weight=self.CVAR_WEIGHT, cvar_alpha=self.CVAR_ALPHA))

        e_rn = _farmer_expected_cost(ef_rn, names)
        e_cvar = _farmer_expected_cost(ef_cvar, names)
        # The expectation-only part of the objective is strictly worse under CVaR.
        self.assertGreater(e_cvar, e_rn + 100.0)

        # ... and that price buys a genuine hedge: a strictly better worst case.
        self.assertLess(_farmer_worst_cost(ef_cvar, names),
                        _farmer_worst_cost(ef_rn, names))


if __name__ == "__main__":
    unittest.main()
