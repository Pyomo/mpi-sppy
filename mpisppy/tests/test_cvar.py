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

import math
import unittest
import pyomo.environ as pyo

import mpisppy.opt.ph
import mpisppy.utils.sputils as sputils
import mpisppy.utils.cvar as cvar
import mpisppy.tests.examples.farmer as farmer
from mpisppy import MPI
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


# ---------------------------------------------------------------------------
# The maximization mirror of the tiny instance: same numbers {10,20,30,40},
# uniform, alpha = 0.6, but now *rewards* (maximize), so risk aversion is on the
# LOWER tail:
#   E[Reward]            = 25
#   lower VaR (eta*)     = 20
#   lower CVaR_0.6       = 20 - (1/(1-0.6))*(1/4)*(20-10) = 13.75
# ---------------------------------------------------------------------------
TINY_MAX_VAR = 20.0
TINY_MAX_CVAR = 13.75


def tiny_max_scenario_creator(sname, **kwargs):
    model = pyo.ConcreteModel(name=sname)
    model.x = pyo.Var(bounds=(0.0, 0.0))
    reward_expr = TINY_COSTS[sname] + model.x
    model.reward = pyo.Objective(expr=reward_expr, sense=pyo.maximize)
    model._mpisppy_probability = 1.0 / len(TINY_COSTS)
    sputils.attach_root_node(model, reward_expr, [model.x])
    return model


def unbounded_cost_scenario_creator(sname, **kwargs):
    # cost = 5 + x with x >= 0 unbounded above: min cost = 5, max cost = +inf
    model = pyo.ConcreteModel(name=sname)
    model.x = pyo.Var(domain=pyo.NonNegativeReals)
    cost = 5.0 + model.x
    model.cost = pyo.Objective(expr=cost, sense=pyo.minimize)
    model._mpisppy_probability = 1.0
    sputils.attach_root_node(model, cost, [model.x])
    return model


def integer_tail_scenario_creator(sname, **kwargs):
    # minimize cost = 10 y + 3 x, y binary with 3y <= 2 (so y = 0 in the MIP),
    # x in [0, 2].  The MIP cost range is [0, 6]; the LP relaxation would let
    # y = 2/3 and report a max of 10*(2/3) + 6 = 12.67, so the worst-case (upper)
    # bound distinguishes the MIP tail solve from a mere relaxation.
    model = pyo.ConcreteModel(name=sname)
    model.y = pyo.Var(domain=pyo.Binary)
    model.x = pyo.Var(bounds=(0.0, 2.0))
    model.con = pyo.Constraint(expr=3 * model.y <= 2)
    cost = 10 * model.y + 3 * model.x
    model.cost = pyo.Objective(expr=cost, sense=pyo.minimize)
    model._mpisppy_probability = 1.0
    sputils.attach_root_node(model, cost, [model.y, model.x])
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

    def test_maximize_mirrors_to_lower_tail(self):
        # A maximization objective gets the mirrored (lower-tail) formulation:
        # the excess var is NonPositiveReals and WITH_CVAR keeps the maximize
        # sense; the constant objective expression is unchanged.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0.0, 0.0))
        reward = 10.0 + m.x
        m.reward = pyo.Objective(expr=reward, sense=pyo.maximize)
        m._mpisppy_probability = 1.0
        sputils.attach_root_node(m, reward, [m.x])
        cvar.add_cvar(m, cvar_weight=1.0, cvar_alpha=0.9)
        self.assertIs(m._mpisppy_cvar_excess.domain, pyo.NonPositiveReals)
        active = sputils.find_active_objective(m)
        self.assertIs(active, m.WITH_CVAR)
        self.assertEqual(active.sense, pyo.maximize)
        self.assertFalse(m.reward.active)


class EtaBoundTests(unittest.TestCase):
    """The eta bound machinery; these do not need a solver.

    add_cvar stashes each scenario's cost range (FBBT) and applies any explicit
    override immediately; set_cvar_eta_bounds reduces the stashed ranges to a
    global [lb, ub] and bounds eta.  The tiny scenarios have a constant cost
    (x is fixed to 0), so each scenario's FBBT cost range is a single point equal
    to its cost.
    """

    def _tiny_dict(self, **cvar_kwargs):
        wrapped = cvar.cvar_scenario_creator(
            tiny_scenario_creator, cvar_weight=1.0, cvar_alpha=TINY_ALPHA,
            **cvar_kwargs)
        return {sname: wrapped(sname) for sname in TINY_COSTS}

    def test_add_cvar_stashes_fbbt_cost_range(self):
        s = tiny_scenario_creator("s2")            # constant cost 30
        cvar.add_cvar(s, cvar_weight=1.0, cvar_alpha=TINY_ALPHA)
        self.assertEqual(s._mpisppy_cvar_eta_cost_bounds, (30.0, 30.0))
        # eta itself is not bounded yet (that is the reduction's job)
        self.assertIsNone(s._mpisppy_cvar_eta.lb)
        self.assertIsNone(s._mpisppy_cvar_eta.ub)

    def test_auto_eta_bound_off_stashes_infinite(self):
        s = tiny_scenario_creator("s2")
        cvar.add_cvar(s, cvar_weight=1.0, cvar_alpha=TINY_ALPHA,
                      auto_eta_bound=False)
        lb, ub = s._mpisppy_cvar_eta_cost_bounds
        self.assertEqual(lb, -math.inf)
        self.assertEqual(ub, math.inf)

    def test_user_override_applied_immediately(self):
        s = tiny_scenario_creator("s2")
        cvar.add_cvar(s, cvar_weight=1.0, cvar_alpha=TINY_ALPHA,
                      eta_lb=-5.0, eta_ub=99.0)
        self.assertEqual(s._mpisppy_cvar_eta.lb, -5.0)
        self.assertEqual(s._mpisppy_cvar_eta.ub, 99.0)

    def test_set_bounds_reduces_to_global_cost_range(self):
        scendict = self._tiny_dict()               # costs {10,20,30,40}
        cvar.set_cvar_eta_bounds(scendict, MPI.COMM_SELF)
        for s in scendict.values():
            self.assertEqual(s._mpisppy_cvar_eta.bounds, (10.0, 40.0))

    def test_user_bound_survives_reduction(self):
        # an explicit ub must win over the (looser) auto ub of 40
        scendict = self._tiny_dict(eta_ub=25.0)
        cvar.set_cvar_eta_bounds(scendict, MPI.COMM_SELF)
        for s in scendict.values():
            self.assertEqual(s._mpisppy_cvar_eta.bounds, (10.0, 25.0))

    def test_auto_off_leaves_eta_free(self):
        scendict = self._tiny_dict(auto_eta_bound=False)
        cvar.set_cvar_eta_bounds(scendict, MPI.COMM_SELF)
        for s in scendict.values():
            self.assertEqual(s._mpisppy_cvar_eta.bounds, (None, None))

    def test_unbounded_cost_leaves_side_free(self):
        # a scenario whose cost can grow without bound gets no finite auto bound
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)   # unbounded above
        cost = 5.0 + m.x
        m.cost = pyo.Objective(expr=cost, sense=pyo.minimize)
        m._mpisppy_probability = 1.0
        sputils.attach_root_node(m, cost, [m.x])
        cvar.add_cvar(m, cvar_weight=1.0, cvar_alpha=TINY_ALPHA)
        lb, ub = m._mpisppy_cvar_eta_cost_bounds
        self.assertEqual(lb, 5.0)
        self.assertEqual(ub, math.inf)
        cvar.set_cvar_eta_bounds({"s": m}, MPI.COMM_SELF)
        # lower bound set (5.0); upper stays free
        self.assertEqual(m._mpisppy_cvar_eta.lb, 5.0)
        self.assertIsNone(m._mpisppy_cvar_eta.ub)

    def test_set_bounds_noop_without_cvar(self):
        # a plain scenario (no eta) must be left untouched
        s = tiny_scenario_creator("s0")
        cvar.set_cvar_eta_bounds({"s0": s}, MPI.COMM_SELF)  # must not raise
        self.assertFalse(hasattr(s, "_mpisppy_cvar_eta"))


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
class EFEtaBoundTests(unittest.TestCase):
    """The full ExtensiveForm path (through SPBase) bounds eta without changing
    the optimum."""

    def test_ef_bounds_eta_and_keeps_optimum(self):
        from mpisppy.opt.ef import ExtensiveForm
        names = list(TINY_COSTS.keys())
        creator = cvar.cvar_scenario_creator(
            tiny_scenario_creator, cvar_weight=1.0, cvar_alpha=TINY_ALPHA)
        ef = ExtensiveForm({"solver": solver_name}, names, creator)
        # SPBase set a valid global bound on every scenario's eta
        for s in ef.local_scenarios.values():
            self.assertEqual(s._mpisppy_cvar_eta.bounds, (10.0, 40.0))
        ef.solve_extensive_form()
        self.assertAlmostEqual(pyo.value(ef.ef.EF_Obj),
                               TINY_EXPECTED_COST + TINY_CVAR, places=4)
        self.assertAlmostEqual(
            pyo.value(ef.local_scenarios["s0"]._mpisppy_cvar_eta),
            TINY_VAR, places=4)


@unittest.skipIf(not solver_available, "no solver is available")
class SolveBoundTests(unittest.TestCase):
    """compute_cvar_eta_bounds_by_solve: relaxation on the slack side, MIP (loose
    gap, dual bound) on the worst-case tail."""

    def test_solve_matches_cost_range_on_tiny(self):
        lb, ub = cvar.compute_cvar_eta_bounds_by_solve(
            list(TINY_COSTS.keys()), tiny_scenario_creator, solver_name,
            comm=MPI.COMM_SELF)
        self.assertAlmostEqual(lb, 10.0, places=4)
        self.assertAlmostEqual(ub, 40.0, places=4)

    def test_solve_on_maximize(self):
        # For a maximize the tail is the LOW (reward) side and the slack is the
        # high side; the global reward range is still [10, 40].
        lb, ub = cvar.compute_cvar_eta_bounds_by_solve(
            list(TINY_COSTS.keys()), tiny_max_scenario_creator, solver_name,
            comm=MPI.COMM_SELF)
        self.assertAlmostEqual(lb, 10.0, places=4)
        self.assertAlmostEqual(ub, 40.0, places=4)

    def test_tail_solves_the_mip_not_the_relaxation(self):
        # With a tight gap the worst-case (upper) bound is the MIP max (6.0),
        # NOT the LP relaxation value (12.67); the slack (lower) side is 0.
        lb, ub = cvar.compute_cvar_eta_bounds_by_solve(
            ["s0"], integer_tail_scenario_creator, solver_name,
            comm=MPI.COMM_SELF, mipgap=0.0)
        self.assertAlmostEqual(lb, 0.0, places=4)
        self.assertAlmostEqual(ub, 6.0, places=4)

    def test_loose_gap_tail_bound_is_valid(self):
        # A loose gap only makes the dual bound looser, never invalid: it must
        # still enclose the true MIP max of 6.0 and stay finite.
        lb, ub = cvar.compute_cvar_eta_bounds_by_solve(
            ["s0"], integer_tail_scenario_creator, solver_name,
            comm=MPI.COMM_SELF, mipgap=0.9)
        self.assertGreaterEqual(ub, 6.0 - 1e-6)
        self.assertTrue(ub < float("inf"))

    def test_solve_bounds_profit_side_of_farmer(self):
        # farmer's cost is unbounded above (unlimited purchases), so the tail MIP
        # is unbounded and there is no upper bound; the "how cheap" (profit) side
        # is a finite LP relaxation, which FBBT cannot find at all.
        names = farmer.scenario_names_creator(3)
        lb, ub = cvar.compute_cvar_eta_bounds_by_solve(
            names, farmer.scenario_creator, solver_name,
            scenario_creator_kwargs={"num_scens": 3}, comm=MPI.COMM_SELF)
        self.assertIsNotNone(lb)          # a finite profit bound exists
        self.assertLess(lb, 0.0)
        self.assertIsNone(ub)             # buying is unbounded, so no upper bound


@unittest.skipIf(not solver_available, "no solver is available")
class EFMaxClosedFormTests(unittest.TestCase):
    """Maximization EF-CVaR (lower-tail mirror) on the tiny instance."""

    def _eta_value(self, ef):
        return pyo.value(getattr(ef, "s0")._mpisppy_cvar_eta)

    def test_ef_cvar_max_matches_closed_form(self):
        # lambda = 1, beta = 1  =>  E[Reward] + lower-tail CVaR
        ef = sputils.create_EF(
            list(TINY_COSTS.keys()),
            cvar.cvar_scenario_creator(
                tiny_max_scenario_creator, cvar_weight=1.0, cvar_alpha=TINY_ALPHA),
            suppress_warnings=True)
        _solve(ef)
        self.assertAlmostEqual(pyo.value(ef.EF_Obj),
                               TINY_EXPECTED_COST + TINY_MAX_CVAR, places=4)
        self.assertAlmostEqual(self._eta_value(ef), TINY_MAX_VAR, places=4)

    def test_pure_cvar_max(self):
        # lambda = 0, beta = 1  =>  lower-tail CVaR only
        ef = sputils.create_EF(
            list(TINY_COSTS.keys()),
            cvar.cvar_scenario_creator(
                tiny_max_scenario_creator, cvar_weight=1.0, cvar_alpha=TINY_ALPHA,
                cvar_mean_weight=0.0),
            suppress_warnings=True)
        _solve(ef)
        self.assertAlmostEqual(pyo.value(ef.EF_Obj), TINY_MAX_CVAR, places=4)
        self.assertAlmostEqual(self._eta_value(ef), TINY_MAX_VAR, places=4)


@unittest.skipIf(not solver_available, "no solver is available")
class SerialPHTests(unittest.TestCase):
    """PH (run serially) accepts a CVaR-transformed model and yields a valid bound.

    Because eta is appended to the root node it is "just another first-stage
    variable", so PH builds and iterates with no algorithm changes.  We assert
    the trivial outer bound brackets the EF-CVaR optimum (a rho-independent
    guarantee); the EF closed-form tests above verify the CVaR value itself, and
    the cylinder bound-sandwich test verifies the full decomposition.
    """

    def test_ph_on_cvar_runs_and_bounds(self):
        names = farmer.scenario_names_creator(3)
        kwargs = {"num_scens": 3}
        creator = cvar.cvar_scenario_creator(
            farmer.scenario_creator, cvar_weight=2.0, cvar_alpha=0.8)

        ef = sputils.create_EF(
            names, creator, scenario_creator_kwargs=kwargs, suppress_warnings=True)
        _solve(ef)
        ef_obj = pyo.value(ef.EF_Obj)

        options = {
            "solver_name": solver_name,
            "PHIterLimit": 10,
            "defaultPHrho": 1.0,
            "convthresh": 1e-8,
            "verbose": False,
            "display_timing": False,
            "display_progress": False,
            "iter0_solver_options": None,
            "iterk_solver_options": None,
        }
        ph = mpisppy.opt.ph.PH(
            options, names, creator, lambda rank, sname, scenario: None,
            scenario_creator_kwargs=kwargs)
        conv, obj, tbound = ph.ph_main()

        # eta is being treated as a nonant by PH
        for s in ph.local_scenarios.values():
            self.assertTrue(hasattr(s, "_mpisppy_cvar_eta"))
        # the trivial (iter0) bound is a valid outer bound for this minimization
        self.assertLessEqual(tbound, ef_obj + 1e-6)


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
