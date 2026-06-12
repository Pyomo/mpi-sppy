###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for PySP-style chance constraints (mpisppy/utils/chance_constraint.py):
#   - structural checks of the EF transform (no solver needed)
#   - EF-CC vs a closed-form "satisfy the cheapest (1-alpha) mass" on a tiny
#     deterministic instance
#   - alpha = 0 forces every scenario to be satisfied (robust)
#   - removing the chance constraint reduces to the risk-neutral EF
#   - indexed indicator => one chance constraint per index
#   - input validation

import math
import unittest
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.utils.chance_constraint as cc
from mpisppy.tests.utils import get_solver, limit_solver_threads

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()


# ---------------------------------------------------------------------------
# A tiny deterministic instance with a known closed form.
#
# Satisfying scenario s (its binary indicator z_s = 1) costs a known amount a_s;
# leaving it unsatisfied (z_s = 0) is free.  The risk-neutral optimum is therefore
# z_s = 0 for all s (cost 0).  The chance constraint
#
#     Sum_s p_s z_s >= 1 - alpha
#
# forces enough scenarios satisfied; to minimize cost the EF satisfies the
# CHEAPEST ceil((1-alpha) * N) of them (uniform probabilities).
#
# Costs {1, 2, 3, 4}, uniform p_s = 1/4:
#   alpha = 0     -> satisfy 4 (all): cost (1+2+3+4)/4 = 2.5
#   alpha = 0.5   -> satisfy 2      : cost (1+2)/4     = 0.75
#   alpha = 0.75  -> satisfy 1      : cost (1)/4       = 0.25
# ---------------------------------------------------------------------------
TINY_COSTS = {"s0": 1.0, "s1": 2.0, "s2": 3.0, "s3": 4.0}


def tiny_scenario_creator(sname, **kwargs):
    costs = kwargs.get("costs", TINY_COSTS)
    model = pyo.ConcreteModel(name=sname)
    model.x = pyo.Var(bounds=(0.0, 0.0))            # trivial first-stage nonant
    model.z = pyo.Var(domain=pyo.Binary)            # indicator: 1 == satisfied
    cost_expr = costs[sname] * model.z + model.x
    model.cost = pyo.Objective(expr=cost_expr, sense=pyo.minimize)
    model._mpisppy_probability = 1.0 / len(costs)
    sputils.attach_root_node(model, model.x, [model.x])
    return model


# Indexed indicator: each scenario carries z[0], z[1].  The cost of satisfying
# index j in scenario s is COSTS_IDX[j][s]; the two indices are independent, so
# each gets its own chance constraint and its own cheapest-k selection.
COSTS_IDX = {
    0: {"s0": 1.0, "s1": 2.0, "s2": 3.0, "s3": 4.0},   # cheapest: s0, s1, ...
    1: {"s0": 4.0, "s1": 3.0, "s2": 2.0, "s3": 1.0},   # cheapest: s3, s2, ...
}


def indexed_scenario_creator(sname, **kwargs):
    model = pyo.ConcreteModel(name=sname)
    model.x = pyo.Var(bounds=(0.0, 0.0))
    model.z = pyo.Var([0, 1], domain=pyo.Binary)
    cost_expr = sum(COSTS_IDX[j][sname] * model.z[j] for j in (0, 1)) + model.x
    model.cost = pyo.Objective(expr=cost_expr, sense=pyo.minimize)
    model._mpisppy_probability = 1.0 / 4
    sputils.attach_root_node(model, model.x, [model.x])
    return model


def _make_ef(scenario_creator, names, scenario_creator_kwargs=None):
    return sputils.create_EF(
        names, scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs, suppress_warnings=True)


def _solve(model):
    solver = pyo.SolverFactory(solver_name)
    limit_solver_threads(solver, solver_name)
    if "_persistent" in solver_name:
        solver.set_instance(model)
    results = solver.solve(model, tee=False)
    pyo.assert_optimal_termination(results)
    return results


def _z_values(ef, names, index=None):
    out = {}
    for sname in names:
        z = getattr(ef, sname).z
        out[sname] = round(pyo.value(z if index is None else z[index]))
    return out


class StructureTests(unittest.TestCase):
    """The EF transform's structure; these do not need a solver."""

    def test_scalar_constraint_added(self):
        ef = _make_ef(tiny_scenario_creator, list(TINY_COSTS))
        con = cc.add_chance_constraint(ef, cc_indicator_var_name="z", cc_alpha=0.5)
        self.assertIs(con, ef._mpisppy_chance_constraint)
        self.assertFalse(con.is_indexed())
        # RHS = (1 - alpha) * total_prob = 0.5 * 1.0
        self.assertAlmostEqual(pyo.value(con.lower), 0.5, places=6)

    def test_indexed_one_constraint_per_index(self):
        ef = _make_ef(indexed_scenario_creator, list(TINY_COSTS))
        con = cc.add_chance_constraint(ef, cc_indicator_var_name="z", cc_alpha=0.5)
        self.assertTrue(con.is_indexed())
        self.assertEqual(set(con.keys()), {0, 1})

    def test_missing_indicator_raises(self):
        ef = _make_ef(tiny_scenario_creator, list(TINY_COSTS))
        with self.assertRaises(ValueError):
            cc.add_chance_constraint(ef, cc_indicator_var_name="nope", cc_alpha=0.5)

    def test_bad_alpha_raises(self):
        for bad in (1.0, -0.1, 1.5):
            ef = _make_ef(tiny_scenario_creator, list(TINY_COSTS))
            with self.assertRaises(ValueError):
                cc.add_chance_constraint(ef, cc_indicator_var_name="z", cc_alpha=bad)


@unittest.skipIf(not solver_available, "no solver is available")
class EFClosedFormTests(unittest.TestCase):
    """EF-CC on the tiny instance vs the closed-form cheapest-k selection."""

    def _solve_cc(self, alpha):
        names = list(TINY_COSTS)
        ef = _make_ef(tiny_scenario_creator, names)
        cc.add_chance_constraint(ef, cc_indicator_var_name="z", cc_alpha=alpha)
        _solve(ef)
        return ef, names

    def _expected(self, alpha):
        # satisfy the cheapest k = ceil((1-alpha) * N) scenarios
        k = math.ceil((1.0 - alpha) * len(TINY_COSTS))
        cheapest = sorted(TINY_COSTS, key=lambda s: TINY_COSTS[s])[:k]
        obj = sum(TINY_COSTS[s] for s in cheapest) / len(TINY_COSTS)
        return set(cheapest), obj

    def test_alpha_half(self):
        ef, names = self._solve_cc(0.5)
        chosen, obj = self._expected(0.5)               # {s0, s1}, 0.75
        self.assertAlmostEqual(pyo.value(ef.EF_Obj), obj, places=4)
        z = _z_values(ef, names)
        self.assertEqual({s for s, v in z.items() if v == 1}, chosen)

    def test_alpha_large_picks_fewer(self):
        ef, names = self._solve_cc(0.75)
        chosen, obj = self._expected(0.75)              # {s0}, 0.25
        self.assertAlmostEqual(pyo.value(ef.EF_Obj), obj, places=4)
        self.assertEqual({s for s, v in _z_values(ef, names).items() if v == 1},
                         chosen)

    def test_alpha_zero_is_robust(self):
        ef, names = self._solve_cc(0.0)
        chosen, obj = self._expected(0.0)               # all four, 2.5
        self.assertAlmostEqual(pyo.value(ef.EF_Obj), obj, places=4)
        self.assertEqual(set(_z_values(ef, names).values()), {1})

    def test_no_chance_constraint_is_risk_neutral(self):
        # Without the chance constraint the optimum leaves everything unsatisfied.
        ef = _make_ef(tiny_scenario_creator, list(TINY_COSTS))
        _solve(ef)
        self.assertAlmostEqual(pyo.value(ef.EF_Obj), 0.0, places=4)


@unittest.skipIf(not solver_available, "no solver is available")
class IndexedTests(unittest.TestCase):
    """An indexed indicator yields an independent cheapest-k per index."""

    def test_each_index_selects_its_own_cheapest(self):
        names = list(TINY_COSTS)
        ef = _make_ef(indexed_scenario_creator, names)
        cc.add_chance_constraint(ef, cc_indicator_var_name="z", cc_alpha=0.5)
        _solve(ef)
        # alpha = 0.5 => satisfy the 2 cheapest scenarios for each index
        sat0 = {s for s, v in _z_values(ef, names, index=0).items() if v == 1}
        sat1 = {s for s, v in _z_values(ef, names, index=1).items() if v == 1}
        self.assertEqual(sat0, {"s0", "s1"})            # cheapest for index 0
        self.assertEqual(sat1, {"s3", "s2"})            # cheapest for index 1
        # objective: (1+2)/4 for index 0 plus (1+2)/4 for index 1
        self.assertAlmostEqual(pyo.value(ef.EF_Obj), 0.75 + 0.75, places=4)


if __name__ == "__main__":
    unittest.main()
