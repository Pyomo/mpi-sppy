###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the --*-try-jensens-first options.

Serial unit tests only: we exercise the pieces that do not need an MPI
wheel (sputils helper, farmer's expected_value_creator, _JensensMixin's
build/assert/solve, and the cfg_vanilla wiring errors). The end-to-end
MPI path is smoke-tested manually and by straight_tests.
"""

import os
import sys
import unittest

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy.cylinders._jensens_mixin import _JensensMixin
from mpisppy.tests.utils import get_solver

# make examples/farmer importable for the positive tests
_FARMER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "farmer")
sys.path.insert(0, _FARMER_DIR)
import farmer  # noqa: E402

solver_available, solver_name, _, _ = get_solver()


class TestAssertJensenIntegerSafe(unittest.TestCase):
    """Unit tests for sputils.assert_jensen_integer_safe."""

    def _mk_model(self, nonant_is_int=False, recourse_is_int=False):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.Integers if nonant_is_int else pyo.Reals,
                      bounds=(0, 10))
        m.y = pyo.Var(within=pyo.Integers if recourse_is_int else pyo.Reals,
                      bounds=(0, 10))
        m.fsc = pyo.Expression(expr=m.x)
        m.obj = pyo.Objective(expr=m.x + m.y)
        sputils.attach_root_node(m, m.fsc, [m.x])
        return m

    def test_continuous_recourse_passes(self):
        m = self._mk_model(nonant_is_int=False, recourse_is_int=False)
        sputils.assert_jensen_integer_safe(m)

    def test_integer_nonant_passes(self):
        m = self._mk_model(nonant_is_int=True, recourse_is_int=False)
        sputils.assert_jensen_integer_safe(m)

    def test_integer_recourse_raises(self):
        m = self._mk_model(nonant_is_int=False, recourse_is_int=True)
        with self.assertRaises(RuntimeError) as ctx:
            sputils.assert_jensen_integer_safe(m)
        self.assertIn("integer/binary Var", str(ctx.exception))

    def test_binary_recourse_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.b = pyo.Var(within=pyo.Binary)
        m.fsc = pyo.Expression(expr=m.x)
        m.obj = pyo.Objective(expr=m.x + m.b)
        sputils.attach_root_node(m, m.fsc, [m.x])
        with self.assertRaises(RuntimeError):
            sputils.assert_jensen_integer_safe(m)


class TestFarmerExpectedValueCreator(unittest.TestCase):
    """Unit tests for the farmer expected_value_creator."""

    def test_requires_num_scens(self):
        with self.assertRaises(ValueError):
            farmer.expected_value_creator("EV")

    def test_probability_is_one(self):
        ev = farmer.expected_value_creator("EV", num_scens=6)
        self.assertEqual(ev._mpisppy_probability, 1.0)

    def test_two_stage_tree(self):
        ev = farmer.expected_value_creator("EV", num_scens=6)
        self.assertEqual(len(ev._mpisppy_node_list), 1)
        self.assertEqual(ev._mpisppy_node_list[0].name, "ROOT")

    def test_yields_are_mean_of_per_scenario_yields(self):
        num_scens = 6
        ev = farmer.expected_value_creator("EV", num_scens=num_scens)
        snames = farmer.scenario_names_creator(num_scens)
        per = [farmer._scenario_data(s) for s in snames]
        for crop in ev.CROPS:
            expected = sum(d["Yield"][crop] for d in per) / num_scens
            self.assertAlmostEqual(pyo.value(ev.Yield[crop]), expected)

    def test_seedoffset_is_threaded(self):
        ev0 = farmer.expected_value_creator("EV", num_scens=6, seedoffset=0)
        ev7 = farmer.expected_value_creator("EV", num_scens=6, seedoffset=7)
        y0 = {c: pyo.value(ev0.Yield[c]) for c in ev0.CROPS}
        y7 = {c: pyo.value(ev7.Yield[c]) for c in ev7.CROPS}
        self.assertNotEqual(y0, y7)

    def test_deterministic(self):
        a = farmer.expected_value_creator("EV", num_scens=6)
        b = farmer.expected_value_creator("EV", num_scens=6)
        ya = {c: pyo.value(a.Yield[c]) for c in a.CROPS}
        yb = {c: pyo.value(b.Yield[c]) for c in b.CROPS}
        self.assertEqual(ya, yb)


class _FakeSpoke(_JensensMixin):
    """Minimal stand-in that exposes the options interface _JensensMixin needs."""

    def __init__(self, jensens_dict, all_scenario_names, solver_name):
        class _Opt:
            pass
        self.opt = _Opt()
        self.opt.options = {
            "solver_name": solver_name,
            "iter0_solver_options": {},
            "jensens": jensens_dict,
        }
        self.opt.all_scenario_names = all_scenario_names


@unittest.skipUnless(solver_available, "no Pyomo-compatible MIP solver")
class TestJensensMixinEndToEnd(unittest.TestCase):
    """Light integration: build the EV model, solve it, pack the cache."""

    def _spoke(self):
        jdict = {
            "expected_value_creator": farmer.expected_value_creator,
            "scenario_creator_kwargs": {"num_scens": 3},
        }
        return _FakeSpoke(jdict, farmer.scenario_names_creator(3), solver_name)

    def test_enabled_flag(self):
        self.assertTrue(self._spoke()._jensens_enabled())

    def test_build_ev_is_two_stage(self):
        ev = self._spoke()._jensens_build_ev()
        self.assertEqual(len(ev._mpisppy_node_list), 1)

    def test_outer_bound_safe_for_continuous_recourse(self):
        sp = self._spoke()
        ev = sp._jensens_build_ev()
        # farmer has continuous recourse — safe for outer bound
        sp._jensens_assert_safe_for_outer_bound(ev)

    def test_solve_returns_finite_obj_and_root_nonants(self):
        sp = self._spoke()
        ev = sp._jensens_build_ev()
        obj, nonants = sp._jensens_solve(ev)
        self.assertTrue(pyo.value(obj) == pyo.value(obj))  # not NaN
        self.assertEqual(len(nonants),
                         len(ev._mpisppy_node_list[0].nonant_vardata_list))

    def test_pack_nonant_cache_is_root_keyed(self):
        sp = self._spoke()
        cache = sp._jensens_pack_nonant_cache([1.0, 2.0, 3.0])
        self.assertEqual(list(cache.keys()), ["ROOT"])
        self.assertEqual(list(cache["ROOT"]), [1.0, 2.0, 3.0])


class TestMissingExpectedValueCreator(unittest.TestCase):
    """cfg_vanilla must raise when the flag is on but the module lacks
    expected_value_creator."""

    def test_maybe_attach_jensens_raises_when_callable_is_none(self):
        import mpisppy.utils.config as cfg_mod
        from mpisppy.utils.cfg_vanilla import _maybe_attach_jensens

        cfg = cfg_mod.Config()
        cfg.lagrangian_args()
        cfg.lagrangian_try_jensens_first = True

        spoke_dict = {"opt_kwargs": {"options": {}}}
        with self.assertRaises(RuntimeError) as ctx:
            _maybe_attach_jensens(spoke_dict, cfg, "lagrangian",
                                  expected_value_creator=None,
                                  scenario_creator_kwargs={})
        self.assertIn("expected_value_creator", str(ctx.exception))

    def test_maybe_attach_jensens_noop_when_flag_off(self):
        import mpisppy.utils.config as cfg_mod
        from mpisppy.utils.cfg_vanilla import _maybe_attach_jensens

        cfg = cfg_mod.Config()
        cfg.lagrangian_args()
        # flag defaults to False; make sure a missing creator is harmless
        spoke_dict = {"opt_kwargs": {"options": {}}}
        _maybe_attach_jensens(spoke_dict, cfg, "lagrangian",
                              expected_value_creator=None,
                              scenario_creator_kwargs={})
        self.assertNotIn("jensens", spoke_dict["opt_kwargs"]["options"])


if __name__ == "__main__":
    unittest.main()
