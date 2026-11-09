###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
###############################################################################
"""Tests for the experimental centered-reduced-cost PH initialization."""

import unittest

import pyomo.environ as pyo

import mpisppy.opt.ph
import mpisppy.utils.sputils as sputils
from mpisppy.tests.utils import get_solver

_PROBS = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
_RCS = {"s0": 1.0, "s1": 4.0, "s2": -2.0}
_SOLVER_AVAILABLE, _SOLVER_NAME, _, _ = get_solver()


def _scenario_creator(name, integer=False, sense=pyo.minimize):
    model = pyo.ConcreteModel(name=name)
    domain = pyo.Integers if integer else pyo.Reals
    model.x = pyo.Var(domain=domain, bounds=(0, 10), initialize=0)
    model.obj = pyo.Objective(expr=model.x, sense=sense)
    model._mpisppy_probability = _PROBS[name]
    sputils.attach_root_node(model, model.obj.expr, [model.x])
    return model


def _priced_scenario_creator(name):
    model = _scenario_creator(name)
    model.obj.set_value(_RCS[name] * model.x)
    return model


def _options(**overrides):
    options = {
        "solver_name": "unused",
        "PHIterLimit": 2,
        "defaultPHrho": 2.0,
        "convthresh": 0,
        "verbose": False,
        "display_progress": False,
        "reduced_costs_w_init": 0.5,
    }
    options.update(overrides)
    return options


class TestReducedCostsWInit(unittest.TestCase):

    def _ph(self, **options):
        return mpisppy.opt.ph.PH(
            _options(**options), list(_PROBS), _scenario_creator)

    def test_probability_centered_weights_and_first_update_are_blended(self):
        ph = self._ph()
        ph.PH_Prep()
        ph._attach_reduced_cost_suffixes()
        for name, scenario in ph.local_scenarios.items():
            scenario.rc[scenario.x] = _RCS[name]

        ph._initialize_W_from_reduced_costs()

        for scenario in ph.local_scenarios.values():
            self.assertFalse(hasattr(scenario, "rc"))

        rbar = sum(_PROBS[s] * _RCS[s] for s in _PROBS)
        for name, scenario in ph.local_scenarios.items():
            expected = -0.5 * (_RCS[name] - rbar)
            self.assertAlmostEqual(
                pyo.value(scenario._mpisppy_model.W[("ROOT", 0)]), expected)

        weighted_w = sum(
            _PROBS[name]
            * pyo.value(scenario._mpisppy_model.W[("ROOT", 0)])
            for name, scenario in ph.local_scenarios.items())
        self.assertAlmostEqual(weighted_w, 0.0)

        before = {
            name: pyo.value(s._mpisppy_model.W[("ROOT", 0)])
            for name, s in ph.local_scenarios.items()}
        for scenario in ph.local_scenarios.values():
            scenario.x.set_value(3.0)
            scenario._mpisppy_model.xbars[("ROOT", 0)]._value = 1.0
        ph._PHIter = 1
        ph._update_W_for_iteration(False)
        # rho * (x - xbar) = 2 * 2 = 4, blended at (1 - eta)=0.5.
        after_first = {
            name: pyo.value(s._mpisppy_model.W[("ROOT", 0)])
            for name, s in ph.local_scenarios.items()}
        for name in before:
            self.assertAlmostEqual(after_first[name], before[name] + 2.0)

        ph._PHIter = 2
        ph._update_W_for_iteration(False)
        for name, scenario in ph.local_scenarios.items():
            self.assertAlmostEqual(
                pyo.value(scenario._mpisppy_model.W[("ROOT", 0)]),
                before[name] + 6.0)

    @unittest.skipIf(not _SOLVER_AVAILABLE, "no LP solver is available")
    def test_iter0_imports_solver_reduced_costs(self):
        ph = mpisppy.opt.ph.PH(
            _options(solver_name=_SOLVER_NAME, PHIterLimit=0,
                     reduced_costs_w_init=1.0),
            list(_PROBS), _priced_scenario_creator)
        ph.PH_Prep(attach_prox=False)
        ph.Iter0()

        rbar = sum(_PROBS[s] * _RCS[s] for s in _PROBS)
        for name, scenario in ph.local_scenarios.items():
            self.assertFalse(hasattr(scenario, "rc"))
            self.assertAlmostEqual(
                pyo.value(scenario._mpisppy_model.W[("ROOT", 0)]),
                -(_RCS[name] - rbar), places=7)

    def test_preexisting_reduced_cost_suffix_is_preserved(self):
        ph = self._ph()
        ph.PH_Prep()
        for name, scenario in ph.local_scenarios.items():
            scenario.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            scenario.rc[scenario.x] = _RCS[name]
        ph._attach_reduced_cost_suffixes()

        ph._initialize_W_from_reduced_costs()

        for scenario in ph.local_scenarios.values():
            self.assertTrue(hasattr(scenario, "rc"))

    def test_eta_validation(self):
        for eta in (-0.1, 1.1, float("inf")):
            with self.subTest(eta=eta):
                with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                    self._ph(reduced_costs_w_init=eta)

    def test_zero_disables_initialization_and_uses_ordinary_update(self):
        ph = self._ph(reduced_costs_w_init=0)
        self.assertIsNone(ph.options["reduced_costs_w_init"])
        ph.PH_Prep()
        for scenario in ph.local_scenarios.values():
            self.assertFalse(hasattr(scenario, "rc"))
            scenario.x.set_value(3.0)
            scenario._mpisppy_model.xbars[("ROOT", 0)]._value = 1.0

        ph._PHIter = 1
        ph._update_W_for_iteration(False)

        for scenario in ph.local_scenarios.values():
            self.assertAlmostEqual(
                pyo.value(scenario._mpisppy_model.W[("ROOT", 0)]), 4.0)

    def test_discrete_and_maximization_models_are_refused(self):
        with self.assertRaisesRegex(RuntimeError, "continuous"):
            mpisppy.opt.ph.PH(
                _options(), list(_PROBS), _scenario_creator,
                scenario_creator_kwargs={"integer": True})
        with self.assertRaisesRegex(RuntimeError, "minimization"):
            mpisppy.opt.ph.PH(
                _options(), list(_PROBS), _scenario_creator,
                scenario_creator_kwargs={"sense": pyo.maximize})


if __name__ == "__main__":
    unittest.main()
