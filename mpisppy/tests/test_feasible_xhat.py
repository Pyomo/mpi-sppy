###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the ``feasible_xhat_creator`` convention and its
``average_xhat_nonants`` helper.

Two prototypes are exercised: netdes (LP-relax + ceil, via the helper)
and sslp (LP-relax + average + round, rolls own). The convention's
contract is that the returned candidate must be feasible to fix in
every real scenario's per-scenario subproblem; the netdes test
verifies this directly.
"""

import os
import sys
import unittest

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import mpisppy.utils.sputils as sputils
from mpisppy.utils.xhat_helpers import average_xhat_nonants, lp_xbar_nonants
from mpisppy.tests.utils import get_solver, limit_solver_threads

_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
)
sys.path.insert(0, os.path.join(_EXAMPLES_DIR, "farmer"))
sys.path.insert(0, os.path.join(_EXAMPLES_DIR, "netdes"))
sys.path.insert(0, os.path.join(_EXAMPLES_DIR, "sslp"))

import farmer  # noqa: E402
import farmer_auxiliary  # noqa: E402
import netdes  # noqa: E402
import netdes_auxiliary  # noqa: E402
import sslp_auxiliary  # noqa: E402

solver_available, solver_name, _, _ = get_solver()

_NETDES_DATA = os.path.join(
    _EXAMPLES_DIR, "netdes", "data", "network-10-20-L-01.dat"
)
_SSLP_DATA_DIR = os.path.join(
    _EXAMPLES_DIR, "sslp", "data", "sslp_15_45_5", "scenariodata"
)


class TestAverageXhatNonantsContract(unittest.TestCase):
    """Solver-free contract checks for the helper."""

    def test_rejects_no_node_list(self):
        def bad_creator(name):
            m = pyo.ConcreteModel()
            m.x = pyo.Var()
            m.obj = pyo.Objective(expr=m.x)
            return m

        with self.assertRaises(RuntimeError) as ctx:
            average_xhat_nonants(bad_creator, solver_name="cplex")
        self.assertIn("_mpisppy_node_list", str(ctx.exception))

    def test_rejects_multi_stage(self):
        def two_node_creator(name):
            m = pyo.ConcreteModel()
            m.x = pyo.Var()
            m.y = pyo.Var()
            m.fsc = pyo.Expression(expr=m.x)
            m.obj = pyo.Objective(expr=m.x + m.y)
            sputils.attach_root_node(m, m.fsc, [m.x])
            # tack on a fake second node
            from mpisppy.scenario_tree import ScenarioNode
            m._mpisppy_node_list.append(
                ScenarioNode("STAGE2", 1.0, 2, m.fsc, [m.y], m,
                             parent_name="ROOT")
            )
            return m

        with self.assertRaises(RuntimeError) as ctx:
            average_xhat_nonants(two_node_creator, solver_name="cplex")
        self.assertIn("two-stage only", str(ctx.exception))


class TestLpXbarNonantsContract(unittest.TestCase):
    """Solver-free input-validation checks for the LP-xbar helper."""

    def test_rejects_empty_scenario_names(self):
        def never_called(name, **k):
            raise AssertionError("scenario_creator should not be called")

        with self.assertRaises(ValueError) as ctx:
            lp_xbar_nonants(never_called, [], solver_name="cplex")
        self.assertIn("empty", str(ctx.exception))

    def test_rejects_all_zero_probabilities(self):
        # Build a one-var two-stage scenario whose probability is 0.
        def zero_prob_creator(name, **k):
            m = pyo.ConcreteModel()
            m.x = pyo.Var(bounds=(0, 1))
            m.fsc = pyo.Expression(expr=m.x)
            m.obj = pyo.Objective(expr=m.x)
            sputils.attach_root_node(m, m.fsc, [m.x])
            m._mpisppy_probability = 0.0
            return m

        # Stub out the solve so we never touch a real solver: monkey-patch
        # the internal helper to return a dummy array. The validation is
        # post-loop, so we only need every iteration to "succeed."
        import mpisppy.utils.xhat_helpers as xh

        original = xh._solve_and_extract_root
        xh._solve_and_extract_root = lambda m, *a, **k: np.array([0.5])
        try:
            with self.assertRaises(ValueError) as ctx:
                lp_xbar_nonants(
                    zero_prob_creator,
                    ["S1", "S2"],
                    solver_name="cplex",
                )
        finally:
            xh._solve_and_extract_root = original
        self.assertIn("probabilities sum", str(ctx.exception))


@unittest.skipIf(not solver_available, "no solver available")
class TestAverageXhatNonantsOnFarmer(unittest.TestCase):
    """End-to-end on farmer (whose first stage is continuous)."""

    def test_returns_array_of_correct_length(self):
        arr = average_xhat_nonants(
            farmer.average_scenario_creator,
            solver_name=solver_name,
            scenario_creator_kwargs={"num_scens": 6},
        )
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (3,))  # DEVOTED_ACRES over 3 crops

    def test_relax_integrality_is_no_op_on_continuous_first_stage(self):
        a = average_xhat_nonants(
            farmer.average_scenario_creator,
            solver_name=solver_name,
            scenario_creator_kwargs={"num_scens": 6},
        )
        b = average_xhat_nonants(
            farmer.average_scenario_creator,
            solver_name=solver_name,
            scenario_creator_kwargs={"num_scens": 6},
            relax_integrality=True,
        )
        np.testing.assert_allclose(a, b, atol=1e-6)


@unittest.skipIf(not solver_available, "no solver available")
class TestFarmerFeasibleXhatCreator(unittest.TestCase):
    """Continuous first-stage: convention is satisfied by the
    average-scenario optimum unchanged."""

    def test_returns_root_array(self):
        cache = farmer_auxiliary.feasible_xhat_creator(
            solver_name=solver_name, num_scens=6,
        )
        self.assertIn("ROOT", cache)
        arr = cache["ROOT"]
        self.assertEqual(arr.shape, (3,))  # DEVOTED_ACRES over 3 crops
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertTrue(np.all(arr >= -1e-9))

    def test_kwargs_thread_through(self):
        a = farmer_auxiliary.feasible_xhat_creator(
            solver_name=solver_name, num_scens=6, seedoffset=0,
        )
        b = farmer_auxiliary.feasible_xhat_creator(
            solver_name=solver_name, num_scens=6, seedoffset=7,
        )
        self.assertFalse(np.allclose(a["ROOT"], b["ROOT"]),
                         "seedoffset did not change the candidate")


@unittest.skipIf(not solver_available, "no solver available")
class TestNetdesFeasibleXhatCreator(unittest.TestCase):
    """End-to-end check: netdes_auxiliary.feasible_xhat_creator returns
    a candidate that is integer-valued and feasible to fix in every
    real scenario."""

    def setUp(self):
        from parse import parse
        full = parse(_NETDES_DATA, scenario_ix=None)
        self.K = full["K"]
        self.cache = netdes_auxiliary.feasible_xhat_creator(
            solver_name=solver_name, path=_NETDES_DATA,
        )

    def test_returns_root_dict(self):
        self.assertIn("ROOT", self.cache)
        self.assertEqual(set(self.cache.keys()), {"ROOT"})

    def test_values_are_integer(self):
        arr = self.cache["ROOT"]
        self.assertTrue(np.allclose(arr, np.round(arr), atol=1e-9),
                        f"ceil output is not integer-valued: {arr}")

    def test_fixes_are_feasible_in_every_real_scenario(self):
        arr = self.cache["ROOT"]
        for k in range(self.K):
            sname = f"Scenario{k}"
            scen = netdes.scenario_creator(sname, path=_NETDES_DATA)
            root = scen._mpisppy_node_list[0]
            for v, val in zip(root.nonant_vardata_list, arr):
                v.fix(val)
            solver = pyo.SolverFactory(solver_name)
            limit_solver_threads(solver, solver_name)
            if sputils.is_persistent(solver):
                solver.set_instance(scen)
                results = solver.solve(tee=False)
            else:
                results = solver.solve(scen, tee=False)
            tc = results.solver.termination_condition
            self.assertIn(
                tc, (TerminationCondition.optimal, TerminationCondition.feasible),
                f"netdes fix infeasible on {sname}: tc={tc}, xhat={arr}",
            )


@unittest.skipIf(not solver_available, "no solver available")
@unittest.skipIf(not os.path.isdir(_SSLP_DATA_DIR), "sslp data not found")
class TestSslpFeasibleXhatCreator(unittest.TestCase):

    def test_returns_integer_root_dict(self):
        cache = sslp_auxiliary.feasible_xhat_creator(
            solver_name=solver_name,
            num_scens=5,
            data_dir=_SSLP_DATA_DIR,
        )
        self.assertIn("ROOT", cache)
        arr = cache["ROOT"]
        self.assertEqual(arr.shape, (15,))  # 15 servers in sslp_15_45_5
        self.assertTrue(np.allclose(arr, np.round(arr), atol=1e-9))
        self.assertTrue(np.all((arr >= 0) & (arr <= 1)),
                        f"FacilityOpen out of [0,1]: {arr}")


class TestMaybeAttachFeasibleXhat(unittest.TestCase):
    """cfg_vanilla._maybe_attach_feasible_xhat behavior."""

    def _cfg_with_flag(self, prefix, flag_on=False, jensens_on=False):
        import mpisppy.utils.config as cfg_mod
        cfg = cfg_mod.Config()
        if prefix == "xhatshuffle":
            cfg.xhatshuffle_args()
        elif prefix == "xhatxbar":
            cfg.xhatxbar_args()
        elif prefix == "xhatlooper":
            cfg.xhatlooper_args()
        elif prefix == "xhatspecific":
            cfg.xhatspecific_args()
        if flag_on:
            cfg[f"{prefix}_try_feasible_xhat_first"] = True
        if jensens_on:
            cfg[f"{prefix}_try_jensens_first"] = True
        return cfg

    def test_noop_when_flag_off(self):
        from mpisppy.utils.cfg_vanilla import _maybe_attach_feasible_xhat
        cfg = self._cfg_with_flag("xhatshuffle", flag_on=False)
        spoke_dict = {"opt_kwargs": {"options": {}}}
        _maybe_attach_feasible_xhat(spoke_dict, cfg, "xhatshuffle",
                                    feasible_xhat_creator=None,
                                    scenario_creator_kwargs={})
        self.assertNotIn("feasible_xhat", spoke_dict["opt_kwargs"]["options"])

    def test_raises_when_creator_is_none(self):
        from mpisppy.utils.cfg_vanilla import _maybe_attach_feasible_xhat
        cfg = self._cfg_with_flag("xhatshuffle", flag_on=True)
        spoke_dict = {"opt_kwargs": {"options": {}}}
        with self.assertRaises(RuntimeError) as ctx:
            _maybe_attach_feasible_xhat(spoke_dict, cfg, "xhatshuffle",
                                        feasible_xhat_creator=None,
                                        scenario_creator_kwargs={})
        self.assertIn("feasible_xhat_creator", str(ctx.exception))

    def test_raises_when_both_first_flags_set(self):
        from mpisppy.utils.cfg_vanilla import _maybe_attach_feasible_xhat
        cfg = self._cfg_with_flag("xhatshuffle", flag_on=True, jensens_on=True)
        spoke_dict = {"opt_kwargs": {"options": {}}}
        with self.assertRaises(RuntimeError) as ctx:
            _maybe_attach_feasible_xhat(spoke_dict, cfg, "xhatshuffle",
                                        feasible_xhat_creator=lambda **k: None,
                                        scenario_creator_kwargs={})
        msg = str(ctx.exception)
        self.assertIn("mutually exclusive", msg)
        self.assertIn("try-jensens-first", msg)
        self.assertIn("try-feasible-xhat-first", msg)

    def test_installs_dict_when_flag_on(self):
        from mpisppy.utils.cfg_vanilla import _maybe_attach_feasible_xhat
        cfg = self._cfg_with_flag("xhatshuffle", flag_on=True)
        spoke_dict = {"opt_kwargs": {"options": {}}}
        kwargs = {"path": "/x"}
        _maybe_attach_feasible_xhat(
            spoke_dict, cfg, "xhatshuffle",
            feasible_xhat_creator=netdes_auxiliary.feasible_xhat_creator,
            scenario_creator_kwargs=kwargs,
        )
        f = spoke_dict["opt_kwargs"]["options"]["feasible_xhat"]
        self.assertIs(f["feasible_xhat_creator"],
                      netdes_auxiliary.feasible_xhat_creator)
        self.assertIs(f["scenario_creator_kwargs"], kwargs)


class TestFindFeasibleXhatCreator(unittest.TestCase):
    """Discovery helper: tries main module, falls back to <name>_auxiliary."""

    def _cfg(self, **flags):
        import mpisppy.utils.config as cfg_mod
        cfg = cfg_mod.Config()
        cfg.xhatshuffle_args()
        cfg.xhatxbar_args()
        cfg.xhatlooper_args()
        cfg.xhatspecific_args()
        for name, val in flags.items():
            cfg[name] = val
        return cfg

    def test_returns_none_when_no_flag_set(self):
        from mpisppy.utils.cfg_vanilla import _find_feasible_xhat_creator
        cfg = self._cfg()
        # netdes is the test module; even though it has an _auxiliary,
        # no flag is set so we never reach for it
        result = _find_feasible_xhat_creator(netdes, cfg)
        self.assertIsNone(result)

    def test_falls_back_to_auxiliary(self):
        from mpisppy.utils.cfg_vanilla import _find_feasible_xhat_creator
        cfg = self._cfg(xhatshuffle_try_feasible_xhat_first=True)
        # netdes does NOT export feasible_xhat_creator on the main
        # module; only netdes_auxiliary does.
        self.assertIsNone(getattr(netdes, "feasible_xhat_creator", None))
        result = _find_feasible_xhat_creator(netdes, cfg)
        self.assertIs(result, netdes_auxiliary.feasible_xhat_creator)

    def test_finds_on_main_module_when_present(self):
        # Synthesize a fake module that exposes feasible_xhat_creator
        # directly, to confirm the main-module lookup wins.
        import types
        from mpisppy.utils.cfg_vanilla import _find_feasible_xhat_creator
        marker = lambda **k: {"ROOT": np.array([0.0])}  # noqa: E731
        fake = types.ModuleType("fake_module_with_feas")
        fake.feasible_xhat_creator = marker
        cfg = self._cfg(xhatxbar_try_feasible_xhat_first=True)
        result = _find_feasible_xhat_creator(fake, cfg)
        self.assertIs(result, marker)

    def test_raises_when_neither_has_it(self):
        # farmer's main module has no feasible_xhat_creator either,
        # but farmer_auxiliary does -- so swap to a module without an
        # auxiliary file at all.
        import types
        from mpisppy.utils.cfg_vanilla import _find_feasible_xhat_creator
        bare = types.ModuleType("bare_module_for_feas_test")
        cfg = self._cfg(xhatlooper_try_feasible_xhat_first=True)
        with self.assertRaises(RuntimeError) as ctx:
            _find_feasible_xhat_creator(bare, cfg)
        self.assertIn("feasible_xhat_creator", str(ctx.exception))

    def test_unrelated_importerror_in_aux_is_not_masked(self):
        # If <module>_auxiliary itself exists but its own imports fail
        # (e.g. missing optional dependency), the real ImportError must
        # bubble up unchanged -- not be reframed as "could not be imported."
        import importlib
        import types
        from mpisppy.utils import cfg_vanilla

        main_name = "feas_aux_main_test_mod"
        main = types.ModuleType(main_name)
        sys.modules[main_name] = main
        self.addCleanup(sys.modules.pop, main_name, None)

        original_import = importlib.import_module

        def fake_import(name, *a, **k):
            if name == f"{main_name}_auxiliary":
                # Simulate aux module raising on import of an unrelated dep.
                raise ModuleNotFoundError(
                    "No module named 'totally_unrelated_dep'",
                    name="totally_unrelated_dep",
                )
            return original_import(name, *a, **k)

        cfg = self._cfg(xhatshuffle_try_feasible_xhat_first=True)
        importlib.import_module = fake_import
        try:
            with self.assertRaises(ModuleNotFoundError) as ctx:
                cfg_vanilla._find_feasible_xhat_creator(main, cfg)
        finally:
            importlib.import_module = original_import
        self.assertEqual(ctx.exception.name, "totally_unrelated_dep")


class _StubFeasOpt:
    """Stand-in for self.opt that captures _fix_nonants / solve_loop /
    Eobjective calls so we can exercise _try_feasible_xhat without MPI."""

    def __init__(self, *, infeas_prob, eobj, options=None):
        self._infeas_prob = infeas_prob
        self._eobj = eobj
        self.options = options if options is not None else {}
        self.fix_calls = []
        self.solve_loop_calls = []

    def _fix_nonants(self, cache):
        self.fix_calls.append(cache)

    def solve_loop(self, **kwargs):
        self.solve_loop_calls.append(kwargs)

    def no_incumbent_prob(self):
        return self._infeas_prob

    def Eobjective(self, verbose=False):
        return self._eobj


class _StubFeasSpoke:
    """Mixin host wired to capture update_if_improving calls."""

    def __init__(self, opt):
        from mpisppy.cylinders._preloop_xhat_mixin import _PreLoopXhatMixin
        self.opt = opt
        self.bound_updates = []
        # bind the mixin methods
        self._feasible_xhat_enabled = lambda: _PreLoopXhatMixin._feasible_xhat_enabled(self)
        self._evaluate_xhat = lambda c: _PreLoopXhatMixin._evaluate_xhat(self, c)
        self._try_feasible_xhat = lambda: _PreLoopXhatMixin._try_feasible_xhat(self)

    def update_if_improving(self, Eobj):
        self.bound_updates.append(Eobj)


class TestTryFeasibleXhat(unittest.TestCase):
    """Unit tests for _PreLoopXhatMixin._try_feasible_xhat."""

    def test_noop_when_flag_off(self):
        opt = _StubFeasOpt(infeas_prob=0.0, eobj=1.0)  # no "feasible_xhat" key
        sp = _StubFeasSpoke(opt)
        sp._try_feasible_xhat()
        self.assertEqual(sp.bound_updates, [])
        self.assertEqual(opt.fix_calls, [])

    def test_updates_bound_when_feasible(self):
        creator = lambda **kw: {"ROOT": np.array([1.0, 2.0])}  # noqa: E731
        opt = _StubFeasOpt(
            infeas_prob=0.0, eobj=42.0,
            options={
                "feasible_xhat": {
                    "feasible_xhat_creator": creator,
                    "scenario_creator_kwargs": {},
                },
                "solver_name": "fake",
                "iterk_solver_options": None,
            },
        )
        sp = _StubFeasSpoke(opt)
        sp._try_feasible_xhat()
        self.assertEqual(sp.bound_updates, [42.0])
        self.assertEqual(len(opt.fix_calls), 1)
        np.testing.assert_array_equal(opt.fix_calls[0]["ROOT"],
                                      np.array([1.0, 2.0]))

    def test_skips_silently_when_infeasible(self):
        creator = lambda **kw: {"ROOT": np.array([0.0])}  # noqa: E731
        opt = _StubFeasOpt(
            infeas_prob=0.5, eobj=999.0,  # ignored on infeasible path
            options={
                "feasible_xhat": {
                    "feasible_xhat_creator": creator,
                    "scenario_creator_kwargs": {},
                },
                "solver_name": "fake",
            },
        )
        sp = _StubFeasSpoke(opt)
        sp._try_feasible_xhat()
        self.assertEqual(sp.bound_updates, [])

    def test_raises_on_bad_return_shape(self):
        # creator returns a bare array, not a dict
        creator = lambda **kw: np.array([1.0])  # noqa: E731
        opt = _StubFeasOpt(
            infeas_prob=0.0, eobj=1.0,
            options={
                "feasible_xhat": {
                    "feasible_xhat_creator": creator,
                    "scenario_creator_kwargs": {},
                },
                "solver_name": "fake",
            },
        )
        sp = _StubFeasSpoke(opt)
        with self.assertRaises(RuntimeError) as ctx:
            sp._try_feasible_xhat()
        self.assertIn("ROOT", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
