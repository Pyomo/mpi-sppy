###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unit tests for mpisppy.utils.proper_bundler.ProperBundler.

The fix in proper_bundler.scenario_creator (firstnum -> firstnum - inum)
matters for modules whose scenario_names_creator expects "start = count of
already-used scenarios" while bundle names embed absolute scenario labels.
For 0-based modules (e.g. farmer) the two were the same, so the bug was
silent. For 1-based modules (e.g. uc_funcs) the bundle path produced an
off-by-one set of scenarios.

These tests use stub modules so no solver / egret is required.
"""

import unittest

import pyomo.environ as pyo

import mpisppy.opt.ph
import mpisppy.utils.sputils as sputils
from mpisppy.utils import config
from mpisppy.utils.proper_bundler import ProperBundler
from mpisppy.tests.utils import get_solver

solver_available, solver_name, _, _ = get_solver()


def _trivial_scenario(sname):
    m = pyo.ConcreteModel(name=sname)
    m.x = pyo.Var(bounds=(0, 1))
    m.fsc = pyo.Expression(expr=m.x)
    m.obj = pyo.Objective(expr=m.x)
    sputils.attach_root_node(m, m.fsc, [m.x])
    m._mpisppy_probability = "uniform"
    return m


def _make_cfg(num_scens, scenarios_per_bundle):
    cfg = config.Config()
    cfg.popular_args()
    cfg.num_scens_required()
    cfg.proper_bundle_config()
    cfg.num_scens = num_scens
    cfg.scenarios_per_bundle = scenarios_per_bundle
    return cfg


class _CapturingModule:
    """Stub that records the snames its scenario_creator is asked for."""

    def __init__(self, naming):
        self.naming = naming  # "one_based" or "zero_based"
        self.captured = []

    def scenario_names_creator(self, num_scens, start=None):
        if start is None:
            start = 0
        if self.naming == "one_based":
            return [f"Scenario{i + 1}" for i in range(start, start + num_scens)]
        return [f"scen{i}" for i in range(start, start + num_scens)]

    def kw_creator(self, cfg):
        return {}

    def scenario_creator(self, sname, **kwargs):
        self.captured.append(sname)
        return _trivial_scenario(sname)


class TestProperBundlerBundleSplit(unittest.TestCase):
    """Verify the ProperBundler.scenario_creator firstnum/inum translation."""

    def _build_bundle(self, naming, bundle_name, num_scens=3,
                      scenarios_per_bundle=3):
        mod = _CapturingModule(naming)
        cfg = _make_cfg(num_scens, scenarios_per_bundle)
        pb = ProperBundler(mod)
        pb.set_bunBFs(cfg)
        kwargs = pb.kw_creator(cfg)
        bundle = pb.scenario_creator(bundle_name, **kwargs)
        return mod, bundle

    def test_one_based_bundle_uses_correct_start(self):
        # uc_funcs convention: scenarios are Scenario1..ScenarioN.
        # bundle_names_creator(num_buns=1, ...) on a 1-based module gives
        # "Bundle_1_3"; the bundler must split that back into
        # ["Scenario1", "Scenario2", "Scenario3"], not Scenario2..Scenario4
        # (which is what the pre-fix `firstnum` start would have produced).
        mod, bundle = self._build_bundle("one_based", "Bundle_1_3")
        self.assertEqual(mod.captured,
                         ["Scenario1", "Scenario2", "Scenario3"])
        self.assertIsNotNone(bundle)
        self.assertTrue(hasattr(bundle, "_mpisppy_node_list"))

    def test_zero_based_bundle_uses_correct_start(self):
        # farmer/aircond convention: scenarios are scen0..scenN-1.
        # bundle_names_creator on a 0-based module gives "Bundle_0_2";
        # split back into ["scen0", "scen1", "scen2"].
        mod, bundle = self._build_bundle("zero_based", "Bundle_0_2")
        self.assertEqual(mod.captured, ["scen0", "scen1", "scen2"])
        self.assertIsNotNone(bundle)

    def test_bundle_names_round_trip_one_based(self):
        # bundle_names_creator -> scenario_creator should be self-consistent.
        mod = _CapturingModule("one_based")
        cfg = _make_cfg(num_scens=6, scenarios_per_bundle=3)
        pb = ProperBundler(mod)
        names = pb.bundle_names_creator(num_buns=2, cfg=cfg)
        self.assertEqual(names, ["Bundle_1_3", "Bundle_4_6"])

        kwargs = pb.kw_creator(cfg)
        pb.scenario_creator("Bundle_4_6", **kwargs)
        self.assertEqual(mod.captured,
                         ["Scenario4", "Scenario5", "Scenario6"])

    def test_bundle_names_round_trip_zero_based(self):
        mod = _CapturingModule("zero_based")
        cfg = _make_cfg(num_scens=6, scenarios_per_bundle=3)
        pb = ProperBundler(mod)
        names = pb.bundle_names_creator(num_buns=2, cfg=cfg)
        self.assertEqual(names, ["Bundle_0_2", "Bundle_3_5"])

        kwargs = pb.kw_creator(cfg)
        pb.scenario_creator("Bundle_3_5", **kwargs)
        self.assertEqual(mod.captured, ["scen3", "scen4", "scen5"])


def _scenario_with_fixed_nonants(sname, **kwargs):
    """Two-stage scenario with three nonants; the middle one is fixed.

    Mimics the structure that triggers issue #668: bundle.ref_vars (and
    thus the bundle's nonant_indices) skips the fixed position, so
    bundle nonant index k != per-scenario position k.
    """
    m = pyo.ConcreteModel(name=sname)
    m.x = pyo.Var([0, 1, 2], bounds=(0, 10))
    m.x[1].fix(5.0)  # the middle position is fixed
    m.fsc = pyo.Expression(expr=m.x[0] + m.x[1] + m.x[2])
    m.obj = pyo.Objective(expr=2.0 * m.x[0] + 3.0 * m.x[1] + 5.0 * m.x[2])
    sputils.attach_root_node(m, m.fsc, [m.x[0], m.x[1], m.x[2]])
    m._mpisppy_probability = "uniform"
    return m


class _ModuleWithFixedNonant:
    @staticmethod
    def scenario_names_creator(num_scens, start=None):
        if start is None:
            start = 0
        return [f"scen{i}" for i in range(start, start + num_scens)]

    @staticmethod
    def kw_creator(cfg):
        return {}

    scenario_creator = staticmethod(_scenario_with_fixed_nonants)


class TestNonantCostCoeffsBundleWithFixedNonants(unittest.TestCase):
    """Regression for issue #668.

    `sputils.nonant_cost_coeffs` used to enumerate per-scenario nonants
    with sequential positions (0..N-1) and look those positions up in
    cost_coefs (keyed by bundle nonant indices, 0..M-1 with M < N when
    create_EF skips fixed-var positions). The lookup KeyError'd for any
    per-scenario position whose bundle index didn't exist. Trips when
    --linearize-proximal-terms is set with proper bundles on a model
    with fixed first-stage vars.
    """

    def _build_bundle(self):
        cfg = _make_cfg(num_scens=3, scenarios_per_bundle=3)
        pb = ProperBundler(_ModuleWithFixedNonant)
        pb.set_bunBFs(cfg)
        kwargs = pb.kw_creator(cfg)
        bundle = pb.scenario_creator("Bundle_0_2", **kwargs)
        return bundle

    def test_bundle_skips_fixed_var_in_nonantlist(self):
        # Sanity: the bundle's nonant_vardata_list excludes the fixed var.
        bundle = self._build_bundle()
        node = bundle._mpisppy_node_list[0]
        self.assertEqual(len(node.nonant_vardata_list), 2)
        # Per-scenario list still has all three.
        scen0 = bundle.component(bundle._ef_scenario_names[0])
        self.assertEqual(len(scen0._mpisppy_node_list[0].nonant_vardata_list), 3)

    def test_nonant_cost_coeffs_does_not_keyerror(self):
        bundle = self._build_bundle()
        # Mimic the SPBase setup that nonant_cost_coeffs needs (normally
        # done by SPBase._attach_nonant_indices).
        node = bundle._mpisppy_node_list[0]
        bundle._mpisppy_data.nonant_indices = {
            (node.name, i): v for i, v in enumerate(node.nonant_vardata_list)
        }
        bundle._mpisppy_data.varid_to_nonant_index = {
            id(v): k for k, v in bundle._mpisppy_data.nonant_indices.items()
        }
        # Pre-fix this raised KeyError(("ROOT", j)) for any per-scenario
        # position j that wasn't in the bundle's filtered nonantlist.
        coefs = sputils.nonant_cost_coeffs(bundle)
        # Bundle nonants are x[0] and x[2] (x[1] was fixed away).
        # The bundle EF objective is sum_s prob_s * scen_obj_s, divided by
        # total probability. With uniform probability over 3 scenarios,
        # the coefficient on (ROOT, 0) is 3 * (1/3) * 2.0 = 2.0, and on
        # (ROOT, 1) it is 3 * (1/3) * 5.0 = 5.0.
        self.assertEqual(set(coefs.keys()),
                         {(node.name, 0), (node.name, 1)})
        self.assertAlmostEqual(coefs[(node.name, 0)], 2.0)
        self.assertAlmostEqual(coefs[(node.name, 1)], 5.0)



# ---------------------------------------------------------------------------
# End-to-end PH regression: bundled vs unbundled with a fixed first-stage
# variable. This is the scenario from issue #668.
#
# Model choice matters: the bug only triggers when the EF objective's
# standard_repn expands down to the raw per-scenario nonant vars. In
# models where FirstStageCost is a Pyomo Var linked by a constraint
# (e.g. sizes) the objective's linear vars are the cost-aggregation
# Vars, not the per-scenario DevotedAcreage-style vars, so the buggy
# code path is never reached. Models where FirstStageCost is a
# pyo.Expression (e.g. farmer, uc) do expand, so those are what the
# test needs.
#
# We fix DevotedAcreage["CORN0"] on purpose: it is position 0 in
# farmer's per-scenario nonant_vardata_list ordering, which leaves
# DevotedAcreage["WHEAT0"] unfixed at per-scenario position 2. With
# only two surviving bundle nonants, the pre-fix code would look up
# cost_coefs[('ROOT', 2)] against a dict keyed only {('ROOT', 0),
# ('ROOT', 1)} and KeyError. Fixing a position near the end would
# leave every unfixed var below M and silently pass.
# ---------------------------------------------------------------------------


def _farmer_fix_creator(scenario_name, **kwargs):
    """Farmer scenario with DevotedAcreage['CORN0'] fixed to 80."""
    from mpisppy.tests.examples.farmer import scenario_creator as base_sc
    model = base_sc(scenario_name, num_scens=3, crops_multiplier=1)
    model.DevotedAcreage["CORN0"].fix(80.0)
    return model


class _FarmerModuleWithFixedFirstStage:
    """Minimal ProperBundler-compatible module wrapper around farmer."""

    @staticmethod
    def scenario_names_creator(num_scens, start=None):
        if start is None:
            start = 0
        # farmer is zero-based (scen0..scen{N-1}).
        return [f"scen{i}" for i in range(start, start + num_scens)]

    @staticmethod
    def kw_creator(cfg):
        return {}

    scenario_creator = staticmethod(_farmer_fix_creator)


def _ph_options_for_farmer():
    return {
        "solver_name": solver_name,
        "PHIterLimit": 3,
        "defaultPHrho": 1,
        "convthresh": 1e-3,
        "verbose": False,
        "display_timing": False,
        "display_progress": False,
        "asynchronousPH": False,
        "smoothed": False,
        "linearize_proximal_terms": True,
        "proximal_linearization_tolerance": 1e-1,
        "iter0_solver_options": None,
        "iterk_solver_options": None,
    }


class TestPHWithFixedFirstStageBundledAndUnbundled(unittest.TestCase):
    """End-to-end regression for issue #668.

    Runs PH with ``linearize_proximal_terms=True`` on a farmer-3
    instance whose ``DevotedAcreage['CORN0']`` is fixed. The bundle
    branch of ``sputils.nonant_cost_coeffs`` is where the pre-fix
    code raised ``KeyError(('ROOT', k))``: the per-scenario walker
    assigned keys ``(ndn, 0..N-1)`` while the bundle's
    ``nonant_indices`` was keyed ``0..M-1`` with ``M<N`` when
    ``create_EF(nonant_for_fixed_vars=False)`` skipped the fixed
    position.

    ``TestNonantCostCoeffsBundleWithFixedNonants`` above exercises
    ``nonant_cost_coeffs`` in isolation with a synthetic bundle.
    *This* test drives the real PH engine on a real stochastic LP,
    so it also catches any future regression in how the fix is
    wired into ``attach_PH_to_objective``. Both the bundled and
    unbundled PH runs must complete and produce an iter-0 trivial
    bound in the same ballpark as the EF optimum.
    """

    @unittest.skipUnless(solver_available, "no solver available")
    def test_bundled_matches_unbundled_with_fixed_first_stage(self):
        # Ground truth: EF solution with the fixed var.
        ef = sputils.create_EF(
            _FarmerModuleWithFixedFirstStage.scenario_names_creator(3),
            _farmer_fix_creator,
            scenario_creator_kwargs={},
        )
        ef_solver = pyo.SolverFactory(solver_name)
        ef_solver.solve(ef)
        ef_obj = pyo.value(ef.EF_Obj)
        self.assertTrue(_math_isfinite(ef_obj))

        # --- Unbundled PH.
        # Sanity path: without bundling, nonant_cost_coeffs uses
        # s._mpisppy_data.varid_to_nonant_index directly and is not
        # affected by the fix.
        ph_unbundled = mpisppy.opt.ph.PH(
            _ph_options_for_farmer(),
            _FarmerModuleWithFixedFirstStage.scenario_names_creator(3),
            _farmer_fix_creator,
            None,
            scenario_creator_kwargs={},
        )
        _, _, tbound_unbundled = ph_unbundled.ph_main()
        self.assertTrue(_math_isfinite(tbound_unbundled))
        for _, s in ph_unbundled.local_scenarios.items():
            self.assertTrue(s.DevotedAcreage["CORN0"].is_fixed())
            self.assertEqual(pyo.value(s.DevotedAcreage["CORN0"]), 80.0)

        # --- Bundled PH (1 bundle of 3 scenarios).
        # Pre-fix this branch crashed with KeyError(('ROOT', 2))
        # inside nonant_cost_coeffs because bundle nonant_indices ==
        # {('ROOT', 0), ('ROOT', 1)} while the per-scenario walker
        # assigned position 2 to the unfixed WHEAT0 nonant.
        cfg = config.Config()
        cfg.popular_args()
        cfg.num_scens_required()
        cfg.proper_bundle_config()
        cfg.num_scens = 3
        cfg.scenarios_per_bundle = 3
        pb = ProperBundler(_FarmerModuleWithFixedFirstStage)
        pb.set_bunBFs(cfg)
        bundle_names = pb.bundle_names_creator(num_buns=1, cfg=cfg)
        self.assertEqual(bundle_names, ["Bundle_0_2"])

        ph_bundled = mpisppy.opt.ph.PH(
            _ph_options_for_farmer(),
            bundle_names,
            pb.scenario_creator,
            None,
            scenario_creator_kwargs=pb.kw_creator(cfg),
        )
        _, _, tbound_bundled = ph_bundled.ph_main()
        self.assertTrue(_math_isfinite(tbound_bundled))
        # The fix must still be present inside every scenario of every
        # local bundle.
        for _, bundle in ph_bundled.local_scenarios.items():
            for sname in bundle._ef_scenario_names:
                scen = bundle.component(sname)
                v = scen.DevotedAcreage["CORN0"]
                self.assertTrue(v.is_fixed(),
                                f"{sname}: DevotedAcreage[CORN0] lost its fix")
                self.assertEqual(pyo.value(v), 80.0)

        # --- Compare objectives.
        # With all 3 scenarios in one bundle the bundled iter-0
        # trivial bound is effectively an EF solve, so it must match
        # the EF ground truth to solver tolerance.
        self.assertAlmostEqual(
            tbound_bundled, ef_obj,
            delta=1e-3 * max(1.0, abs(ef_obj)),
            msg=f"bundled tbound {tbound_bundled} not near EF optimum {ef_obj}",
        )
        # For minimization the unbundled iter-0 trivial bound is a
        # Jensen-style lower bound for the EF optimum (tolerate tiny
        # solver overshoot).
        self.assertLessEqual(
            tbound_unbundled, ef_obj + 1e-3 * max(1.0, abs(ef_obj)),
            msg=f"unbundled tbound {tbound_unbundled} exceeds EF optimum {ef_obj}",
        )


def _math_isfinite(x):
    import math
    return isinstance(x, (int, float)) and math.isfinite(x)


if __name__ == "__main__":
    unittest.main()
