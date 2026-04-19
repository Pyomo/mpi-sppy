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

import mpisppy.utils.sputils as sputils
from mpisppy.utils import config
from mpisppy.utils.proper_bundler import ProperBundler


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



if __name__ == "__main__":
    unittest.main()
