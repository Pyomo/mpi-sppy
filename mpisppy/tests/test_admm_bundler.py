###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import unittest
import mpisppy.tests.examples.stoch_distr.stoch_distr as stoch_distr
from mpisppy.utils.admm_bundler import AdmmBundler
from mpisppy.utils import config
from mpisppy.tests.utils import get_solver

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()


class TestAdmmBundler(unittest.TestCase):

    def _make_bundler(self, num_stoch_scens, num_admm_subproblems,
                      scenarios_per_bundle):
        cfg = config.Config()
        stoch_distr.inparser_adder(cfg)
        cfg.num_stoch_scens = num_stoch_scens
        cfg.num_admm_subproblems = num_admm_subproblems

        admm_subproblem_names = stoch_distr.admm_subproblem_names_creator(
            cfg)
        stoch_scenario_names = stoch_distr.stoch_scenario_names_creator(
            cfg)
        scenario_creator_kwargs = stoch_distr.kw_creator(cfg)
        stoch_scenario_name = stoch_scenario_names[0]
        consensus_vars = stoch_distr.consensus_vars_creator(
            admm_subproblem_names, stoch_scenario_name,
            **scenario_creator_kwargs)

        bundler = AdmmBundler(
            module=stoch_distr,
            scenarios_per_bundle=scenarios_per_bundle,
            admm_subproblem_names=admm_subproblem_names,
            stoch_scenario_names=stoch_scenario_names,
            consensus_vars=consensus_vars,
            combining_fn=stoch_distr.combining_names,
            split_fn=stoch_distr.split_admm_stoch_subproblem_scenario_name,
            scenario_creator_kwargs=scenario_creator_kwargs,
        )
        return bundler

    def test_bundle_names(self):
        """Test that bundle names are generated correctly (full bundling)."""
        bundler = self._make_bundler(4, 2, 4)
        names = bundler.bundle_names_creator()
        # 2 subproblems × 1 bundle each = 2 bundles
        self.assertEqual(len(names), 2)
        self.assertIn("Bundle_ADMM_Region1_0", names)
        self.assertIn("Bundle_ADMM_Region2_0", names)

    def test_partial_bundling_raises(self):
        """Test error when scenarios_per_bundle != num_stoch_scens."""
        bundler = self._make_bundler(4, 2, 2)
        with self.assertRaises(RuntimeError):
            bundler.bundle_names_creator()

    def test_bundle_names_three_subproblems(self):
        """Test bundling with three subproblems."""
        bundler = self._make_bundler(4, 3, 4)
        names = bundler.bundle_names_creator()
        # 3 subproblems × 1 bundle each = 3 bundles
        self.assertEqual(len(names), 3)

    @unittest.skipUnless(solver_available, "no solver available")
    def test_bundle_creation(self):
        """Test that a bundle can be created and has correct structure."""
        bundler = self._make_bundler(4, 2, 4)
        names = bundler.bundle_names_creator()
        bundle = bundler.scenario_creator(names[0])

        # Bundle should have _mpisppy_node_list with ROOT
        self.assertTrue(hasattr(bundle, "_mpisppy_node_list"))
        self.assertEqual(len(bundle._mpisppy_node_list), 1)
        self.assertEqual(bundle._mpisppy_node_list[0].name, "ROOT")

        # Bundle should have EF_Obj
        self.assertTrue(hasattr(bundle, "EF_Obj"))

        # Bundle should have _mpisppy_probability
        self.assertTrue(hasattr(bundle, "_mpisppy_probability"))
        self.assertGreater(bundle._mpisppy_probability, 0)

        # Bundle should have ref_vars
        self.assertTrue(hasattr(bundle, "ref_vars"))

    @unittest.skipUnless(solver_available, "no solver available")
    def test_var_prob_list(self):
        """Test that variable probabilities are correct for bundles."""
        # Use 3 subproblems to ensure some bundles have dummy vars
        bundler = self._make_bundler(4, 3, 4)
        names = bundler.bundle_names_creator()
        bundle = bundler.scenario_creator(names[0])

        vpl = bundler.var_prob_list(bundle)
        self.assertIsInstance(vpl, list)
        self.assertGreater(len(vpl), 0)

        # Each entry should be (id, prob) with prob >= 0
        for vid, prob in vpl:
            self.assertIsInstance(vid, int)
            self.assertGreaterEqual(prob, 0)

        # At least some variables should have prob > 0 (real consensus vars)
        probs = [p for _, p in vpl]
        self.assertTrue(any(p > 0 for p in probs), "No variables with positive probability")

        # With 3+ subproblems, some bundles should have dummy vars (prob=0)
        all_probs = []
        for bname in names:
            if bname != names[0]:
                b = bundler.scenario_creator(bname)
                all_probs.extend([p for _, p in bundler.var_prob_list(b)])
            else:
                all_probs.extend(probs)
        self.assertTrue(any(p == 0 for p in all_probs),
                        "No dummy variables (prob=0) found in any bundle")

    @unittest.skipUnless(solver_available, "no solver available")
    def test_all_bundles_same_nonant_count(self):
        """Test that all bundles have the same number of nonants at ROOT."""
        bundler = self._make_bundler(4, 2, 4)
        names = bundler.bundle_names_creator()

        nonant_counts = []
        for bname in names:
            bundle = bundler.scenario_creator(bname)
            root_node = bundle._mpisppy_node_list[0]
            nonant_counts.append(len(root_node.nonant_vardata_list))

        # All bundles should have the same number of nonants
        # (because nonant_for_fixed_vars=True)
        self.assertTrue(
            all(c == nonant_counts[0] for c in nonant_counts),
            f"Nonant counts differ across bundles: {nonant_counts}"
        )

    @unittest.skipUnless(solver_available, "no solver available")
    def test_var_prob_consistent_lengths(self):
        """Test that var_prob_list length matches nonant count for all bundles."""
        bundler = self._make_bundler(4, 2, 4)
        names = bundler.bundle_names_creator()

        for bname in names:
            bundle = bundler.scenario_creator(bname)
            vpl = bundler.var_prob_list(bundle)
            root_node = bundle._mpisppy_node_list[0]
            self.assertEqual(
                len(vpl), len(root_node.nonant_vardata_list),
                f"var_prob_list length ({len(vpl)}) != nonant count "
                f"({len(root_node.nonant_vardata_list)}) for {bname}"
            )


class TestAdmmArgs(unittest.TestCase):
    """Test admm_args, _count_cylinders, and _check_admm_compatibility."""

    def test_admm_args_registers_flags(self):
        from mpisppy.generic.admm import admm_args
        cfg = config.Config()
        admm_args(cfg)
        self.assertIn("admm", cfg)
        self.assertIn("stoch_admm", cfg)
        self.assertIn("num_admm_subproblems", cfg)
        self.assertIn("num_stoch_scens", cfg)

    def test_admm_args_skips_existing(self):
        """admm_args should not re-register keys already present."""
        from mpisppy.generic.admm import admm_args
        cfg = config.Config()
        cfg.add_to_config("num_admm_subproblems",
                          description="pre-existing", domain=int, default=5)
        admm_args(cfg)
        # The pre-existing default should be preserved
        self.assertEqual(cfg["num_admm_subproblems"], 5)

    def test_count_cylinders_hub_only(self):
        from mpisppy.generic.admm import _count_cylinders
        cfg = config.Config()
        self.assertEqual(_count_cylinders(cfg), 1)

    def test_count_cylinders_with_spokes(self):
        from mpisppy.generic.admm import _count_cylinders
        cfg = config.Config()
        cfg.add_to_config("lagrangian", description="", domain=bool, default=True)
        cfg.add_to_config("xhatshuffle", description="", domain=bool, default=True)
        self.assertEqual(_count_cylinders(cfg), 3)

    def test_check_both_admm_and_stoch_admm(self):
        from mpisppy.generic.admm import _check_admm_compatibility
        cfg = config.Config()
        cfg.add_to_config("admm", description="", domain=bool, default=True)
        cfg.add_to_config("stoch_admm", description="", domain=bool, default=True)
        with self.assertRaises(RuntimeError):
            _check_admm_compatibility(cfg)

    def test_check_fwph_incompatible(self):
        from mpisppy.generic.admm import _check_admm_compatibility
        cfg = config.Config()
        cfg.add_to_config("admm", description="", domain=bool, default=True)
        cfg.add_to_config("fwph", description="", domain=bool, default=True)
        with self.assertRaises(RuntimeError):
            _check_admm_compatibility(cfg)

    def test_check_admm_bundles_incompatible(self):
        from mpisppy.generic.admm import _check_admm_compatibility
        cfg = config.Config()
        cfg.add_to_config("admm", description="", domain=bool, default=True)
        cfg.add_to_config("scenarios_per_bundle", description="", domain=int, default=2)
        with self.assertRaises(RuntimeError):
            _check_admm_compatibility(cfg)

    def test_check_stoch_admm_bundles_with_pickling(self):
        from mpisppy.generic.admm import _check_admm_compatibility
        cfg = config.Config()
        cfg.add_to_config("stoch_admm", description="", domain=bool, default=True)
        cfg.add_to_config("scenarios_per_bundle", description="", domain=int, default=4)
        cfg.add_to_config("pickle_bundles_dir", description="", domain=str, default="/tmp")
        with self.assertRaises(RuntimeError):
            _check_admm_compatibility(cfg)

    def test_check_admm_pickling_without_bundles(self):
        from mpisppy.generic.admm import _check_admm_compatibility
        cfg = config.Config()
        cfg.add_to_config("admm", description="", domain=bool, default=True)
        cfg.add_to_config("unpickle_scenarios_dir", description="", domain=str, default="/tmp")
        with self.assertRaises(RuntimeError):
            _check_admm_compatibility(cfg)

    def test_check_stoch_admm_no_bundles_ok(self):
        """stoch_admm without bundles or pickling should pass."""
        from mpisppy.generic.admm import _check_admm_compatibility
        cfg = config.Config()
        cfg.add_to_config("stoch_admm", description="", domain=bool, default=True)
        # Should not raise
        _check_admm_compatibility(cfg)


class TestNameListsAdmmPath(unittest.TestCase):
    """Test the ADMM early-return path in parsing.name_lists."""

    def test_admm_names_returned(self):
        from mpisppy.generic.parsing import name_lists
        cfg = config.Config()
        cfg.num_scens_required()
        cfg.quick_assign("num_scens", int, 3)
        names = ["Region1", "Region2"]
        nodenames = ["ROOT"]
        object.__setattr__(cfg, "_admm_scenario_names", names)
        object.__setattr__(cfg, "_admm_nodenames", nodenames)
        result_names, result_nodes = name_lists(stoch_distr, cfg)
        self.assertEqual(result_names, names)
        self.assertEqual(result_nodes, nodenames)


class TestMultistageXhatShuffleWarning(unittest.TestCase):
    """Guard the warning emitted when multistage xhatshuffle is used
    without --stage2-ef-solver-name (relaxed from assert in PR #651)."""

    def _make_multistage_cfg(self, stage2=None):
        import types
        cfg = config.Config()
        cfg.multistage()
        cfg.xhatshuffle_args()
        cfg.proper_bundle_config()
        cfg.quick_assign("branching_factors", list, [2, 2])
        cfg.quick_assign("xhatshuffle", bool, True)
        if stage2 is not None:
            cfg.quick_assign("stage2_ef_solver_name", str, stage2)
        module = types.SimpleNamespace(
            scenario_names_creator=lambda n: [f"Scen{i+1}" for i in range(n)]
        )
        return cfg, module

    def test_warn_when_stage2_missing(self):
        import warnings
        from mpisppy.generic.parsing import name_lists
        cfg, module = self._make_multistage_cfg(stage2=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            name_lists(module, cfg)
        msgs = [str(w.message) for w in caught]
        self.assertTrue(
            any("stage2_ef_solver_name" in m for m in msgs),
            f"Expected a stage2_ef_solver_name warning, got: {msgs}",
        )

    def test_no_warn_when_stage2_set(self):
        import warnings
        from mpisppy.generic.parsing import name_lists
        cfg, module = self._make_multistage_cfg(stage2="gurobi")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            name_lists(module, cfg)
        self.assertFalse(
            any("stage2_ef_solver_name" in str(w.message) for w in caught),
            "Did not expect a stage2_ef_solver_name warning when set",
        )


class TestEfOptionsNameCheckForwarding(unittest.TestCase):
    """Guard that cfg.turn_off_names_check reaches the EF options dict.

    generic_cylinders auto-sets this for --admm/--stoch-admm runs (the
    wrappers synthesize dummy nonants with different names), but the
    flag only helps if ef_options forwards it into ef_dict['options'].
    """

    def _base_cfg(self):
        from mpisppy.utils import config
        cfg = config.Config()
        cfg.popular_args()
        cfg.EF_base()
        cfg.quick_assign("EF_solver_name", str, "cplex")
        return cfg

    def test_forwarded_when_true(self):
        from mpisppy.utils.cfg_vanilla import ef_options
        cfg = self._base_cfg()
        cfg.quick_assign("turn_off_names_check", bool, True)
        d = ef_options(cfg, lambda *a, **k: None, lambda *a, **k: None,
                       ["Scen1"], scenario_creator_kwargs={})
        self.assertTrue(d["options"]["turn_off_names_check"])

    def test_false_by_default(self):
        from mpisppy.utils.cfg_vanilla import ef_options
        cfg = self._base_cfg()
        d = ef_options(cfg, lambda *a, **k: None, lambda *a, **k: None,
                       ["Scen1"], scenario_creator_kwargs={})
        self.assertFalse(d["options"]["turn_off_names_check"])


if __name__ == '__main__':
    unittest.main()
