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
            num_admm_subproblems)
        stoch_scenario_names = stoch_distr.stoch_scenario_names_creator(
            num_stoch_scens)
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


if __name__ == '__main__':
    unittest.main()
