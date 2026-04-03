###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for mpisppy/utils/config.py Config class."""

import unittest

from mpisppy.utils.config import Config


class TestConfigAddToConfig(unittest.TestCase):
    """Tests for Config.add_to_config()."""

    def setUp(self):
        self.cfg = Config()

    def test_add_simple_entry(self):
        self.cfg.add_to_config("my_int",
                               description="an integer",
                               domain=int,
                               default=42)
        self.assertIn("my_int", self.cfg)

    def test_default_value_accessible(self):
        self.cfg.add_to_config("my_str",
                               description="a string",
                               domain=str,
                               default="hello")
        self.assertEqual(self.cfg.my_str, "hello")

    def test_value_assignment(self):
        self.cfg.add_to_config("my_float",
                               description="a float",
                               domain=float,
                               default=0.0)
        self.cfg.my_float = 3.14
        self.assertAlmostEqual(self.cfg.my_float, 3.14)

    def test_duplicate_is_ignored_with_complain(self):
        self.cfg.add_to_config("dup",
                               description="first",
                               domain=int,
                               default=1)
        # Adding again with complain=True should not raise; just print a message
        self.cfg.add_to_config("dup",
                               description="second",
                               domain=int,
                               default=2,
                               complain=True)
        # Original entry should remain
        self.assertEqual(self.cfg.dup, 1)

    def test_duplicate_without_complain_is_silently_ignored(self):
        self.cfg.add_to_config("dup2",
                               description="first",
                               domain=int,
                               default=10)
        self.cfg.add_to_config("dup2",
                               description="second",
                               domain=int,
                               default=99)
        self.assertEqual(self.cfg.dup2, 10)

    def test_bool_domain(self):
        self.cfg.add_to_config("flag",
                               description="a boolean flag",
                               domain=bool,
                               default=False)
        self.assertFalse(self.cfg.flag)
        self.cfg.flag = True
        self.assertTrue(self.cfg.flag)

    def test_no_argparse(self):
        # argparse=False: entry is added but not exposed as CLI arg
        self.cfg.add_to_config("hidden",
                               description="hidden from CLI",
                               domain=int,
                               default=0,
                               argparse=False)
        self.assertIn("hidden", self.cfg)


class TestConfigAddAndAssign(unittest.TestCase):
    """Tests for Config.add_and_assign()."""

    def setUp(self):
        self.cfg = Config()

    def test_adds_and_assigns(self):
        self.cfg.add_and_assign("my_val",
                                description="test val",
                                domain=int,
                                default=0,
                                value=99)
        self.assertEqual(self.cfg.my_val, 99)

    def test_duplicate_raises(self):
        self.cfg.add_and_assign("dup",
                                description="first",
                                domain=int,
                                default=0,
                                value=1)
        with self.assertRaises(RuntimeError):
            self.cfg.add_and_assign("dup",
                                    description="second",
                                    domain=int,
                                    default=0,
                                    value=2)

    def test_duplicate_no_complain_does_not_raise(self):
        self.cfg.add_and_assign("dup2",
                                description="first",
                                domain=int,
                                default=0,
                                value=1,
                                complain=False)
        # With complain=False a second add_and_assign silently skips
        self.cfg.add_and_assign("dup2",
                                description="second",
                                domain=int,
                                default=0,
                                value=2,
                                complain=False)
        self.assertEqual(self.cfg.dup2, 1)


class TestConfigDictAssign(unittest.TestCase):
    """Tests for Config.dict_assign()."""

    def setUp(self):
        self.cfg = Config()

    def test_creates_new_entry(self):
        self.cfg.dict_assign("newkey",
                             description="new",
                             domain=str,
                             default="",
                             value="world")
        self.assertEqual(self.cfg.newkey, "world")

    def test_updates_existing_entry(self):
        self.cfg.add_to_config("existing",
                               description="existing",
                               domain=int,
                               default=0)
        self.cfg.dict_assign("existing",
                             description="existing",
                             domain=int,
                             default=0,
                             value=42)
        self.assertEqual(self.cfg.existing, 42)


class TestConfigQuickAssign(unittest.TestCase):
    """Tests for Config.quick_assign()."""

    def setUp(self):
        self.cfg = Config()

    def test_creates_and_assigns(self):
        self.cfg.quick_assign("answer", int, 42)
        self.assertEqual(self.cfg.answer, 42)

    def test_updates_existing(self):
        self.cfg.quick_assign("x", float, 1.0)
        self.cfg.quick_assign("x", float, 2.0)
        self.assertAlmostEqual(self.cfg.x, 2.0)

    def test_string_value(self):
        self.cfg.quick_assign("label", str, "test")
        self.assertEqual(self.cfg.label, "test")


class TestConfigGet(unittest.TestCase):
    """Tests for Config.get()."""

    def setUp(self):
        self.cfg = Config()

    def test_get_existing_key(self):
        self.cfg.add_to_config("k",
                               description="k",
                               domain=int,
                               default=7)
        self.assertEqual(self.cfg.get("k"), 7)

    def test_get_missing_key_returns_none(self):
        self.assertIsNone(self.cfg.get("nonexistent"))

    def test_get_missing_key_returns_default(self):
        self.assertEqual(self.cfg.get("nonexistent", ifmissing=99), 99)

    def test_get_after_assignment(self):
        self.cfg.add_to_config("val",
                               description="val",
                               domain=float,
                               default=0.0)
        self.cfg.val = 3.14
        self.assertAlmostEqual(self.cfg.get("val"), 3.14)


class TestConfigPopularArgs(unittest.TestCase):
    """Tests for Config.popular_args() which registers commonly-used options."""

    def setUp(self):
        self.cfg = Config()
        self.cfg.popular_args()

    def test_max_iterations_exists(self):
        self.assertIn("max_iterations", self.cfg)

    def test_max_iterations_default(self):
        self.assertEqual(self.cfg.max_iterations, 1)

    def test_solver_name_exists(self):
        self.assertIn("solver_name", self.cfg)

    def test_seed_default(self):
        self.assertEqual(self.cfg.seed, 1134)

    def test_verbose_default_false(self):
        self.assertFalse(self.cfg.verbose)


class TestConfigAddSolverSpecs(unittest.TestCase):
    """Tests for Config.add_solver_specs()."""

    def setUp(self):
        self.cfg = Config()

    def test_adds_solver_name(self):
        self.cfg.add_solver_specs()
        self.assertIn("solver_name", self.cfg)

    def test_adds_solver_options(self):
        self.cfg.add_solver_specs()
        self.assertIn("solver_options", self.cfg)

    def test_prefix_adds_prefixed_names(self):
        self.cfg.add_solver_specs(prefix="ef")
        self.assertIn("ef_solver_name", self.cfg)
        self.assertIn("ef_solver_options", self.cfg)

    def test_defaults_are_none(self):
        self.cfg.add_solver_specs()
        self.assertIsNone(self.cfg.solver_name)
        self.assertIsNone(self.cfg.solver_options)


class TestConfigPhArgs(unittest.TestCase):
    """Tests for Config.ph_args()."""

    def setUp(self):
        self.cfg = Config()
        self.cfg.ph_args()

    def test_linearize_binary_proximal_terms_added(self):
        self.assertIn("linearize_binary_proximal_terms", self.cfg)

    def test_linearize_binary_proximal_terms_default_false(self):
        self.assertFalse(self.cfg.linearize_binary_proximal_terms)

    def test_linearize_proximal_terms_added(self):
        self.assertIn("linearize_proximal_terms", self.cfg)

    def test_linearize_proximal_terms_default_false(self):
        self.assertFalse(self.cfg.linearize_proximal_terms)

    def test_proximal_linearization_tolerance_added(self):
        self.assertIn("proximal_linearization_tolerance", self.cfg)

    def test_proximal_linearization_tolerance_default(self):
        self.assertAlmostEqual(self.cfg.proximal_linearization_tolerance, 1e-1)


class TestConfigAphArgs(unittest.TestCase):
    """Tests for Config.aph_args()."""

    def setUp(self):
        self.cfg = Config()
        self.cfg.aph_args()

    def test_aph_flag_added(self):
        self.assertIn("APH", self.cfg)

    def test_aph_flag_default_false(self):
        self.assertFalse(self.cfg.APH)

    def test_aph_gamma_default(self):
        self.assertAlmostEqual(self.cfg.aph_gamma, 1.0)

    def test_aph_nu_default(self):
        self.assertAlmostEqual(self.cfg.aph_nu, 1.0)

    def test_aph_frac_needed_default(self):
        self.assertAlmostEqual(self.cfg.aph_frac_needed, 1.0)

    def test_aph_dispatch_frac_default(self):
        self.assertAlmostEqual(self.cfg.aph_dispatch_frac, 1.0)

    def test_aph_sleep_seconds_default(self):
        self.assertAlmostEqual(self.cfg.aph_sleep_seconds, 0.01)


class TestConfigTwoSidedArgs(unittest.TestCase):
    """Tests for Config.two_sided_args()."""

    def setUp(self):
        self.cfg = Config()
        self.cfg.two_sided_args()

    def test_rel_gap_added(self):
        self.assertIn("rel_gap", self.cfg)

    def test_rel_gap_default(self):
        self.assertAlmostEqual(self.cfg.rel_gap, 0.05)

    def test_abs_gap_added(self):
        self.assertIn("abs_gap", self.cfg)

    def test_abs_gap_default(self):
        self.assertAlmostEqual(self.cfg.abs_gap, 0.0)

    def test_max_stalled_iters_added(self):
        self.assertIn("max_stalled_iters", self.cfg)

    def test_max_stalled_iters_default(self):
        self.assertEqual(self.cfg.max_stalled_iters, 100)


class TestConfigAddMipgapSpecs(unittest.TestCase):
    """Tests for Config.add_mipgap_specs()."""

    def setUp(self):
        self.cfg = Config()
        self.cfg.add_mipgap_specs()

    def test_iter0_mipgap_added(self):
        self.assertIn("iter0_mipgap", self.cfg)

    def test_iter0_mipgap_default_none(self):
        self.assertIsNone(self.cfg.iter0_mipgap)

    def test_iterk_mipgap_added(self):
        self.assertIn("iterk_mipgap", self.cfg)

    def test_iterk_mipgap_default_none(self):
        self.assertIsNone(self.cfg.iterk_mipgap)

    def test_prefix_mipgap_specs(self):
        cfg2 = Config()
        cfg2.add_mipgap_specs(prefix="EF")
        self.assertIn("EF_iter0_mipgap", cfg2)
        self.assertIn("EF_iterk_mipgap", cfg2)


class TestConfigNumScens(unittest.TestCase):
    """Tests for Config.num_scens_optional() and num_scens_required()."""

    def test_num_scens_optional_adds_entry(self):
        cfg = Config()
        cfg.num_scens_optional()
        self.assertIn("num_scens", cfg)

    def test_num_scens_optional_default_none(self):
        cfg = Config()
        cfg.num_scens_optional()
        self.assertIsNone(cfg.num_scens)

    def test_num_scens_optional_can_be_set(self):
        cfg = Config()
        cfg.num_scens_optional()
        cfg.num_scens = 10
        self.assertEqual(cfg.num_scens, 10)

    def test_num_scens_required_adds_entry(self):
        cfg = Config()
        cfg.num_scens_required()
        self.assertIn("num_scens", cfg)


class TestConfigChecker(unittest.TestCase):
    """Tests for Config.checker()."""

    def _make_rho_cfg(self, **flags):
        """Build a Config with rho-setter flags.

        Note: the key lists below mirror the checks inside Config.checker().
        If new rho setters are added to checker(), this helper must be updated
        to match.
        """
        cfg = Config()
        # Rho-setter keys checked by checker() via get() -- keep in sync with
        # the condition in Config.checker()
        rho_keys = ["grad_rho", "sensi_rho", "coeff_rho",
                    "reduced_costs_rho", "sep_rho"]
        dynamic_keys = ["dynamic_rho_primal_crit", "dynamic_rho_dual_crit"]
        other_keys = ["ph_primal_hub", "ph_dual", "relaxed_ph",
                      "rc_fixer", "reduced_costs"]
        for k in rho_keys + dynamic_keys + other_keys:
            cfg.add_to_config(k, description=k, domain=bool,
                              default=flags.get(k, False), argparse=False)
        return cfg

    def test_valid_config_does_not_raise(self):
        cfg = self._make_rho_cfg()
        # all flags False => no conflict => should not raise
        cfg.checker()

    def test_two_rho_setters_raises(self):
        cfg = self._make_rho_cfg(grad_rho=True, sensi_rho=True)
        with self.assertRaises(ValueError):
            cfg.checker()

    def test_dynamic_rho_without_setter_raises(self):
        cfg = self._make_rho_cfg(dynamic_rho_primal_crit=True)
        with self.assertRaises(ValueError):
            cfg.checker()

    def test_dynamic_rho_with_setter_does_not_raise(self):
        cfg = self._make_rho_cfg(grad_rho=True, dynamic_rho_primal_crit=True)
        # grad_rho is set so dynamic_rho is allowed
        cfg.checker()

    def test_ph_primal_hub_without_ph_dual_raises(self):
        cfg = self._make_rho_cfg(ph_primal_hub=True)
        with self.assertRaises(ValueError):
            cfg.checker()

    def test_ph_primal_hub_with_ph_dual_does_not_raise(self):
        cfg = self._make_rho_cfg(ph_primal_hub=True, ph_dual=True)
        cfg.checker()

    def test_rc_fixer_without_reduced_costs_raises(self):
        cfg = self._make_rho_cfg(rc_fixer=True)
        with self.assertRaises(ValueError):
            cfg.checker()

    def test_rc_fixer_with_reduced_costs_does_not_raise(self):
        cfg = self._make_rho_cfg(rc_fixer=True, reduced_costs=True)
        cfg.checker()


class TestConfigFixerArgs(unittest.TestCase):
    """Tests for Config.fixer_args() and related extension args."""

    def test_fixer_args_adds_fixer(self):
        cfg = Config()
        cfg.fixer_args()
        self.assertIn("fixer", cfg)

    def test_fixer_default_false(self):
        cfg = Config()
        cfg.fixer_args()
        self.assertFalse(cfg.fixer)

    def test_fixer_tol_default(self):
        cfg = Config()
        cfg.fixer_args()
        self.assertAlmostEqual(cfg.fixer_tol, 1e-4)

    def test_grad_rho_args_added(self):
        cfg = Config()
        cfg.gradient_args()
        self.assertIn("grad_rho", cfg)

    def test_sep_rho_args_added(self):
        cfg = Config()
        cfg.sep_rho_args()
        self.assertIn("sep_rho", cfg)

    def test_sep_rho_multiplier_default(self):
        cfg = Config()
        cfg.sep_rho_args()
        self.assertAlmostEqual(cfg.sep_rho_multiplier, 1.0)

    def test_sensi_rho_args_added(self):
        cfg = Config()
        cfg.sensi_rho_args()
        self.assertIn("sensi_rho", cfg)

    def test_coeff_rho_args_added(self):
        cfg = Config()
        cfg.coeff_rho_args()
        self.assertIn("coeff_rho", cfg)


if __name__ == "__main__":
    unittest.main()
