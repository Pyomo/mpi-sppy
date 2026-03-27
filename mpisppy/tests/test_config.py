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


if __name__ == "__main__":
    unittest.main()
