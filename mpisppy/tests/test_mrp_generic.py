###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for mrp_generic (sequential sampling generic driver)

import unittest

import mpisppy.tests.examples.farmer as farmer

from mpisppy.tests.utils import get_solver
from mpisppy.utils import config
from mpisppy.generic.mrp import mrp_args, _ef_xhat_generator, do_mrp


solver_available, solver_name, persistent_available, persistent_solver_name \
    = get_solver()

_refmodelname = "mpisppy.tests.examples.farmer"


def _get_BM_cfg():
    """Create a Config with BM stopping criterion for farmer."""
    cfg = config.Config()
    # Sequential sampling args
    mrp_args(cfg)
    cfg.stopping_criterion = "BM"
    # Solver
    cfg.quick_assign("solver_name", str, solver_name)
    cfg.quick_assign("EF_solver_name", str, solver_name)
    # Model-specific
    cfg.quick_assign("use_integer", bool, False)
    cfg.quick_assign("crops_multiplier", int, 1)
    cfg.quick_assign("num_scens", int, 3)
    cfg.quick_assign("EF_2stage", bool, True)
    cfg.quick_assign("solving_type", str, "EF_2stage")
    # BM parameters (loose enough to converge quickly)
    cfg.BM_h = 2.0
    cfg.BM_hprime = 0.5
    cfg.BM_eps = 0.5
    cfg.BM_eps_prime = 0.4
    cfg.BM_p = 0.2
    cfg.BM_q = 1.2
    # xhat_gen_kwargs from model
    scenario_creator_kwargs = farmer.kw_creator(cfg)
    cfg.quick_assign("xhat_gen_kwargs", dict, scenario_creator_kwargs)
    return cfg


def _get_BPL_cfg():
    """Create a Config with BPL stopping criterion for farmer."""
    cfg = config.Config()
    mrp_args(cfg)
    cfg.stopping_criterion = "BPL"
    cfg.quick_assign("solver_name", str, solver_name)
    cfg.quick_assign("EF_solver_name", str, solver_name)
    cfg.quick_assign("use_integer", bool, False)
    cfg.quick_assign("crops_multiplier", int, 1)
    cfg.quick_assign("num_scens", int, 3)
    cfg.quick_assign("EF_2stage", bool, True)
    cfg.quick_assign("solving_type", str, "EF_2stage")
    # BPL parameters (loose enough to converge quickly)
    cfg.BPL_eps = 100.0
    cfg.BPL_c0 = 25
    cfg.BPL_n0min = 0
    scenario_creator_kwargs = farmer.kw_creator(cfg)
    cfg.quick_assign("xhat_gen_kwargs", dict, scenario_creator_kwargs)
    return cfg


class Test_mrp_config(unittest.TestCase):
    """Test mrp_args configuration registration."""

    def test_mrp_args_registers_all(self):
        cfg = config.Config()
        mrp_args(cfg)
        # Check key options are registered
        self.assertIn("stopping_criterion", cfg)
        self.assertIn("mrp_max_iterations", cfg)
        self.assertIn("xhat_method", cfg)
        self.assertIn("confidence_level", cfg)
        self.assertIn("sample_size_ratio", cfg)
        self.assertIn("BM_h", cfg)
        self.assertIn("BPL_eps", cfg)

    def test_mrp_args_defaults(self):
        cfg = config.Config()
        mrp_args(cfg)
        self.assertEqual(cfg.stopping_criterion, "BM")
        self.assertEqual(cfg.mrp_max_iterations, 200)
        self.assertEqual(cfg.xhat_method, "EF")
        self.assertEqual(cfg.confidence_level, 0.95)


class Test_mrp_xhat_generator(unittest.TestCase):
    """Test the generic EF xhat generator."""

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ef_xhat_generator_structure(self):
        """Verify _ef_xhat_generator returns a dict with a ROOT key."""
        cfg = _get_BM_cfg()
        scenario_names = farmer.scenario_names_creator(3)
        xhat = _ef_xhat_generator(
            scenario_names,
            solver_name=solver_name,
            solver_options=None,
            cfg=cfg,
            module_name=_refmodelname,
        )
        self.assertIn("ROOT", xhat)
        self.assertEqual(len(xhat["ROOT"]), 3)  # farmer has 3 first-stage vars

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ef_xhat_generator_values(self):
        """Verify _ef_xhat_generator returns reasonable values for farmer."""
        cfg = _get_BM_cfg()
        scenario_names = farmer.scenario_names_creator(3)
        xhat = _ef_xhat_generator(
            scenario_names,
            solver_name=solver_name,
            solver_options=None,
            cfg=cfg,
            module_name=_refmodelname,
        )
        # All planting decisions should be non-negative
        for val in xhat["ROOT"]:
            self.assertGreaterEqual(val, -1e-6)


class Test_mrp_do_mrp(unittest.TestCase):
    """Test the do_mrp orchestrator."""

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_do_mrp_BM_result_structure(self):
        """Verify do_mrp returns the expected dict structure."""
        cfg = _get_BM_cfg()
        result = do_mrp(_refmodelname, farmer, cfg)
        self.assertIn("T", result)
        self.assertIn("Candidate_solution", result)
        self.assertIn("CI", result)
        self.assertIsInstance(result["T"], int)
        self.assertGreater(result["T"], 0)
        self.assertEqual(len(result["CI"]), 2)
        self.assertEqual(result["CI"][0], 0)
        self.assertGreater(result["CI"][1], 0)
        self.assertIn("ROOT", result["Candidate_solution"])

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_do_mrp_BPL_result_structure(self):
        """Verify do_mrp with BPL stopping criterion."""
        cfg = _get_BPL_cfg()
        result = do_mrp(_refmodelname, farmer, cfg)
        self.assertIn("T", result)
        self.assertIn("Candidate_solution", result)
        self.assertIn("CI", result)
        self.assertEqual(result["CI"][0], 0)
        self.assertEqual(result["CI"][1], 100.0)  # BPL upper bound = BPL_eps
        self.assertIn("ROOT", result["Candidate_solution"])


if __name__ == '__main__':
    unittest.main()
