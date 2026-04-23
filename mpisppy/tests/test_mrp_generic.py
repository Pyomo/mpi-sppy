###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for mrp_generic (sequential sampling generic driver)

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

import mpisppy.tests.examples.farmer as farmer
from mpisppy.confidence_intervals import ciutils

from mpisppy.tests.utils import get_solver
from mpisppy.utils import config
from mpisppy.generic.mrp import mrp_args, _ef_xhat_generator, \
    _cylinder_xhat_generator, do_mrp


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

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ef_xhat_generator_with_solver_options(self):
        """Verify _ef_xhat_generator works with solver_options dict."""
        cfg = _get_BM_cfg()
        scenario_names = farmer.scenario_names_creator(3)
        xhat = _ef_xhat_generator(
            scenario_names,
            solver_name=solver_name,
            solver_options={"threads": 1},
            cfg=cfg,
            module_name=_refmodelname,
        )
        self.assertIn("ROOT", xhat)
        self.assertEqual(len(xhat["ROOT"]), 3)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_ef_xhat_generator_with_start_seed(self):
        """Verify _ef_xhat_generator passes start_seed through."""
        cfg = _get_BM_cfg()
        scenario_names = farmer.scenario_names_creator(3)
        xhat = _ef_xhat_generator(
            scenario_names,
            solver_name=solver_name,
            solver_options=None,
            cfg=cfg,
            module_name=_refmodelname,
            start_seed=42,
        )
        self.assertIn("ROOT", xhat)
        self.assertEqual(len(xhat["ROOT"]), 3)


class Test_cylinder_xhat_generator(unittest.TestCase):
    """Test _cylinder_xhat_generator with mocked MPI and decomp."""

    def test_cylinder_xhat_generator_mocked(self):
        """Cover _cylinder_xhat_generator code paths with mocked decomp."""
        cfg = _get_BM_cfg()
        scenario_names = farmer.scenario_names_creator(3)

        # Write a fake xhat file that ciutils.read_xhat will find
        xhat_data = {"ROOT": np.array([74.0, 245.0, 181.0])}
        tmp_path = tempfile.mktemp(suffix=".npy")
        ciutils.write_xhat(xhat_data, tmp_path)

        mock_wheel = MagicMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.bcast.return_value = tmp_path

        with patch("mpisppy.generic.decomp.do_decomp",
                   return_value=mock_wheel), \
             patch("mpisppy.generic.mrp.MPI") as mock_mpi:
            mock_mpi.COMM_WORLD = mock_comm

            result = _cylinder_xhat_generator(
                scenario_names,
                solver_name=solver_name,
                cfg=cfg,
                module_name=_refmodelname,
                module=farmer,
            )

        self.assertIn("ROOT", result)
        self.assertEqual(len(result["ROOT"]), 3)
        mock_wheel.write_first_stage_solution.assert_called_once()
        # Two barriers: one after write_first_stage_solution, one after
        # read_xhat (so rank 0 can't remove the tmp file before peers finish).
        self.assertEqual(mock_comm.Barrier.call_count, 2)
        # tmp_path should have been removed by the function (rank 0)
        self.assertFalse(os.path.exists(tmp_path))


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

    def test_do_mrp_invalid_stopping_criterion(self):
        """Verify do_mrp raises ValueError for invalid stopping criterion."""
        cfg = _get_BM_cfg()
        cfg.stopping_criterion = "INVALID"
        with self.assertRaises(ValueError):
            do_mrp(_refmodelname, farmer, cfg)

    def test_do_mrp_invalid_xhat_method(self):
        """Verify do_mrp raises ValueError for unknown xhat_method."""
        cfg = _get_BM_cfg()
        cfg.xhat_method = "bogus"
        with self.assertRaises(ValueError):
            do_mrp(_refmodelname, farmer, cfg)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_do_mrp_infers_EF_solver(self):
        """Verify do_mrp defaults EF_solver_name to solver_name."""
        cfg = _get_BM_cfg()
        # Remove explicit EF_solver_name so do_mrp must infer it
        del cfg["EF_solver_name"]
        result = do_mrp(_refmodelname, farmer, cfg)
        self.assertIn("T", result)

    def test_do_mrp_cylinders_branch(self):
        """Cover the cylinders xhat_method code path in do_mrp."""
        cfg = _get_BM_cfg()
        cfg.xhat_method = "cylinders"

        mock_result = {
            "T": 2,
            "Candidate_solution": {"ROOT": [1, 2, 3]},
            "CI": [0, 5.0],
        }
        mock_sampler = MagicMock()
        mock_sampler.run.return_value = mock_result

        with patch("mpisppy.confidence_intervals.seqsampling.SeqSampling",
                   return_value=mock_sampler):
            result = do_mrp(_refmodelname, farmer, cfg)

        self.assertEqual(result["T"], 2)
        self.assertEqual(result["CI"], [0, 5.0])

    def test_do_mrp_multistage_branch(self):
        """Cover the multistage code path in do_mrp."""
        cfg = _get_BM_cfg()
        cfg.quick_assign("branching_factors", list, [3, 3])

        mock_result = {
            "T": 1,
            "Candidate_solution": {"ROOT": [1, 2, 3]},
            "CI": [0, 10.0],
        }
        mock_sampler = MagicMock()
        mock_sampler.run.return_value = mock_result

        with patch(
            "mpisppy.confidence_intervals.multi_seqsampling"
            ".IndepScens_SeqSampling",
            return_value=mock_sampler,
        ):
            result = do_mrp(_refmodelname, farmer, cfg)

        self.assertEqual(result["T"], 1)
        self.assertEqual(result["CI"], [0, 10.0])


class Test_mrp_generic_parse(unittest.TestCase):
    """Test parse_mrp_args from mrp_generic."""

    def test_parse_mrp_args_registers_config(self):
        """Verify parse_mrp_args registers all needed config groups."""
        from mpisppy.mrp_generic import parse_mrp_args

        test_argv = [
            "prog",
            "--module-name", "mpisppy.tests.examples.farmer",
            "--num-scens", "3",
            "--solver-name", solver_name,
            "--stopping-criterion", "BM",
            "--default-rho", "1.0",
            "--max-iterations", "2",
        ]
        with patch("sys.argv", test_argv):
            cfg = parse_mrp_args(farmer)
        self.assertEqual(cfg.num_scens, 3)
        self.assertEqual(cfg.stopping_criterion, "BM")
        self.assertIn("BM_h", cfg)
        self.assertIn("lagrangian", cfg)


class Test_mrp_generic_main(unittest.TestCase):
    """Test the __main__ block of mrp_generic."""

    def test_main_no_args_exits(self):
        """Cover the early-exit path when no arguments are given."""
        import runpy
        with patch("sys.argv", ["mrp_generic"]):
            with self.assertRaises(SystemExit):
                runpy.run_module("mpisppy.mrp_generic", run_name="__main__")

    def test_main_runs_with_mocked_do_mrp(self):
        """Cover the main execution path with mocked do_mrp."""
        import runpy

        mock_result = {
            "T": 2,
            "Candidate_solution": {"ROOT": [74.0, 245.0, 181.0]},
            "CI": [0, 5.0],
        }
        test_argv = [
            "mrp_generic",
            "--module-name", "mpisppy.tests.examples.farmer",
            "--num-scens", "3",
            "--solver-name", solver_name,
            "--stopping-criterion", "BM",
            "--default-rho", "1.0",
            "--max-iterations", "2",
        ]
        with patch("sys.argv", test_argv), \
             patch("mpisppy.generic.mrp.do_mrp", return_value=mock_result):
            runpy.run_module("mpisppy.mrp_generic",
                             run_name="__main__",
                             alter_sys=True)


if __name__ == '__main__':
    unittest.main()
