###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for the empirical bootstrap confidence-interval code (bootsp), using
# the small, fully deterministic schultz example. Run serially:
#
#   python -m pytest mpisppy/tests/test_boot_sp.py
#
# schultz has integer data that is a function of the scenario number, so the
# extensive-form optimum and the (numpy-seeded) bootstrap draws are the same
# for every correct solver; the locked values below are compared with
# round_pos_sig to absorb floating-point noise.

import os
import sys
import types
import unittest

import mpisppy.utils.sputils as sputils
from mpisppy.tests.utils import get_solver, round_pos_sig

import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp
import mpisppy.confidence_intervals.bootsp.user_boot as user_boot
import mpisppy.confidence_intervals.bootsp.simulate_boot as simulate_boot

sputils.disable_tictoc_output()

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

module_dir = os.path.dirname(os.path.abspath(__file__))
bootsp_examples = os.path.join(module_dir, "..", "..", "examples", "bootsp")
example_dir = os.path.join(bootsp_examples, "schultz")
data_example_dir = os.path.join(bootsp_examples, "schultz_data")
for _d in (example_dir, data_example_dir):
    if not os.path.exists(_d):
        raise RuntimeError(f"Directory not found: {_d}")
    if _d not in sys.path:
        sys.path.insert(0, _d)

MODULE_NAME = "unique_schultz"
DATA_MODULE_NAME = "schultz_data"

empirical_methods = ["Classical_gaussian",
                     "Classical_quantile",
                     "Subsampling",
                     "Bagging_with_replacement",
                     "Bagging_without_replacement",
                     "Extended"]

# ci_optimal locked at these params with seed_offset=100 (serial, one MPI rank)
locked_ci_optimal = {
    "Classical_gaussian": [-55.15313724796903, -49.780196085364324],
    "Classical_quantile": [-54.94166666666668, -50.03166666666667],
}

# same, for the data-file example (schultz_data), serial, seed_offset=100
locked_ci_optimal_data = {
    "Classical_gaussian": [-68.27054940987843, -66.1294505901215],
    "Classical_quantile": [-68.08299999999997, -66.22149999999996],
}
# coverage harness (rate, length) for schultz_data, serial, seed base 0, 4 reps
locked_coverage_data = (1.0, 2.5269999999999975)


def _make_cfg(method="Classical_quantile"):
    cfg = boot_utils._process_module(MODULE_NAME)
    cfg.module_name = MODULE_NAME
    cfg.max_count = 50
    cfg.candidate_sample_size = 1
    cfg.sample_size = 30
    cfg.subsample_size = 10
    cfg.nB = 20
    cfg.alpha = 0.1
    cfg.seed_offset = 100
    cfg.xhat_fname = "None"
    cfg.optimal_fname = "None"
    cfg.trace_fname = None
    cfg.coverage_replications = 4
    cfg.solver_name = solver_name
    cfg.boot_method = method
    return cfg


def _make_data_cfg(method="Classical_quantile", seed=100, reps=4):
    cfg = boot_utils._process_module(DATA_MODULE_NAME)
    cfg.module_name = DATA_MODULE_NAME
    cfg.max_count = 200          # the number of rows in schultz_data.csv
    cfg.candidate_sample_size = 5
    cfg.sample_size = 100
    cfg.subsample_size = 20
    cfg.nB = 20
    cfg.alpha = 0.1
    cfg.seed_offset = seed
    cfg.xhat_fname = "None"
    cfg.optimal_fname = "None"
    cfg.trace_fname = None
    cfg.coverage_replications = reps
    cfg.solver_name = solver_name
    cfg.boot_method = method
    cfg.data_file = "schultz_data.csv"
    return cfg


#*****************************************************************************
class Test_boot_sp(unittest.TestCase):
    """ Test the empirical bootstrap code on the schultz example. """

    def _assert_close_list(self, got, expected, sig=4):
        got = list(got)
        self.assertEqual(len(got), len(expected))
        for g, e in zip(got, expected):
            self.assertEqual(round_pos_sig(g, sig), round_pos_sig(e, sig))

    # -- pieces that do not need a solver -------------------------------------

    def test_boot_methods_enum_complete(self):
        members = boot_utils.BootMethods.list_of_members()
        self.assertEqual(len(members), 11)  # 6 empirical + 5 smoothed
        for m in empirical_methods:
            self.assertIn(m, members)
        for m in ["Smoothed_boot_epi", "Smoothed_boot_kernel",
                  "Smoothed_boot_epi_quantile", "Smoothed_boot_kernel_quantile",
                  "Smoothed_bagging"]:
            self.assertIn(m, members)
        with self.assertRaises(ValueError):
            boot_utils.BootMethods.check_for_it("not_a_method")

    def test_is_smoothed(self):
        self.assertTrue(boot_utils.is_smoothed("Smoothed_bagging"))
        self.assertFalse(boot_utils.is_smoothed("Classical_quantile"))
        self.assertEqual(len(boot_utils.empirical_members()), 6)

    def test_compute_xhat_requires_generator(self):
        # a module with no xhat_generator (fixed or legacy) must raise, and the
        # message must name both spellings
        fake = types.ModuleType("no_generator_module")
        fake.kw_creator = lambda cfg: {}
        fake.scenario_names_creator = lambda n, start=None: []
        cfg = _make_cfg()
        cfg.module_name = "no_generator_module"
        with self.assertRaises(RuntimeError) as ctx:
            boot_utils.compute_xhat(cfg, fake)
        msg = str(ctx.exception)
        self.assertIn("xhat_generator", msg)
        self.assertIn("xhat_generator_no_generator_module", msg)

    def test_smoothed_not_yet_merged_boot_sp(self):
        cfg = _make_cfg("Smoothed_boot_kernel")
        with self.assertRaises(RuntimeError) as ctx:
            boot_sp.compute_ci(cfg, None, {"ROOT": [0.0, 5.0]})
        self.assertIn("smoothed", str(ctx.exception).lower())

    # -- pieces that need a solver --------------------------------------------

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_compute_xhat_fixed_name(self):
        # unique_schultz supplies the fixed-name xhat_generator
        cfg = _make_cfg()
        module = boot_utils.module_name_to_module(MODULE_NAME)
        xhat = boot_utils.compute_xhat(cfg, module)
        self.assertIn("ROOT", xhat)
        self.assertEqual(len(xhat["ROOT"]), 2)
        self.assertAlmostEqual(abs(xhat["ROOT"][0]), 0.0, places=5)
        self.assertAlmostEqual(xhat["ROOT"][1], 5.0, places=5)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_empirical_methods_wellformed(self):
        module = boot_utils.module_name_to_module(MODULE_NAME)
        xhat = boot_utils.compute_xhat(_make_cfg(), module)
        for method in empirical_methods:
            cfg = _make_cfg(method)
            res = boot_sp.compute_ci(cfg, module, xhat)
            self.assertEqual(len(res), 6)
            ci_optimal, ci_upper, ci_gap = res[0], res[1], res[2]
            # confidence intervals must be ordered (low, high)
            self.assertLessEqual(ci_optimal[0], ci_optimal[1], msg=method)
            self.assertLessEqual(ci_upper[0], ci_upper[1], msg=method)
            self.assertLessEqual(ci_gap[0], ci_gap[1], msg=method)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_empirical_locked_values(self):
        module = boot_utils.module_name_to_module(MODULE_NAME)
        xhat = boot_utils.compute_xhat(_make_cfg(), module)
        for method, expected in locked_ci_optimal.items():
            cfg = _make_cfg(method)
            res = boot_sp.compute_ci(cfg, module, xhat)
            self._assert_close_list(res[0], expected)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_user_boot_main_routine(self):
        # the end-user entry point returns the 6-tuple and clamps ci_gap[0]>=0
        module = boot_utils.module_name_to_module(MODULE_NAME)
        cfg = _make_cfg("Classical_quantile")
        res = user_boot.main_routine(cfg, module)
        self.assertEqual(len(res), 6)
        self._assert_close_list(res[0], locked_ci_optimal["Classical_quantile"])
        self.assertGreaterEqual(res[2][0], 0.0)  # ci_gap[0] clamped to >= 0

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_user_boot_smoothed_raises(self):
        module = boot_utils.module_name_to_module(MODULE_NAME)
        cfg = _make_cfg("Smoothed_bagging")
        with self.assertRaises(RuntimeError):
            user_boot.main_routine(cfg, module)


#*****************************************************************************
class Test_boot_sp_data(unittest.TestCase):
    """ Test the data-file example (schultz_data), which reads its scenario
        data from a committed CSV rather than generating it on the fly. """

    def _assert_close_list(self, got, expected, sig=4):
        got = list(got)
        self.assertEqual(len(got), len(expected))
        for g, e in zip(got, expected):
            self.assertEqual(round_pos_sig(g, sig), round_pos_sig(e, sig))

    def test_load_data(self):
        # no solver needed: just the dataset reader
        module = boot_utils.module_name_to_module(DATA_MODULE_NAME)
        data = module.load_data()
        self.assertEqual(data.shape, (200, 2))
        with self.assertRaises(FileNotFoundError):
            module.load_data("does_not_exist.csv")

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_data_file_xhat(self):
        cfg = _make_data_cfg()
        module = boot_utils.module_name_to_module(DATA_MODULE_NAME)
        xhat = boot_utils.compute_xhat(cfg, module)
        self.assertIn("ROOT", xhat)
        self.assertEqual(len(xhat["ROOT"]), 2)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_data_file_locked_values(self):
        module = boot_utils.module_name_to_module(DATA_MODULE_NAME)
        xhat = boot_utils.compute_xhat(_make_data_cfg(), module)
        for method, expected in locked_ci_optimal_data.items():
            res = boot_sp.compute_ci(_make_data_cfg(method), module, xhat)
            self._assert_close_list(res[0], expected)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_data_file_coverage(self):
        module = boot_utils.module_name_to_module(DATA_MODULE_NAME)
        cfg = _make_data_cfg("Classical_quantile", seed=0, reps=4)
        # Read a precomputed optimum so the coverage simulation skips the full
        # max_count-scenario extensive form (process_optimal). That single solve
        # is large enough to exceed size-limited solvers on CI (e.g. the
        # community edition of cplex, error 1016); the resampled batches are
        # smaller and solve fine. The stored value is exactly what process_optimal
        # would compute, so the coverage result is unchanged.
        cfg.optimal_fname = os.path.join(data_example_dir, "schultz_data_optimal.npy")
        rate, length = simulate_boot.main(cfg, module)
        self.assertEqual(rate, locked_coverage_data[0])
        self.assertEqual(round_pos_sig(length, 4),
                         round_pos_sig(locked_coverage_data[1], 4))


if __name__ == '__main__':
    unittest.main()
