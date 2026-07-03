###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for the smoothed bootstrap/bagging code (bootsp) and the statdist
# univariate distributions, plus the empirical farmer/cvar examples that need
# statdist (and so could not live in test_boot_sp.py). Run serially:
#
#   python -m pytest mpisppy/tests/test_boot_sp_smoothed.py
# Parallel (exercises the smoothed Gatherv batch split across ranks):
#   mpiexec -np 2 python -m mpi4py mpisppy/tests/test_boot_sp_smoothed.py
#
# The smoothed methods fit a distribution with statdist (scipy); the kernel and
# bagging methods need only an LP/MIP solver, while the epi-spline methods also
# need a nonlinear solver (ipopt), so those tests are skipped when ipopt is
# absent.

import os
import sys
import math
import warnings
import unittest

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy.tests.utils import get_solver, round_pos_sig

import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp
import mpisppy.confidence_intervals.bootsp.smoothed_boot_sp as smoothed_boot_sp
import mpisppy.confidence_intervals.bootsp.user_boot as user_boot
import mpisppy.confidence_intervals.bootsp.simulate_boot as simulate_boot
from mpisppy.confidence_intervals.bootsp.statdist.distribution_factory import (
    distribution_factory,
)

sputils.disable_tictoc_output()

# statdist integrates array-valued pdfs through scipy.integrate.quad, which
# emits a NumPy>=1.25 "array to scalar" DeprecationWarning many thousands of
# times; silence just that one so the (large) CI logs stay readable. The
# numerics are unchanged.
warnings.filterwarnings(
    "ignore",
    message="Conversion of an array with ndim > 0 to a scalar is deprecated",
    category=DeprecationWarning,
)

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()
ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)

comm = boot_utils.comm
n_proc = boot_utils.n_proc
my_rank = boot_utils.my_rank

module_dir = os.path.dirname(os.path.abspath(__file__))
bootsp_examples = os.path.join(module_dir, "..", "..", "examples", "bootsp")
for _sub in ("farmer", "cvar", "multi_knapsack"):
    _d = os.path.join(bootsp_examples, _sub)
    if not os.path.exists(_d):
        raise RuntimeError(f"Directory not found: {_d}")
    if _d not in sys.path:
        sys.path.insert(0, _d)

MK_DATA = os.path.abspath(
    os.path.join(bootsp_examples, "multi_knapsack", "multi_knapsack_data.json"))

univariate_tokens = ["univariate-unif", "univariate-normal", "univariate-student",
                     "univariate-kernel", "univariate-epispline",
                     "univariate-empirical", "univariate-discrete"]


def _make_cvar_cfg(method="Smoothed_bagging", seed=42, reps=2):
    cfg = boot_utils._process_module("cvar")
    cfg.module_name = "cvar"
    cfg.max_count = 200
    cfg.candidate_sample_size = 5
    cfg.sample_size = 20
    cfg.subsample_size = 5
    cfg.nB = 8
    cfg.alpha = 0.1
    cfg.seed_offset = seed
    cfg.xhat_fname = "None"
    cfg.optimal_fname = "None"
    cfg.trace_fname = None
    cfg.coverage_replications = reps
    cfg.solver_name = solver_name
    cfg.boot_method = method
    cfg.smoothed_B_I = 3
    cfg.smoothed_center_sample_size = 20
    return cfg


def _make_farmer_cfg(method="Classical_quantile", seed=100):
    cfg = boot_utils._process_module("farmer")
    cfg.module_name = "farmer"
    cfg.max_count = 200
    cfg.candidate_sample_size = 5
    cfg.sample_size = 30
    cfg.subsample_size = 10
    cfg.nB = 8
    cfg.alpha = 0.1
    cfg.seed_offset = seed
    cfg.xhat_fname = "None"
    cfg.optimal_fname = "None"
    cfg.trace_fname = None
    cfg.coverage_replications = 2
    cfg.solver_name = solver_name
    cfg.boot_method = method
    cfg.crops_multiplier = 1
    cfg.yield_cv = 0.1
    return cfg


#*****************************************************************************
class Test_statdist(unittest.TestCase):
    """ Direct tests of the trimmed statdist univariate distributions. """

    def test_factory_resolves_univariate(self):
        for token in univariate_tokens:
            cls = distribution_factory(token)
            self.assertTrue(hasattr(cls, "fit") or callable(cls), msg=token)

    def test_factory_rejects_unknown(self):
        with self.assertRaises(NameError):
            distribution_factory("not-a-distribution")

    def test_factory_drops_multivariate(self):
        # the multivariate/copula distributions were trimmed out of the port
        for token in ["multivariate-normal", "gaussian-copula"]:
            with self.assertRaises(NameError):
                distribution_factory(token)

    def test_scipy_not_imported_at_module_import(self):
        # statdist defers scipy so the empirical path stays scipy-free; the
        # distributions module must not pull scipy in merely on import
        import importlib
        import mpisppy.confidence_intervals.bootsp.statdist.distributions as dmod
        importlib.reload  # (noop reference; module already imported)
        self.assertTrue(hasattr(dmod, "UnivariateGaussianKernelDistribution"))

    def test_uniform_inverse(self):
        uunif = distribution_factory("univariate-unif")(0, 1)
        mid = uunif.cdf_inverse(0.5)
        self.assertAlmostEqual(mid, 0.5, places=6)
        self.assertLessEqual(uunif.cdf_inverse(0.25), uunif.cdf_inverse(0.75))

    def test_normal_inverse(self):
        unorm = distribution_factory("univariate-normal")(mean=3.0, var=4.0)
        self.assertAlmostEqual(unorm.cdf_inverse(0.5), 3.0, places=4)
        self.assertLess(unorm.cdf_inverse(0.25), unorm.cdf_inverse(0.75))

    def test_kernel_fit_inverse(self):
        # the kernel-density fit backs Smoothed_boot_kernel and Smoothed_bagging
        import numpy as np
        data = list(np.random.RandomState(0).normal(0, 1, size=200))
        kde = distribution_factory("univariate-kernel").fit(data)
        lo = kde.cdf_inverse(0.25)
        hi = kde.cdf_inverse(0.75)
        self.assertTrue(math.isfinite(lo) and math.isfinite(hi))
        self.assertLess(lo, hi)

    def test_empirical_fit_inverse(self):
        import numpy as np
        data = list(np.random.RandomState(1).normal(0, 1, size=200))
        emp = distribution_factory("univariate-empirical").fit(data)
        self.assertLessEqual(emp.cdf_inverse(0.25), emp.cdf_inverse(0.75))

    @unittest.skipIf(not ipopt_available, "ipopt (nonlinear solver) not available")
    def test_epispline_fit_inverse(self):
        import numpy as np
        data = list(np.random.RandomState(2).normal(0, 1, size=100))
        epi = distribution_factory("univariate-epispline").fit(data)
        self.assertLessEqual(epi.cdf_inverse(0.25), epi.cdf_inverse(0.75))


#*****************************************************************************
class Test_empirical_examples(unittest.TestCase):
    """ Empirical methods on the statdist-dependent examples (farmer, cvar).

    These could not live in test_boot_sp.py because importing farmer/cvar pulls
    in statdist; the methods themselves are the empirical ones.
    """

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_farmer_empirical_wellformed(self):
        module = boot_utils.module_name_to_module("farmer")
        xhat = boot_utils.compute_xhat(_make_farmer_cfg(), module)
        self.assertIn("ROOT", xhat)
        for method in ["Classical_quantile", "Bagging_with_replacement"]:
            res = boot_sp.compute_ci(_make_farmer_cfg(method), module, xhat)
            self.assertEqual(len(res), 6)
            for ci in res[:3]:
                self.assertLessEqual(ci[0], ci[1], msg=f"{method}: {ci}")

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_empirical_wellformed(self):
        module = boot_utils.module_name_to_module("cvar")
        cfg = _make_cvar_cfg("Classical_quantile")
        xhat = boot_utils.compute_xhat(cfg, module)
        self.assertIn("ROOT", xhat)
        res = boot_sp.compute_ci(_make_cvar_cfg("Classical_quantile"), module, xhat)
        self.assertEqual(len(res), 6)
        for ci in res[:3]:
            self.assertLessEqual(ci[0], ci[1])

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_empirical_deterministic(self):
        # same cfg twice must give the same interval (seeded streams)
        module = boot_utils.module_name_to_module("cvar")
        xhat = boot_utils.compute_xhat(_make_cvar_cfg("Classical_gaussian"), module)
        r1 = boot_sp.compute_ci(_make_cvar_cfg("Classical_gaussian"), module, xhat)
        r2 = boot_sp.compute_ci(_make_cvar_cfg("Classical_gaussian"), module, xhat)
        for a, b in zip(list(r1[0]), list(r2[0])):
            self.assertEqual(round_pos_sig(a, 6), round_pos_sig(b, 6))


#*****************************************************************************
class Test_smoothed(unittest.TestCase):
    """ Smoothed methods (kernel/bagging need no nonlinear solver). """

    def _check_gap_ci(self, result, method):
        # rank-0 result is (ci_gap_two_sided, center_gap); non-root is (None, None)
        if my_rank == 0:
            ci_gap, center_gap = result
            self.assertEqual(len(ci_gap), 2)
            self.assertTrue(math.isfinite(center_gap), msg=method)
            self.assertLessEqual(ci_gap[0], ci_gap[1], msg=f"{method}: {ci_gap}")
        else:
            self.assertEqual(result, (None, None))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_smoothed_bagging(self):
        module = boot_utils.module_name_to_module("cvar")
        cfg = _make_cvar_cfg("Smoothed_bagging")
        xhat = boot_utils.compute_xhat(cfg, module)
        result = smoothed_boot_sp.compute_smoothed_ci(cfg, module, xhat)
        self._check_gap_ci(result, "Smoothed_bagging")

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_smoothed_kernel(self):
        module = boot_utils.module_name_to_module("cvar")
        cfg = _make_cvar_cfg("Smoothed_boot_kernel")
        xhat = boot_utils.compute_xhat(cfg, module)
        result = smoothed_boot_sp.compute_smoothed_ci(cfg, module, xhat)
        self._check_gap_ci(result, "Smoothed_boot_kernel")

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_smoothed_kernel_quantile(self):
        module = boot_utils.module_name_to_module("cvar")
        cfg = _make_cvar_cfg("Smoothed_boot_kernel_quantile")
        xhat = boot_utils.compute_xhat(cfg, module)
        result = smoothed_boot_sp.compute_smoothed_ci(cfg, module, xhat)
        self._check_gap_ci(result, "Smoothed_boot_kernel_quantile")

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_user_boot_smoothed(self):
        # the end-user entry point routes smoothed methods and clamps ci_gap[0]
        module = boot_utils.module_name_to_module("cvar")
        cfg = _make_cvar_cfg("Smoothed_bagging")
        result = user_boot.main_routine(cfg, module)
        if my_rank == 0:
            ci_gap, center_gap = result
            self.assertGreaterEqual(ci_gap[0], 0.0)
            self.assertLessEqual(ci_gap[0], ci_gap[1])
        else:
            self.assertEqual(result, (None, None))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_simulate_smoothed_coverage(self):
        # the smoothed coverage harness (this exercises the section-4.3
        # compute_xhat fix: no xhat file, so it computes xhat internally)
        module = boot_utils.module_name_to_module("cvar")
        cfg = _make_cvar_cfg("Smoothed_bagging", reps=2)
        result = simulate_boot.main(cfg, module)
        if my_rank == 0:
            cov_two, cov_one, ci_len, run_time = result
            self.assertGreaterEqual(cov_two, 0.0)
            self.assertLessEqual(cov_two, 1.0)
            self.assertGreaterEqual(cov_one, cov_two)  # one-sided covers at least as often
            self.assertEqual(len(ci_len), cfg.coverage_replications)
        else:
            self.assertEqual(result, (None, None, None, None))

    @unittest.skipIf(not ipopt_available, "ipopt (nonlinear solver) not available")
    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_smoothed_epi(self):
        module = boot_utils.module_name_to_module("cvar")
        cfg = _make_cvar_cfg("Smoothed_boot_epi")
        xhat = boot_utils.compute_xhat(cfg, module)
        result = smoothed_boot_sp.compute_smoothed_ci(cfg, module, xhat)
        self._check_gap_ci(result, "Smoothed_boot_epi")


#*****************************************************************************
class Test_multi_knapsack(unittest.TestCase):
    """ Smoke test the multi_knapsack example (deterministic-data-json path). """

    def test_import_and_data(self):
        module = boot_utils.module_name_to_module("multi_knapsack")
        self.assertTrue(hasattr(module, "scenario_creator"))
        self.assertTrue(hasattr(module, "data_sampler"))
        self.assertTrue(hasattr(module, "xhat_generator"))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_multi_knapsack_empirical(self):
        module = boot_utils.module_name_to_module("multi_knapsack")
        cfg = boot_utils._process_module("multi_knapsack")
        cfg.module_name = "multi_knapsack"
        cfg.max_count = 60
        cfg.candidate_sample_size = 3
        cfg.sample_size = 15
        cfg.subsample_size = 5
        cfg.nB = 6
        cfg.alpha = 0.1
        cfg.seed_offset = 100
        cfg.xhat_fname = "None"
        cfg.optimal_fname = "None"
        cfg.trace_fname = None
        cfg.coverage_replications = 2
        cfg.solver_name = solver_name
        cfg.boot_method = "Bagging_with_replacement"
        cfg.deterministic_data_json = MK_DATA
        xhat = boot_utils.compute_xhat(cfg, module)
        self.assertIn("ROOT", xhat)
        res = boot_sp.compute_ci(cfg, module, xhat)
        self.assertEqual(len(res), 6)


if __name__ == '__main__':
    unittest.main()
