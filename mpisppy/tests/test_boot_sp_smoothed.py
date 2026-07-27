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
import importlib.util
import subprocess
import tempfile
import unittest
from collections import OrderedDict

import numpy as np
from pyomo.common.dependencies import scipy
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy.tests.utils import get_solver, round_pos_sig

import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp
import mpisppy.confidence_intervals.bootsp.smoothed_boot_sp as smoothed_boot_sp
import mpisppy.confidence_intervals.bootsp.user_boot as user_boot
import mpisppy.confidence_intervals.bootsp.simulate_boot as simulate_boot
import mpisppy.confidence_intervals.bootsp.statdist.distributions as statdist_distributions
import mpisppy.confidence_intervals.bootsp.statdist.utilities as statdist_utilities
from mpisppy.confidence_intervals.bootsp.statdist.base_distribution import (
    Parameter,
    UnivariateDistribution,
)
from mpisppy.confidence_intervals.bootsp.statdist.distribution_factory import (
    distribution_factory,
)

sputils.disable_tictoc_output()

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()
ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)
# matplotlib is the optional [plot] extra; only the plotting test needs it
matplotlib_available = importlib.util.find_spec("matplotlib") is not None

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

    def test_registry_metadata(self):
        # every univariate distribution registers under its lower-cased name
        # and declares one dimension; the lookup itself is case insensitive
        for token in univariate_tokens:
            cls = distribution_factory(token)
            self.assertEqual(cls.registered_name, token)
            self.assertEqual(cls.registered_ndim, 1)
            self.assertIs(distribution_factory(token.upper()), cls)

    # the subprocess imports mpi-sppy, so it initializes MPI; under an mpiexec
    # launch it would inherit this job's environment and join it
    @unittest.skipIf(n_proc > 1, "spawns a plain (non-MPI) python subprocess")
    def test_scipy_not_imported_at_module_import(self):
        # statdist defers scipy so the empirical path stays scipy-free; the
        # distributions module must not pull scipy in merely on import, so ask
        # a fresh interpreter (this one has scipy loaded by the tests below)
        code = ("import sys;"
                "import mpisppy.confidence_intervals.bootsp.statdist.distributions;"
                "print('scipy loaded:', 'scipy' in sys.modules)")
        done = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, msg=done.stderr)
        self.assertIn("scipy loaded: False", done.stdout)

    def test_uniform_inverse(self):
        uunif = distribution_factory("univariate-unif")(0, 1)
        mid = uunif.cdf_inverse(0.5)
        self.assertAlmostEqual(mid, 0.5, places=6)
        self.assertLessEqual(uunif.cdf_inverse(0.25), uunif.cdf_inverse(0.75))

    def test_uniform_rejects_degenerate_support(self):
        with self.assertRaises(ValueError):
            distribution_factory("univariate-unif")(1.0, 1.0)

    def test_uniform_density_and_cdf(self):
        unif = distribution_factory("univariate-unif")(2.0, 6.0)
        self.assertEqual(unif.pdf(1.0), 0)          # outside the support
        self.assertEqual(unif.pdf(7.0), 0)
        self.assertAlmostEqual(unif.pdf(4.0), 0.25)
        self.assertEqual(unif.cdf(1.0), 0)
        self.assertEqual(unif.cdf(2.0), 0)
        self.assertAlmostEqual(unif.cdf(3.0), 0.25)
        self.assertEqual(unif.cdf(6.0), 1)
        self.assertEqual(unif.cdf(9.0), 1)
        for q in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(unif.cdf(unif.cdf_inverse(q)), q)

    def test_uniform_fit_spans_the_data(self):
        data = [3.0, -1.0, 2.5, 7.25]
        unif = distribution_factory("univariate-unif").fit(data)
        self.assertEqual((unif.a, unif.b), (min(data), max(data)))
        self.assertEqual([p.value for p in unif.parameters],
                         [min(data), max(data)])

    def test_uniform_generates_X_in_the_support(self):
        unif = distribution_factory("univariate-unif")(2.0, 6.0)
        unif.seed_reset(13)
        draws = unif.generates_X(50)
        self.assertEqual(len(draws), 50)
        self.assertTrue(all(2.0 <= d <= 6.0 for d in draws))

    def test_normal_inverse(self):
        unorm = distribution_factory("univariate-normal")(mean=3.0, var=4.0)
        self.assertAlmostEqual(unorm.cdf_inverse(0.5), 3.0, places=4)
        self.assertLess(unorm.cdf_inverse(0.25), unorm.cdf_inverse(0.75))

    def test_normal_fit_recovers_the_moments(self):
        data = list(np.random.RandomState(3).normal(5.0, 2.0, size=500))
        norm = distribution_factory("univariate-normal").fit(data)
        self.assertAlmostEqual(norm.mean, float(np.mean(data)))
        self.assertAlmostEqual(norm.var, float(np.var(data)))

    def test_normal_matches_the_closed_form(self):
        norm = distribution_factory("univariate-normal")(var=4.0, mean=3.0)
        self.assertAlmostEqual(norm.cdf(3.0), 0.5)
        self.assertAlmostEqual(norm.pdf(3.0), 1.0/math.sqrt(2*math.pi*4.0))
        self.assertAlmostEqual(norm.pdf(1.0), norm.pdf(5.0))   # symmetry
        # the mass within one standard deviation of the mean
        self.assertAlmostEqual(norm.cdf(5.0) - norm.cdf(1.0), 0.6826894921,
                               places=6)
        self.assertAlmostEqual(norm.cdf(norm.cdf_inverse(0.975)), 0.975)

    def test_normal_generates_X(self):
        norm = distribution_factory("univariate-normal")(var=1.0, mean=0.0)
        norm.seed_reset(42)
        draws = norm.generates_X(1000)
        self.assertEqual(len(draws), 1000)
        self.assertLess(abs(float(np.mean(draws))), 0.25)

    def test_student_fit_matches_the_data_moments(self):
        # the fit promises the distribution's mean and variance are the data's;
        # the constructor derives scipy's scale from (var, df), so the reported
        # variance is var -- not var*df/(df-2), which is what passing sqrt(var)
        # as the scale would have produced
        data = list(np.random.RandomState(4).normal(0.0, 2.0, size=500))
        var = float(np.var(data))
        st = distribution_factory("univariate-student").fit(data)
        self.assertAlmostEqual(st.distribution.mean(), float(np.mean(data)))
        self.assertAlmostEqual(st.distribution.var(), var)

    def test_student_fit_sets_df_from_kurtosis(self):
        # df is method of moments on the excess kurtosis: a t with df > 4 has
        # excess kurtosis 6/(df-4), so df = 4 + 6/excess_kurtosis. Data drawn
        # from a t with df=5 (excess kurtosis 6) should recover a df near 5.
        data = list(scipy.stats.t(df=5).rvs(size=3000, random_state=7))
        ek = float(scipy.stats.kurtosis(data, fisher=True))
        self.assertGreater(ek, 0.0)                     # heavier than normal
        st = distribution_factory("univariate-student").fit(data)
        self.assertAlmostEqual(st.df, 4.0 + 6.0/ek)     # the rule, exactly
        self.assertGreater(st.df, 4.0)
        self.assertLess(st.df, 10.0)                    # sane recovery of df=5
        self.assertAlmostEqual(st.distribution.var(), float(np.var(data)))

    def test_student_fit_falls_back_to_normal_for_light_tails(self):
        # data with tails no heavier than the normal (here uniform, whose
        # excess kurtosis is negative) has no finite-variance t that matches
        # it, so df falls back to the large "effectively normal" value
        data = list(np.random.RandomState(5).uniform(0.0, 1.0, size=1000))
        self.assertLess(float(scipy.stats.kurtosis(data, fisher=True)), 0.0)
        st = distribution_factory("univariate-student").fit(data)
        self.assertEqual(st.df, statdist_distributions.
                         UnivariateStudentDistribution._FIT_DF_MAX)
        self.assertAlmostEqual(st.distribution.var(), float(np.var(data)))

    def test_student_variance_is_honored_for_any_df(self):
        for df in (2.5, 4.0, 30.0):
            st = distribution_factory("univariate-student")(
                df=df, mean=-2.0, var=9.0)
            self.assertAlmostEqual(st.distribution.var(), 9.0, msg=f"df={df}")
            self.assertAlmostEqual(st.distribution.mean(), -2.0)

    def test_student_needs_a_df_that_has_a_variance(self):
        for df in (1, 2.0):
            with self.assertRaises(ValueError):
                distribution_factory("univariate-student")(
                    df=df, mean=0.0, var=1.0)

    def test_student_fit_accepts_low_variance_data(self):
        # because the scale is derived from (var, df), a variance of one or
        # less is no longer special: the old 2v/(v-1) rule could not fit it,
        # but the kurtosis rule can, and the fitted moments still match
        data = list(np.random.RandomState(5).normal(0.0, 0.1, size=200))
        var = float(np.var(data))
        self.assertLessEqual(var, 1.0)
        st = distribution_factory("univariate-student").fit(data)
        self.assertGreater(st.df, 2.0)
        self.assertAlmostEqual(st.distribution.mean(), float(np.mean(data)))
        self.assertAlmostEqual(st.distribution.var(), var)

    def test_student_is_symmetric_with_heavier_tails(self):
        st = distribution_factory("univariate-student")(df=3.0, mean=1.0, var=4.0)
        self.assertAlmostEqual(st.cdf(1.0), 0.5)
        self.assertAlmostEqual(st.pdf(0.0), st.pdf(2.0))       # symmetry
        self.assertAlmostEqual(st.cdf(st.cdf_inverse(0.9)), 0.9)
        # same mean and variance as a normal, but with the fatter tails
        norm = distribution_factory("univariate-normal")(var=4.0, mean=1.0)
        self.assertAlmostEqual(st.distribution.var(), norm.var)
        self.assertLess(st.cdf_inverse(0.01), norm.cdf_inverse(0.01))
        self.assertGreater(st.cdf_inverse(0.99), norm.cdf_inverse(0.99))

    def test_student_generates_X(self):
        st = distribution_factory("univariate-student")(df=5.0, mean=0.0, var=1.0)
        st.seed_reset(7)
        self.assertEqual(len(st.generates_X(500)), 500)

    def test_kernel_fit_inverse(self):
        # the kernel-density fit backs Smoothed_boot_kernel and Smoothed_bagging
        data = list(np.random.RandomState(0).normal(0, 1, size=200))
        kde = distribution_factory("univariate-kernel").fit(data)
        lo = kde.cdf_inverse(0.25)
        hi = kde.cdf_inverse(0.75)
        self.assertTrue(math.isfinite(lo) and math.isfinite(hi))
        self.assertLess(lo, hi)

    def test_kernel_pdf_returns_a_python_float(self):
        # gaussian_kde.evaluate answers with an array, but the base-class cdf
        # hands the density to scipy.integrate.quad, which wants a scalar
        data = list(np.random.RandomState(6).normal(0, 1, size=100))
        kde = distribution_factory("univariate-kernel").fit(data)
        self.assertIsInstance(kde.pdf(0.0), float)

    def test_kernel_honors_bw_method(self):
        data = list(np.random.RandomState(7).normal(0, 1, size=100))
        cls = distribution_factory("univariate-kernel")
        default = cls.fit(data)
        wide = cls.fit(data, bw_method=2.0)
        self.assertAlmostEqual(wide.kernel.factor, 2.0)
        self.assertNotAlmostEqual(default.kernel.factor, wide.kernel.factor)

    def test_kernel_pads_the_domain(self):
        data = [1.0, 2.0, 3.0, 4.0]
        kde = distribution_factory("univariate-kernel")(data, dom_std=2)
        sd = float(np.std(data))
        self.assertAlmostEqual(kde.alpha, 1.0 - 2*sd)
        self.assertAlmostEqual(kde.beta, 4.0 + 2*sd)

    def test_kernel_cdf_is_a_distribution(self):
        # the kernel class leans on the base-class cdf, which integrates the
        # density numerically between alpha and beta
        data = list(np.random.RandomState(8).normal(0, 1, size=60))
        kde = distribution_factory("univariate-kernel").fit(data)
        self.assertEqual(kde.cdf(kde.alpha), 0)
        self.assertEqual(kde.cdf(kde.beta), 1)
        grid = [float(x) for x in np.linspace(kde.alpha, kde.beta, 12)]
        values = [kde.cdf(x) for x in grid]
        for lo, hi in zip(values, values[1:]):
            self.assertLessEqual(lo, hi + 1e-6)
        self.assertTrue(all(kde.pdf(x) >= 0 for x in grid))
        # the padded domain holds essentially all of the mass
        self.assertGreater(kde.cdf(grid[-2]), 0.9)

    def test_kernel_generates_X_is_seeded(self):
        data = list(np.random.RandomState(9).normal(0, 1, size=40))
        kde = distribution_factory("univariate-kernel").fit(data)
        drawn = kde.generates_X(5, seed=11)
        self.assertEqual(np.shape(drawn), (1, 5))
        np.testing.assert_allclose(drawn, kde.generates_X(5, seed=11))

    def test_empirical_fit_inverse(self):
        data = list(np.random.RandomState(1).normal(0, 1, size=200))
        emp = distribution_factory("univariate-empirical").fit(data)
        self.assertLessEqual(emp.cdf_inverse(0.25), emp.cdf_inverse(0.75))

    def test_empirical_rejects_empty_data(self):
        with self.assertRaises(ValueError):
            distribution_factory("univariate-empirical").fit([])

    def test_empirical_uses_plotting_positions(self):
        # the contract of the interpolated empirical cdf: the i-th smallest of
        # n records sits at quantile (i+1)/(n+1), and cdf_inverse inverts that
        data = [4.0, 1.0, 3.0, 2.0, 5.0]
        emp = distribution_factory("univariate-empirical").fit(data)
        n = len(data)
        for i, value in enumerate(sorted(data)):
            self.assertAlmostEqual(emp.cdf(value), (i + 1)/(n + 1))
            self.assertAlmostEqual(emp.cdf_inverse((i + 1)/(n + 1)), value)
        # and it interpolates linearly between two records
        self.assertAlmostEqual(emp.cdf(2.5), 2.5/(n + 1))

    def test_empirical_cdf_is_monotone_and_bounded(self):
        data = list(np.random.RandomState(10).normal(0, 1, size=30))
        emp = distribution_factory("univariate-empirical").fit(data)
        values = [emp.cdf(float(x))
                  for x in np.linspace(min(data) - 1, max(data) + 1, 40)]
        self.assertTrue(all(0 <= v <= 1 for v in values))
        for lo, hi in zip(values, values[1:]):
            self.assertLessEqual(lo, hi)

    def test_empirical_pdf_is_the_relative_frequency(self):
        emp = distribution_factory("univariate-empirical").fit(
            [1.0, 2.0, 2.0, 3.0])
        self.assertAlmostEqual(emp.pdf(2.0), 0.5)
        self.assertAlmostEqual(emp.pdf(1.0), 0.25)
        self.assertEqual(emp.pdf(9.0), 0)

    def test_empirical_respects_explicit_bounds(self):
        emp = distribution_factory("univariate-empirical").fit([1.0, 2.0, 3.0])
        self.assertEqual(emp.cdf(-1.0, lower_bound=0.0), 0)  # past the bound
        self.assertEqual(emp.cdf(9.0, upper_bound=4.0), 1)
        # inside the bound the cdf interpolates towards it
        self.assertGreater(emp.cdf(0.5, lower_bound=0.0), 0)
        self.assertLess(emp.cdf(3.5, upper_bound=4.0), 1)
        self.assertGreaterEqual(emp.cdf_inverse(0.01, lower_bound=0.0), 0.0)
        self.assertLessEqual(emp.cdf_inverse(0.99, upper_bound=4.0), 4.0)

    def test_empirical_extrapolates_below_the_smallest_record(self):
        # with no lower bound given, a quantile under 1/(n+1) follows the line
        # through the first two plotting positions, (0.25, 1.0) and (0.5, 2.0)
        emp = distribution_factory("univariate-empirical").fit([1.0, 2.0, 3.0])
        self.assertAlmostEqual(emp.cdf_inverse(0.1), 0.4)
        self.assertLess(emp.cdf_inverse(0.2), 1.0)   # below the smallest record

    def test_empirical_cdf_inverse_rejects_bad_quantiles(self):
        emp = distribution_factory("univariate-empirical").fit([1.0, 2.0, 3.0])
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                emp.cdf_inverse(bad)

    def test_empirical_handles_a_degenerate_sample(self):
        # a resample can easily come out constant (or hold a single record):
        # every quantile is then that value, and neither tail has a second
        # point to take a slope from
        for data in ([5.0], [5.0, 5.0, 5.0]):
            emp = distribution_factory("univariate-empirical").fit(data)
            for q in (0.0, 0.1, 0.5, 0.9, 1.0):
                self.assertEqual(emp.cdf_inverse(q), 5.0, msg=f"{data}: {q}")
            self.assertEqual(emp.cdf(4.0), 0)
            self.assertEqual(emp.cdf(6.0), 1)

    def test_empirical_extrapolates_past_a_repeated_extreme(self):
        # the largest value is repeated, so the upper extrapolation has to walk
        # down to the bottom of that run to find a slope
        emp = distribution_factory("univariate-empirical").fit(
            [1.0, 2.0, 3.0, 3.0])
        self.assertGreater(emp.cdf_inverse(0.99), 3.0)
        self.assertEqual(emp.cdf(9.0), 1)    # the extrapolated line is clamped
        self.assertEqual(emp.cdf(-9.0), 0)

    def test_interpolate_line(self):
        line = statdist_distributions.interpolate_line(0.0, 1.0, 2.0, 5.0)
        self.assertAlmostEqual(line(1.0), 3.0)
        with self.assertRaises(ValueError):
            statdist_distributions.interpolate_line(1.0, 0.0, 1.0, 5.0)

    def _discrete(self, pairs):
        return distribution_factory("univariate-discrete")(OrderedDict(pairs))

    def test_discrete_moments(self):
        # a fair two-point distribution on {0, 2}: mean 1, variance 1
        fair = self._discrete([(0.0, 0.5), (2.0, 0.5)])
        self.assertAlmostEqual(fair.mean, 1.0)
        self.assertAlmostEqual(fair.var, 1.0)
        # and a three-point one, against E[X^2] - E[X]^2
        d = self._discrete([(1.0, 0.2), (2.0, 0.3), (5.0, 0.5)])
        mean = 0.2*1 + 0.3*2 + 0.5*5
        self.assertAlmostEqual(d.mean, mean)
        self.assertAlmostEqual(d.var, 0.2*1 + 0.3*4 + 0.5*25 - mean**2)
        self.assertGreater(d.var, 0.0)

    def test_discrete_validates_its_breakpoints(self):
        with self.assertRaises(RuntimeError):        # not a dict at all
            distribution_factory("univariate-discrete")([(0.0, 1.0)])
        with self.assertRaises(RuntimeError):        # values out of order
            self._discrete([(2.0, 0.5), (1.0, 0.5)])
        for bad in ([(0.0, 0.25), (1.0, 0.25)], [(0.0, 0.9), (1.0, 0.9)]):
            with self.assertRaises(ValueError):      # probabilities not one
                self._discrete(bad)

    def test_discrete_cdf_is_a_step_function(self):
        d = self._discrete([(1.0, 0.2), (2.0, 0.3), (5.0, 0.5)])
        self.assertEqual(d.cdf(0.0), 0)
        self.assertAlmostEqual(d.cdf(1.0), 0.2)
        self.assertAlmostEqual(d.cdf(1.5), 0.2)     # flat between breakpoints
        self.assertAlmostEqual(d.cdf(2.0), 0.5)
        self.assertAlmostEqual(d.cdf(4.9), 0.5)
        self.assertAlmostEqual(d.cdf(5.0), 1.0)
        self.assertAlmostEqual(d.cdf(6.0), 1.0)
        self.assertAlmostEqual(d.rect_prob(1.0, 5.0), 0.8)

    def test_discrete_has_no_density_or_inverse(self):
        d = self._discrete([(1.0, 0.5), (2.0, 0.5)])
        with self.assertRaises(RuntimeError):
            d.pdf(1.0)
        with self.assertRaises(RuntimeError):
            d.cdf_inverse(0.5)

    def test_discrete_sample_one_draws_from_the_breakpoints(self):
        d = self._discrete([(1.0, 0.25), (2.0, 0.75)])
        d.seed_reset(4)
        draws = [d.sample_one() for _ in range(400)]
        self.assertEqual(set(draws), {1.0, 2.0})
        self.assertAlmostEqual(draws.count(2.0)/len(draws), 0.75, places=1)

    @unittest.skipIf(not ipopt_available, "ipopt (nonlinear solver) not available")
    def test_epispline_fit_inverse(self):
        data = list(np.random.RandomState(2).normal(0, 1, size=100))
        epi = distribution_factory("univariate-epispline").fit(data)
        self.assertLessEqual(epi.cdf_inverse(0.25), epi.cdf_inverse(0.75))


#*****************************************************************************
class _RampDistribution(UnivariateDistribution):
    """ A closed-form distribution for testing the base-class machinery.

    The density is 2x on [0, 1], so cdf(x) = x**2, cdf_inverse(q) = sqrt(q),
    and the mean is 2/3.
    """

    def __init__(self, declare_support=True):
        self.alpha = 0.0
        self.beta = 1.0
        params = [Parameter("slope", 2.0)]
        if declare_support:
            UnivariateDistribution.__init__(self, params, self.alpha, self.beta)
        else:
            UnivariateDistribution.__init__(self, params)

    @classmethod
    def fit(cls, data):
        return cls()

    def pdf(self, x):
        if x < self.alpha or x > self.beta:
            return 0.0
        return 2.0 * x


class _Interval:
    """ The minimal interval protocol conditional_expectation expects. """

    def __init__(self, a, b, cutouts=None):
        self.a = a
        self.b = b
        if cutouts is not None:
            self.cutouts = cutouts


class Test_statdist_base(unittest.TestCase):
    """ The generic univariate machinery in base_distribution.py: the numeric
    cdf and its inversion, expectations, sampling and parameter bookkeeping. """

    def setUp(self):
        self.d = _RampDistribution()

    def test_support_defaults_to_unbounded(self):
        self.assertEqual((self.d.lower, self.d.upper), (0.0, 1.0))
        undeclared = _RampDistribution(declare_support=False)
        self.assertEqual((undeclared.lower, undeclared.upper),
                         (-np.inf, np.inf))
        self.assertEqual(undeclared.dimension, 1)

    def test_cdf_integrates_the_density(self):
        for x in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(self.d.cdf(x), x**2, places=5)
        self.assertEqual(self.d.cdf(self.d.alpha), 0)
        self.assertEqual(self.d.cdf(-1.0), 0)
        self.assertEqual(self.d.cdf(self.d.beta), 1)
        self.assertEqual(self.d.cdf(2.0), 1)

    def test_cdf_inverse_inverts_the_cdf(self):
        for q in (0.1, 0.25, 0.5, 0.81):
            self.assertAlmostEqual(self.d.cdf_inverse(q), math.sqrt(q),
                                   places=3)
        # the ends of the support, and quantiles that are not quantiles
        self.assertEqual(self.d.cdf_inverse(0.0), self.d.alpha)
        self.assertEqual(self.d.cdf_inverse(1.0), self.d.beta)
        self.assertIsNone(self.d.cdf_inverse(-0.1))
        self.assertIsNone(self.d.cdf_inverse(1.1))

    def test_cdf_is_cached_per_tolerance(self):
        # the cdf is memoized, and a different accuracy is a different question
        self.assertAlmostEqual(self.d.cdf(0.5, epsabs=1e-2), 0.25, places=2)
        self.assertAlmostEqual(self.d.cdf(0.5, epsabs=1e-12), 0.25, places=9)

    def test_mean_and_region_expectation(self):
        self.assertAlmostEqual(self.d.mean(), 2/3, places=5)
        self.assertAlmostEqual(self.d.region_expectation((0.0, 1.0)), 2/3,
                               places=5)
        self.assertAlmostEqual(self.d.region_expectation((0.0, 0.5)), 1/12,
                               places=5)
        self.assertAlmostEqual(self.d.region_probability((0.0, 0.5)), 0.25,
                               places=5)
        self.assertAlmostEqual(self.d.region_probability((0.0, 1.0)), 1.0,
                               places=5)

    def test_region_arguments_are_validated(self):
        with self.assertRaises(ValueError):
            self.d.region_expectation((0.75, 0.25))    # upper below lower
        # a region has to be a tuple, and the complaint about that has to
        # survive the memoization wrapper (a list is not hashable)
        for not_a_region in ([0.0, 1.0], "region"):
            with self.assertRaises(TypeError):
                self.d.region_expectation(not_a_region)
            with self.assertRaises(ValueError):
                self.d.region_probability(not_a_region)

    def test_conditional_expectation(self):
        # conditioning on the whole support is just the mean
        self.assertAlmostEqual(
            self.d.conditional_expectation(_Interval(0.0, 1.0)), 2/3, places=3)
        # cutting the lower half out conditions on the upper half, which pulls
        # the expectation up; E[X | X > median] = (2/3)(1 - 0.5**1.5)/0.5
        upper_half = self.d.conditional_expectation(
            _Interval(0.0, 1.0, cutouts=[_Interval(0.0, 0.5)]))
        self.assertAlmostEqual(upper_half, (2/3)*(1 - 0.5**1.5)/0.5, places=3)
        self.assertGreater(upper_half, 2/3)

    def test_log_likelihood(self):
        data = [0.25, 0.5, 0.75]
        self.assertAlmostEqual(self.d.log_likelihood(data),
                               sum(math.log(2*x) for x in data))

    def test_sampling_stays_in_the_support(self):
        # the inversion is numeric, so allow it a little slack at the ends
        slack = 1e-3
        self.d.seed_reset(12)
        for _ in range(20):
            drawn = self.d.sample_one()
            self.assertGreaterEqual(drawn, self.d.alpha - slack)
            self.assertLessEqual(drawn, self.d.beta + slack)
        for _ in range(10):
            drawn = self.d.sample_on_interval(0.25, 0.75)
            self.assertGreaterEqual(drawn, 0.25 - slack)
            self.assertLessEqual(drawn, 0.75 + slack)
            # a quantile range maps to the matching range of values
            between = self.d.sample_between_quantiles(0.1, 0.2)
            self.assertGreaterEqual(between, math.sqrt(0.1) - slack)
            self.assertLessEqual(between, math.sqrt(0.2) + slack)

    def test_str_and_repr_name_the_parameters(self):
        self.assertIn("slope", str(self.d))
        self.assertIn("2.0", str(self.d))
        self.assertEqual(repr(self.d), "Distribution(_RampDistribution)")

    def test_parameter_bookkeeping(self):
        p = Parameter("mean", 3.0, bounds=(0, None))
        self.assertTrue(p.instantiated)          # it has a value
        self.assertEqual(p.bounds, (0, None))
        self.assertIs(p.kind, float)
        self.assertEqual(repr(p), "Parameter(mean,3.0)")
        self.assertEqual(str(p), repr(p))
        unset = Parameter("variance")
        self.assertFalse(unset.instantiated)      # and this one does not
        unset.set_value(2.5)
        self.assertEqual(unset.value, 2.5)
        self.assertTrue(unset.instantiated)

    @unittest.skipIf(not matplotlib_available, "matplotlib is not installed")
    def test_plot_writes_a_file(self):
        import matplotlib
        matplotlib.use("Agg")   # no display in a test run
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_dir = os.path.join(tmpdir, "plots")
            self.d.plot(output_file="ramp.png", title="ramp", xlabel="x",
                        ylabel="density", output_directory=plot_dir)
            self.assertTrue(os.path.exists(os.path.join(plot_dir, "ramp.png")))
            # an unbounded support falls back to a [-5, 5] window, and the
            # directory this time already exists
            _RampDistribution(declare_support=False).plot(
                plot_cdf=False, output_file="unbounded.png",
                output_directory=plot_dir)
            self.assertTrue(
                os.path.exists(os.path.join(plot_dir, "unbounded.png")))


#*****************************************************************************
class Test_statdist_utilities(unittest.TestCase):
    """ The memoization helpers and the argv context manager in
    statdist/utilities.py. """

    def test_memoize_caches_by_value(self):
        calls = []

        @statdist_utilities.memoize
        def total(xs, offset=0):
            calls.append(1)
            return sum(xs) + offset

        self.assertEqual(total([1, 2, 3]), 6)
        self.assertEqual(total([1, 2, 3]), 6)
        self.assertEqual(len(calls), 1)          # the second call was cached
        # an unhashable list argument normalizes to the tuple's key
        self.assertEqual(total((1, 2, 3)), 6)
        self.assertEqual(len(calls), 1)
        self.assertEqual(total([1, 2, 3], offset=10), 16)
        self.assertEqual(len(calls), 2)

    def test_memoize_normalizes_dictionary_arguments(self):
        calls = []

        @statdist_utilities.memoize
        def size(mapping):
            calls.append(1)
            return len(mapping)

        self.assertEqual(size({"a": 1, "b": 2}), 2)
        self.assertEqual(size({"b": 2, "a": 1}), 2)   # equal dict, new object
        self.assertEqual(len(calls), 1)

    def test_normalize_args_maps_positionals_to_names(self):
        def f(a, b, c=0):
            return a

        args = statdist_utilities.normalize_args(f, (1, [2, 3]), {"c": {"k": 4}})
        self.assertEqual(args["a"], 1)
        self.assertEqual(args["b"], (2, 3))           # list -> tuple
        self.assertEqual(args["c"], (("k", 4),))      # dict -> sorted pairs

    def test_memoize_method_caches_per_instance(self):
        class Counter:
            def __init__(self):
                self.calls = 0

            @statdist_utilities.memoize_method
            def squared(self, x):
                self.calls += 1
                return x * x

        one, two = Counter(), Counter()
        self.assertEqual(one.squared(3), 9)
        self.assertEqual(one.squared(3), 9)
        self.assertEqual(one.calls, 1)
        self.assertEqual(two.squared(3), 9)      # a cache of its own
        self.assertEqual(two.calls, 1)
        # reached through the class the method is the undecorated one
        self.assertEqual(Counter.squared(one, 4), 16)
        self.assertEqual(one.calls, 2)

    def test_memoize_method_keys_on_keyword_values(self):
        class Rounder:
            def __init__(self):
                self.calls = 0

            @statdist_utilities.memoize_method
            def value(self, x, places=2):
                self.calls += 1
                return round(x, places)

        r = Rounder()
        self.assertEqual(r.value(1.23456, places=2), 1.23)
        self.assertEqual(r.value(1.23456, places=4), 1.2346)
        self.assertEqual(r.calls, 2)             # not one answer for both
        self.assertEqual(r.value(1.23456, places=4), 1.2346)
        self.assertEqual(r.calls, 2)             # and now it is cached

    def test_memoize_method_passes_unhashable_arguments_through(self):
        class Sizer:
            def __init__(self):
                self.calls = 0

            @statdist_utilities.memoize_method
            def size(self, thing):
                self.calls += 1
                if not isinstance(thing, tuple):
                    raise TypeError("tuples only")
                return len(thing)

        s = Sizer()
        self.assertEqual(s.size((1, 2, 3)), 3)
        # an unhashable argument cannot be cached, but the method still runs
        # and its own error is what comes back
        with self.assertRaises(TypeError):
            s.size([1, 2, 3])
        self.assertEqual(s.calls, 2)

    def test_set_arguments_restores_argv(self):
        saved = list(sys.argv)
        with statdist_utilities.set_arguments(["prog", "--flag"]):
            self.assertEqual(sys.argv, ["prog", "--flag"])
        self.assertEqual(sys.argv, saved)


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
            # every rank participates in the collective inside compute_ci
            res = boot_sp.compute_ci(_make_farmer_cfg(method), module, xhat)
            self.assertEqual(len(res), 6)
            if my_rank == 0:
                for ci in res[:3]:
                    self.assertLessEqual(ci[0], ci[1], msg=f"{method}: {ci}")
            else:
                self.assertEqual(res, (None, None, None, None, None, None))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_empirical_wellformed(self):
        module = boot_utils.module_name_to_module("cvar")
        cfg = _make_cvar_cfg("Classical_quantile")
        xhat = boot_utils.compute_xhat(cfg, module)
        self.assertIn("ROOT", xhat)
        res = boot_sp.compute_ci(_make_cvar_cfg("Classical_quantile"), module, xhat)
        self.assertEqual(len(res), 6)
        if my_rank == 0:
            for ci in res[:3]:
                self.assertLessEqual(ci[0], ci[1])
        else:
            self.assertEqual(res, (None, None, None, None, None, None))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_cvar_empirical_deterministic(self):
        # same cfg twice must give the same interval (seeded streams)
        module = boot_utils.module_name_to_module("cvar")
        xhat = boot_utils.compute_xhat(_make_cvar_cfg("Classical_gaussian"), module)
        # both runs are collectives on every rank; only rank 0 gets real values
        r1 = boot_sp.compute_ci(_make_cvar_cfg("Classical_gaussian"), module, xhat)
        r2 = boot_sp.compute_ci(_make_cvar_cfg("Classical_gaussian"), module, xhat)
        if my_rank == 0:
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

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_smoothed_bootstrap_draws_are_disjoint_and_fitted(self):
        # Two properties of the smoothed bootstrap that are easy to lose:
        #  (1) every batch is an independent set of draws from the fitted
        #      distribution, so the per-batch record blocks are pairwise
        #      disjoint and disjoint from the center's block. Overlapping
        #      blocks reuse draws and collapse the estimated spread.
        #  (2) the center is drawn from the *fitted* distribution, not from
        #      the raw sample; drawing it raw makes it the purely empirical
        #      point estimate.
        module = boot_utils.module_name_to_module("cvar")
        cfg = _make_cvar_cfg("Smoothed_boot_kernel")
        xhat = boot_utils.compute_xhat(cfg, module)

        pools = []
        fitted_at_center = []
        real_eval = boot_sp.evaluate_scenarios
        real_center = smoothed_boot_sp.center_smoothed

        # the smoothed callers never pass a communicator, so the spy does not
        # need one either -- and not taking one keeps this working whether or
        # not evaluate_scenarios has grown an mpicomm argument
        def spy_eval(cfg_, module_, scenarios, xhat_, duplication=True):
            pools.append(list(scenarios))
            return real_eval(cfg_, module_, scenarios, xhat_,
                             duplication=duplication)

        def spy_center(cfg_, module_, xhat_):
            fitted_at_center.append(cfg_.use_fitted)
            return real_center(cfg_, module_, xhat_)

        boot_sp.evaluate_scenarios = spy_eval
        smoothed_boot_sp.center_smoothed = spy_center
        try:
            smoothed_boot_sp.compute_smoothed_ci(cfg, module, xhat)
        finally:
            boot_sp.evaluate_scenarios = real_eval
            smoothed_boot_sp.center_smoothed = real_center

        self.assertEqual(fitted_at_center, [True])  # (2)

        # pools[0] is the center; the rest are this rank's batches
        center_pool, batch_pools = set(pools[0]), [set(p) for p in pools[1:]]
        self.assertEqual(len(center_pool), cfg.smoothed_center_sample_size)
        for i, bp in enumerate(batch_pools):                                # (1)
            self.assertEqual(len(bp), cfg.sample_size)
            self.assertEqual(bp & center_pool, set(),
                             msg=f"batch {i} reuses the center's draws")
            for j, other in enumerate(batch_pools[i + 1:], start=i + 1):
                self.assertEqual(bp & other, set(),
                                 msg=f"batches {i} and {j} share draws")

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
