###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Smoothed bootstrap/bagging for data-based, two-stage stochastic programs.
# These methods fit a (univariate) distribution to the sampled data using the
# statdist library and then resample from the fitted distribution. They are the
# counterpart to the empirical methods in boot_sp.py.

from contextlib import contextmanager

import numpy as np
from statistics import NormalDist
import pyomo.environ as pyo

from mpisppy import global_toc
import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp
import mpisppy.confidence_intervals.bootsp.statdist as statdist

# The communicators live in boot_utils so there is a single source of truth.
comm = boot_utils.comm
n_proc = boot_utils.n_proc
my_rank = boot_utils.my_rank
rankcomm = boot_utils.rankcomm


def fit_options(cfg, distr_type):
    """ The fit options a distribution type takes from cfg.

    Args:
        cfg (Config): parameters
        distr_type (str): a statdist univariate distribution token
    Returns:
        dict of keyword arguments for the distribution's fit()

    Only the epi-spline fit solves anything, so it is the only one with a
    solver to choose; the rest take no options from cfg and would reject them.
    """
    if distr_type == "univariate-epispline":
        return {"nonlinear_solver": cfg.smoothed_nonlinear_solver}
    return {}


def fit_distribution(sample_data, distr_type='univariate-epispline',
                     **distr_options):
    """ Fit a (univariate) distribution to sample data.

    Args:
        sample_data (list or list of dict): a list of scalars (one variable) or
            a list of dicts (multivariate, keyed by variable name)
        distr_type (str): a statdist univariate distribution token
        distr_options: keyword arguments for the distribution's fit()
            (see fit_options)
    Returns:
        the fitted distribution (or a dict of them, keyed as the input dicts)
    """
    distr_func = statdist.distribution_factory(distr_type)
    # Test for the dict, not for float/int: numpy scalars are the ordinary
    # return type of a data_sampler and only some of them subclass the python
    # builtins (np.float64 does, np.int64 and np.float32 do not), so an
    # integer-valued sampler would fall into the multivariate branch and fail
    # inside the library with no hint that its return type was the cause.
    if not isinstance(sample_data[0], dict):  # 1-dim
        fitted_distr = distr_func.fit(sample_data, **distr_options)
    else:
        fitted_distr = {}
        for key in sample_data[0]:
            data = [data_dict[key] for data_dict in sample_data]
            fitted_distr[key] = distr_func.fit(data, **distr_options)
    return fitted_distr


def _batch_barrier(serial):
    """ Synchronize the ranks between batch phases, unless this run is serial.

    Args:
        serial (bool): one rank is doing every batch

    In serial mode the other ranks are not in this call at all, so a collective
    on COMM_WORLD blocks forever waiting for them -- the same reason the gather
    that follows is guarded.
    """
    if not serial:
        comm.Barrier()


def draw_space_origin(cfg):
    """ The first record number the fitted-distribution draws may use.

    Args:
        cfg (Config): parameters
    Returns:
        int

    The draws must not reuse the record numbers the data occupies. A record
    number is the seed the model gives a draw, and the fitted distribution is
    close to the one the data came from, so a draw at a data record number is
    the same uniform variate pushed through two nearly identical inverse CDFs:
    it reproduces that data point instead of being independent of it. Batches
    built from such draws are correlated with the sample the interval is
    measuring spread around, which understates the width.

    Starting the draw space past max_count keeps the two spaces disjoint for
    any seed_offset. Do not fold this back to seed_offset alone.
    """
    return cfg.max_count + cfg.seed_offset


def center_smoothed(cfg, module, xhat):
    """ Estimate the CI center (the optimality gap) from the fitted distribution.

    The smoothed methods are single-rank-per-solve, so the solves here go to the
    module globals' view; there is no communicator to thread through (an earlier
    signature took one and ignored it).
    """
    if cfg.smoothed_center_sample_size is None:
        # an assert here would vanish under python -O, and the next line would
        # then add None to an int somewhere inside the library
        raise ValueError(
            "smoothed_center_sample_size is required for the smoothed methods; "
            "it is the number of draws the gap center is estimated from.")
    origin = draw_space_origin(cfg)
    scenario_pool = list(range(origin,
                               origin + cfg.smoothed_center_sample_size))

    center_upper = boot_sp.evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=False)
    center_ef = boot_sp.solve_routine(cfg, module, scenario_pool, num_threads=2, duplication=False)
    center_optimal = pyo.value(center_ef.EF_Obj)
    center_gap = center_upper - center_optimal

    if my_rank == 0:
        return center_gap
    else:
        return None


def smoothed_resample_helper(cfg, module, xhat, serial=False):
    """ Get local gaps for the smoothed bootstrap (the fitted-distribution
        analog of boot_sp._bootstrap_resample).

    Every batch is an *independent* set of cfg.subsample_size draws from the
    fitted distribution, so the batches take disjoint blocks of the draw index
    space: batch b covers [start + b*m, start + (b+1)*m). A record number is
    the draw's seed, so striding by anything less than m (the batch size) would
    hand consecutive batches most of the same draws and collapse the spread the
    interval is built from. The block the center estimate draws from,
    [seed_offset, seed_offset + smoothed_center_sample_size) (see
    center_smoothed), is reserved ahead of the batches because it samples the
    same fitted distribution and must not reuse their draws.
    """
    if serial:
        local_nB = cfg.nB
    else:
        local_nB = boot_sp.slice_lens(cfg.nB)[my_rank]

    local_boot_gaps = np.empty(local_nB, dtype=np.float64)

    m = cfg.subsample_size
    start = draw_space_origin(cfg) + (cfg.smoothed_center_sample_size or 0)
    # this rank's first batch in the global 0..nB-1 numbering
    first_batch = 0 if serial else sum(boot_sp.slice_lens(cfg.nB)[:my_rank])

    for iter in range(local_nB):
        b = first_batch + iter
        scenario_pool = list(range(start + b * m, start + (b + 1) * m))

        local_boot_upper = boot_sp.evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=False)
        local_boot_ef = boot_sp.solve_routine(cfg, module, scenario_pool, num_threads=2, duplication=False)
        local_boot_optimal = pyo.value(local_boot_ef.EF_Obj)
        local_boot_gaps[iter] = local_boot_upper - local_boot_optimal

    return local_boot_gaps


@contextmanager
def _fitted_on_cfg(cfg, module, scenario_pool, distr_type, serial=False):
    """ Fit a distribution to the sampled data and expose it on cfg, then put
        cfg back the way it was found.

    Args:
        cfg (Config): parameters (temporarily modified)
        module (Python module): supplies data_sampler
        scenario_pool (iterable): the records to fit to
        distr_type (str): a statdist univariate distribution token
        serial (bool): this rank is working alone, so it fits for itself and
            takes part in no collective

    The estimators flip ``use_fitted`` so the module's scenario_creator draws
    from the fitted distribution rather than the real data, and the bootstrap
    also overwrites ``subsample_size``. The cfg belongs to the caller, and every
    example guards on ``getattr(cfg, "use_fitted", False)``, so leaving either
    behind would silently make some later model build -- a second interval, a
    zhat evaluation, the caller's own scenario_creator call -- sample from a
    distribution fitted for something else.
    """
    saved = {"use_fitted": cfg.use_fitted,
             "fitted_distribution": cfg.fitted_distribution,
             "subsample_size": cfg.subsample_size}
    try:
        cfg.use_fitted = False
        # Fit once and hand the result round. Every rank used to fit the same
        # data to the same distribution -- n_proc identical ipopt NLPs on the
        # epi-spline path, for one answer -- and if a solve ever came out
        # differently on one rank, the batches gathered from them would be
        # draws from two different densities.
        def _fit():
            sample_data = [module.data_sampler(scenario, cfg)
                           for scenario in scenario_pool]
            return fit_distribution(sample_data, distr_type=distr_type,
                                    **fit_options(cfg, distr_type))

        if serial:
            # no other rank is here to broadcast to
            cfg.fitted_distribution = _fit()
        else:
            if my_rank == 0:
                try:
                    fitted, failure = _fit(), None
                except Exception as e:
                    fitted, failure = None, f"{type(e).__name__}: {e}"
            else:
                fitted, failure = None, None
            fitted, failure = comm.bcast((fitted, failure), root=0)
            if failure is not None:
                # every rank raises: rank 0 leaving alone would strand the
                # others in the next collective
                raise RuntimeError(
                    f"fitting the {distr_type} distribution failed on rank 0 "
                    f"({failure}); every rank is stopping so the run does not "
                    "hang")
            cfg.fitted_distribution = fitted
        # From here on the draws come from the fitted distribution. Estimating
        # from the raw sample instead would give up the smoothing that is the
        # whole point of these methods.
        cfg.use_fitted = True
        yield
    finally:
        for name, value in saved.items():
            setattr(cfg, name, value)


def smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-epispline', quantile=False, serial=False):
    """ fit a distribution to the sample, then draw both the center and the
        batches from it to get a smoothed point estimate and interval width
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        distr_type (str): a statdist univariate distribution token to fit
        quantile (bool): use the quantile method (else the gaussian method)
        serial (bool): indicates that only one MPI rank should be used
    Returns:
        tuple (ci_gap_two_sided, center_gap) if on MPI rank 0, else None

    """
    # the interval width comes from the spread *among* the batches, so one
    # batch leaves it undefined: np.std(one element, ddof=1) is nan, and the
    # nan then survives user_boot's max(0, .) clamp as a reported [0, nan].
    # Same reasoning as the smoothed_B_I >= 2 check in bagging.
    if cfg.nB is None or cfg.nB < 2:
        raise ValueError(
            "nB (the number of bootstrap batches) must be at least 2 for the "
            f"smoothed bootstrap; got {cfg.nB}. The interval width is the "
            "spread among the batch gaps, which a single batch does not have.")

    scenario_pool = boot_sp.draw_scenario_pool(cfg)

    with _fitted_on_cfg(cfg, module, scenario_pool, distr_type, serial):
        # the center: one replication at a large resample size
        # (smoothed_center_sample_size) drawn from the fitted distribution
        dag_gap = center_smoothed(cfg, module, xhat)
        _batch_barrier(serial)

        # each batch is a fresh set of cfg.sample_size draws from the same
        # fitted distribution, so the bootstrap batch size is the full sample
        # size (restored on the way out by _fitted_on_cfg)
        cfg.subsample_size = cfg.sample_size
        local_boot_gaps = smoothed_resample_helper(cfg, module, xhat, serial)
        _batch_barrier(serial)

    if serial:
        # one rank did every batch, so there is nothing to gather (and the
        # collective below would hang waiting for ranks that never call it)
        boot_gap = local_boot_gaps
    else:
        # do analysis only on rank 0
        if my_rank == 0:
            boot_gap = np.empty(cfg.nB, dtype=np.float64)
        else:
            boot_gap = None

        # but everyone needs to send to the gather
        lenlist = boot_sp.slice_lens(cfg.nB)
        comm.Gatherv(sendbuf=local_boot_gaps, recvbuf=(boot_gap, lenlist), root=0)

    if my_rank == 0:
        global_toc("Done smoothed bootstrap")

        if not quantile:
            s_g = np.std(boot_gap, ddof=1)
            ppf = NormalDist().inv_cdf(1 - cfg.alpha / 2)
            error = s_g * ppf
            ci_gap_two_sided = [dag_gap - error, dag_gap + error]
        else:
            alpha = cfg.alpha / 2
            eps = np.quantile(boot_gap - dag_gap, [alpha, 1 - alpha])
            ci_gap_two_sided = [dag_gap - eps[1], dag_gap - eps[0]]
        return ci_gap_two_sided, dag_gap
    else:
        # non-root ranks return a matching arity so callers can unpack safely
        return None, None


def smoothed_bagging(cfg, module, xhat, distr_type='univariate-kernel', serial=False):
    """ perform a bagging-based estimation of confidence intervals using a fitted distribution
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        distr_type (str): a statdist univariate distribution token to fit
        serial (bool): indicates that only one MPI rank should be used
    Returns:
        tuple (ci_gap_two_sided, center_gap) if on MPI rank 0, else None
    """
    scenario_pool = boot_sp.draw_scenario_pool(cfg)

    with _fitted_on_cfg(cfg, module, scenario_pool, distr_type, serial):
        return _bagging_from_fitted(cfg, module, xhat, serial)


def _bagging_from_fitted(cfg, module, xhat, serial):
    """ The bagging replications, with the fitted distribution already on cfg.

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): a candidate solution in mpi-sppy nonant format
        serial (bool): this rank does every bag, with no collectives
    Returns:
        tuple (ci_gap_two_sided, center_gap) if on MPI rank 0, else None
    """
    # serial means this rank does all nB bags itself, so it neither slices the
    # work nor takes part in the collectives below
    local_nB = cfg.nB if serial else boot_sp.slice_lens(cfg.nB)[my_rank]
    first_bag = 0 if serial else sum(boot_sp.slice_lens(cfg.nB)[:my_rank])
    local_gaps = np.empty(local_nB, dtype=np.float64)

    if my_rank == 0:
        bagging_gap = np.empty(cfg.nB, dtype=np.float64)
        all_gaps = []
        avg_gaps = []
    else:
        bagging_gap = None
        all_gaps = None
        avg_gaps = None

    # B_I is the number of initial seed points; s1 below is the variance *among*
    # their averages, so fewer than two of them leaves it undefined
    if cfg.smoothed_B_I is None or cfg.smoothed_B_I < 2:
        raise ValueError(
            "smoothed_B_I (the number of initial seed points) must be at least "
            f"2 for smoothed bagging; got {cfg.smoothed_B_I}. The variance of "
            "the per-seed-point averages is what estimates the between-point "
            "term of the interval width.")

    if cfg.subsample_size is None:
        raise ValueError(
            "subsample_size (the number of draws per bag) is required for "
            "smoothed bagging; it is unset. Only the smoothed bootstrap may "
            "leave it out, because it uses the full sample size per batch.")

    B_I = cfg.smoothed_B_I
    for i in range(B_I):
        seed_offset_base = draw_space_origin(cfg) + cfg.nB * cfg.subsample_size * i

        for j in range(local_nB):
            seed_offset = seed_offset_base + (first_bag + j) * cfg.subsample_size
            scenario_pool = list(range(seed_offset, seed_offset + cfg.subsample_size))
            scenario_pool[0] = seed_offset_base

            local_upper = boot_sp.evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=False)
            local_ef = boot_sp.solve_routine(cfg, module, scenario_pool, num_threads=2, duplication=False)
            local_optimal = pyo.value(local_ef.EF_Obj)
            local_gaps[j] = local_upper - local_optimal
        _batch_barrier(serial)
        if serial:
            bagging_gap = local_gaps
        else:
            lenlist = boot_sp.slice_lens(cfg.nB)
            comm.Gatherv(sendbuf=local_gaps, recvbuf=(bagging_gap, lenlist), root=0)

        if my_rank == 0:
            all_gaps = all_gaps + bagging_gap.tolist()
            avg_gaps.append(np.mean(bagging_gap))

    if my_rank == 0:
        global_toc("Done Smoothed Bagging")

        dag_gap = np.mean(avg_gaps)

        # sample variances (ddof=1), as the algorithm specifies and as the
        # empirical estimators already use for their Gaussian half-width;
        # ddof=0 understates s1 by (B_I-1)/B_I, which is a third at B_I=3
        s1 = np.var(avg_gaps, ddof=1)
        s2 = np.var(all_gaps, ddof=1)
        ppf = NormalDist().inv_cdf(1 - cfg.alpha / 2)
        s_g_2 = (cfg.subsample_size**2) * s1 / cfg.sample_size + s2 / (B_I * cfg.nB)
        error = np.sqrt(s_g_2) * ppf
        ci_gap_two_sided = [dag_gap - error, dag_gap + error]

        return ci_gap_two_sided, dag_gap
    else:
        # non-root ranks return a matching arity so callers can unpack safely
        return None, None


def _ensure_smoothed_cfg(cfg):
    """ Idempotently attach the run-time config entries the smoothed methods need.

    The smoothed estimators toggle ``use_fitted`` and stash a
    ``fitted_distribution`` on the cfg. This may be called repeatedly (e.g.
    once per replication in a coverage simulation), so every add is guarded.

    Reading a module's own data is deliberately not done here: a module that
    needs deterministic data knows its own option name and where the file
    lives relative to itself, which this cannot.
    """
    if "use_fitted" not in cfg:
        cfg.add_to_config(name="use_fitted",
                          description="a boolean to control use of fitted distribution",
                          domain=bool,
                          default=None,
                          argparse=False)
    cfg.use_fitted = False
    if "fitted_distribution" not in cfg:
        cfg.add_to_config(name="fitted_distribution",
                          description="a fitted distribution from sample data",
                          domain=None,
                          default=None,
                          argparse=False)


def compute_smoothed_ci(cfg, module, xhat):
    """ Dispatch to the requested smoothed bootstrap/bagging method.

    Args:
        cfg (Config): parameters (cfg.boot_method selects the method)
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): a candidate solution in mpi-sppy nonant format
    Returns:
        (ci_gap_two_sided, center_gap) on MPI rank 0, else None

    Note:
        This is the single smoothed-dispatch point shared by user_boot and
        simulate_boot (the counterpart to boot_sp.compute_ci for the empirical
        methods).
    """
    _ensure_smoothed_cfg(cfg)
    method = cfg.boot_method
    boot_utils.BootMethods.check_for_it(method)
    # every smoothed method fits a distribution to module.data_sampler output.
    # Checked here, before anything is solved: without it the run dies on a
    # bare AttributeError, and only after the candidate xhat EF has been solved.
    if not hasattr(module, "data_sampler"):
        raise RuntimeError(
            f"\nModule {cfg.module_name} must contain a function data_sampler "
            f"to use a smoothed method ({method}); it is what supplies the "
            "sample the distribution is fitted to. The empirical methods do "
            "not need it.")
    if method == "Smoothed_boot_epi":
        return smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-epispline')
    elif method == "Smoothed_boot_kernel":
        return smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-kernel')
    elif method == "Smoothed_boot_epi_quantile":
        return smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-epispline', quantile=True)
    elif method == "Smoothed_boot_kernel_quantile":
        return smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-kernel', quantile=True)
    elif method == "Smoothed_bagging":
        return smoothed_bagging(cfg, module, xhat, distr_type='univariate-kernel')
    else:
        raise ValueError(f"boot_method={method} is not a smoothed method.")


if __name__ == "__main__":
    print("smoothed_boot_sp contains only functions and is not directly runnable.")
    print("Try, e.g., user_boot.py")
