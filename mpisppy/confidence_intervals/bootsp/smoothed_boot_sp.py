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

import json

import numpy as np
from numpy.random import default_rng
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


def fit_distribution(sample_data, distr_type='univariate-epispline'):
    """ Fit a (univariate) distribution to sample data.

    Args:
        sample_data (list or list of dict): a list of scalars (one variable) or
            a list of dicts (multivariate, keyed by variable name)
        distr_type (str): a statdist univariate distribution token
    Returns:
        the fitted distribution (or a dict of them, keyed as the input dicts)
    """
    distr_func = statdist.distribution_factory(distr_type)
    if isinstance(sample_data[0], (float, int)):  # 1-dim
        fitted_distr = distr_func.fit(sample_data)
    else:
        fitted_distr = {}
        for key in sample_data[0]:
            data = [data_dict[key] for data_dict in sample_data]
            fitted_distr[key] = distr_func.fit(data)
    return fitted_distr


def center_smoothed(cfg, module, xhat, mpicomm):
    """ Estimate the CI center (the optimality gap) from the fitted distribution. """
    assert cfg.smoothed_center_sample_size is not None, \
        "need a sample size for smoothed bootstrap center estimation"
    scenario_pool = list(range(cfg.seed_offset,
                               cfg.seed_offset + cfg.smoothed_center_sample_size))

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
        analog of boot_sp._bootstrap_resample). """
    if serial:
        local_nB = cfg.nB
    else:
        local_nB = boot_sp.slice_lens(cfg.nB)[my_rank]

    local_boot_gaps = np.empty(local_nB, dtype=np.float64)

    boot_cfg = cfg()  # for ephemeral changes to deal with seed_offset
    boot_cfg.use_fitted = True

    for iter in range(local_nB):
        # seed_offset makes unique samples
        if serial:
            seed_offset = iter
        else:
            seed_offset = sum(boot_sp.slice_lens(boot_cfg.nB)[:my_rank]) + iter
        boot_cfg.seed_offset = cfg.seed_offset + seed_offset

        scenario_pool = list(range(boot_cfg.seed_offset,
                                   boot_cfg.seed_offset + cfg.subsample_size))

        local_boot_upper = boot_sp.evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=False)
        local_boot_ef = boot_sp.solve_routine(cfg, module, scenario_pool, num_threads=2, duplication=False)
        local_boot_optimal = pyo.value(local_boot_ef.EF_Obj)
        local_boot_gaps[iter] = local_boot_upper - local_boot_optimal

    return local_boot_gaps


def smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-epispline', quantile=False, serial=False):
    """ use the original data to estimate the center, then perform a smoothed estimation of width of confidence intervals
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
    rng = default_rng(cfg.seed_offset)
    scenario_pool = rng.choice(cfg.max_count, size=cfg.sample_size, replace=False)

    cfg.use_fitted = False
    sample_data = [module.data_sampler(scenario, cfg) for scenario in scenario_pool]
    cfg.fitted_distribution = fit_distribution(sample_data, distr_type=distr_type)

    # estimation of CI center
    dag_gap = center_smoothed(cfg, module, xhat, mpicomm=comm)
    comm.Barrier()
    cfg.use_fitted = True

    # conduct an m out of n bootstrap, with B = cfg.nB
    cfg.subsample_size = cfg.sample_size
    local_boot_gaps = smoothed_resample_helper(cfg, module, xhat, serial)
    comm.Barrier()

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
        print(f"{ci_gap_two_sided = }")
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
    rng = default_rng(cfg.seed_offset)
    scenario_pool = rng.choice(cfg.max_count, size=cfg.sample_size, replace=False)

    cfg.use_fitted = False
    sample_data = [module.data_sampler(scenario, cfg) for scenario in scenario_pool]
    cfg.fitted_distribution = fit_distribution(sample_data, distr_type=distr_type)
    cfg.use_fitted = True

    local_nB = boot_sp.slice_lens(cfg.nB)[my_rank]
    local_gaps = np.empty(local_nB, dtype=np.float64)

    if my_rank == 0:
        bagging_gap = np.empty(cfg.nB, dtype=np.float64)
        all_gaps = []
        avg_gaps = []
    else:
        bagging_gap = None
        all_gaps = None
        avg_gaps = None

    assert cfg.smoothed_B_I is not None, "B_I required for smoothed bagging"

    B_I = cfg.smoothed_B_I
    for i in range(B_I):
        seed_offset_base = cfg.seed_offset + cfg.nB * cfg.subsample_size * i

        for j in range(local_nB):
            seed_offset = seed_offset_base + (sum(boot_sp.slice_lens(cfg.nB)[:my_rank]) + j) * cfg.subsample_size
            scenario_pool = list(range(seed_offset, seed_offset + cfg.subsample_size))
            scenario_pool[0] = seed_offset_base

            local_upper = boot_sp.evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=False)
            local_ef = boot_sp.solve_routine(cfg, module, scenario_pool, num_threads=2, duplication=False)
            local_optimal = pyo.value(local_ef.EF_Obj)
            local_gaps[j] = local_upper - local_optimal
        comm.Barrier()
        lenlist = boot_sp.slice_lens(cfg.nB)
        comm.Gatherv(sendbuf=local_gaps, recvbuf=(bagging_gap, lenlist), root=0)

        if my_rank == 0:
            all_gaps = all_gaps + bagging_gap.tolist()
            avg_gaps.append(np.mean(bagging_gap))

    if my_rank == 0:
        global_toc("Done Smoothed Bagging")

        dag_gap = np.mean(avg_gaps)

        s1 = np.var(avg_gaps)
        s2 = np.var(all_gaps)
        ppf = NormalDist().inv_cdf(1 - cfg.alpha / 2)
        s_g_2 = (cfg.subsample_size**2) * s1 / cfg.sample_size + s2 / (B_I * cfg.nB)
        error = np.sqrt(s_g_2) * ppf
        ci_gap_two_sided = [dag_gap - error, dag_gap + error]

        print(f"{ci_gap_two_sided = }")
        return ci_gap_two_sided, dag_gap
    else:
        # non-root ranks return a matching arity so callers can unpack safely
        return None, None


def _ensure_smoothed_cfg(cfg):
    """ Idempotently attach the run-time config entries the smoothed methods need.

    The smoothed estimators toggle ``use_fitted`` and stash a
    ``fitted_distribution`` on the cfg; a module may also supply deterministic
    data via a json file named by ``deterministic_data_json``. This may be
    called repeatedly (e.g. once per replication in a coverage simulation), so
    every add is guarded.
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
    if "deterministic_data_json" in cfg and "detdata" not in cfg:
        json_fname = cfg.deterministic_data_json
        try:
            with open(json_fname, "r") as read_file:
                detdata = json.load(read_file)
        except Exception:
            print(f"Could not read the json file: {json_fname}")
            raise
        cfg.add_to_config("detdata",
                          description="deterministic data from json file",
                          domain=dict,
                          default=detdata)


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
