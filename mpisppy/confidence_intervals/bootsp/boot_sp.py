###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# General-purpose bootstrap code for data-based, two-stage stochastic programs.
# These are the empirical methods (classical, extended, subsampling, bagging);
# the smoothed methods arrive in a follow-on merge.

import os
from statistics import NormalDist
import numpy as np
from numpy.random import default_rng
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils

# The communicators live in boot_utils so there is a single source of truth.
comm = boot_utils.comm
n_proc = boot_utils.n_proc
my_rank = boot_utils.my_rank
rankcomm = boot_utils.rankcomm


def _scenario_creator_w_mapping(scenario_name, module=None, mapping=None, **kwargs):
    """ A wrapper to allow for bootstrap samples to map to actual samples
    Args:
        scenario_name (str): the scenario number will be peeled off the end
        module (Python module): contains the scenario creator function and helpers
        mapping (dict): maps the scenario_name argument to a scenario sent to
                        the module scenario creator
        kwargs (dict): arguments for the module scenario creator
    Returns:
        model (Pyomo ConcreteModel): the instantiated scenarios

    Note: w is *not* the PH W, it is for resampling
    """
    if mapping is not None:
        return module.scenario_creator(mapping[scenario_name], **kwargs)
    else:
        return module.scenario_creator(scenario_name, **kwargs)


def slice_lens(nB):
    """ compute the share of nB for every MPI rank
    Args:
        nB (int): number of batches
    Returns:
        slice_lens (list): an allocation of nB to n_proc (a global) slices
    """

    avg = nB / n_proc
    slice_lens = [int((i + 1) * avg) - int(i * avg) for i in range(n_proc)]
    # we don't really need this assert, but it is harmless
    assert sum(slice_lens) == nB

    return slice_lens


def eligible_scenarios(cfg):
    """ The scenario numbers usable for confidence-interval sampling.

    Args:
        cfg (Config): parameters
    Returns:
        numpy array of scenario numbers

    The candidate_sample_size scenarios starting at sample_size are reserved
    for computing xhat (see boot_utils.compute_xhat and
    boot_general_prep.find_candidate), so the confidence-interval sampling
    must not touch them; this returns all the other scenario numbers.
    """
    if cfg.sample_size + cfg.candidate_sample_size > cfg.max_count:
        raise RuntimeError(
            "sample_size plus candidate_sample_size must be at most max_count "
            f"(got {cfg.sample_size} + {cfg.candidate_sample_size} > {cfg.max_count})")
    return np.concatenate([np.arange(cfg.sample_size),
                           np.arange(cfg.sample_size + cfg.candidate_sample_size,
                                     cfg.max_count)])


def _pool_rng(cfg):
    """ The random stream for pool and center draws.

    Streams are seeded with a (seed_offset, word) pair, which numpy's
    SeedSequence hashes, so no two streams coincide either within a run
    (the second word separates this stream from every rank's batch stream)
    or across seed_offsets (e.g. the sequential offsets used by the
    coverage simulations); summing the words instead can collide.
    """
    return default_rng([cfg.seed_offset, 0])


def _batch_rng(cfg):
    """ The per-rank random stream for batch resampling (see _pool_rng). """
    return default_rng([cfg.seed_offset, my_rank + 1])


def process_optimal(cfg, module):
    """ For simulations we need a known or assumed z*
        Args:
            cfg (Config): parameters
            module (Python module): contains the scenario creator function and helpers
        Returns:
            opt_obj (float): z*
            opt_gap (float): gap if provided by solver
    """

    if cfg.optimal_fname is not None and cfg.optimal_fname != "None":
        if not os.path.exists(cfg.optimal_fname):
            raise ValueError(f"File {cfg.optimal_fname} does not exist.\n"
                             "Maybe you need to run bootsp.boot_general_prep")
        print(f"Reading pre-computed optimal value from {cfg.optimal_fname}", flush=True)
        tmp = np.load(cfg["optimal_fname"], 'r')
        opt_obj = tmp[0]
        opt_gap = tmp[1]
        print(f"   ...optimal value: {opt_obj}")
        print(f"   ...optimality gap: {opt_gap}")
    else:
        print('No calculated optimal found, starting computing the "actual" optimal')
        print("Computing optimal function value on Rank 0 only")
        opt_ef = solve_routine(cfg, module, range(cfg.max_count), num_threads=2)
        opt_obj = pyo.value(opt_ef.EF_Obj)
        print(f"optimal EF objective: {opt_obj}; using zero gap (this should be verified visually)")
        opt_gap = 0
    return opt_obj, opt_gap


def solve_routine(cfg, module, scenarios, num_threads=None, duplication=False):
    """
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenarios (iterable; e.g., list): scenario numbers
        num_threads (int): number of solver threads
        duplication (bool): sample with duplication
    Returns:
        ef (EF): the full extensive form from mpi-sppy
    """

    tee_rank0_solves = False
    scenario_creator = _scenario_creator_w_mapping
    scenario_creator_kwargs = module.kw_creator(cfg)  # we get a new one every time...
    scenario_creator_kwargs['module'] = module  # we are going to call a wrapper

    if duplication:
        scenario_names = ['SampleScenario' + str(i) for i in range(len(scenarios))]
        scenario_creator_kwargs['mapping'] = {'SampleScenario' + str(i): 'Scenario' + str(scenarios[i]) for i in range(len(scenarios))}
    else:
        scenario_names = ['Scenario' + str(i) for i in scenarios]
        scenario_creator_kwargs['mapping'] = None

    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    solver = pyo.SolverFactory(cfg.solver_name)
    solver.options["threads"] = num_threads
    teeme = tee_rank0_solves if my_rank == 0 else False
    if 'persistent' in cfg.solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        solver.solve(tee=teeme)
    else:
        solver.solve(ef, tee=teeme, symbolic_solver_labels=True)

    return ef


def evaluate_routine(cfg, module, xhat, scenario_names, sample_mapping):
    """ evaluate a given xhat over given scenario names

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)
        scenario_names (list of str): the scenario number will be peeled off the ends
        sample_mapping (dict): If not None, maps the scenario_name argument to a scenario sent to
            the module scenario creator

    Returns:
        zhat (float): the computed expected value
    """
    xhat_eval_options = {"iter0_solver_options": None,
                         "iterk_solver_options": None,
                         "display_timing": False,
                         "solver_name": cfg.solver_name,
                         "verbose": False,
                         "toc": False
                         }

    scenario_creator = _scenario_creator_w_mapping
    scenario_creator_kwargs = module.kw_creator(cfg)  # we get a new one every time...
    scenario_creator_kwargs['module'] = module  # we are going to call a wrapper
    scenario_creator_kwargs["mapping"] = sample_mapping

    ev = xhat_eval.Xhat_Eval(xhat_eval_options,
                scenario_names,
                scenario_creator,
                mpicomm=rankcomm,
                scenario_creator_kwargs=scenario_creator_kwargs
                )

    zhat = ev.evaluate(xhat)

    return zhat


def evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True):
    """ evaluate xhat using a list of (sampled) scenario numbers

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenarios (iterable; e.g., list): scenario numbers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)
        duplication (bool): indicates scenarios may be duplicated in the scenarios list

    Returns:
        zhat (float): the computed expectation

    """
    # Take a list of indices of the original scenarios with possible replications
    # If need mapping, create a set of scenario names and a mapping function that maps the scenario names to the original ones
    # Return the function value evaluated for a given xhat

    if duplication:
        scenario_names = ['SampleScenario' + str(i) for i in range(len(scenarios))]
        sample_mapping = {'SampleScenario' + str(i): 'Scenario' + str(scenarios[i]) for i in range(len(scenarios))}
    else:
        scenario_names = ['Scenario' + str(i) for i in scenarios]
        sample_mapping = None

    return evaluate_routine(cfg, module, xhat, scenario_names, sample_mapping)


def _bootstrap_resample(cfg, module, scenario_pool, xhat, serial=False):
    """ Get gaps and optimal values for classic bootstrap.
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenario_pool (iterable; e.g., list): scenario numbers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        serial (bool): indicates that only one MPI rank should be used
    Returns:
        numpy arrays (vector) with gaps and optimal values that are *local* if serial is False

    """
    # loop over batches

    rng = _batch_rng(cfg)
    if serial:
        local_nB = cfg.nB
    else:
        local_nB = slice_lens(cfg.nB)[my_rank]
    local_boot_gaps = np.empty(local_nB, dtype=np.float64)
    local_boot_optimals = np.empty(local_nB, dtype=np.float64)
    local_boot_uppers = np.empty(local_nB, dtype=np.float64)
    for iter in range(local_nB):
        scenarios = rng.choice(scenario_pool, size=cfg.sample_size, replace=True)
        boot_ev = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True)
        boot_ef = solve_routine(cfg, module, scenarios, num_threads=2, duplication=True)
        local_boot_optimals[iter] = pyo.value(boot_ef.EF_Obj)
        local_boot_uppers[iter] = boot_ev
        local_boot_gaps[iter] = local_boot_uppers[iter] - local_boot_optimals[iter]

    return local_boot_gaps, local_boot_optimals, local_boot_uppers


def classical_bootstrap(cfg, module, xhat, quantile=True):
    """ perform a classic bootstrap estimation of confidence intervals

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        quantile (bool): use the quantile method (else the gaussian method)

    Returns:
        tuple with confidence interval if on MPI rank 0

    """
    rng = _pool_rng(cfg)

    scenario_pool = rng.choice(eligible_scenarios(cfg), size=cfg.sample_size, replace=False)
    dag_upper = evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=False)
    dag_ef = solve_routine(cfg, module, scenario_pool, num_threads=2, duplication=False)

    dag_optimal = pyo.value(dag_ef.EF_Obj)
    dag_gap = dag_upper - dag_optimal  # this is gamma(D) in the note

    # tron is a "secret" way to turn on internal trace information
    if cfg.get("tron", False):
        print(f"rank {my_rank} at dag barrier", flush=True)
    comm.Barrier()

    # bootstrap from pool
    local_boot_gaps, local_boot_optimals, local_boot_uppers = _bootstrap_resample(cfg, module, scenario_pool, xhat, serial=False)

    comm.Barrier()

    # do analysis only on rank 0
    if my_rank == 0:
        boot_gaps = np.empty(cfg.nB, dtype=np.float64)
        boot_optimals = np.empty(cfg.nB, dtype=np.float64)
        boot_uppers = np.empty(cfg.nB, dtype=np.float64)
    else:
        boot_gaps = None
        boot_optimals = None
        boot_uppers = None

    # but everyone needs to send to the gather
    lenlist = slice_lens(cfg.nB)
    comm.Gatherv(sendbuf=local_boot_gaps, recvbuf=(boot_gaps, lenlist), root=0)
    comm.Gatherv(sendbuf=local_boot_optimals, recvbuf=(boot_optimals, lenlist), root=0)
    comm.Gatherv(sendbuf=local_boot_uppers, recvbuf=(boot_uppers, lenlist), root=0)
    if cfg.get("tron", False) and my_rank == 0:
        print("*** rank 0 ends gather", flush=True)

    if my_rank == 0:
        if quantile:
            alpha = cfg.alpha / 2
            ci_optimal = np.quantile(2 * dag_optimal - boot_optimals, [alpha, 1 - alpha])
            ci_upper = np.quantile(2 * dag_upper - boot_uppers, [alpha, 1 - alpha])
            ci_gap = np.quantile(2 * dag_gap - boot_gaps, [alpha, 1 - alpha])
        else:
            dd = NormalDist().inv_cdf(1 - cfg.alpha / 2)
            std_optimal = np.std(boot_optimals, ddof=1)
            std_gap = np.std(boot_gaps, ddof=1)
            std_upper = np.std(boot_uppers, ddof=1)

            ci_optimal = [dag_optimal - dd * std_optimal, dag_optimal + dd * std_optimal]
            ci_upper = [dag_upper - dd * std_upper, dag_upper + dd * std_upper]
            ci_gap = [dag_gap - dd * std_gap, dag_gap + dd * std_gap]

        return ci_optimal, ci_upper, ci_gap, dag_optimal, dag_upper, dag_gap
    else:
        return None, None, None, None, None, None


def _sub_resample(cfg, module, scenario_pool, xhat, serial=False):
    """ Get gaps and optimal values for subsampling method.
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenario_pool (iterable; e.g., list): scenario numbers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        serial (bool): indicates that only one MPI rank should be used
    Returns:
        numpy arrays (vector) with gaps and optimal values that are *local* if serial is False

    """
    # loop over batches
    # only difference between this and bootstrap_sampling is the size of the scenarios: one is subsample_size, the other is sample_size

    rng = _batch_rng(cfg)
    if serial:
        local_nB = cfg.nB
    else:
        local_nB = slice_lens(cfg.nB)[my_rank]
    local_boot_gaps = np.empty(local_nB, dtype=np.float64)
    local_boot_optimals = np.empty(local_nB, dtype=np.float64)
    local_boot_uppers = np.empty(local_nB, dtype=np.float64)
    for iter in range(local_nB):
        scenarios = rng.choice(scenario_pool, size=cfg.subsample_size, replace=False)
        boot_ev = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True)
        boot_ef = solve_routine(cfg, module, scenarios, num_threads=2, duplication=True)
        if cfg.get("tron", False) and my_rank == 0:
            print(f"_sub_resample using EF_obj: {pyo.value(boot_ef.EF_Obj)}")
            print(f"   using evaluation: {boot_ev}")
        local_boot_optimals[iter] = pyo.value(boot_ef.EF_Obj)
        local_boot_uppers[iter] = boot_ev
        local_boot_gaps[iter] = local_boot_uppers[iter] - local_boot_optimals[iter]

    return local_boot_gaps, local_boot_optimals, local_boot_uppers


def subsampling(cfg, module, xhat):
    """ perform a subsampling estimation of confidence intervals

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)

    Returns:
        tuple with confidence interval if on MPI rank 0

    """
    rng = _pool_rng(cfg)

    scenario_pool = rng.choice(eligible_scenarios(cfg), size=cfg.sample_size, replace=False)
    dag_upper = evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=False)
    dag_ef = solve_routine(cfg, module, scenario_pool, num_threads=2, duplication=False)
    dag_optimal = pyo.value(dag_ef.EF_Obj)
    dag_gap = dag_upper - dag_optimal  # this is gamma(D) in the note

    comm.Barrier()

    # subsampling from pool
    local_boot_gaps, local_boot_optimals, local_boot_uppers = _sub_resample(cfg, module, scenario_pool, xhat, serial=False)
    comm.Barrier()

    # do analysis only on rank 0
    if my_rank == 0:
        boot_gaps = np.empty(cfg.nB, dtype=np.float64)
        boot_optimals = np.empty(cfg.nB, dtype=np.float64)
        boot_uppers = np.empty(cfg.nB, dtype=np.float64)
    else:
        boot_gaps = None
        boot_optimals = None
        boot_uppers = None

    # but everyone needs to send to the gather
    lenlist = slice_lens(cfg.nB)
    comm.Gatherv(sendbuf=local_boot_gaps, recvbuf=(boot_gaps, lenlist), root=0)
    comm.Gatherv(sendbuf=local_boot_optimals, recvbuf=(boot_optimals, lenlist), root=0)
    comm.Gatherv(sendbuf=local_boot_uppers, recvbuf=(boot_uppers, lenlist), root=0)

    if my_rank == 0:
        alpha = cfg.alpha / 2
        err_optimal = np.sqrt(cfg.subsample_size / cfg.sample_size) * np.quantile(boot_optimals - dag_optimal, [1 - alpha, alpha])
        ci_optimal = dag_optimal - err_optimal

        err_upper = np.sqrt(cfg.subsample_size / cfg.sample_size) * np.quantile(boot_uppers - dag_upper, [1 - alpha, alpha])
        ci_upper = dag_upper - err_upper

        err_gap = np.sqrt(cfg.subsample_size / cfg.sample_size) * np.quantile(boot_gaps - dag_gap, [1 - alpha, alpha])
        ci_gap = dag_gap - err_gap

        return ci_optimal, ci_upper, ci_gap, dag_optimal, dag_upper, dag_gap
    else:
        return None, None, None, None, None, None


def _extended_resample(cfg, module, xhat, serial=False):
    """ Get gaps and optimal values differences for extended bootstrap.
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        serial (bool): indicates that only one MPI rank should be used
    Returns:
        numpy arrays (vector) with gaps and optimal values differences that are *local* if serial is False

    """
    # loop over batches

    rng = _batch_rng(cfg)
    if serial:
        local_nB = cfg.nB
    else:
        local_nB = slice_lens(cfg.nB)[my_rank]

    local_boot_optimals_diff = np.empty(local_nB, dtype=np.float64)
    local_boot_uppers_diff = np.empty(local_nB, dtype=np.float64)
    local_boot_gaps_diff = np.empty(local_nB, dtype=np.float64)

    eligible = eligible_scenarios(cfg)
    for iter in range(local_nB):
        scenario_pool = rng.choice(eligible, size=cfg.sample_size, replace=True)
        dag_optimal_ef = solve_routine(cfg, module, scenario_pool, num_threads=2, duplication=True)
        dag_upper = evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=True)

        scenarios = rng.choice(scenario_pool, size=cfg.sample_size, replace=True)
        boot_optimal_ef = solve_routine(cfg, module, scenarios, num_threads=2, duplication=True)
        boot_upper = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True)

        local_boot_optimals_diff[iter] = pyo.value(boot_optimal_ef.EF_Obj) - pyo.value(dag_optimal_ef.EF_Obj)
        local_boot_uppers_diff[iter] = boot_upper - dag_upper

        local_boot_gaps_diff[iter] = local_boot_uppers_diff[iter] - local_boot_optimals_diff[iter]

    return local_boot_gaps_diff, local_boot_optimals_diff, local_boot_uppers_diff


def extended_bootstrap(cfg, module, xhat):
    """ perform an extended bootstrap estimation of confidence intervals

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)

    Returns:
        tuple with confidence interval if on MPI rank 0

    """
    rng = _pool_rng(cfg)

    # extended bootstrap
    local_boot_gaps_diff, local_boot_optimals_diff, local_boot_uppers_diff = _extended_resample(cfg, module, xhat, serial=False)
    comm.Barrier()

    # do analysis only on rank 0
    if my_rank == 0:
        boot_gaps_diff = np.empty(cfg.nB, dtype=np.float64)
        boot_optimals_diff = np.empty(cfg.nB, dtype=np.float64)
        boot_uppers_diff = np.empty(cfg.nB, dtype=np.float64)
    else:
        boot_gaps_diff = None
        boot_optimals_diff = None
        boot_uppers_diff = None

    # but everyone needs to send to the gather
    lenlist = slice_lens(cfg.nB)
    comm.Gatherv(sendbuf=local_boot_gaps_diff, recvbuf=(boot_gaps_diff, lenlist), root=0)
    comm.Gatherv(sendbuf=local_boot_optimals_diff, recvbuf=(boot_optimals_diff, lenlist), root=0)
    comm.Gatherv(sendbuf=local_boot_uppers_diff, recvbuf=(boot_uppers_diff, lenlist), root=0)

    if my_rank == 0:

        # get center
        eligible = eligible_scenarios(cfg)
        scenarios = rng.choice(eligible, size=cfg.sample_size, replace=True)
        dag_optimal_ef = solve_routine(cfg, module, scenarios, num_threads=2, duplication=True)
        dag_optimal = pyo.value(dag_optimal_ef.EF_Obj)
        dag_upper = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True)

        scenarios_ = rng.choice(eligible, size=cfg.sample_size, replace=True)
        scenarios_combined = np.concatenate([scenarios, scenarios_])

        dag_optimal_ef_combined = solve_routine(cfg, module, scenarios_combined, num_threads=2, duplication=True)
        dag_optimal_combined = pyo.value(dag_optimal_ef_combined.EF_Obj)
        dag_upper_combined = evaluate_scenarios(cfg, module, scenarios_combined, xhat, duplication=True)

        center_optimal = 2 * dag_optimal_combined - dag_optimal
        center_upper = 2 * dag_upper_combined - dag_upper
        center_gap = center_upper - center_optimal

        alpha = cfg.alpha / 2
        ci_optimal = center_optimal - np.quantile(boot_optimals_diff, [1 - alpha, alpha])
        ci_upper = center_upper - np.quantile(boot_uppers_diff, [1 - alpha, alpha])
        ci_gap = center_gap - np.quantile(boot_gaps_diff, [1 - alpha, alpha])

        return ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap
    else:
        return None, None, None, None, None, None


def _bagging_resample(cfg, module, scenario_pool, xhat, serial=False, replacement=True):
    """ Get gaps and optimal values differences for bagging.
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenario_pool (iterable; e.g., list): scenario numbers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        serial (bool): indicates that only one MPI rank should be used
    Returns:
        numpy arrays (vector) with gaps, optimal values, and boot counts that are *local* if serial is False

    """
    # loop over batches

    rng = _batch_rng(cfg)
    if serial:
        local_nB = cfg.nB
    else:
        local_nB = slice_lens(cfg.nB)[my_rank]
    local_boot_gaps = np.empty(local_nB, dtype=np.float64)
    local_boot_optimals = np.empty(local_nB, dtype=np.float64)
    local_boot_uppers = np.empty(local_nB, dtype=np.float64)
    local_boot_counts = np.zeros((local_nB, cfg.sample_size))
    for iter in range(local_nB):
        scenarios_index = rng.choice(len(scenario_pool), size=cfg.subsample_size, replace=replacement)
        scenarios = [scenario_pool[index] for index in scenarios_index]
        boot_ev = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=replacement)
        boot_ef = solve_routine(cfg, module, scenarios, num_threads=2, duplication=replacement)

        local_boot_optimals[iter] = pyo.value(boot_ef.EF_Obj)
        local_boot_uppers[iter] = boot_ev
        local_boot_gaps[iter] = local_boot_uppers[iter] - local_boot_optimals[iter]

        for index in scenarios_index:
            local_boot_counts[iter, index] += 1

    local_boot_counts = np.reshape(local_boot_counts, local_nB * cfg.sample_size)

    return local_boot_gaps, local_boot_optimals, local_boot_uppers, local_boot_counts


def bagging_bootstrap(cfg, module, xhat, replacement=True):
    """ perform a bagging-based estimation of confidence intervals

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)
        replacement (bool): sample the subsample with replacement

    Returns:
        tuple with confidence interval if on MPI rank 0
    """

    rng = _pool_rng(cfg)
    scenario_pool = rng.choice(eligible_scenarios(cfg), size=cfg.sample_size, replace=False)

    # bootstrap from pool
    local_boot_gaps, local_boot_optimals, local_boot_uppers, local_boot_counts = _bagging_resample(cfg, module, scenario_pool, xhat, serial=False, replacement=replacement)
    comm.Barrier()

    # do analysis only on rank 0
    if my_rank == 0:
        boot_gaps = np.empty(cfg.nB, dtype=np.float64)
        boot_optimals = np.empty(cfg.nB, dtype=np.float64)
        boot_uppers = np.empty(cfg.nB, dtype=np.float64)
        boot_counts = np.empty(cfg.nB * cfg.sample_size, dtype=np.float64)
    else:
        boot_gaps = None
        boot_optimals = None
        boot_uppers = None
        boot_counts = None

    # but everyone needs to send to the gather
    lenlist = slice_lens(cfg.nB)
    comm.Gatherv(sendbuf=local_boot_gaps, recvbuf=(boot_gaps, lenlist), root=0)
    comm.Gatherv(sendbuf=local_boot_optimals, recvbuf=(boot_optimals, lenlist), root=0)
    comm.Gatherv(sendbuf=local_boot_uppers, recvbuf=(boot_uppers, lenlist), root=0)

    receive_len = [x * cfg.sample_size for x in lenlist]
    comm.Gatherv(sendbuf=local_boot_counts, recvbuf=(boot_counts, receive_len), root=0)

    if my_rank == 0:
        center_gap = np.mean(boot_gaps)
        center_optimal = np.mean(boot_optimals)
        center_upper = np.mean(boot_uppers)
        boot_counts = np.reshape(boot_counts, (cfg.nB, cfg.sample_size))

        cov_gap = np.matmul(boot_gaps - center_gap, boot_counts - cfg.subsample_size / cfg.sample_size) / cfg.nB
        cov_gap = np.linalg.norm(cov_gap)

        cov_optimal = np.matmul(boot_optimals - center_optimal, boot_counts - cfg.subsample_size / cfg.sample_size) / cfg.nB
        cov_optimal = np.linalg.norm(cov_optimal)

        cov_upper = np.matmul(boot_uppers - center_upper, boot_counts - cfg.subsample_size / cfg.sample_size) / cfg.nB
        cov_upper = np.linalg.norm(cov_upper)

        if not replacement:
            cov_gap *= cfg.sample_size / (cfg.sample_size - cfg.subsample_size)
            cov_optimal *= cfg.sample_size / (cfg.sample_size - cfg.subsample_size)
            cov_upper *= cfg.sample_size / (cfg.sample_size - cfg.subsample_size)

        if cfg.get("tron", False) and my_rank == 0:
            print(f"cov_gap:{cov_gap}")
            print(f"cov_optimal: {cov_optimal}")
            print(f"cov_upper: {cov_upper}")

        dd = NormalDist().inv_cdf(1 - cfg.alpha / 2)
        ci_optimal = [center_optimal - dd * cov_optimal, center_optimal + dd * cov_optimal]
        ci_upper = [center_upper - dd * cov_upper, center_upper + dd * cov_upper]
        ci_gap = [center_gap - dd * cov_gap, center_gap + dd * cov_gap]

        return ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap
    else:
        return None, None, None, None, None, None


def compute_ci(cfg, module, xhat):
    """ Dispatch to the requested bootstrap method and return its result.

    Args:
        cfg (Config): parameters (cfg.boot_method selects the method)
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): a candidate solution in mpi-sppy nonant format

    Returns:
        (ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap);
        the ci_* entries are None on MPI ranks other than 0.

    Note:
        This is the single dispatch point shared by user_boot and
        simulate_boot. A smoothed method raises a friendly "not yet merged"
        error (the smoothed methods land in a follow-on merge).
    """
    method = cfg.boot_method
    boot_utils.BootMethods.check_for_it(method)
    if boot_utils.is_smoothed(method):
        boot_utils.smoothed_not_yet_merged(method)
    if method == "Extended":
        return extended_bootstrap(cfg, module, xhat)
    elif method == "Bagging_with_replacement":
        return bagging_bootstrap(cfg, module, xhat, replacement=True)
    elif method == "Bagging_without_replacement":
        return bagging_bootstrap(cfg, module, xhat, replacement=False)
    elif method == "Classical_quantile":
        return classical_bootstrap(cfg, module, xhat, quantile=True)
    elif method == "Classical_gaussian":
        return classical_bootstrap(cfg, module, xhat, quantile=False)
    elif method == "Subsampling":
        return subsampling(cfg, module, xhat)
    else:
        raise ValueError(f"boot_method={method} is not supported.")


if __name__ == "__main__":
    print("boot_sp contains only functions and is not directly runnable.")
    print("Try, e.g., user_boot.py")
