###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# A driver for the bootstrap code aimed at researchers running coverage
# simulations.
#
#   python -m mpisppy.confidence_intervals.bootsp.simulate_boot <json>

import sys
import time
import mpisppy.confidence_intervals.ciutils as ciutils
import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp
import mpisppy.confidence_intervals.bootsp.smoothed_boot_sp as smoothed_boot_sp

my_rank = boot_utils.my_rank


def empirical_main_routine(cfg, module):
    """ The empirical coverage harness; called by main() and by test drivers.

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
    Returns:
        coverage_rate (float): the coverage detected in the simulations
        average_length (float): the average width of the interval around z*
        (both None on MPI ranks other than 0)
    """
    if my_rank == 0:
        opt_obj, opt_gap = boot_sp.process_optimal(cfg, module)
    else:
        opt_obj = None  # only rank 0 should use the opt_obj in analysis anyway
        opt_gap = None

    if cfg["xhat_fname"] is not None and cfg["xhat_fname"] != "None":
        xhat = ciutils.read_xhat(cfg["xhat_fname"])
    else:
        xhat = boot_utils.compute_xhat(cfg, module)

    coverage_cnt = 0
    total_len = 0
    seed_list = range(cfg.coverage_replications)
    for seed in seed_list:
        cfg.seed_offset = seed

        ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap = \
            boot_sp.compute_ci(cfg, module, xhat)

        if my_rank == 0:
            if cfg.trace_fname is not None:
                with open(cfg.trace_fname, "a+") as f:
                    f.write(f"method: {cfg.boot_method}\n")
                    f.write(f"seed: {seed}\n")
                    f.write(f"optimal function value z^*: {opt_obj}\n")
                    f.write(f"ci for optimal function value z^*: {ci_optimal}\n")
                    f.write(f"function value evaluated at xhat: {opt_obj + opt_gap} \n")
                    f.write(f"ci for function value at xhat: {ci_upper}\n")
                    f.write(f"optimality gap: {opt_gap}\n")
                    f.write(f"ci for optimality gap: {ci_gap}\n")
            if (ci_optimal[0] <= opt_obj) and (opt_obj <= ci_optimal[1]):
                coverage_cnt += 1
            total_len += ci_optimal[1] - ci_optimal[0]

    # only rank 0 gets accumulated confidence interval
    if my_rank == 0:
        assert cfg.coverage_replications != 0
        return coverage_cnt / cfg.coverage_replications, total_len / cfg.coverage_replications
    else:
        return None, None


def smoothed_main_routine(cfg, module):
    """ The smoothed-method coverage harness; called by main() and test drivers.

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
    Returns:
        (coverage_two_sided, coverage_one_sided, ci_lengths, run_times);
        all None on MPI ranks other than 0.

    Note:
        The smoothed estimators report only the optimality-gap interval, so the
        coverage counts are against opt_gap (from process_optimal) rather than
        against z* as in the empirical harness.
    """
    if my_rank == 0:
        # only opt_gap is used by the smoothed coverage counting
        _, opt_gap = boot_sp.process_optimal(cfg, module)
    else:
        opt_gap = None

    if cfg["xhat_fname"] is not None and cfg["xhat_fname"] != "None":
        xhat = ciutils.read_xhat(cfg["xhat_fname"])
    else:
        # boot-sp called an undefined fit_resample_utils.compute_xhat here; the
        # intended call is boot_utils.compute_xhat (design doc section 4.3).
        xhat = boot_utils.compute_xhat(cfg, module)

    coverage_cnt_one_sided, coverage_cnt_two_sided = 0, 0
    ci_len = []
    run_time = []
    seed_offset = cfg.seed_offset  # store the original offset
    seed_list = [i * cfg.nB * 100 + seed_offset for i in range(cfg.coverage_replications)]

    for seed in seed_list:
        cfg.seed_offset = seed
        if my_rank == 0:
            st_time = time.time()
        ci_gap_two_sided, _ = smoothed_boot_sp.compute_smoothed_ci(cfg, module, xhat)
        if my_rank == 0:
            en_time = time.time()
            if cfg.trace_fname is not None:
                with open(cfg.trace_fname, "a+") as f:
                    f.write(f"seed: {cfg.seed_offset}\n")
                    f.write(f"optimality gap: {opt_gap}\n")
                    f.write(f"ci for optimality gap: {ci_gap_two_sided}\n")
            if (ci_gap_two_sided[0] <= opt_gap) and (opt_gap <= ci_gap_two_sided[1]):
                coverage_cnt_two_sided += 1
            if (opt_gap <= ci_gap_two_sided[1]):
                coverage_cnt_one_sided += 1
            ci_len.append(ci_gap_two_sided[1] - ci_gap_two_sided[0])
            run_time.append(en_time - st_time)

    if my_rank == 0:
        assert cfg.coverage_replications != 0
        return (coverage_cnt_two_sided / cfg.coverage_replications,
                coverage_cnt_one_sided / cfg.coverage_replications,
                ci_len, run_time)
    else:
        return None, None, None, None


def main(cfg, module):
    """ Dispatch to the appropriate coverage harness for cfg.boot_method.

    The empirical methods run the empirical coverage harness (returns a
    (rate, length) pair); the smoothed methods run the smoothed harness
    (returns a (cov_two, cov_one, ci_lengths, run_times) tuple).
    """
    if boot_utils.is_smoothed(cfg.boot_method):
        return smoothed_main_routine(cfg, module)
    return empirical_main_routine(cfg, module)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("need json file")
        print("usage, e.g.: python -m mpisppy.confidence_intervals.bootsp.simulate_boot farmer.json")
        quit()

    json_fname = sys.argv[1]
    cfg = boot_utils.cfg_from_json(json_fname)
    boot_utils.check_BFs(cfg)

    module = boot_utils.module_name_to_module(cfg.module_name)

    coverage = main(cfg, module)
    if my_rank == 0:
        print("Coverage", coverage)
