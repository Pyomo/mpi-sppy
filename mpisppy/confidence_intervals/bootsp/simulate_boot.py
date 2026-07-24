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
import mpisppy.confidence_intervals.ciutils as ciutils
import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp

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


def main(cfg, module):
    """ Dispatch to the appropriate coverage harness for cfg.boot_method.

    A smoothed method raises a friendly "not yet merged" error; the empirical
    methods run the empirical coverage harness.
    """
    if boot_utils.is_smoothed(cfg.boot_method):
        boot_utils.smoothed_not_yet_merged(cfg.boot_method)
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
