###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# A command-line driver for the bootstrap confidence-interval code.
#
#   python -m mpisppy.confidence_intervals.bootsp.user_boot <module> --options

import sys
import mpisppy.confidence_intervals.ciutils as ciutils
import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp
import mpisppy.confidence_intervals.bootsp.smoothed_boot_sp as smoothed_boot_sp

my_rank = boot_utils.my_rank


def _empirical_report(cfg, module, xhat):
    """ Run and print an empirical bootstrap CI; return the 6-tuple. """
    ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap = \
        boot_sp.compute_ci(cfg, module, xhat)

    if my_rank == 0:
        # print result
        print(f"point estimator for optimal function value: {center_optimal}")
        print(f"point estimator for function value at xhat: {center_upper}")
        print(f"point estimator for optimality gap: {center_gap}")
        ci_gap[0] = max(0, ci_gap[0])
        print(f"ci for optimal function value: {ci_optimal}")
        print(f"ci for function value at xhat: {ci_upper}")
        print(f"ci for optimality gap: {ci_gap}")

    return ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap


def _smoothed_report(cfg, module, xhat):
    """ Run and print a smoothed bootstrap/bagging CI; return (ci_gap, center_gap).

    The smoothed methods estimate only the optimality-gap interval, so the
    return signature differs from the empirical 6-tuple.
    """
    result = smoothed_boot_sp.compute_smoothed_ci(cfg, module, xhat)
    if my_rank == 0:
        ci_gap_two_sided, center_gap = result
        ci_gap_two_sided[0] = max(0, ci_gap_two_sided[0])
        print(f"point estimator for the optimality gap: {center_gap}")
        print(f"two-sided CI for optimality gap: {ci_gap_two_sided}")
        return ci_gap_two_sided, center_gap
    return result


def main_routine(cfg, module):
    """ The top level of user_boot; called by __main__ and by test drivers.

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
    Returns:
        For an empirical boot_method, the 6-tuple
        (ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap);
        for a smoothed boot_method, the pair (ci_gap_two_sided, center_gap).
        The ci_* entries are None on MPI ranks other than 0.
    Note:
        Prints the confidence-interval results to the terminal on rank 0.
    """
    if cfg["xhat_fname"] is not None and cfg["xhat_fname"] != "None":
        xhat = ciutils.read_xhat(cfg["xhat_fname"])
    else:
        xhat = boot_utils.compute_xhat(cfg, module)

    if boot_utils.is_smoothed(cfg.boot_method):
        return _smoothed_report(cfg, module, xhat)
    return _empirical_report(cfg, module, xhat)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("need module name")
        print("usage python -m mpisppy.confidence_intervals.bootsp.user_boot module --options")
        print("usage (e.g.): python -m mpisppy.confidence_intervals.bootsp.user_boot farmer --help")
        print("   note: module name should not end in .py")
        quit()

    module_name = sys.argv[1]
    cfg = boot_utils.cfg_from_parse(module_name, name="user_boot")
    boot_utils.check_BFs(cfg)

    module = boot_utils.module_name_to_module(cfg["module_name"])

    main_routine(cfg, module)
