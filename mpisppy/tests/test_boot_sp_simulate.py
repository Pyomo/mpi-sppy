###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for the bootstrap coverage-simulation harness (bootsp) and its
# Gatherv-based batch parallelism, using the deterministic schultz example.
#
# Serial:
#   python -m pytest mpisppy/tests/test_boot_sp_simulate.py
# Parallel (exercises the Gatherv batch split across ranks):
#   mpiexec -np 2 python -m mpi4py mpisppy/tests/test_boot_sp_simulate.py
#
# Because each rank seeds its own bootstrap stream, the assembled result on
# rank 0 depends on the number of ranks; the locked values below are keyed by
# comm size so the same file asserts the correct value serially and under
# mpiexec -np 2.

import os
import sys
import unittest

import mpisppy.utils.sputils as sputils
from mpisppy.tests.utils import get_solver, round_pos_sig

import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp
import mpisppy.confidence_intervals.bootsp.simulate_boot as simulate_boot

sputils.disable_tictoc_output()

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

comm = boot_utils.comm
n_proc = boot_utils.n_proc
my_rank = boot_utils.my_rank

module_dir = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(module_dir, "..", "..", "examples", "bootsp", "schultz")
if not os.path.exists(example_dir):
    raise RuntimeError(f"Directory not found: {example_dir}")
if example_dir not in sys.path:
    sys.path.insert(0, example_dir)

MODULE_NAME = "unique_schultz"

# classical_bootstrap ci_optimal (seed_offset=100), keyed by number of ranks
locked_ci_optimal = {
    1: [-53.855000000000025, -48.95166666666669],
    2: [-54.26333333333335, -48.53000000000002],
}
# coverage harness result (rate, length) with seed base 0 and 4 replications,
# keyed by number of ranks
locked_coverage = {
    1: (1.0, 5.000000000000005),
    2: (1.0, 5.60000000000001),
}


def _make_cfg(method="Classical_quantile", seed=100, reps=4):
    cfg = boot_utils._process_module(MODULE_NAME)
    cfg.module_name = MODULE_NAME
    cfg.max_count = 50
    cfg.candidate_sample_size = 1
    cfg.sample_size = 30
    cfg.subsample_size = 10
    cfg.nB = 20
    cfg.alpha = 0.1
    cfg.seed_offset = seed
    cfg.xhat_fname = "None"
    cfg.optimal_fname = "None"
    cfg.trace_fname = None
    cfg.coverage_replications = reps
    cfg.solver_name = solver_name
    cfg.boot_method = method
    return cfg


#*****************************************************************************
class Test_boot_sp_simulate(unittest.TestCase):
    """ Test the coverage harness and the Gatherv batch parallelism.

    Every test calls its collective operation on all ranks (so the MPI
    collectives line up) and only asserts on rank 0.
    """

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_gatherv_value(self):
        # classical_bootstrap gathers per-rank batches with Gatherv; the
        # assembled rank-0 result is deterministic for a given rank count.
        module = boot_utils.module_name_to_module(MODULE_NAME)
        cfg = _make_cfg("Classical_quantile", seed=100)
        xhat = boot_utils.compute_xhat(cfg, module)
        res = boot_sp.classical_bootstrap(cfg, module, xhat, quantile=True)
        if my_rank == 0:
            ci_optimal = list(res[0])
            self.assertLessEqual(ci_optimal[0], ci_optimal[1])
            if n_proc in locked_ci_optimal:
                expected = locked_ci_optimal[n_proc]
                for g, e in zip(ci_optimal, expected):
                    self.assertEqual(round_pos_sig(g, 4), round_pos_sig(e, 4))
        else:
            # non-root ranks participate in the collective and get None back
            self.assertEqual(res, (None, None, None, None, None, None))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_coverage_harness(self):
        module = boot_utils.module_name_to_module(MODULE_NAME)
        cfg = _make_cfg("Classical_quantile", seed=0, reps=4)
        coverage = simulate_boot.main(cfg, module)
        if my_rank == 0:
            rate, length = coverage
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)
            self.assertGreater(length, 0.0)
            if n_proc in locked_coverage:
                exp_rate, exp_length = locked_coverage[n_proc]
                self.assertEqual(rate, exp_rate)
                self.assertEqual(round_pos_sig(length, 4), round_pos_sig(exp_length, 4))
        else:
            self.assertEqual(coverage, (None, None))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_bagging_gatherv(self):
        # bagging also gathers a (larger) counts buffer with Gatherv
        module = boot_utils.module_name_to_module(MODULE_NAME)
        cfg = _make_cfg("Bagging_with_replacement", seed=100)
        xhat = boot_utils.compute_xhat(cfg, module)
        res = boot_sp.bagging_bootstrap(cfg, module, xhat, replacement=True)
        if my_rank == 0:
            self.assertLessEqual(res[0][0], res[0][1])
        else:
            self.assertEqual(res, (None, None, None, None, None, None))

    def test_smoothed_not_yet_merged(self):
        # no solver needed; the guard fires before any solve
        module = boot_utils.module_name_to_module(MODULE_NAME)
        cfg = _make_cfg("Smoothed_boot_kernel")
        with self.assertRaises(RuntimeError):
            simulate_boot.main(cfg, module)


if __name__ == '__main__':
    unittest.main()
