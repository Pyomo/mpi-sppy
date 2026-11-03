###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unit tests for the opt-in Slurm-aware soft deadline."""

import os
import unittest
from unittest import mock

from mpisppy.phbase import PHBase
from mpisppy.opt.fwph import FWPH


class TestSlurmTimeLimit(unittest.TestCase):
    def _bare_ph(self):
        return object.__new__(PHBase)

    def test_unset_option_is_inactive_without_slurm(self):
        ph = self._bare_ph()
        with mock.patch.dict(os.environ, {}, clear=True):
            ph._configure_slurm_deadline({})
        self.assertIsNone(ph._slurm_deadline)

    def test_set_option_requires_slurm_end_time(self):
        ph = self._bare_ph()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SLURM_JOB_END_TIME"):
                ph._configure_slurm_deadline(
                    {"slurm_buffer_seconds": 7200}
                )

    def test_negative_buffer_is_rejected(self):
        ph = self._bare_ph()
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ph._configure_slurm_deadline(
                {"slurm_buffer_seconds": -1}
            )

    def test_end_time_is_converted_to_monotonic_deadline(self):
        ph = self._bare_ph()
        with mock.patch.dict(
                os.environ, {"SLURM_JOB_END_TIME": "5000"}, clear=True), \
                mock.patch("mpisppy.phbase.time.time", return_value=1000), \
                mock.patch(
                    "mpisppy.phbase.time.perf_counter", return_value=100
                ):
            ph._configure_slurm_deadline(
                {"slurm_buffer_seconds": 600}
            )

        self.assertEqual(ph._slurm_end_time, 5000.0)
        self.assertEqual(ph._slurm_buffer_seconds, 600.0)
        self.assertEqual(ph._slurm_deadline, 3500.0)

    def test_log_message_uses_time_remaining_when_it_is_emitted(self):
        ph = self._bare_ph()
        ph._slurm_end_time = 5000.0
        ph._slurm_deadline = 3500.0
        ph._slurm_buffer_seconds = 600.0

        with mock.patch("mpisppy.phbase.time.time", return_value=1200), \
                mock.patch(
                    "mpisppy.phbase.time.perf_counter", return_value=300
                ):
            message = ph._slurm_limit_message()

        self.assertIn("3800.0s remaining", message)
        self.assertIn("hub time limit: 3200.0s", message)

    def test_slurm_deadline_and_run_limit_are_both_honored(self):
        ph = self._bare_ph()
        ph.options = {"time_limit": 1000}
        ph.start_time = 100.0
        ph._slurm_deadline = 500.0
        ph._slurm_buffer_seconds = 60.0

        with mock.patch(
                "mpisppy.phbase.time.perf_counter", return_value=501.0):
            self.assertIn("Slurm shutdown buffer",
                          ph._local_time_limit_stop_reason())

        ph._slurm_deadline = 2000.0
        with mock.patch(
                "mpisppy.phbase.time.perf_counter", return_value=1100.0):
            self.assertIn("Time limit", ph._local_time_limit_stop_reason())

    def test_fwph_skips_crossover_after_lp_loop_times_out(self):
        fw = object.__new__(FWPH)
        fw.options = {
            "smoothed": 0,
            "objgap_mode": False,
            "FW_LP_start_iterations": 1,
            "PHIterLimit": 5,
            "display_timing": False,
            "tee-rank0-solves": False,
            "verbose": False,
        }
        fw.FW_options = fw.options
        fw.start_time = 0.0
        fw._PHIter = 0
        fw.local_scenarios = {}
        fw.ph_converger = None
        fw.cylinder_rank = 0

        for name in (
                "PH_Prep", "_output_header", "_attach_MIP_vars",
                "_cache_nonant_var_swap_mip", "_output", "_disable_W",
                "_initialize_QP_subproblems", "_attach_indices",
                "_attach_MIP_QP_maps", "_set_QP_objective",
                "_initialize_QP_var_values", "_cache_nonant_var_swap_qp",
                "_setup_shared_column_generation",
                "_generate_initial_columns_if_needed", "_reenable_W",
                "attach_PH_to_objective", "_reenable_prox", "_disable_prox",
        ):
            setattr(fw, name, mock.Mock())
        fw.Iter0 = mock.Mock(return_value=0.0)
        fw._is_timed_out = mock.Mock(return_value=True)
        fw._local_time_limit_stop_reason = mock.Mock(
            return_value="Slurm shutdown buffer reached."
        )
        fw.fwph_solve_loop = mock.Mock()

        def finish_lp_loop():
            fw._PHIter = 1

        with mock.patch.object(PHBase, "iterk_loop", side_effect=finish_lp_loop):
            result = fw.fwph_main(finalize=False)

        self.assertEqual(result, 1)
        fw.fwph_solve_loop.assert_not_called()
        self.assertEqual(fw.options["PHIterLimit"], 5)


if __name__ == "__main__":
    unittest.main()
