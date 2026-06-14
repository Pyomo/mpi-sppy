###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for the --incumbent-on-improvement-filename-prefix feature (issue #285).

Covers three surfaces touched by the feature:

  1. Config.popular_args() registers the option with default None.
  2. cfg_vanilla.shared_options forwards cfg.incumbent_on_improvement_filename_prefix
     into the options dict consumed by spokes.
  3. InnerBoundSpoke._maybe_write_incumbent_on_improvement does the right thing
     across the disabled / happy / fail-soft branches.

The spoke tests call the unbound method against a SimpleNamespace stub so we
don't have to stand up MPI infrastructure to exercise pure file-IO control flow.
"""

import types
import unittest
import warnings
from unittest.mock import MagicMock

from mpisppy.cylinders.spoke import InnerBoundSpoke
from mpisppy.utils import sputils
from mpisppy.utils.config import Config


class TestConfigRegistration(unittest.TestCase):
    """Config.popular_args must register the new option and default it to None."""

    def test_option_is_registered(self):
        cfg = Config()
        cfg.popular_args()
        self.assertIn("incumbent_on_improvement_filename_prefix", cfg)

    def test_option_defaults_to_none(self):
        cfg = Config()
        cfg.popular_args()
        self.assertIsNone(cfg.incumbent_on_improvement_filename_prefix)


class TestSharedOptionsForwarding(unittest.TestCase):
    """cfg_vanilla.shared_options must forward the prefix into the options
    dict so InnerBoundSpoke can see it via self.opt.options.get(...)."""

    def _make_cfg(self, prefix=None):
        cfg = Config()
        cfg.popular_args()
        cfg.solver_name = "gurobi"
        if prefix is not None:
            cfg.incumbent_on_improvement_filename_prefix = prefix
        return cfg

    def test_prefix_forwarded_when_set(self):
        import mpisppy.utils.cfg_vanilla as vanilla
        cfg = self._make_cfg(prefix="/tmp/incumbent")
        opts = vanilla.shared_options(cfg, is_hub=False)
        self.assertEqual(
            opts["incumbent_on_improvement_filename_prefix"],
            "/tmp/incumbent",
        )

    def test_prefix_none_by_default(self):
        import mpisppy.utils.cfg_vanilla as vanilla
        cfg = self._make_cfg(prefix=None)
        opts = vanilla.shared_options(cfg, is_hub=False)
        self.assertIsNone(opts["incumbent_on_improvement_filename_prefix"])

    def test_prefix_present_on_hub_too(self):
        import mpisppy.utils.cfg_vanilla as vanilla
        cfg = self._make_cfg(prefix="/tmp/hub_inc")
        opts = vanilla.shared_options(cfg, is_hub=True)
        self.assertEqual(
            opts["incumbent_on_improvement_filename_prefix"],
            "/tmp/hub_inc",
        )


class TestMaybeWriteIncumbent(unittest.TestCase):
    """Behavioral tests for InnerBoundSpoke._maybe_write_incumbent_on_improvement.

    We call the method against a minimal stub (SimpleNamespace) rather than
    construct a real spoke. The method only touches:

      self.opt.options.get(...)
      self.opt.write_first_stage_solution(...)
      self.cylinder_rank
      self._incumbent_write_counter
      self._incumbent_write_disabled

    so a stub with those attributes is enough.
    """

    def _stub(self, prefix=None, rank=0, disabled=False, counter=0):
        stub = types.SimpleNamespace()
        stub.cylinder_rank = rank
        stub._incumbent_write_counter = counter
        stub._incumbent_write_disabled = disabled
        stub.opt = types.SimpleNamespace()
        stub.opt.options = {"incumbent_on_improvement_filename_prefix": prefix}
        stub.opt.write_first_stage_solution = MagicMock()
        return stub

    def _run(self, stub):
        return InnerBoundSpoke._maybe_write_incumbent_on_improvement(stub)

    # ---- disabled branches ----

    def test_noop_when_prefix_is_none(self):
        stub = self._stub(prefix=None)
        self._run(stub)
        stub.opt.write_first_stage_solution.assert_not_called()
        self.assertEqual(stub._incumbent_write_counter, 0)
        self.assertFalse(stub._incumbent_write_disabled)

    def test_noop_when_already_disabled(self):
        stub = self._stub(prefix="/tmp/p", disabled=True)
        self._run(stub)
        stub.opt.write_first_stage_solution.assert_not_called()
        # counter untouched, disabled stays True
        self.assertEqual(stub._incumbent_write_counter, 0)
        self.assertTrue(stub._incumbent_write_disabled)

    # ---- happy path ----

    def test_happy_path_writes_csv_and_npy_and_bumps_counter(self):
        stub = self._stub(prefix="/tmp/inc", counter=0)
        self._run(stub)

        # csv goes first, with default writer (no kwarg)
        call_csv, call_npy = stub.opt.write_first_stage_solution.call_args_list
        self.assertEqual(call_csv.args, ("/tmp/inc_0000.csv",))
        self.assertEqual(call_csv.kwargs, {})

        # npy follows with the explicit npy serializer
        self.assertEqual(call_npy.args, ("/tmp/inc_0000.npy",))
        self.assertEqual(
            call_npy.kwargs,
            {"first_stage_solution_writer": sputils.first_stage_nonant_npy_serializer},
        )

        self.assertEqual(stub._incumbent_write_counter, 1)
        self.assertFalse(stub._incumbent_write_disabled)

    def test_counter_zero_pads_to_four_digits(self):
        stub = self._stub(prefix="/tmp/inc", counter=42)
        self._run(stub)
        names = [c.args[0] for c in stub.opt.write_first_stage_solution.call_args_list]
        self.assertEqual(names, ["/tmp/inc_0042.csv", "/tmp/inc_0042.npy"])
        self.assertEqual(stub._incumbent_write_counter, 43)

    def test_repeated_calls_increment_counter(self):
        stub = self._stub(prefix="/tmp/inc", counter=0)
        self._run(stub)
        self._run(stub)
        self._run(stub)
        names = [c.args[0] for c in stub.opt.write_first_stage_solution.call_args_list]
        # 3 calls × 2 files each = 6 filenames, counters 0/0/1/1/2/2
        self.assertEqual(names, [
            "/tmp/inc_0000.csv", "/tmp/inc_0000.npy",
            "/tmp/inc_0001.csv", "/tmp/inc_0001.npy",
            "/tmp/inc_0002.csv", "/tmp/inc_0002.npy",
        ])
        self.assertEqual(stub._incumbent_write_counter, 3)

    # ---- fail-soft branches ----

    def test_runtime_error_warns_on_rank0_and_disables(self):
        stub = self._stub(prefix="/tmp/inc", rank=0, counter=5)
        stub.opt.write_first_stage_solution.side_effect = RuntimeError(
            "No first stage solution available"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # must not propagate
            self._run(stub)
        self.assertTrue(stub._incumbent_write_disabled)
        # counter must NOT be incremented — the increment is past the try block
        self.assertEqual(stub._incumbent_write_counter, 5)
        # exactly one warning, and it mentions the option name
        self.assertEqual(len(w), 1)
        self.assertIn("incumbent_on_improvement_filename_prefix", str(w[0].message))
        self.assertIn("Disabling", str(w[0].message))

    def test_runtime_error_silent_on_nonzero_rank(self):
        stub = self._stub(prefix="/tmp/inc", rank=1, counter=0)
        stub.opt.write_first_stage_solution.side_effect = RuntimeError("boom")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._run(stub)
        # disabled flag still set so future calls short-circuit on every rank
        self.assertTrue(stub._incumbent_write_disabled)
        # but no warning printed on non-rank-0
        self.assertEqual(len(w), 0)

    def test_disabled_after_failure_makes_next_call_noop(self):
        stub = self._stub(prefix="/tmp/inc", rank=0)
        stub.opt.write_first_stage_solution.side_effect = RuntimeError("nope")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._run(stub)  # first call: trips fail-soft
            stub.opt.write_first_stage_solution.reset_mock()
            self._run(stub)  # second call: short-circuits before writer
        stub.opt.write_first_stage_solution.assert_not_called()


if __name__ == "__main__":
    unittest.main()
