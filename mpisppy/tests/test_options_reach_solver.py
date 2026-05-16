###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""End-to-end tests that verify solver_options_layers actually reach the
solver. Each test sets ONE option through a specific layer path (CLI
flag, JSON options file, etc.) via the Config / cfg_vanilla surface,
runs farmer PH for a couple of iterations against gurobi_persistent
with solver_log_dir pointing at a tmpdir, and then greps the
per-solve Gurobi log file for the ``Set parameter X to value Y`` line
that proves the parameter was actually applied.

Gurobi-only because it's the solver we install via pip in CI (the
size-limited trial license is sufficient for farmer) AND because the
gurobi log format is stable enough to grep cheaply. These tests are at
the mercy of upstream gurobi log-format changes; treat that as a
known cost.
"""

import copy
import json
import os
import re
import shutil
import tempfile
import unittest

import pyomo.environ as pyo

import mpisppy.opt.ph
import mpisppy.tests.examples.farmer as farmer
from mpisppy.utils import config
from mpisppy.utils.cfg_vanilla import (
    apply_solver_specs,
    shared_options,
)


SOLVER_NAME = "gurobi_persistent"


def setUpModule():
    # A stale GRB_LICENSE_FILE on the developer's machine (pointing at
    # a license bound to a different hostid) silently overrides the
    # pip-installed gurobipy trial license and causes
    # ``s.available()`` to raise. Pop it for this process so that
    # local runs use the trial license like CI does. No-op in CI
    # where the env var is unset.
    os.environ.pop("GRB_LICENSE_FILE", None)


# Module-level probe for gurobi_persistent. ``s.available()`` actually
# attempts to construct a Gurobi model, so this catches a stale or
# missing license too.
try:
    _gurobi_available = pyo.SolverFactory(SOLVER_NAME).available()
except Exception:
    _gurobi_available = False


def _bare_cfg():
    """Minimal Config with the flags shared_options reads."""
    cfg = config.Config()
    cfg.popular_args()
    cfg.add_mipgap_specs()
    cfg.add_solver_specs("")
    cfg.gapper_args()
    return cfg


def _spoke_cfg(spoke_name):
    cfg = _bare_cfg()
    cfg.add_solver_specs(prefix=spoke_name)
    cfg.add_mipgap_specs(prefix=spoke_name)
    cfg.gapper_args(name=spoke_name)
    return cfg


@unittest.skipUnless(
    _gurobi_available,
    "gurobi_persistent is required for these end-to-end tests",
)
class TestOptionsReachSolver(unittest.TestCase):
    """Common harness shared by every option-path test in this file."""

    def setUp(self):
        # Fresh tempdir per test; solver_log_dir must not exist before
        # the SPOpt constructor creates it.
        self._tmpdir = tempfile.mkdtemp(prefix="mpisppy_solver_log_")
        self.log_dir = os.path.join(self._tmpdir, "logs")
        self.scenario_names = [f"Scenario{i+1}" for i in range(3)]
        self.creator_kwargs = {"crops_multiplier": 1}

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ----- harness primitives -----

    def _options_from(self, cfg, *, iter_limit=2, extra=None):
        """Build the options dict shared_options would produce and
        layer the PH-only knobs (rho, iter limit, etc.) on top. Any
        keys in *extra* override what shared_options put there.
        """
        shoptions = shared_options(cfg, is_hub=True)
        shoptions["solver_name"] = SOLVER_NAME
        shoptions["solver_log_dir"] = self.log_dir
        shoptions["PHIterLimit"] = iter_limit
        shoptions["defaultPHrho"] = 1.0
        shoptions["convthresh"] = 0  # don't terminate on convergence
        shoptions["verbose"] = False
        shoptions["display_timing"] = False
        shoptions["display_progress"] = False
        shoptions["smoothed"] = 0
        shoptions["asynchronousPH"] = False
        shoptions["subsolvedirectives"] = None
        shoptions["toc"] = False
        if extra:
            shoptions.update(extra)
        return shoptions

    def _spoke_options_from(self, cfg, spoke_name, **kw):
        """Drive apply_solver_specs to fold per-spoke layers into a
        spoke dict's options, then return those options as if they
        were the hub's options. In-process PH uses them to verify
        that the per-spoke fold reaches the solver — the spoke's
        actual solve path is layered identically to the hub's.
        """
        sh = self._options_from(cfg, **kw)
        spoke = {"opt_kwargs": {"options": copy.deepcopy(sh)}}
        apply_solver_specs(spoke_name, spoke, cfg)
        return spoke["opt_kwargs"]["options"]

    def _run_ph(self, options):
        ph = mpisppy.opt.ph.PH(
            options,
            self.scenario_names,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=self.creator_kwargs,
        )
        ph.ph_main()

    def _log_path(self, iter_idx, scen_name="Scenario1"):
        # spopt.solve_one names log files
        # "{cylinder_name}_{scenario_name}_{idx}.log" where idx counts
        # the times that scenario has been solved (so it is 0 at PH
        # iteration 0, 1 at iteration 1, ...). Standalone PH's cylinder
        # name is "PH".
        return os.path.join(
            self.log_dir, f"PH_{scen_name}_{iter_idx}.log")

    def _log_text(self, iter_idx, scen_name="Scenario1"):
        with open(self._log_path(iter_idx, scen_name)) as f:
            return f.read()

    def _gurobi_param(self, log_text, param):
        """Pull the value Gurobi reported it set the given parameter
        to. Returns the value as a string (Gurobi prints e.g. "0.001"
        or "1e-05") or None if no such line is present (which Gurobi
        does when the parameter is left at its default).
        """
        m = re.search(rf"Set parameter {param} to value\s+(\S+)", log_text)
        return m.group(1) if m else None

    # ----- inline --solver-options -----

    def test_inline_mipgap_reaches_solver(self):
        cfg = _bare_cfg()
        cfg.solver_options = "mipgap=0.005"
        self._run_ph(self._options_from(cfg))
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "MIPGap"), "0.005")
        self.assertEqual(
            self._gurobi_param(self._log_text(1), "MIPGap"), "0.005")

    def test_inline_threads_translates_to_native_name(self):
        # mpi-sppy stores threads canonically; gurobi's native key is
        # "Threads" (capital T). The translation table in sputils
        # should rename it before reaching the solver.
        cfg = _bare_cfg()
        cfg.solver_options = "threads=2"
        self._run_ph(self._options_from(cfg))
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "Threads"), "2")

    def test_inline_presolve_passes_through_unchanged(self):
        cfg = _bare_cfg()
        cfg.solver_options = "presolve=2"
        self._run_ph(self._options_from(cfg))
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "Presolve"), "2")

    # ----- iter0 / iterk sugar flags -----

    def test_iter0_iterk_mipgap_apply_to_correct_iterations(self):
        cfg = _bare_cfg()
        cfg.iter0_mipgap = 0.001
        cfg.iterk_mipgap = 0.05
        self._run_ph(self._options_from(cfg))
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "MIPGap"), "0.001")
        self.assertEqual(
            self._gurobi_param(self._log_text(1), "MIPGap"), "0.05")

    # ----- --mipgaps-json (planned-deprecation path) -----

    def test_mipgaps_json_after_iter_layer_applies(self):
        # Per-iter schedule. Three iters: k=0 maps to default-layer
        # mipgap=0.1 (since N=0 is routed to `default` predicate);
        # k=2 onwards uses 0.001.
        import warnings
        path = os.path.join(self._tmpdir, "mipgaps.json")
        with open(path, "w") as f:
            json.dump({"0": 0.1, "2": 0.001}, f)
        cfg = _bare_cfg()
        cfg.mipgaps_json = path
        # add_gapper is what wires --mipgaps-json into layers; call it
        # the way the spoke / hub orchestrators do.
        from mpisppy.utils.cfg_vanilla import add_gapper
        sh = self._options_from(cfg, iter_limit=3)
        hub_dict = {"opt_kwargs": {"options": sh}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            add_gapper(hub_dict, cfg)
        self._run_ph(sh)
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "MIPGap"), "0.1")
        self.assertEqual(
            self._gurobi_param(self._log_text(1), "MIPGap"), "0.1")
        self.assertEqual(
            self._gurobi_param(self._log_text(2), "MIPGap"), "0.001")

    # ----- --solver-options-file: each sub-block -----

    def _write_file(self, payload):
        path = os.path.join(self._tmpdir, "opts.json")
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def test_options_file_default_subblock_applies_everywhere(self):
        cfg = _bare_cfg()
        cfg.solver_options_file = self._write_file(
            {"default": {"mipgap": 0.01}})
        self._run_ph(self._options_from(cfg))
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "MIPGap"), "0.01")
        self.assertEqual(
            self._gurobi_param(self._log_text(1), "MIPGap"), "0.01")

    def test_options_file_iter0_and_iterk_subblocks_fire_separately(self):
        # iter0 sub-block applies only at k=0; iterk applies only at
        # k>=1. Pair them with distinct values so each iter's value is
        # observable in the per-solve log. (Asserting that iterk has
        # NO MIPGap line when iter0 sets one is unreliable: gurobi
        # persists parameter state across solves on the same model
        # instance, and re-emits "Set parameter MIPGap" whenever
        # LogFile is changed for the next solve. The clean signal is
        # that the value differs between iters per the layer fold.)
        cfg = _bare_cfg()
        cfg.solver_options_file = self._write_file({
            "iter0": {"mipgap": 0.0001},
            "iterk": {"mipgap": 0.005},
        })
        self._run_ph(self._options_from(cfg))
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "MIPGap"), "0.0001")
        self.assertEqual(
            self._gurobi_param(self._log_text(1), "MIPGap"), "0.005")

    def test_options_file_starting_at_iter_fires_from_N_onward(self):
        cfg = _bare_cfg()
        cfg.solver_options_file = self._write_file({
            "default": {"mipgap": 0.01},
            "starting_at_iter": {"2": {"mipgap": 0.0001}},
        })
        self._run_ph(self._options_from(cfg, iter_limit=3))
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "MIPGap"), "0.01")
        self.assertEqual(
            self._gurobi_param(self._log_text(1), "MIPGap"), "0.01")
        self.assertEqual(
            self._gurobi_param(self._log_text(2), "MIPGap"), "0.0001")

    def test_options_file_below_cli_sugar_in_precedence(self):
        # Axis 2: --iter0-mipgap and --iterk-mipgap override the
        # file's iter0/iterk entries at the same predicate.
        cfg = _bare_cfg()
        cfg.solver_options_file = self._write_file({
            "iter0": {"mipgap": 0.5},
            "iterk": {"mipgap": 0.5},
        })
        cfg.iter0_mipgap = 0.001
        cfg.iterk_mipgap = 0.005
        self._run_ph(self._options_from(cfg))
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "MIPGap"), "0.001")
        self.assertEqual(
            self._gurobi_param(self._log_text(1), "MIPGap"), "0.005")

    # ----- per-spoke files (exercised via apply_solver_specs) -----

    def test_spokes_subblock_inside_global_file_reaches_spoke_solver(self):
        cfg = _spoke_cfg("lagrangian")
        cfg.solver_options_file = self._write_file({
            "spokes": {
                "lagrangian": {"default": {"mipgap": 0.005}},
            },
        })
        spoke_opts = self._spoke_options_from(cfg, "lagrangian")
        self._run_ph(spoke_opts)
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "MIPGap"), "0.005")

    def test_dedicated_per_spoke_file_reaches_spoke_solver(self):
        cfg = _spoke_cfg("lagrangian")
        cfg.lagrangian_solver_options_file = self._write_file(
            {"default": {"mipgap": 0.002}})
        spoke_opts = self._spoke_options_from(cfg, "lagrangian")
        self._run_ph(spoke_opts)
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "MIPGap"), "0.002")

    # ----- --max-solver-threads cap -----

    def test_max_solver_threads_wins_over_user_threads(self):
        # System-level cap must beat any user-supplied threads value.
        cfg = _bare_cfg()
        cfg.solver_options = "threads=8"
        cfg.max_solver_threads = 2
        self._run_ph(self._options_from(cfg))
        self.assertEqual(
            self._gurobi_param(self._log_text(0), "Threads"), "2")


if __name__ == "__main__":
    unittest.main()
