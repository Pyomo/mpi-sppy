###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for checkpoint/resume (doc/designs/checkpointing_design.md), phase 1a.

The load-bearing test is the A/B harness: run A is an uninterrupted run of N
iterations; run B stops at k < N with a checkpoint, then resumes and continues
to N. On farmer -- a deterministic LP -- the two must agree bit-for-bit, which
is the strong "nothing was lost" check.

The rest pin the things that would make a resume quietly wrong rather than
loudly broken: that resume does not solve the fresh models at iteration 0
(which would throw away the checkpointed iterate and, for a large MIP, cost
hours), that a geometry or structural-option mismatch is refused instead of
producing nonsense, and that the initially-fixed-nonant baseline survives the
model swap -- without it a resumed run silently stops updating its best bound.
"""

import os
import tempfile
import unittest

import mpisppy.utils.checkpointing as checkpointing
import mpisppy.tests.examples.farmer as farmer
from mpisppy.extensions.checkpointer import Checkpointer
from mpisppy.opt.ph import PH
from mpisppy.tests.utils import get_solver
from mpisppy.utils.config import Config

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()

SCENARIO_NAMES = ["scen0", "scen1", "scen2"]
CREATOR_KWARGS = {"use_integer": False, "crops_multiplier": 1}


def _options(max_iters, ckpt_dir=None, resume_from=None, **overrides):
    options = {
        "solver_name": solver_name,
        "PHIterLimit": max_iters,
        "defaultPHrho": 1.0,
        # Never converge early: the A/B comparison needs a fixed iteration
        # count on both sides.
        "convthresh": -1.0,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "display_convergence_detail": False,
        "iter0_solver_options": None,
        "iterk_solver_options": None,
        "tee-rank0-solves": False,
        "smoothed": 0,
        "time_limit": None,
    }
    if ckpt_dir is not None:
        options["checkpoint_dir"] = ckpt_dir
        options["checkpoint_at_termination"] = True
        options["checkpoint_backend"] = checkpointing.DILL_RELOAD_BACKEND
    if resume_from is not None:
        options["resume_from"] = resume_from
    options.update(overrides)
    return options


def _make_ph(options, scenario_names=None):
    extensions = Checkpointer if "checkpoint_dir" in options else None
    return PH(
        options,
        scenario_names if scenario_names is not None else SCENARIO_NAMES,
        farmer.scenario_creator,
        farmer.scenario_denouement,
        scenario_creator_kwargs=CREATOR_KWARGS,
        extensions=extensions,
    )


def _primal_snapshot(ph):
    """W, rho and nonant values for every local scenario, keyed by name."""
    snap = {}
    for sname, s in ph.local_scenarios.items():
        for ndn_i, v in s._mpisppy_data.nonant_indices.items():
            snap[(sname, "x", v.name)] = v._value
            snap[(sname, "W", str(ndn_i))] = \
                float(s._mpisppy_model.W[ndn_i]._value)
            snap[(sname, "rho", str(ndn_i))] = \
                float(s._mpisppy_model.rho[ndn_i]._value)
    return snap


@unittest.skipIf(not solver_available,
                 "no solver is available for the A/B resume harness")
class TestResumeABFarmer(unittest.TestCase):
    """Uninterrupted vs stop-and-resume on a deterministic LP."""

    N = 6
    STOP = 3

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _run_b(self):
        """Stop at STOP with a checkpoint, then resume and finish."""
        stopped = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        stopped.ph_main()
        resumed = _make_ph(_options(self.N, resume_from=self.ckpt_dir))
        resumed.ph_main()
        return stopped, resumed

    def test_resume_is_bit_identical(self):
        reference = _make_ph(_options(self.N))
        reference.ph_main()
        _, resumed = self._run_b()

        want = _primal_snapshot(reference)
        got = _primal_snapshot(resumed)
        self.assertEqual(set(want), set(got))
        for key in want:
            self.assertEqual(
                want[key], got[key],
                msg=f"{key} differs after resume: {want[key]} vs {got[key]}")

    def test_iteration_numbering_is_global(self):
        """A resumed run continues the count instead of restarting at 1."""
        stopped, resumed = self._run_b()
        self.assertEqual(stopped._PHIter, self.STOP)
        self.assertTrue(resumed._resumed_from_checkpoint)
        self.assertEqual(resumed._resume_iteration, self.STOP)
        self.assertEqual(resumed._PHIter, self.N)

    def test_resume_performs_no_iter0_solve(self):
        """The whole point of the in-core branch: no throwaway W = 0 solve.

        For a large MIP that solve is the most expensive in the run -- cold,
        unregularized, no warm start -- and its answer is discarded.
        """
        _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir)).ph_main()

        resumed = _make_ph(_options(self.N, resume_from=self.ckpt_dir))
        calls = []
        original = resumed.solve_loop

        def counting_solve_loop(*args, **kwargs):
            calls.append(resumed._PHIter)
            return original(*args, **kwargs)

        resumed.solve_loop = counting_solve_loop
        resumed.ph_main()

        self.assertNotIn(
            0, calls,
            msg="resume solved the fresh models at iteration 0; the "
                "checkpointed iterate would have been discarded")
        self.assertEqual(calls, list(range(self.STOP + 1, self.N + 1)))

    def test_trivial_bound_is_restored_not_recomputed(self):
        """The trivial bound belongs to iteration 0 of the original run."""
        reference = _make_ph(_options(self.N))
        reference.ph_main()
        _, resumed = self._run_b()
        self.assertEqual(reference.trivial_bound, resumed.trivial_bound)

    def test_writes_one_generation_and_a_manifest(self):
        """Retention is exactly one published generation."""
        _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir)).ph_main()
        self.assertTrue(
            os.path.exists(os.path.join(self.ckpt_dir, "manifest.json")))
        generations = os.listdir(os.path.join(self.ckpt_dir, "hub"))
        self.assertEqual(generations, [f"gen_{self.STOP:04d}"])


@unittest.skipIf(not solver_available,
                 "no solver is available to write a checkpoint to refuse")
class TestResumeRefusesMismatch(unittest.TestCase):
    """A checkpoint that does not fit the current run must be refused."""

    STOP = 2

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir)).ph_main()

    def tearDown(self):
        self._tmp.cleanup()

    def test_structural_option_mismatch_is_refused(self):
        """Changing rho changes the meaning of the state in the checkpoint."""
        options = _options(4, resume_from=self.ckpt_dir, defaultPHrho=2.0)
        with self.assertRaises(checkpointing.CheckpointMismatch) as ctx:
            _make_ph(options).ph_main()
        self.assertIn("structural options", str(ctx.exception))

    def test_scenario_distribution_mismatch_is_refused(self):
        """Resuming with a different scenario set is refused, not guessed at."""
        options = _options(4, resume_from=self.ckpt_dir)
        with self.assertRaises(checkpointing.CheckpointMismatch) as ctx:
            _make_ph(options, scenario_names=["scen0", "scen1"]).ph_main()
        self.assertIn("scenario", str(ctx.exception).lower())

    def test_missing_manifest_is_refused_clearly(self):
        options = _options(4, resume_from=os.path.join(self._tmp.name, "nope"))
        with self.assertRaises(checkpointing.CheckpointMismatch) as ctx:
            _make_ph(options).ph_main()
        self.assertIn("manifest", str(ctx.exception))

    def test_iteration_limit_may_change_on_resume(self):
        """The limit and the clock are deliberately outside the fingerprint.

        Picking a run back up the next morning with a different budget is the
        primary use case, so these must not be treated as a mismatch.
        """
        options = _options(4, resume_from=self.ckpt_dir, time_limit=3600)
        resumed = _make_ph(options)
        resumed.ph_main()
        self.assertEqual(resumed._PHIter, 4)


class TestStructuralFingerprint(unittest.TestCase):
    """Which option changes block a resume, and which do not."""

    def _fingerprint(self, **options):
        base = {"defaultPHrho": 1.0, "linearize_proximal_terms": False}
        base.update(options)
        return checkpointing.structural_fingerprint(base)

    def test_identical_options_match(self):
        self.assertEqual(self._fingerprint(), self._fingerprint())

    def test_structural_change_is_detected(self):
        self.assertNotEqual(self._fingerprint(),
                            self._fingerprint(defaultPHrho=2.0))
        self.assertNotEqual(self._fingerprint(),
                            self._fingerprint(linearize_proximal_terms=True))

    def test_non_structural_change_is_ignored(self):
        """A named subset, so harmless flags do not block a resume."""
        self.assertEqual(
            self._fingerprint(),
            self._fingerprint(PHIterLimit=999, time_limit=60,
                              display_progress=True, verbose=True,
                              solver_name="some_other_solver"))

    def test_structural_cfg_extras_are_covered(self):
        """Settings PH never reads, but which reshape the model, still count."""
        with_cvar = self._fingerprint(
            checkpoint_structural_cfg={"cvar": True, "cvar_alpha": 0.95})
        without = self._fingerprint(
            checkpoint_structural_cfg={"cvar": False, "cvar_alpha": 0.95})
        self.assertNotEqual(with_cvar, without)


class TestConfigRegistration(unittest.TestCase):
    """Config.checkpoint_args registers the phase-1a flags with sane defaults."""

    def setUp(self):
        self.cfg = Config()
        self.cfg.checkpoint_args()

    def test_flags_are_registered(self):
        for name in ("checkpoint_dir", "checkpoint_at_termination",
                     "checkpoint_backend", "resume_from"):
            self.assertIn(name, self.cfg)

    def test_checkpointing_is_off_by_default(self):
        self.assertIsNone(self.cfg.checkpoint_dir)
        self.assertIsNone(self.cfg.resume_from)

    def test_terminal_checkpoint_defaults_on(self):
        self.assertTrue(self.cfg.checkpoint_at_termination)

    def test_backend_defaults_to_dill_reload(self):
        self.assertEqual(self.cfg.checkpoint_backend,
                         checkpointing.DILL_RELOAD_BACKEND)


class TestFilenameSanitizing(unittest.TestCase):
    """File names must never go through extract_num (not unique for ADMM)."""

    def test_wrapped_admm_names_stay_distinct(self):
        first = checkpointing.sanitize_for_filename(
            "ADMM_STOCH__ADMM__region1__ADMM__scen3")
        second = checkpointing.sanitize_for_filename(
            "ADMM_STOCH__ADMM__region2__ADMM__scen3")
        self.assertNotEqual(first, second)

    def test_path_separators_are_removed(self):
        self.assertNotIn("/", checkpointing.sanitize_for_filename("a/b c"))


class TestFixedNonantBaseline(unittest.TestCase):
    """The initially-fixed baseline must survive the model swap, by name.

    `_initial_fixed_varibles` is a ComponentSet of vardata, so a resume that
    replaces the scenario models invalidates it by identity. Both failure
    directions are pinned here: lose the baseline and the gate refuses to
    update the bound; rebuild it from the *current* fixedness and a nonant that
    a fixing extension pinned mid-run passes as original, admitting a bound the
    uninterrupted run would have refused.

    These call the gate directly. A plain PH hub is insulated in practice --
    `PHBase._can_update_best_bound` short-circuits whenever prox is enabled, and
    the one consultation with prox off is the iteration-0 trivial bound, which
    the resume branch replaces -- but `Subgradient` and `FWPH` consult the same
    baseline per iteration, so restoring it correctly is what keeps this from
    becoming a bug the moment resume covers them. See design section 9, item 11.
    """

    def setUp(self):
        # No solve, so no solver is needed -- this is pure bookkeeping.
        # PH_Prep attaches the W/prox parameters that the PHBase override of
        # _can_update_best_bound inspects before delegating to the fixedness
        # check; with the attach deferred, prox is off, which is the state the
        # gate is actually consulted in.
        self.ph = _make_ph(_options(1))
        self.ph.PH_Prep()
        scenario = next(iter(self.ph.local_scenarios.values()))
        self.nonant = next(iter(scenario._mpisppy_data.nonant_indices.values()))
        self.nonant.fix(self.nonant._value if self.nonant._value else 0.0)

    def test_baseline_by_name_allows_bound_updates(self):
        self.ph._restore_fixed_nonant_baseline([self.nonant.name])
        self.assertTrue(
            self.ph._can_update_best_bound(),
            msg="a nonant fixed before the run started must stay part of the "
                "baseline, or the resumed run stops updating its bound")

    def test_lost_baseline_would_block_bound_updates(self):
        """What an identity-keyed cache degrades to after a swap."""
        self.ph._restore_fixed_nonant_baseline([])
        self.assertFalse(self.ph._can_update_best_bound())

    def test_midrun_fixings_are_not_absorbed_into_the_baseline(self):
        """A nonant pinned after the start must not pass as original."""
        others = [v for s in self.ph.local_scenarios.values()
                  for v in s._mpisppy_data.nonant_indices.values()
                  if v is not self.nonant]
        midrun = others[0]
        midrun.fix(midrun._value if midrun._value else 0.0)

        # Only the original is in the checkpointed baseline.
        self.ph._restore_fixed_nonant_baseline([self.nonant.name])
        self.assertFalse(
            self.ph._can_update_best_bound(),
            msg="a mid-run fixing was treated as originally fixed, which "
                "would admit a bound the uninterrupted run would refuse")

    def test_rebuilt_baseline_holds_current_model_objects(self):
        """Rebuilt by name means the objects belong to the live models."""
        self.ph._restore_fixed_nonant_baseline([self.nonant.name])
        live = {id(v) for s in self.ph.local_scenarios.values()
                for v in s._mpisppy_data.nonant_indices.values()}
        for v in self.ph._initial_fixed_varibles:
            self.assertIn(id(v), live)


class TestUnknownBackend(unittest.TestCase):
    def test_require_dill_ignores_other_backends(self):
        # Only the dill-reload backend needs dill; nothing should raise here.
        checkpointing.require_dill(checkpointing.LEAF_BACKEND)


if __name__ == "__main__":
    unittest.main()
