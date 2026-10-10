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
is the strong "nothing was lost" check. (The acceptance matrix, whose stop
generation is timing-dependent, allows last-bit solver noise; see the comment
at its final assertion.)

The rest pin the things that would make a resume quietly wrong rather than
loudly broken: that resume does not solve the fresh models at iteration 0
(which would throw away the checkpointed iterate and, for a large MIP, cost
hours), that a geometry or structural-option mismatch is refused instead of
producing nonsense, and that the initially-fixed-nonant baseline survives the
model swap -- without it a resumed run silently stops updating its best bound.
"""

import errno
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import mpisppy.utils.checkpointing as checkpointing
import mpisppy.tests.examples.farmer as farmer
from mpisppy.extensions.checkpointer import Checkpointer
from mpisppy.extensions.extension import Extension
from mpisppy.cylinders.hub import PHHub
from mpisppy.opt.ph import PH
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import REPO_ROOT, get_solver, subprocess_env
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
        options["checkpoint_backend"] = checkpointing.DILL_RELOAD_BACKEND
        options["checkpoint_every_iterations"] = 1
    if resume_from is not None:
        options["resume_from"] = resume_from
    options.update(overrides)
    return options


#: The iteration at which ClockRewinder trips the --time-limit break.
_LATE_STOP_ITERATION = 3


def _strip_leaf_key(ckpt_dir, key):
    """Delete a key from the published leaf file, standing in for a checkpoint
    written before that key existed."""
    generation = _published_generation(ckpt_dir)
    gen_dir = os.path.join(ckpt_dir, checkpointing.HUB_SUBDIR,
                           checkpointing._generation_dirname(generation))
    path = os.path.join(gen_dir, checkpointing._leaf_filename(0))
    with open(path, "rb") as f:
        leaf = pickle.load(f)
    del leaf[key]
    with open(path, "wb") as f:
        pickle.dump(leaf, f)


def _published_generation(ckpt_dir):
    """The generation the manifest names, or None if nothing was published."""
    path = os.path.join(ckpt_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["generation"]


class StateRecorder(Extension):
    """Record the primal state at exactly the points a checkpoint is written.

    Lets a test say what the checkpoint *should* contain without reaching into
    the Checkpointer, and without assuming the run's final state is the
    checkpointed one -- for an early-break exit it deliberately is not.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.latest = None
        self.latest_iteration = None

    def _record(self):
        self.latest = _primal_snapshot(self.opt)
        self.latest_iteration = int(getattr(self.opt, "_PHIter", 0))

    def post_iter0_after_sync(self):
        self._record()

    def enditer(self):
        self._record()

    # A test probe; nothing of its own to carry across a resume.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


class MidIterMutator(Extension):
    """Mutate model state in miditer, the way shipped extensions do.

    `miditer` runs before every early break in `iterk_loop`, and real
    extensions use it to change rho (the rho updaters), fix nonants (`fixer`,
    `relaxed_ph_fixer`) or relax domains (`integer_relax_then_enforce`). Any
    of those makes the pre-solve half of an iteration observable in the model.

    Using a purpose-built extension rather than a shipped one keeps this test
    about the *behavior* -- state moving mid-iteration -- instead of about a
    particular extension's option schema, and makes it deterministic.
    """

    def miditer(self):
        for s in self.opt.local_scenarios.values():
            for ndn_i in s._mpisppy_data.nonant_indices:
                s._mpisppy_model.rho[ndn_i]._value *= 1.05
            if self.opt._PHIter == 2:
                first = next(iter(s._mpisppy_data.nonant_indices.values()))
                first.fix(first._value)

    # A test probe; nothing of its own to carry across a resume.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


class EndIterMutator(Extension):
    """Mutate model state in enditer, which a user extension is free to do.

    Shipped extensions all keep enditer read-only, so this stands in for the
    one supplied with --user-defined-extensions. It compounds, so a checkpoint
    taken before this hook and one taken after are never equal.
    """

    def enditer(self):
        for s in self.opt.local_scenarios.values():
            for ndn_i in s._mpisppy_data.nonant_indices:
                s._mpisppy_model.rho[ndn_i]._value *= 1.05

    # A test probe; nothing of its own to carry across a resume.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass

class ClockRewinder(Extension):
    """Trip the --time-limit break at a chosen iteration, without a clock race.

    The matrix needs a cell that exits through `--time-limit` *after* some
    iterations have completed. Setting a small wall-clock limit and trusting
    farmer to be fast does not give that: the limit also covers model
    construction, solver startup and Iter0, so on a slow enough host the run
    trips it in iteration 1 and the cell silently becomes the "no checkpoint
    published" case -- inverting what it asserts.

    Rewinding `start_time` instead makes the elapsed time whatever this test
    wants it to be. `miditer` runs immediately before the time-limit check in
    the same iteration, so the break happens in `_LATE_STOP_ITERATION` with
    the iterations before it completed. The real check, allreduce and break
    all still run; only the clock is under control.
    """

    def miditer(self):
        if self.opt._PHIter == _LATE_STOP_ITERATION:
            self.opt.start_time -= (self.opt.options["time_limit"] + 1.0)

    # A test probe; nothing of its own to carry across a resume.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


class DeadlineApproacher(Extension):
    """Put the run one iteration away from a deadline, deterministically.

    `--checkpoint-before-seconds` reads two things: how much wall clock has
    gone, and what the last iteration cost. A test cannot wait out a real
    deadline, and racing one on farmer would make the trigger fire whenever
    the host happened to be slow. So this states both, in `enditer` -- which
    runs immediately before the checkpoint hook -- and the deadline arrives at
    a named iteration instead.

    It keeps stating them on every iteration from `ITERATION` on, so the
    condition stays true afterwards. That is what makes the latch visible: the
    trigger is entitled to fire again on every one of those iterations and
    must not.
    """

    #: The iteration at whose end the deadline comes into view.
    ITERATION = 2
    #: Declared elapsed time and iteration cost. Their sum clears
    #: BEFORE_SECONDS; neither does alone, so the test cannot pass on a
    #: trigger that ignored the iteration duration and just watched the clock.
    ELAPSED_SECONDS = 60.0
    ITERATION_SECONDS = 50.0

    def enditer(self):
        if self.opt._PHIter >= self.ITERATION:
            self.opt.start_time = time.perf_counter() - self.ELAPSED_SECONDS
            self.opt._last_iteration_seconds = self.ITERATION_SECONDS

    # A test probe; nothing of its own to carry across a resume.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


#: --checkpoint-before-seconds for the runs driven by DeadlineApproacher.
_BEFORE_SECONDS = 100.0


class IterationCostStamper(Extension):
    """Stamp an unmistakable iteration duration into the checkpoint.

    `enditer` runs after the solve and before the checkpoint hook, so what it
    stamps is what gets written; the loop overwrites it with the real
    measurement straight afterwards. That is what makes it a usable probe --
    a duration that could not possibly have been measured, sitting in the file
    and nowhere else.
    """

    STAMP = 12345.0

    def enditer(self):
        self.opt._last_iteration_seconds = self.STAMP

    # A test probe; nothing of its own to carry across a resume.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


class PreIter0Tagger(Extension):
    """Tag the models pre_iter0 sees, so a test can tell whether the hook ran
    on the models the run actually iterates or on fresh ones a resume splice
    discarded."""

    def pre_iter0(self):
        for s in self.opt.local_scenarios.values():
            s._pre_iter0_saw_this_model = True

    # A test probe; nothing of its own to carry across a resume.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


def _extension_class(name):
    if name is None:
        return None
    if name == "norm_rho":
        from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
        return NormRhoUpdater
    if name == "miditer_mutator":
        return MidIterMutator
    if name == "enditer_mutator":
        return EndIterMutator
    if name == "recorder":
        return StateRecorder
    if name == "coeff_rho":
        from mpisppy.extensions.coeff_rho import CoeffRho
        return CoeffRho
    if name == "pre_tagger":
        return PreIter0Tagger
    if name == "clock_rewinder":
        return ClockRewinder
    if name == "deadline_approacher":
        return DeadlineApproacher
    if name == "cost_stamper":
        return IterationCostStamper
    if name == "sep_rho":
        from mpisppy.extensions.sep_rho import SepRho
        return SepRho
    raise ValueError(name)


def _make_ph(options, scenario_names=None, extension_name=None,
             extra_names=()):
    classes = []
    if "checkpoint_dir" in options:
        classes.append(Checkpointer)
    for name in (extension_name,) + tuple(extra_names):
        cls = _extension_class(name)
        if cls is not None:
            classes.append(cls)

    extensions = None
    extension_kwargs = None
    if len(classes) == 1:
        extensions = classes[0]
    elif len(classes) > 1:
        from mpisppy.extensions.extension import MultiExtension
        extensions = MultiExtension
        extension_kwargs = {"ext_classes": classes}
    if extension_kwargs is not None:
        return PH(
            options,
            scenario_names if scenario_names is not None else SCENARIO_NAMES,
            farmer.scenario_creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=CREATOR_KWARGS,
            extensions=extensions,
            extension_kwargs=extension_kwargs,
        )
    return PH(
        options,
        scenario_names if scenario_names is not None else SCENARIO_NAMES,
        farmer.scenario_creator,
        farmer.scenario_denouement,
        scenario_creator_kwargs=CREATOR_KWARGS,
        extensions=extensions,
    )


def _flat_snapshot(ph):
    """_primal_snapshot with JSON-safe keys, for crossing a process boundary."""
    return {"|".join(str(part) for part in key): value
            for key, value in _primal_snapshot(ph).items()}


def _resume_and_dump(ckpt_dir, iters_this_run, out_path):
    """Resume in this process and write the final state to out_path.

    The entry point test_resume_survives_a_fresh_process runs in a subprocess;
    it lives at module scope because that subprocess imports this module by
    name to reach it.
    """
    ph = _make_ph(_options(iters_this_run, resume_from=ckpt_dir))
    ph.ph_main()
    with open(out_path, "w") as f:
        json.dump({
            # Matching state is not by itself evidence of a resume: farmer is
            # deterministic, so a run that ignored the checkpoint entirely and
            # did all N iterations from scratch lands on the same answer.
            "resumed": bool(getattr(ph, "_resumed_from_checkpoint", False)),
            "resume_iteration": int(getattr(ph, "_resume_iteration", 0)),
            "state": _flat_snapshot(ph),
        }, f)


def _find_recorder(ph):
    ext = ph.extobject
    for candidate in getattr(ext, "extdict", {}).values():
        if isinstance(candidate, StateRecorder):
            return candidate
    if isinstance(ext, StateRecorder):
        return ext
    raise AssertionError("no StateRecorder attached")


#: Per-nonant Params compared by _primal_snapshot. Beyond x/W/rho these cover
#: the smoothing state, which a rho-scaled smoothing run would otherwise never
#: check.
_SNAPSHOT_PARAMS = ("W", "rho", "xbars", "z", "p")


def _primal_snapshot(ph):
    """Nonant values, fixedness, and the iterate Params, keyed by name."""
    snap = {}
    for sname, s in ph.local_scenarios.items():
        for ndn_i, v in s._mpisppy_data.nonant_indices.items():
            snap[(sname, "x", v.name)] = v._value
            snap[(sname, "fixed", v.name)] = float(v.is_fixed())
            for pname in _SNAPSHOT_PARAMS:
                param = getattr(s._mpisppy_model, pname, None)
                if param is None:
                    continue
                snap[(sname, pname, str(ndn_i))] = float(param[ndn_i]._value)
    return snap


@unittest.skipIf(not solver_available,
                 "no solver is available for the A/B resume harness")
class TestResumeABFarmer(unittest.TestCase):
    """Uninterrupted vs stop-and-resume on a deterministic LP."""

    N = 6
    STOP = 3
    #: PHIterLimit (--max-iterations) bounds the run being started, not the
    #: study, so the resumed leg asks for the iterations that are left rather
    #: than for N. Only --stop-at-iteration-number counts absolutely.
    REMAINING = N - STOP

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _run_b(self):
        """Stop at STOP with a checkpoint, then resume and finish."""
        stopped = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        stopped.ph_main()
        resumed = _make_ph(_options(self.REMAINING, resume_from=self.ckpt_dir))
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

    def test_resume_survives_a_fresh_process(self):
        """The acceptance gate the design actually asks for (section 11.1).

        Every other A/B test here resumes in the interpreter that wrote the
        checkpoint, where module state, registrations and dill's own caches
        are already warm. The real use case is a new job on a new day, so the
        resumed leg runs in a subprocess that shares nothing with the writer
        but the files on disk.
        """
        _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir)).ph_main()

        out_path = os.path.join(self._tmp.name, "resumed_state.json")
        call_args = json.dumps([self.ckpt_dir, self.REMAINING, out_path])
        result = subprocess.run(
            [sys.executable, "-c",
             "import json, mpisppy.tests.test_checkpoint as t; "
             f"t._resume_and_dump(*json.loads({call_args!r}))"],
            capture_output=True, text=True, timeout=900, check=False,
            env=subprocess_env(),
        )
        self.assertEqual(
            result.returncode, 0,
            msg=f"the resumed run failed in a fresh process:\n{result.stderr}")

        with open(out_path) as f:
            payload = json.load(f)

        self.assertTrue(
            payload["resumed"],
            msg="the fresh process ran without resuming; matching state "
                "would then only show that farmer is deterministic")
        self.assertEqual(payload["resume_iteration"], self.STOP)
        got = payload["state"]

        reference = _make_ph(_options(self.N))
        reference.ph_main()
        want = _flat_snapshot(reference)

        self.assertEqual(set(want), set(got))
        for key in want:
            self.assertEqual(
                want[key], got[key],
                msg=f"{key} differs after resuming in a fresh process: "
                    f"{want[key]} vs {got[key]}")

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

        resumed = _make_ph(_options(self.REMAINING, resume_from=self.ckpt_dir))
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

    def test_early_break_exit_resumes_coherently(self):
        """The exit the planned-stop recipe actually takes.

        iterk_loop computes xbar, updates W, *may break* (user converger,
        convergence threshold, --time-limit), and only then solves. Exiting
        through one of those breaks leaves W at iteration k while the nonants
        are still at k-1. Checkpointing that state and resuming applies the
        dual update to the same iterate twice and skips a solve.

        The other A/B tests here set convthresh = -1.0 so the run never
        converges, which means they only ever exercise the clean
        iteration-limit exit. This one converges on purpose.
        """
        stopped = _make_ph(_options(20, ckpt_dir=self.ckpt_dir,
                                    convthresh=20.0))
        stopped.ph_main()
        self.assertLess(stopped._PHIter, 20,
                        msg="test did not exit through an early break")

        # An early break stops at an iteration this test does not know, so the
        # remaining work cannot be counted out as a per-run limit. The study
        # bound says where to end absolutely, whatever the checkpoint holds.
        resumed = _make_ph(_options(self.N, resume_from=self.ckpt_dir,
                                    stop_at_iteration_number=self.N))
        resumed.ph_main()
        self.assertEqual(resumed._PHIter, self.N)

        reference = _make_ph(_options(self.N))
        reference.ph_main()

        want = _primal_snapshot(reference)
        got = _primal_snapshot(resumed)
        for key in want:
            self.assertEqual(
                want[key], got[key],
                msg=f"{key} differs after resuming a run that exited through "
                    f"an early break: {want[key]} vs {got[key]}")

    def test_empty_resume_does_not_republish_as_generation_zero(self):
        """A resume with no iterations to do must not destroy state.

        The resumed loop has nothing to do, so _PHIter would otherwise stay at
        its initial 0, and a terminal write would publish generation 0 and
        delete the real checkpoint -- losing the run.
        """
        _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir)).ph_main()

        resumed = _make_ph(_options(0, resume_from=self.ckpt_dir,
                                    ckpt_dir=self.ckpt_dir))
        resumed.ph_main()
        self.assertEqual(resumed._PHIter, self.STOP)
        generations = os.listdir(os.path.join(self.ckpt_dir, "hub"))
        self.assertEqual(generations, [f"gen_{self.STOP:04d}"])

    def test_a_finished_study_does_not_republish_as_generation_zero(self):
        """The same guarantee by the other route to an empty loop: the study
        bound has already been reached, so there is nothing left to do."""
        _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir)).ph_main()

        resumed = _make_ph(_options(self.N, resume_from=self.ckpt_dir,
                                    ckpt_dir=self.ckpt_dir,
                                    stop_at_iteration_number=self.STOP))
        resumed.ph_main()
        self.assertEqual(resumed._PHIter, self.STOP)
        generations = os.listdir(os.path.join(self.ckpt_dir, "hub"))
        self.assertEqual(generations, [f"gen_{self.STOP:04d}"])

    def test_incumbent_objective_is_restored(self):
        """Otherwise it reads as None and any later xhat is accepted."""
        stopped = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        stopped.ph_main()
        stopped.best_solution_obj_val = -110000.0
        checkpointing.write_checkpoint(stopped, self.ckpt_dir, self.STOP)

        resumed = _make_ph(_options(self.REMAINING, resume_from=self.ckpt_dir))
        resumed.ph_main()
        self.assertEqual(resumed.best_solution_obj_val, -110000.0)
        # And the guarantee that value exists to provide: a worse candidate
        # must now be rejected rather than accepted over the checkpointed one.
        self.assertFalse(
            resumed.update_best_solution_if_improving(-100000.0),
            msg="a worse incumbent was accepted after a resume")

    def _spin_hub(self, options):
        """A PH hub with no spokes, so the run has a hub to keep bounds on."""
        extensions = Checkpointer if "checkpoint_dir" in options else None
        hub_dict = {
            "hub_class": PHHub,
            "hub_kwargs": {"options": {"rel_gap": -1, "abs_gap": -1,
                                       "max_stalled_iters": None}},
            "opt_class": PH,
            "opt_kwargs": {
                "options": options,
                "all_scenario_names": SCENARIO_NAMES,
                "scenario_creator": farmer.scenario_creator,
                "scenario_creator_kwargs": CREATOR_KWARGS,
                "scenario_denouement": farmer.scenario_denouement,
                "extensions": extensions,
            },
        }
        wheel = WheelSpinner(hub_dict, [])
        wheel.spin()
        return wheel.spcomm

    def test_outer_bound_from_a_spoke_is_restored(self):
        """A spoke's bound reaches only spcomm.BestOuterBound, not
        opt.best_bound_obj_val, so the checkpoint has to take it from there."""
        hub = self._spin_hub(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        # Better than the trivial bound, which is all PH computes by itself
        # on farmer; this stands in for what a Lagrangian spoke would send.
        spoke_bound = -110000.0
        self.assertLess(hub.opt.trivial_bound, spoke_bound)
        hub.BestOuterBound = spoke_bound
        checkpointing.write_checkpoint(hub.opt, self.ckpt_dir, self.STOP)

        resumed = self._spin_hub(_options(self.REMAINING,
                                          resume_from=self.ckpt_dir))
        self.assertTrue(resumed.opt._resumed_from_checkpoint)
        self.assertEqual(resumed.BestOuterBound, spoke_bound)

    def test_writes_one_generation_and_a_manifest(self):
        """Retention is exactly one published generation."""
        _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir)).ph_main()
        self.assertTrue(
            os.path.exists(os.path.join(self.ckpt_dir, "manifest.json")))
        generations = os.listdir(os.path.join(self.ckpt_dir, "hub"))
        self.assertEqual(generations, [f"gen_{self.STOP:04d}"])

    def test_sweep_reclaims_interrupted_write_artifacts(self):
        """Leftovers from a killed write must not accumulate.

        A write that dies partway leaves a `.incoming` or `.retiring`
        directory. They are what make the interrupted generation recoverable,
        and the next successful write is what reclaims them -- so if the sweep
        ever stopped matching them, a long-running job would grow a directory
        of dead generations without anything failing.
        """
        opt = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        opt.ph_main()
        hub = os.path.join(self.ckpt_dir, "hub")
        os.makedirs(os.path.join(hub, "gen_0001.incoming"), exist_ok=True)
        os.makedirs(os.path.join(hub, "gen_0002.retiring"), exist_ok=True)
        os.makedirs(os.path.join(hub, "gen_0009"), exist_ok=True)

        checkpointing.write_checkpoint(opt, self.ckpt_dir, self.STOP + 1)
        self.assertEqual(sorted(os.listdir(hub)),
                         [f"gen_{self.STOP + 1:04d}"])

    def test_interrupted_same_generation_write_is_still_loadable(self):
        """The retired copy is the manifest's generation, intact."""
        opt = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        opt.ph_main()
        hub = os.path.join(self.ckpt_dir, "hub")
        live = os.path.join(hub, f"gen_{self.STOP:04d}")
        # Exactly the state a kill between the two renames leaves behind.
        os.rename(live, f"{live}.retiring")

        leaf, models = checkpointing.load_checkpoint(opt, self.ckpt_dir)
        self.assertEqual(leaf["generation"], self.STOP)
        self.assertEqual(sorted(models), sorted(opt.local_scenarios))

    def test_retention_deletes_the_previous_generation(self):
        """Writing a second generation must remove the first, not accumulate."""
        opt = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        opt.ph_main()
        checkpointing.write_checkpoint(opt, self.ckpt_dir, self.STOP + 1)
        generations = sorted(os.listdir(os.path.join(self.ckpt_dir, "hub")))
        self.assertEqual(generations, [f"gen_{self.STOP + 1:04d}"])

    def _assert_failed_write_leaves_only_the_committed_generation(
            self, fail_in):
        """A write that fails at `fail_in` must leave nothing behind.

        On a full disk an orphaned generation is what makes the retry fail
        too, so it has to be reclaimed on the failure path, not by the sweep
        after a successful write.
        """
        opt = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        opt.ph_main()
        hub = os.path.join(self.ckpt_dir, "hub")
        enospc = OSError(errno.ENOSPC, "No space left on device")
        with mock.patch.object(checkpointing, fail_in, side_effect=enospc):
            # OSError from rank 0's publish; a failure while staging is
            # agreed across ranks and re-raised as RuntimeError.
            with self.assertRaises((OSError, RuntimeError)):
                checkpointing.write_checkpoint(opt, self.ckpt_dir,
                                               self.STOP + 1)
        self.assertEqual(sorted(os.listdir(hub)), [f"gen_{self.STOP:04d}"])
        leaf, _ = checkpointing.load_checkpoint(opt, self.ckpt_dir)
        self.assertEqual(leaf["generation"], self.STOP)

        checkpointing.write_checkpoint(opt, self.ckpt_dir, self.STOP + 2)
        self.assertEqual(sorted(os.listdir(hub)),
                         [f"gen_{self.STOP + 2:04d}"])

    def test_failure_after_the_models_leaves_no_staged_generation(self):
        # _fsync_dir first runs right after the leaf write, while the new
        # generation is still staged.
        self._assert_failed_write_leaves_only_the_committed_generation(
            "_fsync_dir")

    def test_failure_before_the_manifest_flip_leaves_no_published_orphan(self):
        # By _publish_manifest the new generation has been renamed into
        # place, but nothing names it.
        self._assert_failed_write_leaves_only_the_committed_generation(
            "_publish_manifest")

    def test_orphans_from_an_earlier_failure_are_reclaimed_before_staging(self):
        """The pre-write sweep, alone: leftovers must not block the retry."""
        opt = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        opt.ph_main()
        hub = os.path.join(self.ckpt_dir, "hub")
        os.makedirs(os.path.join(hub, f"gen_{self.STOP + 1:04d}.tmp"))
        os.makedirs(os.path.join(hub, f"gen_{self.STOP + 1:04d}"))
        seen = []
        real = checkpointing._write_models

        def spy(*args, **kwargs):
            seen.append(sorted(os.listdir(hub)))
            return real(*args, **kwargs)

        with mock.patch.object(checkpointing, "_write_models", spy):
            checkpointing.write_checkpoint(opt, self.ckpt_dir, self.STOP + 2)
        self.assertEqual(seen, [[f"gen_{self.STOP:04d}",
                                 f"gen_{self.STOP + 2:04d}.tmp"]])

    def test_failure_keeps_the_retired_copy_the_manifest_depends_on(self):
        """After a kill between the publishing renames, the manifest's
        generation exists only as `.retiring`; cleaning up a later failed
        write must not delete it."""
        opt = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        opt.ph_main()
        hub = os.path.join(self.ckpt_dir, "hub")
        live = os.path.join(hub, f"gen_{self.STOP:04d}")
        os.rename(live, f"{live}.retiring")
        enospc = OSError(errno.ENOSPC, "No space left on device")
        with mock.patch.object(checkpointing, "_fsync_dir",
                               side_effect=enospc):
            # OSError from rank 0's publish; a failure while staging is
            # agreed across ranks and re-raised as RuntimeError.
            with self.assertRaises((OSError, RuntimeError)):
                checkpointing.write_checkpoint(opt, self.ckpt_dir,
                                               self.STOP + 1)
        self.assertEqual(sorted(os.listdir(hub)),
                         [f"gen_{self.STOP:04d}.retiring"])
        leaf, _ = checkpointing.load_checkpoint(opt, self.ckpt_dir)
        self.assertEqual(leaf["generation"], self.STOP)


@unittest.skipIf(not solver_available,
                 "no solver is available to run the iteration bounds")
class TestIterationBounds(unittest.TestCase):
    """The two bounds, and which of them ends a run.

    --max-iterations counts the iterations of the run being started, so
    resuming with 2 does two more whatever number the checkpoint stopped at.
    --stop-at-iteration-number counts the study, as an absolute iteration
    number across every run linked by checkpoints. A run ends at whichever
    arrives first.
    """

    STOP = 3

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir)).ph_main()

    def tearDown(self):
        self._tmp.cleanup()

    def _resume(self, max_iters, **overrides):
        ph = _make_ph(_options(max_iters, resume_from=self.ckpt_dir,
                               **overrides))
        ph.ph_main()
        self.assertEqual(ph._resume_iteration, self.STOP)
        return ph

    def test_max_iterations_counts_this_run_not_the_study(self):
        """Two more iterations, not two in total: passing the study total is
        the mistake the old reading of this flag invited."""
        self.assertEqual(self._resume(2)._PHIter, self.STOP + 2)

    def test_the_study_bound_can_end_the_run_first(self):
        ph = self._resume(10, stop_at_iteration_number=self.STOP + 1)
        self.assertEqual(ph._PHIter, self.STOP + 1)

    def test_the_per_run_bound_can_end_the_run_first(self):
        ph = self._resume(1, stop_at_iteration_number=self.STOP + 99)
        self.assertEqual(ph._PHIter, self.STOP + 1)

    def test_a_study_bound_already_reached_runs_nothing(self):
        """The study is over; the run must not quietly do another iteration."""
        ph = self._resume(10, stop_at_iteration_number=self.STOP)
        self.assertEqual(ph._PHIter, self.STOP)

    def test_a_study_bound_behind_the_checkpoint_runs_nothing(self):
        ph = self._resume(10, stop_at_iteration_number=self.STOP - 1)
        self.assertEqual(ph._PHIter, self.STOP)

    def test_the_study_bound_is_not_structural(self):
        """Deciding tomorrow morning where the study should end is a budget
        change like any other, so it must not refuse the checkpoint."""
        base = {"defaultPHrho": 1.0}
        self.assertEqual(
            checkpointing.structural_fingerprint(
                {**base, "stop_at_iteration_number": None}),
            checkpointing.structural_fingerprint(
                {**base, "stop_at_iteration_number": 40}))
        self.assertTrue(
            checkpointing._is_non_structural("stop_at_iteration_number"))

    def test_the_last_iteration_is_written_when_the_study_bound_ends_it(self):
        """The always-write rule follows whichever bound stopped the run: off
        cadence, the study's final iterate is exactly the one worth keeping."""
        writes = []
        real = checkpointing.write_checkpoint

        def counting(opt, ckpt_dir, generation, backend):
            writes.append(generation)
            return real(opt, ckpt_dir, generation, backend=backend)

        with mock.patch.object(checkpointing, "write_checkpoint",
                               side_effect=counting):
            self._resume(10, ckpt_dir=self.ckpt_dir,
                         stop_at_iteration_number=self.STOP + 2,
                         checkpoint_every_iterations=5)
        self.assertEqual(writes, [self.STOP + 2])


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
        self.assertIn("describe a different problem", str(ctx.exception))

    def test_scenario_distribution_mismatch_is_refused(self):
        """Resuming with a different scenario set is refused, not guessed at."""
        options = _options(4, resume_from=self.ckpt_dir)
        with self.assertRaises(checkpointing.CheckpointMismatch) as ctx:
            _make_ph(options, scenario_names=["scen0", "scen1"]).ph_main()
        self.assertIn("scenario", str(ctx.exception).lower())

    def test_a_leaf_missing_a_key_the_resume_reads_is_refused_at_load(self):
        """Iter0 reads the bounds outside any agreement, just before a
        collective, so the load -- which runs inside one -- has to refuse a
        leaf without them rather than leave one rank to raise KeyError."""
        manifest = checkpointing._read_manifest(self.ckpt_dir)
        leaf_path = os.path.join(
            self.ckpt_dir, checkpointing.HUB_SUBDIR,
            checkpointing._generation_dirname(manifest["generation"]),
            checkpointing._leaf_filename(0))
        with open(leaf_path, "rb") as f:
            original = f.read()
        try:
            for key in checkpointing.LEAF_KEYS_READ_ON_RESUME:
                with self.subTest(key=key):
                    leaf = pickle.loads(original)
                    del leaf[key]
                    with open(leaf_path, "wb") as f:
                        pickle.dump(leaf, f)
                    resumed = _make_ph(_options(4, resume_from=self.ckpt_dir))
                    with self.assertRaises(
                            checkpointing.CheckpointMismatch) as ctx:
                        checkpointing.load_checkpoint(resumed, self.ckpt_dir)
                    self.assertIn(key, str(ctx.exception))
        finally:
            with open(leaf_path, "wb") as f:
                f.write(original)

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
        # The budget is this run's, so four more on top of the checkpoint.
        self.assertEqual(resumed._PHIter, self.STOP + 4)


@unittest.skipIf(not solver_available,
                 "no solver is available for the acceptance matrix")
class TestResumeAcceptanceMatrix(unittest.TestCase):
    """Every way a run can end, crossed with what can mutate state mid-loop.

    This is the gate for the feature, not a spot check. The mechanism writes
    only at iteration boundaries precisely so that none of these combinations
    needs special handling -- so if any cell fails, the invariant is broken and
    the answer is to fix the mechanism or refuse the configuration, not to add
    a case to it.

    Exit paths matter because `iterk_loop` can break *before* the solve (user
    converger, convergence threshold, --time-limit) or after it (iteration
    limit, cylinder convergence). Extensions matter because `miditer` runs
    before those breaks and can change rho, fix variables, or relax domains.
    """

    N = 6

    #: (label, option overrides, whether a checkpoint should be published)
    #:
    #: A checkpoint describes a completed PH iteration, so a run that ends
    #: before finishing iteration 1 publishes nothing -- there is no iterate to
    #: resume from. That is a guarantee worth pinning, not an accident.
    #: The fourth element is an extension the exit path needs, or None.
    EXITS = (
        ("iteration limit", {}, True, None),
        ("convergence", {"convthresh": 20.0}, True, None),
        # 0.0 trips on the first test, whatever the host: no clock control
        # needed, and no iteration completes.
        ("time limit, iteration 1", {"time_limit": 0.0}, False, None),
        ("time limit, later", {"time_limit": 3600.0}, True, "clock_rewinder"),
    )

    #: (label, option overrides, extension name or None, reproducible)
    #:
    #: `reproducible` says whether a resumed run can be expected to match an
    #: uninterrupted one. It cannot when a stateful extension is attached: the
    #: extension's own accumulated state is not part of the checkpoint (the
    #: design schedules that separately, as the Extension
    #: checkpoint_state/restore_state contract), so the resumed run's extension
    #: starts fresh and the trajectories legitimately part company. Those cells
    #: still assert the property this phase *does* guarantee -- that the
    #: checkpoint round-trips exactly -- which is what tests the mechanism.
    CONFIGS = (
        ("plain PH", {}, None, True),
        # rho != 1 on purpose: the smoothing rescale is p *= rho, which at
        # rho = 1 is the identity and would hide a double-application.
        ("smoothing", {"smoothed": 2, "defaultPHp": 0.1, "defaultPHbeta": 0.1,
                       "defaultPHrho": 2.0}, None, True),
        ("linearized prox", {"linearize_proximal_terms": True}, None, True),
        ("norm rho updater", {}, "norm_rho", False),
        ("miditer rho + fixing", {}, "miditer_mutator", False),
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def test_matrix(self):
        for exit_label, exit_opts, expect_ckpt, exit_ext in self.EXITS:
            for cfg_label, cfg_opts, ext, reproducible in self.CONFIGS:
                with self.subTest(exit=exit_label, config=cfg_label):
                    self._one_cell(exit_opts, cfg_opts, ext, reproducible,
                                   expect_ckpt, exit_ext)

    def _one_cell(self, exit_opts, cfg_opts, ext, reproducible, expect_ckpt,
                  exit_ext=None):
        shutil.rmtree(self.ckpt_dir, ignore_errors=True)

        stop_opts = _options(self.N, ckpt_dir=self.ckpt_dir, **cfg_opts)
        stop_opts.update(exit_opts)
        extras = ("recorder",) if exit_ext is None else ("recorder", exit_ext)
        stopped = _make_ph(stop_opts, extension_name=ext, extra_names=extras)
        stopped.ph_main()
        recorded = _find_recorder(stopped)

        if exit_ext == "clock_rewinder":
            # Without this the cell still passes when the rewinder never
            # fires -- the run just ends on the iteration limit instead, and
            # the matrix quietly stops covering the --time-limit exit.
            self.assertEqual(
                stopped._PHIter, _LATE_STOP_ITERATION,
                msg="the run did not exit through the --time-limit break")

        generation = _published_generation(self.ckpt_dir)
        if not expect_ckpt:
            self.assertIsNone(
                generation,
                msg="a checkpoint was published for a run that completed no "
                    "PH iteration; there is no coherent iterate to save")
            return
        self.assertIsNotNone(
            generation,
            msg="the run ended without publishing any checkpoint")

        # 1. The checkpoint must round-trip exactly. Resuming with a per-run
        #    budget of zero runs no iteration, so whatever state comes back is
        #    precisely what was written -- which is the mechanism under test,
        #    and holds whether or not any extension is stateful.
        rt_opts = _options(0, resume_from=self.ckpt_dir, **cfg_opts)
        roundtrip = _make_ph(rt_opts, extension_name=ext)
        roundtrip.ph_main()
        self.assertEqual(roundtrip._PHIter, generation)

        self.assertEqual(
            recorded.latest_iteration, generation,
            msg="the published generation is not the last completed iteration")
        got = _primal_snapshot(roundtrip)
        self.assertEqual(set(recorded.latest), set(got))
        worst = max((abs(recorded.latest[k] - got[k])
                     for k in recorded.latest), default=0.0)
        self.assertEqual(
            worst, 0.0,
            msg=f"the checkpoint did not round-trip: restored state differs "
                f"from what was written by {worst:.6e}")

        if not reproducible:
            return

        # 2. With no stateful extension in play, a resumed run must also be
        #    indistinguishable from one that was never interrupted.
        resume_opts = _options(self.N - generation, resume_from=self.ckpt_dir,
                               **cfg_opts)
        resumed = _make_ph(resume_opts, extension_name=ext)
        resumed.ph_main()

        ref_opts = _options(resumed._PHIter, **cfg_opts)
        reference = _make_ph(ref_opts, extension_name=ext)
        reference.ph_main()

        want = _primal_snapshot(reference)
        got = _primal_snapshot(resumed)
        self.assertEqual(set(want), set(got))
        worst = max((abs(want[k] - got[k]) for k in want), default=0.0)
        # Not exactly 0.0: a persistent solver that was set_instance'd fresh
        # (the resumed leg) can return a last-bit-different vertex than one
        # updated in place (the reference), measured at 8e-13 on farmer with
        # linearized prox resumed from generation 5. The broken-checkpoint
        # failures this cell exists to catch measured 37.8 and 330, so 1e-9
        # separates the two cleanly. The round-trip check above stays exact:
        # serialization has no solver in the loop.
        self.assertLessEqual(
            worst, 1e-9,
            msg=f"resumed run diverged from the uninterrupted reference by "
                f"{worst:.6e}; the checkpoint did not describe a completed "
                f"iteration")


class TestSetupRefusals(unittest.TestCase):
    """Unsupported configurations must fail at setup, not hours in."""

    def _stub(self, n_proc=1, backend=checkpointing.DILL_RELOAD_BACKEND):
        """A real PH object: the Checkpointer refuses anything else outright.

        Building one costs a scenario_creator call and no solve, which is
        cheap, and it keeps these tests honest -- a hand-rolled stub would sail
        past the hub-type check that exists precisely to reject non-PH hubs.
        """
        opt = _make_ph(_options(1))
        opt.options["checkpoint_dir"] = tempfile.mkdtemp()
        opt.options["checkpoint_backend"] = backend
        opt.n_proc = n_proc
        return opt

    def test_non_ph_hub_is_refused_at_setup(self):
        """APH inherits ph_hub's wiring but breaks the design's invariant.

        Its loop dispatches a fraction of the scenarios per pass, keeps its own
        hardcoded iteration range that no resume offset touches, and runs on a
        worker thread -- so a checkpoint written from it would not describe a
        completed iteration, and a resume would renumber from 1 and overwrite
        the checkpoint it resumed from.
        """
        import mpisppy.phbase

        # Must be a real PHBase subclass that is not PH: asserting against a
        # bare stub would also pass if the check were widened to PHBase, which
        # is the single most plausible future edit (to admit Subgradient or
        # FWPH) and would silently readmit APH.
        class NotPH(mpisppy.phbase.PHBase):
            def __init__(self):          # no scenarios needed for this check
                self.options = {
                    "checkpoint_dir": tempfile.mkdtemp(),
                    "checkpoint_backend": checkpointing.DILL_RELOAD_BACKEND,
                }
                self.n_proc = 1
                self.cylinder_rank = 0
                self.local_scenarios = {}

        self.assertNotIsInstance(NotPH(), PH)
        with self.assertRaises(RuntimeError) as ctx:
            Checkpointer(NotPH())
        self.assertIn("synchronous PH hub", str(ctx.exception))

    def test_non_ph_hub_resume_is_refused(self):
        """The read-side counterpart of the write-side hub-type refusal.

        A resume-only run never constructs a Checkpointer, so without this
        check `--APH --resume-from` would splice a PH checkpoint into APH --
        which attaches its own Params to the models the splice discards and
        iterates a hardcoded range(1, ...) that ignores the resume offset --
        with no error at startup.
        """
        import mpisppy.phbase

        class NotPH(mpisppy.phbase.PHBase):
            def __init__(self):        # the refusal fires before any load
                self.options = {"resume_from": tempfile.mkdtemp()}

        self.assertNotIsInstance(NotPH(), PH)
        with self.assertRaises(RuntimeError) as ctx:
            NotPH()._restore_from_checkpoint_if_resuming()
        self.assertIn("synchronous PH hub", str(ctx.exception))

    def test_colliding_scenario_filenames_are_refused_at_setup(self):
        stub = self._stub()
        model = next(iter(stub.local_scenarios.values()))
        stub.local_scenarios = {"scen 1": model, "scen_1": model}
        with self.assertRaises(RuntimeError) as ctx:
            Checkpointer(stub)
        self.assertIn("checkpoint file names", str(ctx.exception))

    def test_unimplemented_backend_is_refused_at_setup(self):
        with self.assertRaises(RuntimeError) as ctx:
            Checkpointer(self._stub(backend="leaf"))
        self.assertIn("not implemented", str(ctx.exception))

    def test_unimplemented_backend_is_refused_on_a_resume_only_run(self):
        """The read-side counterpart of the refusal above.

        A resume need not have a Checkpointer attached, so that refusal cannot
        be relied on, and add_checkpointing used to forward the backend only
        alongside --checkpoint-dir. `--resume-from ckpt --checkpoint-backend
        leaf` then went ahead on the manifest's backend with no error. Built
        through add_checkpointing so the forwarding is under test too, and
        indifferent to which of the two refusals fires first.
        """
        import mpisppy.utils.cfg_vanilla as vanilla

        cfg = Config()
        cfg.checkpoint_args()
        cfg.resume_from = tempfile.mkdtemp()
        cfg.checkpoint_backend = checkpointing.LEAF_BACKEND
        hub_dict = {"opt_kwargs": {"options": {}}}
        vanilla.add_checkpointing(hub_dict, cfg)

        with self.assertRaises(RuntimeError) as ctx:
            opt = _make_ph(_options(1, **hub_dict["opt_kwargs"]["options"]))
            opt._restore_from_checkpoint_if_resuming()
        self.assertIn("not implemented", str(ctx.exception))

    def test_multirank_is_accepted_at_setup(self):
        """Phase 2 removed the single-rank refusal.

        Only the setup gate is checked here -- a real multi-rank write needs
        real ranks, which `test_checkpoint_multirank.py` supplies under
        mpiexec. This is what keeps the refusal from creeping back in.
        """
        opt = self._stub(n_proc=2)
        # A two-rank cylinder agrees on its setup steps, so it needs a comm
        # that can; the single-rank fallback mpi-sppy uses without mpi4py
        # has no allgather.
        opt.mpicomm = _TwoRankComm()
        ckpt = Checkpointer(opt)
        self.assertTrue(ckpt.write_enabled)

    def test_unwritable_directory_is_refused_at_setup(self):
        stub = self._stub()
        stub.options["checkpoint_dir"] = os.path.join(
            tempfile.gettempdir(), "mpisppy_no_such_parent", "x", "y")
        os.makedirs(os.path.dirname(os.path.dirname(
            stub.options["checkpoint_dir"])), exist_ok=True)
        ro = os.path.dirname(stub.options["checkpoint_dir"])
        os.makedirs(ro, exist_ok=True)
        os.chmod(ro, 0o500)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                Checkpointer(stub)
            self.assertIn("Cannot write", str(ctx.exception))
        finally:
            os.chmod(ro, 0o700)


class TestDillabilityProbe(unittest.TestCase):
    """Setup refuses a run whose models cannot be checkpointed.

    The guarantee is that checkpointing either works or says so at startup.
    A run that got past setup would fail at every write, survive each failure
    by design, and finish hours later having published nothing.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _ph_with_unserializable_scenario(self, target):
        """farmer, with one scenario carrying something dill cannot handle.

        A generator is the payload because dill serializes most of the
        obvious candidates -- thread locks and open files included -- so they
        would not exercise the probe at all.
        """
        def creator(sname, **kwargs):
            model = farmer.scenario_creator(sname, **kwargs)
            if sname == target:
                model._unserializable = (i for i in range(3))
            return model

        return PH(
            _options(1, ckpt_dir=self.ckpt_dir), SCENARIO_NAMES, creator,
            farmer.scenario_denouement,
            scenario_creator_kwargs=CREATOR_KWARGS,
            extensions=Checkpointer,
        )

    def test_refuses_when_the_first_scenario_is_the_bad_one(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._ph_with_unserializable_scenario("scen0").ph_main()
        self.assertIn("no checkpoint could ever be written",
                      str(ctx.exception))

    def test_refuses_when_a_later_scenario_is_the_bad_one(self):
        """Probing only the first scenario waves this through.

        What makes a model undillable is usually something the
        scenario_creator closed over, and a creator that does it for one
        scenario is exactly the case a single-scenario probe misses.
        """
        with self.assertRaises(RuntimeError) as ctx:
            self._ph_with_unserializable_scenario("scen2").ph_main()
        message = str(ctx.exception)
        self.assertIn("no checkpoint could ever be written", message)
        self.assertIn("scen2", message)


class TestCheckpointingSurvivedGuard(unittest.TestCase):
    """A guard whose only job is to fire is exactly what silently stops firing.

    Several hub types never wire the Checkpointer, and configure_extensions
    rebuilds the hub's extension list from scratch, dropping whatever was
    attached earlier. Either way --checkpoint-dir used to produce a run that
    exited 0 having written nothing.
    """

    def _cfg(self, **overrides):
        cfg = Config()
        cfg.checkpoint_args()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def _hub_dict(self, extensions=None, ext_classes=None, options=None):
        opt_kwargs = {"options": options or {}}
        if extensions is not None:
            opt_kwargs["extensions"] = extensions
        if ext_classes is not None:
            opt_kwargs["extension_kwargs"] = {"ext_classes": ext_classes}
        return {"opt_kwargs": opt_kwargs}

    def test_refuses_when_the_extension_was_dropped(self):
        from mpisppy.generic.decomp import _check_checkpointing_survived
        with self.assertRaises(RuntimeError) as ctx:
            _check_checkpointing_survived(
                self._hub_dict(), self._cfg(checkpoint_dir="/tmp/x"))
        self.assertIn("not attached", str(ctx.exception))

    def test_accepts_when_the_extension_is_attached_directly(self):
        from mpisppy.generic.decomp import _check_checkpointing_survived
        _check_checkpointing_survived(
            self._hub_dict(extensions=Checkpointer),
            self._cfg(checkpoint_dir="/tmp/x"))

    def test_accepts_when_composed_under_multiextension(self):
        from mpisppy.generic.decomp import _check_checkpointing_survived
        from mpisppy.extensions.extension import MultiExtension
        _check_checkpointing_survived(
            self._hub_dict(extensions=MultiExtension,
                           ext_classes=[Checkpointer, StateRecorder]),
            self._cfg(checkpoint_dir="/tmp/x"))

    def test_refuses_resume_when_the_hub_ignores_it(self):
        from mpisppy.generic.decomp import _check_checkpointing_survived
        with self.assertRaises(RuntimeError) as ctx:
            _check_checkpointing_survived(
                self._hub_dict(), self._cfg(resume_from="/tmp/x"))
        self.assertIn("silently start from scratch", str(ctx.exception))

    def test_silent_when_checkpointing_was_not_requested(self):
        from mpisppy.generic.decomp import _check_checkpointing_survived
        _check_checkpointing_survived(self._hub_dict(), self._cfg())

    def test_judges_the_hub_dict_the_user_callback_hands_back(self):
        """hub_and_spoke_dict_callback runs late and may rewrite anything.

        A model's callback that rebuilds the hub's extension list drops the
        Checkpointer. Checked before the callback, the guard passes on a
        dictionary the wheel is never built from, and the run writes nothing.
        """
        from mpisppy.generic.decomp import do_decomp
        from mpisppy.generic.parsing import add_decomp_args

        cfg = Config()
        cfg.proper_bundle_config()
        cfg.pickle_scenarios_config()
        cfg.EF_base()
        add_decomp_args(cfg)
        cfg.quick_assign("solution_base_name", str, None)
        cfg.quick_assign("write_scenario_lp_mps_files_dir", str, None)
        cfg.quick_assign("module_name", str, None)
        cfg.quick_assign("num_scens", int, len(SCENARIO_NAMES))
        cfg.solver_name = solver_name or "unused"
        cfg.default_rho = 1.0
        cfg.checkpoint_dir = tempfile.mkdtemp()

        def drop_extensions(hub_dict, list_of_spoke_dict, cfg):
            hub_dict["opt_kwargs"]["extensions"] = None
            hub_dict["opt_kwargs"]["extension_kwargs"] = None

        class FarmerWithCallback:
            hub_and_spoke_dict_callback = staticmethod(drop_extensions)

            def __getattr__(self, name):
                return getattr(farmer, name)

        with mock.patch("mpisppy.generic.decomp.WheelSpinner") as wheel:
            with self.assertRaises(RuntimeError) as ctx:
                do_decomp(FarmerWithCallback(), cfg, farmer.scenario_creator,
                          CREATOR_KWARGS, farmer.scenario_denouement)
        self.assertIn("not attached", str(ctx.exception))
        wheel.assert_not_called()


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

    def test_setup_flags_do_not_block_a_resume(self):
        """How a run was configured is not what problem it is.

        out-of-the-box prints an equivalent command line and invites the user
        to reuse it; with these folded in, resuming from exactly that line
        raised CheckpointMismatch. ph_xfeas_spoke is the one spoke flag that
        was missing from the "which cylinders run" entry above it, and the
        per-cylinder gapper knobs are named by gapper_args as
        <name>_mipgaps_json / <name>_mipgap_ratio, which the suffix rule for
        solver settings did not match.
        """
        for key, value in (
                ("out_of_the_box", ""),
                ("out_of_the_box_minus", ""),
                ("out_of_the_box_plus", ""),
                ("inspect_only", True),
                ("ph_xfeas_spoke", True),
                ("lagrangian_mipgaps_json", "/tmp/gaps.json"),
                ("lagrangian_mipgap_ratio", 0.5),
                ("lagrangian_starting_mipgap", 0.1),
                # Spokes a custom driver can add or drop between legs; the
                # spoke files are named so that this is absorbed.
                ("lagranger", True),
                ("xhatlooper", True),
                ("xhatspecific", True),
                ("slammax", True),
                ("slammin", True),
        ):
            with self.subTest(key=key):
                self.assertTrue(
                    checkpointing._is_non_structural(key),
                    f"{key} would be folded into the fingerprint, so a resume "
                    f"differing only in {key} is refused")
                self.assertNotIn(key, self._folded_cfg(**{key: value}))

    def test_spoke_flags_that_change_the_hub_models_stay_structural(self):
        """Unlike the other spoke flags, each of these also attaches a hub
        extension that changes the hub's own models -- reduced_costs fixes
        variables, cross_scenario_cuts adds cuts -- and a resume without it
        would keep what it did."""
        for key in ("reduced_costs", "cross_scenario_cuts"):
            with self.subTest(key=key):
                self.assertFalse(checkpointing._is_non_structural(key))

    def _folded_cfg(self, **overrides):
        """What cfg_vanilla actually hands the fingerprint."""
        import mpisppy.utils.cfg_vanilla as vanilla
        from mpisppy.utils.config import Config

        cfg = Config()
        cfg.popular_args()
        cfg.checkpoint_args()
        cfg.checkpoint_dir = "/tmp/whatever"
        for key, value in overrides.items():
            cfg.quick_assign(key, type(value), value)
        hub_dict = {"opt_kwargs": {"options": {}}}
        vanilla.add_checkpointing(hub_dict, cfg)
        return hub_dict["opt_kwargs"]["options"]["checkpoint_structural_cfg"]

    def test_structural_cfg_extras_are_covered(self):
        """Settings PH never reads, but which reshape the model, still count."""
        with_cvar = self._fingerprint(
            checkpoint_structural_cfg={"cvar": True, "cvar_alpha": 0.95})
        without = self._fingerprint(
            checkpoint_structural_cfg={"cvar": False, "cvar_alpha": 0.95})
        self.assertNotEqual(with_cvar, without)

    def test_model_specific_options_are_covered_by_default(self):
        """The denylist inversion: an option the fingerprint never heard of.

        Options a model's own inparser_adder registers -- farmer's
        use_integer, say -- never appear in opt.options, so the previous
        allowlist missed them and a farmer LP checkpoint could be resumed as a
        MIP without complaint.
        """
        lp = self._fingerprint(
            checkpoint_structural_cfg={"module_name": "farmer",
                                       "use_integer": False})
        mip = self._fingerprint(
            checkpoint_structural_cfg={"module_name": "farmer",
                                       "use_integer": True})
        self.assertNotEqual(lp, mip)

    def test_denylisted_entries_are_excluded_from_the_cfg_fold(self):
        """Whatever cfg_vanilla omits must not affect the hash."""
        import mpisppy.utils.cfg_vanilla as vanilla
        from mpisppy.utils.config import Config

        def built(max_iters):
            cfg = Config()
            cfg.popular_args()
            cfg.checkpoint_args()
            cfg.max_iterations = max_iters
            cfg.checkpoint_dir = "/tmp/whatever"
            hub_dict = {"opt_kwargs": {"options": {}}}
            vanilla.add_checkpointing(hub_dict, cfg)
            return hub_dict["opt_kwargs"]["options"][
                "checkpoint_structural_cfg"]

        self.assertNotIn("max_iterations", built(10))
        self.assertNotIn("checkpoint_dir", built(10))
        self.assertNotIn("solver_options_file", built(10))
        self.assertNotIn("lagrangian_solver_name", built(10))
        # ... but structural entries must still be folded in.
        self.assertIn("default_rho", built(10))
        self.assertEqual(
            checkpointing.structural_fingerprint(
                {"checkpoint_structural_cfg": built(10)}),
            checkpointing.structural_fingerprint(
                {"checkpoint_structural_cfg": built(999)}))


class TestConfigRegistration(unittest.TestCase):
    """Config.checkpoint_args registers the phase-1a flags with sane defaults."""

    def setUp(self):
        self.cfg = Config()
        self.cfg.checkpoint_args()

    def test_flags_are_registered(self):
        for name in ("checkpoint_dir", "checkpoint_backend",
                     "checkpoint_every_iterations",
                     "checkpoint_before_seconds", "resume_from",
                     "stop_at_iteration_number"):
            self.assertIn(name, self.cfg)

    def test_cadence_defaults_to_every_iteration(self):
        self.assertEqual(self.cfg.checkpoint_every_iterations, 1)

    def test_checkpointing_is_off_by_default(self):
        self.assertIsNone(self.cfg.checkpoint_dir)
        self.assertIsNone(self.cfg.resume_from)
        self.assertIsNone(self.cfg.checkpoint_before_seconds)

        # The refusal of --checkpoint-before-seconds without a directory to
        # write to lives with the other Config.checker tests, in test_config.py.

    def test_obsolete_termination_flag_is_gone(self):
        """It described a trigger that no longer exists.

        Checkpoints are written at every completed iteration now, so there is
        no terminal checkpoint to enable or disable. Leaving the flag
        registered would have meant a documented option that silently did
        nothing -- which is how it was found.
        """
        self.assertNotIn("checkpoint_at_termination", self.cfg)

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

    def test_colliding_names_are_refused(self):
        """'scen 1' and 'scen_1' both sanitize to 'scen_1'; writing both would
        silently land in one file and a resume would restore one scenario's
        model for both."""
        with self.assertRaises(RuntimeError) as ctx:
            checkpointing.check_filename_collisions(["scen 1", "scen_1"])
        self.assertIn("scen_1", str(ctx.exception))

    def test_names_differing_only_in_case_are_refused(self):
        """On a case-insensitive filesystem (the macOS and Windows defaults)
        'Scenario1' and 'scenario1' are one file, so the second write would
        replace the first."""
        with self.assertRaises(RuntimeError) as ctx:
            checkpointing.check_filename_collisions(["Scenario1", "scenario1"])
        self.assertIn("case", str(ctx.exception))

    def test_distinct_names_pass(self):
        checkpointing.check_filename_collisions(SCENARIO_NAMES)


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
        self.sname, scenario = next(iter(self.ph.local_scenarios.items()))
        self.nonant = next(iter(scenario._mpisppy_data.nonant_indices.values()))
        self.nonant.fix(self.nonant._value if self.nonant._value else 0.0)

    def test_baseline_by_name_allows_bound_updates(self):
        self.ph._restore_fixed_nonant_baseline({self.sname: [self.nonant.name]})
        self.assertTrue(
            self.ph._can_update_best_bound(),
            msg="a nonant fixed before the run started must stay part of the "
                "baseline, or the resumed run stops updating its bound")

    def test_lost_baseline_would_block_bound_updates(self):
        """What an identity-keyed cache degrades to after a swap."""
        self.ph._restore_fixed_nonant_baseline({})
        self.assertFalse(self.ph._can_update_best_bound())

    def test_midrun_fixings_are_not_absorbed_into_the_baseline(self):
        """A nonant pinned after the start must not pass as original."""
        others = [v for s in self.ph.local_scenarios.values()
                  for v in s._mpisppy_data.nonant_indices.values()
                  if v is not self.nonant]
        midrun = others[0]
        midrun.fix(midrun._value if midrun._value else 0.0)

        # Only the original is in the checkpointed baseline.
        self.ph._restore_fixed_nonant_baseline({self.sname: [self.nonant.name]})
        self.assertFalse(
            self.ph._can_update_best_bound(),
            msg="a mid-run fixing was treated as originally fixed, which "
                "would admit a bound the uninterrupted run would refuse")

    def test_baseline_does_not_smear_across_scenarios(self):
        """Pyomo component names are not scenario-qualified: every scenario's
        first-stage x is literally named the same. The baseline must key by
        scenario, or one scenario's initial fixing would absorb the same-named
        nonant of every other scenario."""
        other_sname, other_scen = next(
            (k, s) for k, s in self.ph.local_scenarios.items()
            if k != self.sname)
        twin = next(v for v in other_scen._mpisppy_data.nonant_indices.values()
                    if v.name == self.nonant.name)
        self.assertIsNot(twin, self.nonant)
        twin.fix(twin._value if twin._value else 0.0)

        # The checkpoint recorded the fixing in self.sname only.
        self.ph._restore_fixed_nonant_baseline({self.sname: [self.nonant.name]})
        self.assertFalse(
            self.ph._can_update_best_bound(),
            msg="a nonant fixed in one scenario passed as originally fixed "
                "in another, which would admit a bound the uninterrupted run "
                "would refuse")

    def test_rebuilt_baseline_holds_current_model_objects(self):
        """Rebuilt by name means the objects belong to the live models."""
        self.ph._restore_fixed_nonant_baseline({self.sname: [self.nonant.name]})
        live = {id(v) for s in self.ph.local_scenarios.values()
                for v in s._mpisppy_data.nonant_indices.values()}
        for v in self.ph._initial_fixed_varibles:
            self.assertIn(id(v), live)


#: The resume side of the issue-#762 prox/solver capability checks -- they
#: must fire on the first solve of a resumed leg, not only at iteration 1 --
#: is covered in test_prox_solver_compat.py, next to the rest of those checks.


@unittest.skipIf(not solver_available,
                 "no solver is available for the write-cadence tests")
class TestCheckpointEveryIterations(unittest.TestCase):
    """--checkpoint-every-iterations K trades lost iterations for write cost.

    The cadence must not disturb what a checkpoint *means*: writes still land
    only on iteration boundaries, so whatever is published is still a
    completed iteration and still resumes exactly.
    """

    N = 6

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, max_iters, every, **overrides):
        opts = _options(max_iters, ckpt_dir=self.ckpt_dir, **overrides)
        opts["checkpoint_every_iterations"] = every
        ph = _make_ph(opts)
        ph.ph_main()
        return ph

    def test_writes_are_skipped_between_checkpoint_points(self):
        """K=3 over 6 iterations writes at 3 and 6, not six times."""
        writes = []
        real = checkpointing.write_checkpoint

        def counting(opt, ckpt_dir, generation, backend):
            writes.append(generation)
            return real(opt, ckpt_dir, generation, backend=backend)

        with mock.patch.object(checkpointing, "write_checkpoint",
                               side_effect=counting):
            self._run(self.N, 3)
        self.assertEqual(writes, [3, 6])

    def test_a_cadence_checkpoint_resumes(self):
        """What K changes is which generation is published; that generation
        must still resume into an indistinguishable continuation."""
        self._run(3, 3)
        self.assertEqual(_published_generation(self.ckpt_dir), 3)

        resumed = _make_ph(_options(self.N - 3, resume_from=self.ckpt_dir))
        resumed.ph_main()
        self.assertEqual(resumed._resume_iteration, 3)
        self.assertEqual(resumed._PHIter, self.N)

        reference = _make_ph(_options(self.N))
        reference.ph_main()
        want = _primal_snapshot(reference)
        got = _primal_snapshot(resumed)
        worst = max((abs(want[k] - got[k]) for k in want), default=0.0)
        self.assertLessEqual(worst, 1e-9)

    def test_final_iteration_is_written_even_off_cadence(self):
        """Iteration 5 is not a multiple of 3, but it exhausts the limit --
        and resuming with a raised limit is how a study gets extended."""
        writes = []
        real = checkpointing.write_checkpoint

        def counting(opt, ckpt_dir, generation, backend):
            writes.append(generation)
            return real(opt, ckpt_dir, generation, backend=backend)

        with mock.patch.object(checkpointing, "write_checkpoint",
                               side_effect=counting):
            self._run(5, 3, PHIterLimit=5)
        self.assertEqual(writes, [3, 5])
        self.assertEqual(_published_generation(self.ckpt_dir), 5)

    def test_default_writes_every_iteration(self):
        writes = []
        real = checkpointing.write_checkpoint

        def counting(opt, ckpt_dir, generation, backend):
            writes.append(generation)
            return real(opt, ckpt_dir, generation, backend=backend)

        with mock.patch.object(checkpointing, "write_checkpoint",
                               side_effect=counting):
            self._run(4, 1)
        self.assertEqual(writes, [1, 2, 3, 4])


class TestCheckpointCadenceDecision(unittest.TestCase):
    """`_should_write` in isolation: which completed iterations are
    checkpoint points, without depending on when a real run happens to stop.

    A stop that is not the iteration limit -- convergence, the user converger,
    --time-limit -- is decided in the *next* iteration's top half, so the
    iterations after the last checkpoint point are simply lost. That is the
    trade K buys, and it is stated here rather than inferred from a timed run.
    """

    def _checkpointer(self, every, phiter, limit):
        opt = _make_ph(_options(limit))
        opt.options["checkpoint_dir"] = tempfile.mkdtemp()
        opt.options["checkpoint_backend"] = checkpointing.DILL_RELOAD_BACKEND
        opt.options["checkpoint_every_iterations"] = every
        ext = Checkpointer(opt)
        opt._PHIter = phiter
        return ext

    def test_on_cadence_iteration_is_a_checkpoint_point(self):
        self.assertTrue(self._checkpointer(3, phiter=6, limit=100)._should_write())

    def test_off_cadence_iteration_is_skipped(self):
        for phiter in (4, 5, 7, 8):
            with self.subTest(phiter=phiter):
                self.assertFalse(
                    self._checkpointer(3, phiter=phiter, limit=100)._should_write())

    def test_final_iteration_is_a_checkpoint_point_off_cadence(self):
        self.assertTrue(self._checkpointer(3, phiter=100, limit=100)._should_write())

    def test_every_one_writes_at_every_iteration(self):
        for phiter in (1, 2, 3, 4, 5):
            with self.subTest(phiter=phiter):
                self.assertTrue(
                    self._checkpointer(1, phiter=phiter, limit=100)._should_write())

    def test_cadence_counts_absolute_iterations_across_a_resume(self):
        """The counter is global, so a resume does not shift the cadence --
        generations stay on the same multiples whether or not the run stopped.
        """
        ext = self._checkpointer(5, phiter=10, limit=100)
        ext.opt._resume_iteration = 7
        self.assertTrue(ext._should_write())
        ext.opt._PHIter = 12
        self.assertFalse(ext._should_write())


class TestCheckpointEveryIterationsValidation(unittest.TestCase):
    def test_zero_is_refused_at_setup(self):
        opt = _make_ph(_options(1))
        opt.options["checkpoint_dir"] = tempfile.mkdtemp()
        opt.options["checkpoint_backend"] = checkpointing.DILL_RELOAD_BACKEND
        opt.options["checkpoint_every_iterations"] = 0
        with self.assertRaises(RuntimeError) as ctx:
            Checkpointer(opt)
        self.assertIn("at least 1", str(ctx.exception))

    def test_cadence_is_not_structural(self):
        """Changing K must not block a resume: it is a cadence knob, like the
        iteration limit, not a description of the problem."""
        base = {"defaultPHrho": 1.0}
        self.assertEqual(
            checkpointing.structural_fingerprint(
                {**base, "checkpoint_every_iterations": 1}),
            checkpointing.structural_fingerprint(
                {**base, "checkpoint_every_iterations": 25}))
        self.assertTrue(
            checkpointing._is_non_structural("checkpoint_every_iterations"))


class TestCheckpointBeforeSecondsDecision(unittest.TestCase):
    """`--checkpoint-before-seconds S` in isolation.

    The trigger asks whether *another* iteration would carry the run past S
    seconds of elapsed wall clock, so both of its inputs are set here directly:
    a real elapsed time cannot be waited for, and a real farmer iteration is
    too fast to be worth predicting.
    """

    def _checkpointer(self, before_seconds, every=100, phiter=3, limit=100):
        opt = _make_ph(_options(limit))
        opt.options["checkpoint_dir"] = tempfile.mkdtemp()
        opt.options["checkpoint_backend"] = checkpointing.DILL_RELOAD_BACKEND
        opt.options["checkpoint_every_iterations"] = every
        opt.options["checkpoint_before_seconds"] = before_seconds
        ext = Checkpointer(opt)
        opt._PHIter = phiter
        return ext

    def _clock(self, ext, elapsed, last_iteration=10.0):
        ext.opt.start_time = time.perf_counter() - elapsed
        ext.opt._last_iteration_seconds = last_iteration

    def test_fires_before_the_deadline_not_at_it(self):
        """80 seconds gone of 100 is not a stop -- but the next iteration
        costs 30, so this is the last boundary before the deadline."""
        ext = self._checkpointer(100.0)
        self._clock(ext, elapsed=80.0, last_iteration=30.0)
        self.assertTrue(ext._should_write())

    def test_silent_while_another_iteration_still_fits(self):
        ext = self._checkpointer(100.0)
        self._clock(ext, elapsed=80.0, last_iteration=10.0)
        self.assertFalse(ext._should_write())

    def test_fires_at_most_once(self):
        """Past the deadline every later iteration also qualifies. Writing at
        each of them is the per-iteration cost K was set to avoid, at the
        point in the run where the user has said time is short."""
        ext = self._checkpointer(100.0)
        self._clock(ext, elapsed=200.0, last_iteration=30.0)
        self.assertTrue(ext._should_write())
        ext.opt._PHIter += 1
        self.assertFalse(ext._should_write())
        ext.opt._PHIter += 1
        self.assertFalse(ext._should_write())

    def test_off_when_no_deadline_was_given(self):
        ext = self._checkpointer(None)
        self._clock(ext, elapsed=1e6, last_iteration=1e6)
        self.assertFalse(ext._should_write())

    def test_an_unmeasured_iteration_counts_as_zero(self):
        """Nothing has been timed yet, so there is nothing to predict with and
        the test degenerates to the elapsed clock -- it must not raise."""
        ext = self._checkpointer(100.0)
        ext.opt.start_time = time.perf_counter() - 80.0
        ext.opt._last_iteration_seconds = None
        self.assertFalse(ext._should_write())
        ext.opt.start_time = time.perf_counter() - 120.0
        self.assertTrue(ext._should_write())

    def test_the_cadence_still_writes_after_the_deadline_fired(self):
        """The latch is on the deadline trigger alone; K keeps its cadence."""
        ext = self._checkpointer(100.0, every=2, phiter=3)
        self._clock(ext, elapsed=200.0)
        self.assertTrue(ext._should_write())
        ext.opt._PHIter = 4
        self.assertTrue(ext._should_write())

    def test_a_cadence_write_near_the_deadline_is_the_deadline_write(self):
        """A multiple of K that lands near the deadline writes anyway, and
        that write is the one the deadline asked for: writing again at the
        next iteration spends a serialization just when time is short."""
        ext = self._checkpointer(100.0, every=10, phiter=30)
        self._clock(ext, elapsed=95.0, last_iteration=10.0)
        self.assertTrue(ext._should_write())
        ext.opt._PHIter = 31
        self.assertFalse(ext._should_write())

    def test_every_iteration_reaches_the_collective(self):
        """Rank safety: the all-reduce must be reached by every rank or by
        none. It is asked at every completed iteration, so nothing that
        could differ between ranks decides whether it is reached."""
        for phiter in (3, 4):
            with self.subTest(phiter=phiter):
                ext = self._checkpointer(100.0, every=2, phiter=phiter)
                self._clock(ext, elapsed=1.0, last_iteration=1.0)
                with mock.patch.object(ext.opt, "allreduce_or",
                                       return_value=False) as reduce:
                    ext._should_write()
                reduce.assert_called_once()

    def test_the_ranks_decide_together(self):
        """A rank that wrote on its own local clock would hang the cylinder at
        the write barrier, so the all-reduced answer is the only one that
        counts -- including when it overrules this rank."""
        ext = self._checkpointer(100.0)
        self._clock(ext, elapsed=200.0, last_iteration=30.0)
        with mock.patch.object(ext.opt, "allreduce_or", return_value=False):
            self.assertFalse(ext._should_write())
        # ... and having not fired, it is not latched either.
        self.assertTrue(ext._should_write())

    def test_the_failed_write_warning_names_what_retries(self):
        """A user deciding whether to stop a job needs to know whether a
        write is still coming, so the warning says which one, and says so
        when none is."""
        last = self._checkpointer(100.0, phiter=100, limit=100)
        self.assertIn("no later write will try again", last._what_retries())

        pending = self._checkpointer(100.0, phiter=5)
        self.assertIn("the --checkpoint-before-seconds write,",
                      pending._what_retries())
        self.assertNotIn("already fired", pending._what_retries())

        fired = self._checkpointer(100.0, phiter=5)
        fired._before_seconds_fired = True
        self.assertIn("already fired and does not retry",
                      fired._what_retries())
        self.assertNotIn("the --checkpoint-before-seconds write,",
                         fired._what_retries())

        unset = self._checkpointer(None, phiter=5)
        self.assertIn("will try again", unset._what_retries())
        self.assertNotIn("--checkpoint-before-seconds",
                         unset._what_retries())

        # A stop on convergence, the gap or --time-limit is not knowable at
        # the hook, so no mid-run promise is unconditional.
        for ext in (pending, fired, unset):
            self.assertIn("if the run gets that far", ext._what_retries())

    def _failed_write_warning(self, ext):
        """Drive a failing write through maybe_checkpoint, the path a user
        sees, and return the warning it printed."""
        enospc = OSError(errno.ENOSPC, "No space left on device")
        with mock.patch.object(checkpointing, "write_checkpoint",
                               side_effect=enospc), \
             mock.patch("mpisppy.extensions.checkpointer.global_toc") as toc:
            ext.maybe_checkpoint()
        warnings = [c.args[0] for c in toc.call_args_list
                    if c.args[0].startswith("WARNING: checkpoint write failed")]
        self.assertEqual(len(warnings), 1, toc.call_args_list)
        return warnings[0]

    def test_the_printed_warning_names_what_retries(self):
        """The warning the user reads is the one maybe_checkpoint prints, so
        each case is checked there, not only in the helper."""
        # A K write fails while the deadline write is still to come.
        pending = self._checkpointer(100.0, every=5, phiter=5)
        self._clock(pending, elapsed=1.0, last_iteration=1.0)
        warning = self._failed_write_warning(pending)
        self.assertIn("the --checkpoint-before-seconds write,", warning)
        self.assertIn("if the run gets that far", warning)

        # The failed write is itself the deadline write: the trigger set the
        # latch on the way in, so the warning must not offer it as a retry.
        deadline = self._checkpointer(100.0, every=100, phiter=5)
        self._clock(deadline, elapsed=80.0, last_iteration=30.0)
        warning = self._failed_write_warning(deadline)
        self.assertTrue(deadline._before_seconds_fired)
        self.assertIn("already fired and does not retry", warning)
        self.assertNotIn("the --checkpoint-before-seconds write,", warning)

        # The last iteration of the limit has nothing after it.
        last = self._checkpointer(100.0, every=100, phiter=100, limit=100)
        self._clock(last, elapsed=1.0, last_iteration=1.0)
        warning = self._failed_write_warning(last)
        self.assertIn("no later write will try again", warning)

    def test_a_nonpositive_or_non_finite_deadline_is_refused_at_setup(self):
        """NaN and +inf pass "<= 0" and would make the deadline test false
        on every iteration, switching the safeguard off without a word."""
        for bad in (0.0, -5.0, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError) as ctx:
                    self._checkpointer(bad)
                self.assertIn("must be a finite positive number",
                              str(ctx.exception))

    def test_a_non_finite_iteration_duration_is_ignored(self):
        """It is read back from a checkpoint on a resume; a NaN there would
        make the deadline never fire. It is treated as unknown instead."""
        ext = self._checkpointer(100.0)
        self._clock(ext, elapsed=100.5, last_iteration=float("nan"))
        self.assertTrue(ext._should_write())

    def test_the_deadline_is_not_structural(self):
        """A resume may set a different deadline -- the second leg of a study
        usually gets a different slot -- and must not be refused for it."""
        base = {"defaultPHrho": 1.0}
        self.assertEqual(
            checkpointing.structural_fingerprint(
                {**base, "checkpoint_before_seconds": None}),
            checkpointing.structural_fingerprint(
                {**base, "checkpoint_before_seconds": 3500.0}))
        self.assertTrue(
            checkpointing._is_non_structural("checkpoint_before_seconds"))


@unittest.skipIf(not solver_available,
                 "no solver is available to run PH to a deadline")
class TestCheckpointBeforeSeconds(unittest.TestCase):
    """The deadline trigger in a run: it exists to close the gap K opens.

    At K > 1 a run that stops against a wall clock rather than at its
    iteration limit stops at an iteration that is not a multiple of K, and the
    newest checkpoint is up to K-1 iterations old. If K is larger than the
    number of iterations the run completes, there is no checkpoint at all --
    which is the case these tests are built around, because it is the one that
    loses everything rather than a little.
    """

    N = 4
    #: Larger than any iteration these runs reach, so nothing is a checkpoint
    #: point on cadence and every write seen is the deadline's or the final
    #: iteration's.
    K = 100

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, max_iters, before_seconds=_BEFORE_SECONDS,
             extra_names=("deadline_approacher",), **overrides):
        opts = _options(max_iters, ckpt_dir=self.ckpt_dir, **overrides)
        opts["checkpoint_every_iterations"] = self.K
        if before_seconds is not None:
            opts["checkpoint_before_seconds"] = before_seconds
        ph = _make_ph(opts, extra_names=extra_names)
        writes = []
        real = checkpointing.write_checkpoint

        def counting(opt, ckpt_dir, generation, backend):
            writes.append(generation)
            return real(opt, ckpt_dir, generation, backend=backend)

        with mock.patch.object(checkpointing, "write_checkpoint",
                               side_effect=counting):
            ph.ph_main()
        return ph, writes

    def test_writes_at_the_last_boundary_before_the_deadline(self):
        """Iteration 2 is not a multiple of K and is not the last, so the
        deadline is the only reason it is written -- and iterations 3 and 4
        are equally close to it, so only the latch keeps 3 out."""
        _, writes = self._run(self.N)
        self.assertEqual(writes, [DeadlineApproacher.ITERATION, self.N])

    def test_without_the_deadline_only_the_final_iteration_is_written(self):
        """The same run, minus the option: what the deadline write adds."""
        _, writes = self._run(self.N, before_seconds=None)
        self.assertEqual(writes, [self.N])

    def test_a_run_stopped_by_the_time_limit_has_a_checkpoint(self):
        """The whole point, end to end. --time-limit stops the run at an
        iteration that is not a multiple of K and is not the limit, so nothing
        else in the design writes anything at all."""
        _, writes = self._run(
            self.N, extra_names=("deadline_approacher", "clock_rewinder"),
            time_limit=3600)
        self.assertEqual(writes, [DeadlineApproacher.ITERATION])
        self.assertEqual(_published_generation(self.ckpt_dir),
                         DeadlineApproacher.ITERATION)

    def test_without_the_deadline_that_run_checkpoints_nothing(self):
        """The gap, stated: the run stops cleanly, the directory exists, and
        there is no checkpoint in it."""
        _, writes = self._run(
            self.N, before_seconds=None,
            extra_names=("deadline_approacher", "clock_rewinder"),
            time_limit=3600)
        self.assertEqual(writes, [])
        self.assertIsNone(_published_generation(self.ckpt_dir))

    def test_the_deadline_checkpoint_resumes(self):
        """A write earned by a deadline is a write like any other: it lands on
        an iteration boundary, so it must resume into a continuation that is
        indistinguishable from the uninterrupted run."""
        total = 6
        self._run(self.N, extra_names=("deadline_approacher",
                                       "clock_rewinder"),
                  time_limit=3600)
        self.assertEqual(_published_generation(self.ckpt_dir),
                         DeadlineApproacher.ITERATION)

        resumed = _make_ph(_options(total - DeadlineApproacher.ITERATION,
                                    resume_from=self.ckpt_dir))
        resumed.ph_main()
        self.assertEqual(resumed._resume_iteration,
                         DeadlineApproacher.ITERATION)
        self.assertEqual(resumed._PHIter, total)

        reference = _make_ph(_options(total))
        reference.ph_main()
        want = _primal_snapshot(reference)
        got = _primal_snapshot(resumed)
        worst = max((abs(want[k] - got[k]) for k in want), default=0.0)
        self.assertLessEqual(worst, 1e-9)


@unittest.skipIf(not solver_available,
                 "no solver is available to time a PH iteration")
class TestIterationDurationIsRecorded(unittest.TestCase):
    """`PHBase._last_iteration_seconds`: the deadline trigger's estimate.

    It is the plain measured duration of the most recent completed iteration.
    mpi-sppy does not pad it and does not add anything for the write it may
    trigger; the write's own cost is bracketed by `toc` in the log, and
    leaving room for it is the user's to do when choosing S.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def test_iteration_zero_seeds_it(self):
        """The first time the trigger is tested, no iteration of the loop has
        finished yet, so iteration 0 is what there is to go on."""
        ph = _make_ph(_options(0, PHIterLimit=0))
        ph.ph_main()
        self.assertIsNotNone(ph._last_iteration_seconds)
        self.assertGreater(ph._last_iteration_seconds, 0.0)

    def test_it_tracks_the_iterations(self):
        ph = _make_ph(_options(3))
        ph.ph_main()
        self.assertIsNotNone(ph._last_iteration_seconds)
        self.assertGreater(ph._last_iteration_seconds, 0.0)

    def test_it_is_carried_across_a_resume(self):
        """A resumed run's own iteration 0 reloads models instead of solving
        them, so timing it describes a reload and not a PH iteration. The
        checkpoint carries a real measurement, and that is the seed."""
        ph = _make_ph(_options(2, ckpt_dir=self.ckpt_dir),
                      extra_names=("cost_stamper",))
        ph.ph_main()
        self.assertEqual(_published_generation(self.ckpt_dir), 2)

        # Resuming with a per-run budget of zero runs no iterations at all,
        # so what is read here is the seed and nothing else.
        resumed = _make_ph(_options(0, resume_from=self.ckpt_dir))
        resumed.ph_main()
        self.assertEqual(resumed._PHIter, 2)
        self.assertEqual(resumed._last_iteration_seconds,
                         IterationCostStamper.STAMP)

    def test_a_checkpoint_without_one_falls_back_to_iteration_zero(self):
        """Checkpoints written before the duration was carried have no such
        key; a resume from one seeds itself and does not fail."""
        ph = _make_ph(_options(2, ckpt_dir=self.ckpt_dir))
        ph.ph_main()
        _strip_leaf_key(self.ckpt_dir, "last_iteration_seconds")

        # Again with nothing to do, so what is checked is the seed itself and
        # not a duration a resumed iteration happened to measure.
        resumed = _make_ph(_options(0, resume_from=self.ckpt_dir))
        resumed.ph_main()
        self.assertIsNotNone(resumed._last_iteration_seconds)
        self.assertGreater(resumed._last_iteration_seconds, 0.0)


class TestConfigureExtensionsComposes(unittest.TestCase):
    """configure_extensions must compose with what is already attached.

    It used to rebuild the extension list from scratch whenever one of its
    own flags was set, silently dropping anything attached earlier -- the
    Checkpointer included, which turned --checkpoint-dir plus --wtracker into
    a startup refusal (or, off the guarded path, a run that wrote nothing).
    """

    def test_checkpointer_survives_ext_class_flags(self):
        from mpisppy.generic.parsing import add_decomp_args
        from mpisppy.generic.extensions import configure_extensions
        from mpisppy.generic.decomp import _check_checkpointing_survived
        from mpisppy.extensions.extension import MultiExtension
        from mpisppy.extensions.wtracker_extension import Wtracker_extension

        cfg = Config()
        add_decomp_args(cfg)
        # Registered by parsing outside add_decomp_args; configure_extensions
        # reads it, so the test supplies it the way the driver does.
        cfg.add_to_config(name="write_scenario_lp_mps_files_dir",
                          description="", domain=str, default=None)
        cfg.checkpoint_dir = "/tmp/whatever"
        cfg.wtracker = True

        hub_dict = {"opt_kwargs": {"options": {},
                                   "extensions": Checkpointer,
                                   "extension_kwargs": None}}
        configure_extensions(hub_dict, None, cfg)

        self.assertIs(hub_dict["opt_kwargs"]["extensions"], MultiExtension)
        classes = hub_dict["opt_kwargs"]["extension_kwargs"]["ext_classes"]
        self.assertIn(Checkpointer, classes)
        self.assertIn(Wtracker_extension, classes)
        _check_checkpointing_survived(hub_dict, cfg)  # the guard agrees


@unittest.skipIf(not solver_available,
                 "no solver is available for the resume behavior tests")
class TestResumeExtensionBehavior(unittest.TestCase):
    """Extension hooks at the resume boundary act on the checkpointed state."""

    STOP = 3

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def test_rho_extension_keeps_checkpointed_rho(self):
        """CoeffRho stands in for the rho-setting extensions: its post_iter0
        recomputes rho from the objective coefficients, which on a resume
        would clobber the adapted rho the checkpoint carries (the mutator's
        compounding scaling makes the two distinguishable)."""
        stopped = _make_ph(
            _options(self.STOP, ckpt_dir=self.ckpt_dir),
            extension_name="coeff_rho", extra_names=("miditer_mutator",))
        stopped.ph_main()

        # A per-run budget of zero makes this an empty resume: Iter0 restores,
        # the loop body never runs, so what remains is exactly the splice.
        resumed = _make_ph(_options(0, resume_from=self.ckpt_dir),
                           extension_name="coeff_rho")
        resumed.ph_main()

        self.assertEqual(
            _primal_snapshot(resumed), _primal_snapshot(stopped),
            msg="the resumed state differs from the checkpointed state; a "
                "rho extension recomputed rho across the resume")

    def test_pre_iter0_runs_on_the_restored_models(self):
        stopped = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        stopped.ph_main()

        resumed = _make_ph(_options(0, resume_from=self.ckpt_dir),
                           extension_name="pre_tagger")
        resumed.ph_main()

        for sname, s in resumed.local_scenarios.items():
            self.assertTrue(
                getattr(s, "_pre_iter0_saw_this_model", False),
                msg=f"pre_iter0 ran on a model that the resume splice then "
                    f"discarded (scenario {sname}), so its effects were lost")

    def _assert_write_failure_is_survived(self, exc):
        """A transient write failure warns and continues; the previously
        published generation stays resumable and later iterations retry."""
        calls = {"n": 0}
        real = checkpointing.write_checkpoint

        def flaky(opt, ckpt_dir, generation, backend):
            calls["n"] += 1
            if calls["n"] > 1:
                raise exc
            return real(opt, ckpt_dir, generation, backend=backend)

        with mock.patch.object(checkpointing, "write_checkpoint",
                               side_effect=flaky):
            ph = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
            ph.ph_main()    # must not raise

        self.assertEqual(calls["n"], self.STOP,
                         msg="every iteration boundary must retry the write")
        self.assertEqual(_published_generation(self.ckpt_dir), 1)

        # And the surviving generation is genuinely resumable.
        resumed = _make_ph(_options(1, resume_from=self.ckpt_dir))
        resumed.ph_main()
        self.assertEqual(resumed._resume_iteration, 1)

    def test_midrun_write_failure_does_not_kill_the_run(self):
        self._assert_write_failure_is_survived(RuntimeError("synthetic ENOSPC"))

    def test_midrun_oserror_does_not_kill_the_run(self):
        """Only the model dump is wrapped as a RuntimeError; the leaf write,
        the publishing renames and the manifest write all raise a bare
        OSError, so a disk that fills between the last model file and the leaf
        must not take the run down."""
        self._assert_write_failure_is_survived(
            OSError(errno.ENOSPC, "No space left on device"))


class _HookRecorder(Extension):
    """Counts maybe_checkpoint calls without writing anything."""

    def __init__(self, opt=None):
        self.opt = opt
        self.calls = 0

    def maybe_checkpoint(self):
        self.calls += 1

    # A test probe; nothing of its own to carry across a resume.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass


class TestCheckpointHookDispatch(unittest.TestCase):
    """The dedicated hook exists on the extension interface and dispatches."""

    def test_base_extension_hook_is_a_noop(self):
        Extension(None).maybe_checkpoint()   # must not raise

    def test_multiextension_dispatches_to_every_extension(self):
        from mpisppy.extensions.extension import MultiExtension
        multi = MultiExtension(None, [])
        recorders = [_HookRecorder(), _HookRecorder()]
        multi.extdict = {"a": recorders[0], "b": recorders[1]}
        multi.maybe_checkpoint()
        self.assertEqual([r.calls for r in recorders], [1, 1])

    def _spoke_stub(self, extensions):
        from mpisppy.cylinders.xhatbase import XhatInnerBoundBase
        recorder = _HookRecorder()
        stub = types.SimpleNamespace(
            opt=types.SimpleNamespace(extensions=extensions,
                                      extobject=recorder))
        XhatInnerBoundBase.maybe_checkpoint(stub)
        return recorder

    def test_spoke_hook_fires_when_extensions_are_attached(self):
        self.assertEqual(self._spoke_stub(Extension).calls, 1)

    def test_spoke_hook_is_silent_without_extensions(self):
        # A spoke with no extensions is the common case; it must not blow up
        # on the extobject that does not exist.
        self.assertEqual(self._spoke_stub(None).calls, 0)


class TestXhatterLoopsOfferCheckpointPoints(unittest.TestCase):
    """Every xhatter spoke loop reaches the hook on every pass.

    The hub gets its checkpoint points from iterk_loop; the xhatter loops are
    not PH iterations and had no hook at all, so these pin the calls that give
    a spoke somewhere to write its incumbent from. The loops are driven
    against stubs -- a real spoke needs MPI windows and a hub to talk to --
    but the loop bodies themselves are the shipped ones.
    """

    def _drive(self, cls, options, kill_after, prep=None, extra=None):
        """Run cls.main() for kill_after passes; return the hook recorder."""
        spoke = object.__new__(cls)
        recorder = _HookRecorder()
        spoke.opt = types.SimpleNamespace(
            options=options, extensions=Extension, extobject=recorder)
        spoke.global_rank = 0
        spoke.cylinder_rank = 0
        spoke.verbose = False
        # False for kill_after passes, then True to end the loop.
        kills = [False] * kill_after + [True]
        spoke.got_kill_signal = lambda: kills.pop(0)
        spoke.update_nonants = lambda: False
        spoke.xhat_prep = lambda: (prep if prep is not None
                                   else types.SimpleNamespace())
        spoke._try_average_scenario_xhat = lambda: None
        spoke._try_feasible_xhat = lambda: None
        if extra is not None:
            extra(spoke)
        cls.main(spoke)
        return recorder

    def test_xhatlooper(self):
        from mpisppy.cylinders.xhatlooper_bounder import XhatLooperInnerBound
        recorder = self._drive(
            XhatLooperInnerBound,
            {"xhat_looper_options": {"scen_limit": 1}}, kill_after=3)
        self.assertEqual(recorder.calls, 3)

    def test_xhatxbar(self):
        from mpisppy.cylinders.xhatxbar_bounder import XhatXbarInnerBound
        recorder = self._drive(XhatXbarInnerBound, {}, kill_after=3)
        self.assertEqual(recorder.calls, 3)

    def test_xhatspecific(self):
        from mpisppy.cylinders.xhatspecific_bounder import (
            XhatSpecificInnerBound)
        recorder = self._drive(
            XhatSpecificInnerBound,
            {"xhat_specific_options": {"xhat_scenario_dict": {"ROOT": "s0"}}},
            kill_after=3)
        self.assertEqual(recorder.calls, 3)

    def _shuffle_options(self):
        return {"xhat_looper_options": {"reverse": True, "iter_step": None,
                                        "xhat_solver_options": None}}

    def _shuffle_extra(self, spoke, kills=None):
        import random
        spoke.random_seed = 42
        spoke.random_stream = random.Random()
        spoke.opt.all_scenario_names = ["s0", "s1", "s2"]
        # A two-stage tree: ROOT's kids are leaves, so ScenarioCycler stays in
        # its non-multistage branch and needs nothing else from the tree.
        spoke.opt.nonleaves = {"ROOT": types.SimpleNamespace(kids=[])}
        spoke._nonant_len_receive_buffer = types.SimpleNamespace(
            id=lambda: 1)
        spoke.try_scenario_dict = lambda _: False

    def test_xhatshuffle(self):
        from mpisppy.cylinders.xhatshufflelooper_bounder import (
            XhatShuffleInnerBound)
        recorder = self._drive(
            XhatShuffleInnerBound, self._shuffle_options(), kill_after=3,
            extra=self._shuffle_extra)
        self.assertEqual(recorder.calls, 3)

    def test_xhatshuffle_kill_between_tries_still_offers_a_point(self):
        """An exit from the middle of a pass skips the bottom of the loop.

        xhatshuffle re-checks the kill signal between its two tries and
        returns from the middle of the pass. The try just above it may have
        improved the incumbent, and the pass offers it a write before
        returning.
        """
        from mpisppy.cylinders.xhatshufflelooper_bounder import (
            XhatShuffleInnerBound)

        def extra(spoke):
            self._shuffle_extra(spoke)
            spoke.update_nonants = lambda: True
            # localnonants is a read-only property over this buffer.
            spoke._nonant_len_receive_buffer = types.SimpleNamespace(
                id=lambda: 1, value_array=lambda: None)
            spoke.opt._put_nonant_cache = lambda _: None
            spoke.opt._restore_nonants = lambda **kwargs: None
            # while-condition, then the mid-pass re-check.
            kills = [False, True]
            spoke.got_kill_signal = lambda: kills.pop(0)

        recorder = self._drive(
            XhatShuffleInnerBound, self._shuffle_options(), kill_after=0,
            extra=extra)
        self.assertEqual(recorder.calls, 1)


@unittest.skipIf(not solver_available,
                 "no solver is available for the hook placement test")
class TestCheckpointHookPlacement(unittest.TestCase):
    """What the hub writes does not depend on extension attach order."""

    STOP = 3

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def test_enditer_model_changes_are_in_the_checkpoint(self):
        """An extension that changes a model in enditer is checkpointed.

        The Checkpointer is attached first, so dispatching the write from an
        enditer would put it ahead of this extension's: the checkpoint would
        hold the rho of the iteration before, and because a resume starts at
        the *next* iteration, the scaling of the last one would be lost for
        good.
        """
        stopped = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir),
                           extension_name="enditer_mutator")
        stopped.ph_main()

        # A per-run budget of zero: the loop body never runs, so the resumed
        # state is exactly what the checkpoint held.
        resumed = _make_ph(_options(0, resume_from=self.ckpt_dir),
                           extension_name="enditer_mutator")
        resumed.ph_main()

        self.assertEqual(
            _primal_snapshot(resumed), _primal_snapshot(stopped),
            msg="the checkpoint was written before the last enditer, so the "
                "model change that hook made is missing from it")


class _SpokeStub:
    """Stands in for the spoke communicator the Checkpointer reads."""

    def __init__(self, strata_rank=2, best_inner_bound=None, loop_state=None,
                 communicators=None):
        self.strata_rank = strata_rank
        #: The cylinder list WheelSpinner hands every SPCommunicator, or None
        #: for a stub driven outside a wheel -- which is then the only
        #: cylinder of its class as far as the Checkpointer can tell.
        if communicators is not None:
            self.communicators = communicators
        #: The real spoke starts this at the infinity that loses every
        #: comparison and never holds None, so a stub that said None would
        #: not be standing in for anything reachable.
        self.best_inner_bound = (math.inf if best_inner_bound is None
                                 else best_inner_bound)
        self.is_minimizing = True
        self.sent_bounds = []
        self.sent_xhats = 0
        #: What checkpoint_loop_state() reports. None is what every xhatter
        #: but xhatshuffle says: they re-evaluate from scratch when new
        #: nonants arrive, so there is no cursor to carry.
        self.loop_state = loop_state

    def send_bound(self, value):
        self.sent_bounds.append(value)

    def send_best_xhat(self):
        self.sent_xhats += 1

    def checkpoint_loop_state(self):
        return self.loop_state

    def loop_state_progress(self, state):
        # XhatBase's default. A stub that stands in for a spoke has to carry
        # the whole contract, not just the part that was in use when it was
        # written -- the Checkpointer calls this unguarded.
        return state


def _xhat_eval(ckpt_dir=None, resume_from=None, **overrides):
    """An Xhat_Eval on farmer, the object an xhat spoke drives."""
    from mpisppy.utils.xhat_eval import Xhat_Eval
    options = _options(1, ckpt_dir=ckpt_dir, resume_from=resume_from,
                       **overrides)
    return Xhat_Eval(options, SCENARIO_NAMES, farmer.scenario_creator,
                     farmer.scenario_denouement,
                     scenario_creator_kwargs=CREATOR_KWARGS)


def _set_and_cache_solution(opt, base):
    """Give every variable a distinct known value and cache it as the
    incumbent, the way an accepted xhat evaluation does."""
    import pyomo.environ as pyo
    for offset, (sname, s) in enumerate(opt.local_scenarios.items()):
        for i, var in enumerate(s.component_data_objects(pyo.Var)):
            var.set_value(base + offset * 100 + i, skip_validation=True)
        s._mpisppy_data.inner_bound = float(base + offset)
    opt.update_best_solution_if_improving(float(base))


def _publish_best_xhat(opt):
    """What ``send_best_xhat`` puts in the buffer, per scenario.

    Returns one list per scenario: its nonant values followed by the
    objective published with them. The spoke object itself is stubbed --
    what is under test is the method, and standing up a real cylinder would
    need a wheel and an MPI window to read one array back.
    """
    import numpy as np
    from mpisppy.cylinders.spoke import InnerBoundNonantSpoke
    from mpisppy.cylinders.spwindow import Field

    per_scenario = [len(s._mpisppy_data.nonant_indices) + 1
                    for s in opt.local_scenarios.values()]

    class _Stub:
        def __init__(self):
            self.opt = opt
            self.send_buffers = {Field.BEST_XHAT: np.zeros(sum(per_scenario))}
            self.sent = None

        def put_send_buffer(self, buf, field):
            self.sent = (field, list(buf))

    stub = _Stub()
    InnerBoundNonantSpoke.send_best_xhat(stub)
    field, values = stub.sent
    assert field is Field.BEST_XHAT
    out, at = [], 0
    for width in per_scenario:
        out.append(values[at:at + width])
        at += width
    return out


def _solution_by_name(opt):
    import pyomo.environ as pyo
    return {
        sname: {v.name: v.value
                for v in s.component_data_objects(pyo.Var)}
        for sname, s in opt.local_scenarios.items()
    }


class _TwoRankComm:
    """A cylinder comm for two ranks that agree on everything: each
    collective answers as though the other rank sent what this one did."""

    def Get_size(self):
        return 2

    def Get_rank(self):
        return 0

    def allgather(self, value):
        return [value, value]

    def bcast(self, value, root=0):
        return value

    def Barrier(self):
        pass


class TestSpokeIncumbentFile(unittest.TestCase):
    """The spoke's own checkpoint: the best xhat, by variable name.

    The hub checkpoint does not carry the incumbent -- it lives in
    best_solution_cache on the xhat spoke -- so without this file a resumed
    run restores its iterate perfectly and still reports whatever it happens
    to find after the restart.
    """

    CYLINDER = "XhatShuffleInnerBound"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _write_one(self, base=1.0, bound=-42.0):
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(opt, base)
        path = checkpointing.write_spoke_incumbent(
            opt, self.ckpt_dir, self.CYLINDER, 2, best_inner_bound=bound)
        return opt, path

    #: What a solver can report for a solution it accepted: no bound at all,
    #: or (ipopt) an infinite one. NaN is what None becomes in a buffer.
    NOT_AN_OBJECTIVE = (None, math.inf, -math.inf, math.nan)

    def test_a_solution_with_no_objective_is_not_written(self):
        """A solver may accept a solution and report no bound for it.

        The objective travels in the same float64 buffer as the values, and
        assigning None into one stores NaN without raising, so the gap would
        leave here as a number and arrive at FWPH as the recourse cost of a
        QP column. Refuse the file instead; the caller turns this into the
        warning it already prints when a spoke cannot write.
        """
        for bad in self.NOT_AN_OBJECTIVE:
            with self.subTest(objective=bad):
                opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
                _set_and_cache_solution(opt, 1.0)
                for s in opt.local_scenarios.values():
                    s._mpisppy_data.best_solution_inner_bound = bad
                with self.assertRaises(ValueError) as ctx:
                    checkpointing.write_spoke_incumbent(
                        opt, self.ckpt_dir, self.CYLINDER, 2,
                        best_inner_bound=-42.0)
                self.assertIn("no finite objective", str(ctx.exception))
                self.assertFalse(
                    os.path.isdir(os.path.join(self.ckpt_dir, "spokes")),
                    msg="a file that cannot describe a usable incumbent was "
                        "written")

    def test_a_solution_with_no_objective_is_not_restored(self):
        """Files written before the write refused this still exist."""
        opt, _ = self._write_one()
        for bad in self.NOT_AN_OBJECTIVE:
            with self.subTest(objective=bad):
                state = checkpointing.load_spoke_incumbent(
                    opt, self.ckpt_dir, self.CYLINDER, 2)
                for entry in state["solutions"].values():
                    entry["inner_bound"] = bad
                with self.assertRaises(
                        checkpointing.CheckpointMismatch) as ctx:
                    checkpointing.restore_spoke_incumbent(opt, state)
                self.assertIn("no finite objective", str(ctx.exception))

    def test_written_where_the_design_says(self):
        _, path = self._write_one()
        self.assertEqual(
            os.path.relpath(path, self.ckpt_dir),
            os.path.join("spokes",
                         f"spoke_{self.CYLINDER}_ordinal_02_rank_0000.pkl"))

    def test_nothing_written_before_an_incumbent_exists(self):
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        self.assertIsNone(checkpointing.write_spoke_incumbent(
            opt, self.ckpt_dir, self.CYLINDER, 2))
        self.assertFalse(os.path.exists(
            os.path.join(self.ckpt_dir, "spokes")))

    def test_the_file_carries_the_incumbents_own_objective(self):
        """Not the objective of whatever this spoke solved most recently.

        ``inner_bound`` moves on every solve while the cached values move
        only on an improvement, so a file that reads it live pairs this
        incumbent's variable values with a later solve's objective -- and
        the resumed spoke republishes that pair to the hub, where the
        trailing per-scenario objective in BEST_XHAT is what grad rho reads.
        """
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(opt, 10.0)
        incumbent = {sname: s._mpisppy_data.inner_bound
                     for sname, s in opt.local_scenarios.items()}

        # A later solve that does not improve on the incumbent: nothing else
        # in the file moves, so this is the whole difference.
        for offset, s in enumerate(opt.local_scenarios.values()):
            s._mpisppy_data.inner_bound = 9999.0 + offset

        state = checkpointing.spoke_incumbent_state(opt, self.CYLINDER, 2)
        for sname, entry in state["solutions"].items():
            self.assertEqual(
                entry["inner_bound"], incumbent[sname],
                msg=f"{sname}: the file carries the objective of a solve "
                    "that came after the incumbent it stores")

    def test_a_write_is_identified_by_objective_and_cursor(self):
        """Since a spoke also writes when only its cursor moves, the
        objective alone no longer tells two writes apart; ranks holding
        files from different writes must not be taken as one checkpoint."""
        same = checkpointing.spoke_write_id(-10.0, {"cycle_idx": 3})
        self.assertEqual(same, checkpointing.spoke_write_id(
            -10.0, {"cycle_idx": 3}))
        moved = checkpointing.spoke_write_id(-10.0, {"cycle_idx": 4})
        self.assertNotEqual(same, moved)
        verdict, _ = checkpointing._one_write_verdict(
            2, 2, [checkpointing.XHAT_WRITE_KEY({"write_id": same}),
                   checkpointing.XHAT_WRITE_KEY({"write_id": moved})])
        self.assertEqual(verdict, "differ")
        # A file written before there was a write_id falls back to the
        # objective, as before.
        self.assertEqual(
            checkpointing.XHAT_WRITE_KEY({"best_solution_obj_val": -10.0}),
            -10.0)

    def test_the_file_records_its_write_id(self):
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(opt, 10.0)
        state = checkpointing.spoke_incumbent_state(
            opt, self.CYLINDER, 2, progress={"cycle_idx": 1})
        self.assertEqual(state["write_id"], checkpointing.spoke_write_id(
            state["best_solution_obj_val"], {"cycle_idx": 1}))

    def test_an_unchanged_incumbent_is_not_rebuilt(self):
        """A spoke writes on nearly every evaluation, and keying the values
        by name is almost all of a write's cost on a large model. The dict is
        built once per incumbent and reused until the incumbent changes."""
        import pyomo.environ as pyo
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(opt, 10.0)
        first = checkpointing.spoke_incumbent_state(opt, self.CYLINDER, 2)
        # Later solves move the live variables but not the incumbent.
        for s in opt.local_scenarios.values():
            for v in s.component_data_objects(pyo.Var):
                if v.value is not None:
                    v.set_value(v.value + 1.0, skip_validation=True)
        second = checkpointing.spoke_incumbent_state(opt, self.CYLINDER, 2)
        for sname in first["solutions"]:
            self.assertIs(second["solutions"][sname]["values"],
                          first["solutions"][sname]["values"],
                          msg=f"{sname}: rebuilt for an unchanged incumbent")

        # An improvement replaces the cache, so the dict follows it.
        _set_and_cache_solution(opt, 5.0)
        third = checkpointing.spoke_incumbent_state(opt, self.CYLINDER, 2)
        for sname, s in opt.local_scenarios.items():
            values = third["solutions"][sname]["values"]
            self.assertIsNot(values, first["solutions"][sname]["values"])
            self.assertEqual(
                values,
                {v.name: x for v, x in
                 s._mpisppy_data.best_solution_cache.items()})

    def test_the_restored_objective_is_the_one_the_spoke_republishes(self):
        """And it stays that one once the resumed spoke starts working.

        Every xhat evaluation overwrites the live ``inner_bound``, and the
        restored incumbent stays in the cache until the spoke improves on
        it. Asking only what the restore put on the models cannot see that:
        this asks what ``send_best_xhat`` actually puts in the buffer, after
        such an evaluation, which is what reaches FWPH.
        """
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(opt, 10.0)
        incumbent = {sname: s._mpisppy_data.inner_bound
                     for sname, s in opt.local_scenarios.items()}
        # Same later solve as above, so this test also fails if the write
        # goes back to reading the live attribute.
        for offset, s in enumerate(opt.local_scenarios.values()):
            s._mpisppy_data.inner_bound = 9999.0 + offset
        path = checkpointing.write_spoke_incumbent(
            opt, self.ckpt_dir, self.CYLINDER, 2, best_inner_bound=-42.0)
        self.assertIsNotNone(path)

        fresh = _xhat_eval(resume_from=self.ckpt_dir)
        state = checkpointing.load_spoke_incumbent(
            fresh, self.ckpt_dir, self.CYLINDER, 2)
        checkpointing.restore_spoke_incumbent(fresh, state)
        for sname, s in fresh.local_scenarios.items():
            self.assertEqual(s._mpisppy_data.inner_bound, incumbent[sname])
            self.assertEqual(s._mpisppy_data.best_solution_inner_bound,
                             incumbent[sname])

        # The resumed spoke evaluates an xhat of its own, and a worse one
        # leaves the incumbent alone -- and the live attribute on every
        # scenario changed.
        for offset, s in enumerate(fresh.local_scenarios.values()):
            s._mpisppy_data.inner_bound = 5555.0 + offset

        published = _publish_best_xhat(fresh)
        self.assertEqual(
            [entry[-1] for entry in published],
            [incumbent[sname] for sname in fresh.local_scenarios],
            msg="the spoke republished its restored xhat with the objectives "
                "of the evaluation it happened to run first")

    def test_the_published_objective_belongs_to_the_published_xhat(self):
        """On a fresh run too, and this is where it is consumed.

        FWPH reads each scenario's block back as one column of its QP: the
        values are the column and the trailing objective is that column's
        recourse cost. Pairing an incumbent with a later evaluation's
        objective is a wrong coefficient in someone else's optimization,
        with nothing anywhere to reveal it.
        """
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(opt, 10.0)
        incumbent = {sname: s._mpisppy_data.inner_bound
                     for sname, s in opt.local_scenarios.items()}
        cached = _solution_by_name(opt)

        # An evaluation that does not improve on the incumbent: the cache
        # stands and the live objectives move.
        for offset, s in enumerate(opt.local_scenarios.values()):
            s._mpisppy_data.inner_bound = 9999.0 + offset

        published = _publish_best_xhat(opt)
        for entry, (sname, s) in zip(published, opt.local_scenarios.items()):
            self.assertEqual(
                entry[-1], incumbent[sname],
                msg=f"{sname}: the published objective is a later "
                    f"evaluation's, not the published xhat's")
            # And the values really are the incumbent's, so the pair is one
            # solution rather than two halves that happen to agree.
            nonants = s._mpisppy_data.nonant_indices.values()
            self.assertEqual(
                list(entry[:-1]),
                [cached[sname][var.name] for var in nonants])

    def test_restores_every_variable_onto_fresh_models(self):
        """The load-bearing test: a *different* set of models, built by the
        scenario_creator exactly as a resumed spoke builds them, ends up
        holding the checkpointed solution."""
        written, _ = self._write_one(base=7.0, bound=-99.0)
        want = _solution_by_name(written)

        resumed = _xhat_eval(resume_from=self.ckpt_dir)
        state = checkpointing.load_spoke_incumbent(
            resumed, self.ckpt_dir, self.CYLINDER, 2)
        self.assertIsNotNone(state)
        obj = checkpointing.restore_spoke_incumbent(resumed, state)

        self.assertEqual(obj, 7.0)
        self.assertEqual(resumed.best_solution_obj_val, 7.0)
        # load_best_solution is what finalize() calls; after it the models
        # hold the restored answer.
        self.assertTrue(resumed.load_best_solution())
        self.assertEqual(_solution_by_name(resumed), want)

    def test_per_scenario_inner_bound_survives(self):
        """send_best_xhat packs it beside the values, so a resumed spoke that
        published without it would send whatever the fresh models hold."""
        self._write_one(base=3.0)
        resumed = _xhat_eval(resume_from=self.ckpt_dir)
        state = checkpointing.load_spoke_incumbent(
            resumed, self.ckpt_dir, self.CYLINDER, 2)
        checkpointing.restore_spoke_incumbent(resumed, state)
        for offset, s in enumerate(resumed.local_scenarios.values()):
            self.assertEqual(s._mpisppy_data.inner_bound, 3.0 + offset)

    def test_missing_file_is_not_an_error(self):
        """A run may have stopped before this spoke found anything."""
        resumed = _xhat_eval(resume_from=self.ckpt_dir)
        self.assertIsNone(checkpointing.load_spoke_incumbent(
            resumed, self.ckpt_dir, self.CYLINDER, 2))

    def test_a_different_spoke_does_not_read_this_one(self):
        self._write_one()
        resumed = _xhat_eval(resume_from=self.ckpt_dir)
        self.assertIsNone(checkpointing.load_spoke_incumbent(
            resumed, self.ckpt_dir, "XhatXbarInnerBound", 3))

    def test_structural_mismatch_is_refused(self):
        """Values from a differently configured model are wrong answers, not
        stale ones."""
        self._write_one()
        resumed = _xhat_eval(resume_from=self.ckpt_dir)
        resumed.options["checkpoint_structural_cfg"] = {"crops_multiplier": 2}
        with self.assertRaises(checkpointing.CheckpointMismatch):
            checkpointing.load_spoke_incumbent(
                resumed, self.ckpt_dir, self.CYLINDER, 2)

    def test_resume_may_add_the_certified_outer_bound_spoke(self):
        """Like --lagrangian, --certified-outer-bound changes which cylinders
        run, not what problem the checkpoint describes, and its cushion only
        loosens the bound that spoke reports. The cfg is folded from the
        options the spoke really registers, so a renamed option shows up
        here rather than as a refused resume."""
        import mpisppy.utils.cfg_vanilla as vanilla
        from mpisppy.utils.config import Config

        def folded(**overrides):
            cfg = Config()
            cfg.popular_args()
            cfg.checkpoint_args()
            cfg.certified_outer_bound_args()
            cfg.checkpoint_dir = self.ckpt_dir
            for key, value in overrides.items():
                cfg[key] = value
            hub_dict = {"opt_kwargs": {"options": {}}}
            vanilla.add_checkpointing(hub_dict, cfg)
            return hub_dict["opt_kwargs"]["options"][
                "checkpoint_structural_cfg"]

        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        opt.options["checkpoint_structural_cfg"] = folded()
        _set_and_cache_solution(opt, 1.0)
        checkpointing.write_spoke_incumbent(
            opt, self.ckpt_dir, self.CYLINDER, 2, best_inner_bound=-42.0)

        resumed = _xhat_eval(resume_from=self.ckpt_dir)
        resumed.options["checkpoint_structural_cfg"] = folded(
            certified_outer_bound=True, certified_outer_bound_cushion=1e-6)
        state = checkpointing.load_spoke_incumbent(
            resumed, self.ckpt_dir, self.CYLINDER, 2)
        self.assertIsNotNone(state)

    def test_a_file_missing_an_agreed_key_is_refused_at_load(self):
        """The agreed restore relies on these two, so a file without one has
        to be refused by the load, which runs inside an agreement, rather
        than have None stand in for what the file recorded."""
        for key in ("best_solution_obj_val", "best_inner_bound"):
            with self.subTest(key=key):
                _, path = self._write_one()
                with open(path, "rb") as f:
                    state = pickle.load(f)
                del state[key]
                with open(path, "wb") as f:
                    pickle.dump(state, f)
                resumed = _xhat_eval(resume_from=self.ckpt_dir)
                with self.assertRaises(checkpointing.CheckpointMismatch) \
                        as ctx:
                    checkpointing.load_spoke_incumbent(
                        resumed, self.ckpt_dir, self.CYLINDER, 2)
                self.assertIn(key, str(ctx.exception))

    def test_a_dual_file_missing_a_key_read_outside_the_agreement_is_refused(
            self):
        """The dual restore reads generation and Wbar outside any agreement,
        just before a collective, so the load -- which runs inside one --
        has to refuse a file without them."""
        opt = _make_ph(_options(1))
        cylinder, ordinal = "PHDualSpoke", 0
        spokes_dir = os.path.join(self.ckpt_dir, checkpointing.SPOKES_SUBDIR)
        os.makedirs(spokes_dir)
        path = os.path.join(spokes_dir, checkpointing._spoke_filename(
            cylinder, ordinal, opt.cylinder_rank))
        for key in ("generation", "Wbar"):
            with self.subTest(key=key):
                state = {
                    "format_version": checkpointing.FORMAT_VERSION,
                    "kind": "dual-spoke-ph-state",
                    "structural_fingerprint":
                        checkpointing.structural_fingerprint(opt.options),
                    "geometry": {
                        "scenario_names": sorted(opt.local_scenarios)},
                    "generation": 3,
                    "Wbar": {},
                }
                del state[key]
                with open(path, "wb") as f:
                    pickle.dump(state, f)
                with self.assertRaises(checkpointing.CheckpointMismatch) \
                        as ctx:
                    checkpointing.load_dual_spoke_state(
                        opt, self.ckpt_dir, cylinder, ordinal)
                self.assertIn(key, str(ctx.exception))

    def test_a_variable_the_model_no_longer_has_is_refused(self):
        """A partially restored incumbent is a solution that was never
        feasible for anything."""
        _, path = self._write_one()
        with open(path, "rb") as f:
            state = pickle.load(f)
        state["solutions"]["scen0"]["values"]["NoSuchVar[0]"] = 1.0
        resumed = _xhat_eval(resume_from=self.ckpt_dir)
        with self.assertRaises(checkpointing.CheckpointMismatch):
            checkpointing.restore_spoke_incumbent(resumed, state)


class TestCheckpointerSpokeMode(unittest.TestCase):
    """The extension half: when the spoke writes, and what it tells the hub."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _attach(self, opt, spoke=None):
        # SPBase.spcomm is a weakref, so the stub needs an owner that outlives
        # the call or it is collected before the extension ever reads it.
        self.spoke = spoke if spoke is not None else _SpokeStub()
        opt.spcomm = self.spoke
        return Checkpointer(opt), self.spoke

    def test_attaches_to_an_xhat_spoke(self):
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        ext, _ = self._attach(opt)
        self.assertTrue(ext.spoke_mode)
        self.assertTrue(ext.write_enabled)

    def test_writes_only_when_the_incumbent_improves(self):
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        ext, spoke = self._attach(opt)

        ext.maybe_checkpoint()          # nothing found yet
        self.assertFalse(os.path.exists(os.path.join(self.ckpt_dir, "spokes")))

        _set_and_cache_solution(opt, 5.0)
        spoke.best_inner_bound = 5.0
        ext.maybe_checkpoint()
        path = os.path.join(
            self.ckpt_dir, "spokes",
            "spoke__SpokeStub_ordinal_00_rank_0000.pkl")
        self.assertTrue(os.path.exists(path))
        first = os.stat(path).st_mtime_ns

        # A pass that found nothing new must not rewrite the file; the loop
        # calls this every time round while it waits on the hub.
        ext.maybe_checkpoint()
        self.assertEqual(os.stat(path).st_mtime_ns, first)

    def test_restore_only_run_reads_without_writing(self):
        writer = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(writer, 11.0)
        checkpointing.write_spoke_incumbent(
            writer, self.ckpt_dir, "_SpokeStub", 0, best_inner_bound=11.0)

        # --resume-from with no --checkpoint-dir: the spoke still has to
        # restore, which is why the extension is attached for a read too.
        opt = _xhat_eval(resume_from=self.ckpt_dir)
        ext, spoke = self._attach(opt)
        self.assertFalse(ext.write_enabled)
        ext.pre_iter0()
        self.assertEqual(opt.best_solution_obj_val, 11.0)
        self.assertEqual(spoke.best_inner_bound, 11.0)

    def test_restore_only_run_attempts_no_write(self):
        """--resume-from with no --checkpoint-dir has nowhere to write.

        The write failure path warns rather than raises, so attempting the
        write anyway does not fail a run -- it just warns on every
        improvement for the rest of it, which is how this went unnoticed in a
        cylinders run until the log was read.
        """
        writer = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(writer, 17.0)
        checkpointing.write_spoke_incumbent(
            writer, self.ckpt_dir, "_SpokeStub", 0, best_inner_bound=17.0)

        opt = _xhat_eval(resume_from=self.ckpt_dir)
        ext, _ = self._attach(opt)
        ext.pre_iter0()
        with mock.patch.object(checkpointing, "write_spoke_incumbent") as write:
            ext.maybe_checkpoint()
            _set_and_cache_solution(opt, 3.0)
            ext.maybe_checkpoint()
        write.assert_not_called()

    def test_restored_bound_is_published_to_the_hub_once(self):
        """The hub learns bounds only from what a spoke sends, so a restored
        incumbent that is never published leaves the hub reporting an
        infinite inner bound and gapping against it. The restore runs before
        the spoke's first solve, and that is when the hub must hear it: a
        publish left for later lets the hub run an iteration against an
        infinite inner bound."""
        writer = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(writer, 13.0)
        checkpointing.write_spoke_incumbent(
            writer, self.ckpt_dir, "_SpokeStub", 0, best_inner_bound=13.0)

        opt = _xhat_eval(resume_from=self.ckpt_dir)
        ext, spoke = self._attach(opt)
        ext.pre_iter0()
        self.assertEqual(spoke.sent_bounds, [13.0])
        self.assertEqual(spoke.sent_xhats, 1)

        ext.maybe_checkpoint()                    # not resent every pass
        self.assertEqual(spoke.sent_bounds, [13.0])

    def test_a_resume_into_a_new_directory_writes_the_incumbent_at_once(self):
        """A short resume whose hub finishes while this spoke is still in its
        prep never reaches a loop pass, so a write left for the bottom of
        one leaves the new directory with a hub checkpoint and no incumbent,
        and the next resume starts without one."""
        old = os.path.join(self._tmp.name, "old")
        writer = _xhat_eval(ckpt_dir=old)
        _set_and_cache_solution(writer, 19.0)
        checkpointing.write_spoke_incumbent(
            writer, old, "_SpokeStub", 0, best_inner_bound=19.0)

        opt = _xhat_eval(ckpt_dir=self.ckpt_dir, resume_from=old)
        ext, _ = self._attach(opt)
        ext.pre_iter0()                           # no loop pass follows
        state = checkpointing.load_spoke_incumbent(
            opt, self.ckpt_dir, "_SpokeStub", 0)
        self.assertIsNotNone(state, "the new directory has no incumbent")
        self.assertEqual(state["best_inner_bound"], 19.0)

    def test_the_write_after_a_restore_carries_what_was_read(self):
        """The spoke is handed the loop state after pre_iter0 and the
        extension state at the end of its prep, so the write straight after
        a restore has to carry the read state itself; asking the spoke would
        put a fresh cursor and fresh extension state in the new directory."""
        old = os.path.join(self._tmp.name, "old")
        read_loop = {"xh_iter": 7, "cursor": {"at": 3}}
        read_ext = {"extensions": {"SomeExtension": {"k": 1}}}
        writer = _xhat_eval(ckpt_dir=old)
        _set_and_cache_solution(writer, 29.0)
        checkpointing.write_spoke_incumbent(
            writer, old, "_SpokeStub", 0, best_inner_bound=29.0,
            loop_state=read_loop, extension_state=read_ext)

        opt = _xhat_eval(ckpt_dir=self.ckpt_dir, resume_from=old)
        ext, _ = self._attach(opt, _SpokeStub(loop_state={"fresh": True}))
        ext.pre_iter0()
        state = checkpointing.load_spoke_incumbent(
            opt, self.ckpt_dir, "_SpokeStub", 0)
        self.assertIsNotNone(state, "the new directory has no incumbent")
        self.assertEqual(state["loop_state"], read_loop)
        self.assertEqual(state["extension_state"], read_ext)

    def test_a_resume_in_place_does_not_rewrite_what_it_read(self):
        writer = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(writer, 23.0)
        checkpointing.write_spoke_incumbent(
            writer, self.ckpt_dir, "_SpokeStub", 0, best_inner_bound=23.0)

        opt = _xhat_eval(ckpt_dir=self.ckpt_dir, resume_from=self.ckpt_dir)
        ext, _ = self._attach(opt)
        with mock.patch.object(checkpointing, "write_spoke_incumbent") as write:
            ext.pre_iter0()
        write.assert_not_called()


class TestAnInnerBoundSpokeCheckpointsAtFinalize(unittest.TestCase):
    """A spoke loop can exit at its top check without reaching a bottom,
    where the checkpoint point is, so an incumbent found before the loop
    would never be written. finalize offers one last checkpoint point."""

    def test_finalize_offers_a_checkpoint_point_first(self):
        from mpisppy.cylinders.spoke import InnerBoundNonantSpoke
        calls = []
        # Built without __init__, which wants communicators; finalize reads
        # only these two. The checkpoint comes before load_best_solution,
        # which InnerBoundSpoke.finalize runs through super().
        spoke = InnerBoundNonantSpoke.__new__(InnerBoundNonantSpoke)
        spoke.maybe_checkpoint = lambda: calls.append("checkpoint")
        spoke.opt = types.SimpleNamespace(
            load_best_solution=lambda: calls.append("load") and False)
        self.assertIsNone(spoke.finalize())
        self.assertEqual(calls, ["checkpoint", "load"])


class TestAnEarlierStudysSpokeFilesAreCleared(unittest.TestCase):
    """A spoke overwrites only its own file, so a run started in a directory
    an earlier study used would otherwise leave that study's spoke files
    there, and a later resume would restore one into a spoke of the same name
    -- or die on it when the configuration differs."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        self.stale = os.path.join(self.ckpt_dir, "spokes",
                                  "spoke_XhatShuffleInnerBound_ordinal_00_"
                                  "rank_0000.pkl")
        os.makedirs(os.path.dirname(self.stale))
        with open(self.stale, "wb"):
            pass

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_fresh_run_clears_them(self):
        _make_ph(_options(1, ckpt_dir=self.ckpt_dir))
        self.assertFalse(os.path.exists(self.stale))

    def test_resuming_from_another_directory_clears_them(self):
        other = os.path.join(self._tmp.name, "other")
        _make_ph(_options(1, ckpt_dir=self.ckpt_dir, resume_from=other))
        self.assertFalse(os.path.exists(self.stale))

    def test_resuming_in_place_keeps_them(self):
        """They are this study's, and the spokes are about to read them."""
        _make_ph(_options(1, ckpt_dir=self.ckpt_dir,
                          resume_from=self.ckpt_dir))
        self.assertTrue(os.path.exists(self.stale))

    def test_a_dual_cylinder_does_not_clear_them(self):
        """relaxed_ph and ph_dual run PH without being the hub. A second
        cylinder deleting the same directory at the same moment as the hub
        races it, and the loser's rmtree raises on a file already gone."""
        _make_ph(_options(1, ckpt_dir=self.ckpt_dir,
                          checkpoint_role="dual_spoke"))
        self.assertTrue(os.path.exists(self.stale))

    def test_a_spoke_does_not_clear_them(self):
        """Only the hub clears: it is the one cylinder whose setup is
        certain to come before every spoke's first write."""
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        spoke = _SpokeStub()
        opt.spcomm = spoke
        Checkpointer(opt)
        self.assertTrue(os.path.exists(self.stale))


class TestSpokeInnerBoundsToRestore(unittest.TestCase):
    """What the resumed hub credits, read from the spokes' own files: only a
    bound a spoke holds a solution for may be credited to it, because the
    credited cylinder writes the solution."""

    class Hub:
        pass

    class Lagrangian:
        pass

    class XhatShuffle:
        pass

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = self._tmp.name
        os.makedirs(os.path.join(self.ckpt_dir, "spokes"))

    def tearDown(self):
        self._tmp.cleanup()

    def _file(self, cylinder, ordinal, state, rank=0):
        path = os.path.join(
            self.ckpt_dir, "spokes",
            checkpointing._spoke_filename(cylinder, ordinal, rank))
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def _incumbent(self, cylinder, ordinal, bound, rank=0, n_proc=1,
                   objective=None):
        self._file(cylinder, ordinal, {
            "format_version": checkpointing.FORMAT_VERSION,
            "kind": "spoke-incumbent", "best_inner_bound": bound,
            "best_solution_obj_val": bound if objective is None else objective,
            "geometry": {"n_proc": n_proc, "rank": rank}}, rank=rank)

    def _found(self, *classes):
        return checkpointing.spoke_inner_bounds_to_restore(
            [{"spcomm_class": cls} for cls in classes], self.ckpt_dir)

    def test_each_spoke_gets_its_own_file(self):
        self._incumbent("XhatShuffle", 0, -100.0)
        self._incumbent("XhatShuffle", 1, -90.0)
        self.assertEqual(
            self._found(self.Hub, self.XhatShuffle, self.Lagrangian,
                        self.XhatShuffle),
            [(1, -100.0), (3, -90.0)])

    def test_found_after_an_unrelated_cylinder_is_dropped(self):
        self._incumbent("XhatShuffle", 0, -100.0)
        self.assertEqual(self._found(self.Hub, self.XhatShuffle),
                         [(1, -100.0)])

    def test_nothing_for_a_spoke_this_run_does_not_have(self):
        self._incumbent("XhatXbar", 0, -100.0)
        self.assertEqual(self._found(self.Hub, self.XhatShuffle), [])

    def test_a_file_that_is_not_an_incumbent_is_skipped(self):
        for name, state in (
                ("XhatShuffle", {"format_version": -1,
                                 "kind": "spoke-incumbent",
                                 "best_inner_bound": -100.0}),
                ("Lagrangian", {"format_version":
                                checkpointing.FORMAT_VERSION,
                                "kind": "something-else",
                                "best_inner_bound": -100.0})):
            self._file(name, 0, state)
        self.assertEqual(
            self._found(self.Hub, self.XhatShuffle, self.Lagrangian), [])

    def test_a_file_without_a_usable_bound_is_skipped(self):
        for bound in (None, math.inf, "not a number"):
            with self.subTest(bound=bound):
                self._incumbent("XhatShuffle", 0, bound)
                self.assertEqual(self._found(self.Hub, self.XhatShuffle), [])

    def test_a_spoke_whose_ranks_agree_is_credited(self):
        for rank in (0, 1):
            self._incumbent("XhatShuffle", 0, -100.0, rank=rank, n_proc=2)
        self.assertEqual(self._found(self.Hub, self.XhatShuffle),
                         [(1, -100.0)])

    def test_a_spoke_with_a_rank_missing_is_not_credited(self):
        """The spoke's ranks drop an incumbent one of them has no file for,
        so the hub must not credit it: the spoke would write a solution that
        is not the one whose objective the hub reports."""
        self._incumbent("XhatShuffle", 0, -100.0, rank=0, n_proc=2)
        self.assertEqual(self._found(self.Hub, self.XhatShuffle), [])

    def test_a_spoke_whose_ranks_disagree_is_not_credited(self):
        self._incumbent("XhatShuffle", 0, -100.0, rank=0, n_proc=2)
        self._incumbent("XhatShuffle", 0, -90.0, rank=1, n_proc=2)
        self.assertEqual(self._found(self.Hub, self.XhatShuffle), [])
        self._incumbent("XhatShuffle", 0, -100.0, rank=1, n_proc=2,
                        objective=-99.0)
        self.assertEqual(self._found(self.Hub, self.XhatShuffle), [])

    def test_ranks_agreeing_on_the_objective_are_credited_rank_0s_bound(self):
        """The spoke's ranks compare only the objective (agree_spoke_restore)
        and then all take rank 0's inner bound, which is what they publish,
        so the hub credits exactly that."""
        self._incumbent("XhatShuffle", 0, -100.0, rank=0, n_proc=2)
        self._incumbent("XhatShuffle", 0, -90.0, rank=1, n_proc=2,
                        objective=-100.0)
        self.assertEqual(self._found(self.Hub, self.XhatShuffle),
                         [(1, -100.0)])

    def test_a_pickle_that_is_not_a_dict_is_skipped(self):
        self._file("XhatShuffle", 0, ["not", "a", "dict"])
        self.assertEqual(self._found(self.Hub, self.XhatShuffle), [])

    def test_an_unreadable_file_is_skipped(self):
        """Raising would stop hub rank 0 alone, with the other hub ranks
        waiting on it."""
        path = os.path.join(self.ckpt_dir, "spokes",
                            checkpointing._spoke_filename("XhatShuffle", 0, 0))
        with open(path, "wb") as f:
            f.write(b"not a pickle")
        self.assertEqual(self._found(self.Hub, self.XhatShuffle), [])


class TestUnknownBackend(unittest.TestCase):
    def test_require_dill_ignores_other_backends(self):
        # Only the dill-reload backend needs dill; nothing should raise here.
        checkpointing.require_dill(checkpointing.LEAF_BACKEND)


@unittest.skipIf(not solver_available,
                 "no solver is available for the varid map tests")
class TestVaridToNonantIndexRestored(unittest.TestCase):
    """varid_to_nonant_index is keyed by id(vardata), so it cannot cross a
    checkpoint on its own: dill brings the dict back holding the addresses of
    the objects that were written, which say nothing about the objects that
    came out. It looks intact and maps nothing."""

    STOP = 2

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        stopped = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        stopped.ph_main()
        self.resumed = _make_ph(_options(0, resume_from=self.ckpt_dir))
        self.resumed.ph_main()

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_map_addresses_the_restored_models(self):
        for sname, s in self.resumed.local_scenarios.items():
            varids = s._mpisppy_data.varid_to_nonant_index
            for ndn_i, var in s._mpisppy_data.nonant_indices.items():
                self.assertIn(
                    id(var), varids,
                    msg=f"{sname} nonant {ndn_i} is not in "
                        f"varid_to_nonant_index after the resume; the map "
                        f"still holds the writing process's ids")
                self.assertEqual(varids[id(var)], ndn_i)
            self.assertEqual(len(varids), len(s._mpisppy_data.nonant_indices),
                             msg=f"{sname}: stale ids were left behind")

    def test_the_consumer_that_crashes_does_not(self):
        """is_zero_prob indexes the map rather than testing membership, so a
        stale one raises. It is reached from _check_staleness after every
        solve and from gather_var_values_to_rank0 when a run writes its
        solution, on any run that sets variable_probability."""
        self.resumed.variable_probability = {}     # take the early return out
        for sname, s in self.resumed.local_scenarios.items():
            for var in s._mpisppy_data.nonant_indices.values():
                self.resumed.is_zero_prob(s, var)  # must not raise


class TestIntegerRelaxThenEnforceResume(unittest.TestCase):
    """The transformation records its undo map in a _relaxed_integer_vars
    Suffix. Applying it twice replaces that Suffix with an empty one, so the
    undo restores nothing while still reporting that it enforced integrality
    -- and the run answers the relaxation."""

    def _extension(self, scenarios, resumed):
        import types
        from mpisppy.extensions.integer_relax_then_enforce import (
            IntegerRelaxThenEnforce)
        opt = types.SimpleNamespace(
            options={"integer_relax_then_enforce_options": {"ratio": 0.5}},
            local_scenarios=scenarios,
            cylinder_rank=0,
            _resumed_from_checkpoint=resumed,
        )
        return IntegerRelaxThenEnforce(opt)

    def _model(self):
        import pyomo.environ as pyo
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, 10))
        m.obj = pyo.Objective(expr=m.x)
        m._solver_plugin = None
        return m

    def test_a_relaxed_model_is_not_relaxed_a_second_time(self):
        m = self._model()
        ext = self._extension({"s": m}, resumed=False)
        ext.pre_iter0()                       # the run that wrote the checkpoint
        undo_map = m._relaxed_integer_vars

        resumed = self._extension({"s": m}, resumed=True)
        resumed.integer_relaxer = mock.Mock()
        resumed.pre_iter0()

        resumed.integer_relaxer.apply_to.assert_not_called()
        self.assertTrue(resumed._integers_relaxed,
                        msg="a resumed run must know its models are relaxed")
        self.assertIs(m._relaxed_integer_vars, undo_map,
                      msg="the undo map was replaced, so unrelaxing would "
                          "restore nothing")

    def test_the_integrality_actually_comes_back(self):
        m = self._model()
        self._extension({"s": m}, resumed=False).pre_iter0()
        self.assertFalse(m.x.is_integer())

        resumed = self._extension({"s": m}, resumed=True)
        resumed.pre_iter0()
        resumed._unrelax_integers()

        self.assertTrue(
            m.x.is_integer(),
            msg="the resumed run reports that it enforced integrality while "
                "still solving the relaxation")

    def test_models_that_are_not_relaxed_leave_integrality_alone(self):
        """A checkpoint carries no extension state, so a run that had already
        enforced integrality and one that never had this extension look the
        same from here. Relaxing now would change the algorithm mid-study."""
        m = self._model()
        resumed = self._extension({"s": m}, resumed=True)
        resumed.integer_relaxer = mock.Mock()
        resumed.pre_iter0()

        resumed.integer_relaxer.apply_to.assert_not_called()
        self.assertFalse(resumed._integers_relaxed)
        self.assertTrue(m.x.is_integer())


@unittest.skipIf(not solver_available,
                 "no solver is available for the dynamic rho resume tests")
class TestDynRhoCachesSurviveResume(unittest.TestCase):
    """post_iter0 skips the rho recomputation on a resume, but it is also the
    only thing that seeds primal_conv_cache and the WTracker's first W set --
    and miditer reads both on the very next pass."""

    STOP = 2

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _sep_rho_options(self, *args, **kwargs):
        import types
        options = _options(*args, **kwargs)
        options["sep_rho_options"] = {"cfg": types.SimpleNamespace()}
        return options

    def test_a_resumed_run_keeps_iterating(self):
        stopped = _make_ph(self._sep_rho_options(self.STOP,
                                                 ckpt_dir=self.ckpt_dir),
                           extension_name="sep_rho")
        stopped.ph_main()

        # A non-zero budget is the point: the failure was at the first
        # miditer of the resumed leg, which a zero-iteration resume never
        # reaches.
        resumed = _make_ph(self._sep_rho_options(2,
                                                 resume_from=self.ckpt_dir),
                           extension_name="sep_rho")
        resumed.ph_main()       # must not raise

        self.assertEqual(resumed._PHIter, self.STOP + 2)
        ext = resumed.extobject
        self.assertTrue(ext.primal_conv_cache,
                        msg="the convergence cache was never seeded")
        self.assertGreaterEqual(
            len(ext.wt.local_Ws), 2,
            msg="the WTracker never grabbed a W set on the resumed run")


@unittest.skipIf(not solver_available,
                 "no solver is available for the dynamic rho resume tests")
class TestGradRhoReadsTheUsersGradientOnAResume(unittest.TestCase):
    """GradRho differentiates each scenario's objective when its post_iter0
    hook runs. On a fresh run that objective is the user's, because PH
    attaches its W and prox terms only after the hook. On a resume the
    reloaded models already carry them, so the partials also hold
    W_on*W and prox_on*rho*(x - xbar): the gradient of the PH subproblem
    rather than of the user's cost, and every rho recomputed after the
    resume was scaled from it. The extension masks the two terms while it
    evaluates, which removes them exactly."""

    STOP = 2

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _resumed_grad_rho(self):
        from mpisppy.extensions.grad_rho import GradRho
        stopped = _make_ph(_options(self.STOP, ckpt_dir=self.ckpt_dir))
        stopped.ph_main()
        resumed = _make_ph(_options(0, resume_from=self.ckpt_dir))
        resumed.ph_main()
        self.assertTrue(resumed._resumed_from_checkpoint)
        # The preconditions that let this test discriminate: the reloaded
        # objectives carry the PH terms and they are switched on, and the
        # duals are not all zero, so the W term has something to add.
        for s in resumed.local_scenarios.values():
            self.assertEqual(s._mpisppy_model.W_on.value, 1)
            self.assertEqual(s._mpisppy_model.prox_on.value, 1)
        self.assertTrue(any(w.value != 0
                            for s in resumed.local_scenarios.values()
                            for w in s._mpisppy_model.W.values()))
        cfg = Config()
        cfg.gradient_args()
        cfg.dynamic_rho_args()
        cfg.grad_order_stat = 0.5
        resumed.options["grad_rho_options"] = {"cfg": cfg}
        grad = GradRho(resumed)
        grad._get_grad_exprs()
        return resumed, grad

    def test_the_gradient_is_of_the_users_objective(self):
        import pyomo.environ as pyo
        resumed, grad = self._resumed_grad_rho()
        for s in resumed.local_scenarios.values():
            # Farmer's objective is linear in the nonants, so the partial
            # with respect to each DevotedAcreage is its planting cost --
            # and it does not depend on W, rho or xbar.
            want = {ndn_i: pyo.value(s.PlantingCostPerAcre[v.index()])
                    for ndn_i, v in s._mpisppy_data.nonant_indices.items()}
            self.assertEqual(grad._eval_grad_exprs(s, None), want)

    def test_the_flags_are_off_while_the_partials_are_evaluated(self):
        """And back on afterwards.

        Asserting only that they are back on afterwards does not
        discriminate: code that never touched them passes that too. What
        has to be observed is their value at the moment the partials are
        read, so the partials here are stand-ins that report it.
        """
        resumed, grad = self._resumed_grad_rho()
        for s in resumed.local_scenarios.values():
            ph = s._mpisppy_model
            grad.grad_exprs[s] = {
                ndn_i: ph.W_on + ph.prox_on
                for ndn_i in s._mpisppy_data.nonant_indices
            }
            observed = grad._eval_grad_exprs(s, None)
            self.assertTrue(
                observed and all(v == 0 for v in observed.values()),
                msg=f"the PH terms were live while the user's gradient was "
                    f"evaluated: {observed}")
            self.assertEqual(ph.W_on.value, 1)
            self.assertEqual(ph.prox_on.value, 1)


@unittest.skipIf(not solver_available,
                 "no solver is available for the wtracker resume test")
class TestWtrackerReportDoesNotReachPastTheStop(unittest.TestCase):
    """--wtracker's end-of-run report reads a window of W sets. A resumed run
    holds only the sets it grabbed itself, and the window was indexed as if
    every iteration since 1 were there, so a resumed leg shorter than the
    window died with a KeyError in post_everything -- after every solve had
    been paid for. The window now starts no earlier than the first tracked
    set, and a leg too short for it reports "not enough iterations" the way
    a short uninterrupted run does."""

    STOP = 4
    #: Shorter than the window, so the report reaches back past the stop.
    RESUMED = 3
    WLEN = 3

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _ph(self, max_iters, leg, **ckpt_kwargs):
        from mpisppy.extensions.extension import MultiExtension
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        options = _options(max_iters, **ckpt_kwargs)
        options["wtracker_options"] = {
            "wlen": self.WLEN,
            "file_prefix": os.path.join(self._tmp.name, leg)}
        classes = [Wtracker_extension]
        if "checkpoint_dir" in options:
            classes.insert(0, Checkpointer)
        return PH(options, SCENARIO_NAMES, farmer.scenario_creator,
                  farmer.scenario_denouement,
                  scenario_creator_kwargs=CREATOR_KWARGS,
                  extensions=MultiExtension,
                  extension_kwargs={"ext_classes": classes})

    def test_the_resumed_run_reports_instead_of_raising(self):
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        self.assertLess(self.RESUMED, self.WLEN + 1)
        self._ph(self.STOP, "stopped", ckpt_dir=self.ckpt_dir).ph_main()
        resumed = self._ph(self.RESUMED, "resumed", resume_from=self.ckpt_dir)
        resumed.ph_main()       # raised KeyError from post_everything
        self.assertTrue(resumed._resumed_from_checkpoint)
        summary = os.path.join(
            self._tmp.name,
            f"resumed_summary_iter{self.STOP + self.RESUMED}"
            f"_rank{resumed.global_rank}.txt")
        with open(summary) as f:
            report = f.read()
        self.assertIn(f"W Report at iteration {self.STOP + self.RESUMED}",
                      report)
        # Extension.checkpoint_state itself arrives with a later phase.
        if (getattr(Wtracker_extension, "checkpoint_state", None)
                is getattr(Extension, "checkpoint_state", None)):
            # The tracker holds only the resumed leg's W sets: too few for
            # the window, said the same way a short uninterrupted run says it.
            self.assertIn("Not enough iterations tracked", report)
        else:
            # A later phase carries the window across the checkpoint, and
            # then the resumed run writes the full report.
            self.assertIn("Sorted by windowed stdev", report)


@unittest.skipIf(not solver_available,
                 "no solver is available for the wtracker resume test")
class TestWtrackerAskedForOnlyOnTheResumedLeg(unittest.TestCase):
    """``--wtracker`` on the command that resumes, not the one that stopped.

    Then nothing carried the earlier W sets -- no phase of this stack
    carries what was never tracked -- so the tracker holds only the sets the
    resumed leg grabbed while ``ph_iter`` counts from where the study left
    off. That is the gap the window has to start at rather than index past,
    and it is the same failure whether or not the extension can checkpoint
    its own state, which is what makes this the test that still guards the
    window once the whole stack is merged.

    The window arithmetic itself is tested without a solver, and without
    checkpointing, in test_ph_extensions.py::TestWtrackerWindow. What is
    left here is the end-to-end path: that a real resumed run reaches it.
    """

    STOP = 4
    #: Shorter than the window, so a report that reached back past the stop
    #: would look for sets from before it.
    RESUMED = 3
    WLEN = 3

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def _ph(self, max_iters, leg, wtracker, **ckpt_kwargs):
        from mpisppy.extensions.extension import MultiExtension
        from mpisppy.extensions.wtracker_extension import Wtracker_extension
        options = _options(max_iters, **ckpt_kwargs)
        classes = []
        if "checkpoint_dir" in options:
            classes.append(Checkpointer)
        if wtracker:
            options["wtracker_options"] = {
                "wlen": self.WLEN,
                "file_prefix": os.path.join(self._tmp.name, leg)}
            classes.append(Wtracker_extension)
        return PH(options, SCENARIO_NAMES, farmer.scenario_creator,
                  farmer.scenario_denouement,
                  scenario_creator_kwargs=CREATOR_KWARGS,
                  extensions=MultiExtension,
                  extension_kwargs={"ext_classes": classes})

    def test_the_resumed_run_reports_instead_of_raising(self):
        self._ph(self.STOP, "stopped", wtracker=False,
                 ckpt_dir=self.ckpt_dir).ph_main()
        resumed = self._ph(self.RESUMED, "resumed", wtracker=True,
                           resume_from=self.ckpt_dir)
        resumed.ph_main()       # raised KeyError from post_everything
        self.assertTrue(resumed._resumed_from_checkpoint)
        summary = os.path.join(
            self._tmp.name,
            f"resumed_summary_iter{self.STOP + self.RESUMED}"
            f"_rank{resumed.global_rank}.txt")
        with open(summary) as f:
            report = f.read()
        self.assertIn("Not enough iterations tracked", report)
        # The count and the threshold have to be readable against each
        # other, or "3 tracked for window len 3" reads as enough.
        self.assertIn(f"spans {self.WLEN + 1}", report)


class TestWXBarReaderResume(unittest.TestCase):
    """pre_iter0 runs after the checkpoint's models are spliced in, so reading
    an --init-W-fname there overwrites the checkpointed duals with the values
    the study started from -- and the documented workflow submits the same
    command every morning, so the flag is still on it."""

    def _reader(self, resumed):
        import types
        from mpisppy.utils.w_utils.wxbarreader import WXBarReader
        reader = WXBarReader.__new__(WXBarReader)
        reader.not_active = False
        reader.w_fname = "W0.csv"
        reader.x_fname = None
        reader.sep_files = False
        reader.cylinder_rank = 0
        reader.PHB = types.SimpleNamespace(_resumed_from_checkpoint=resumed)
        return reader

    def test_a_resumed_run_does_not_read_the_files(self):
        import mpisppy.utils.w_utils.wxbarutils as wxbarutils
        with mock.patch.object(wxbarutils, "set_W_from_file") as set_W:
            self._reader(resumed=True).pre_iter0()
        set_W.assert_not_called()

    def test_an_ordinary_run_still_reads_them(self):
        import mpisppy.utils.w_utils.wxbarutils as wxbarutils
        reader = self._reader(resumed=False)
        reader.PHB._reenable_W = mock.Mock()
        with mock.patch.object(wxbarutils, "set_W_from_file") as set_W:
            reader.pre_iter0()
        set_W.assert_called_once()

    def _ph_stub(self, resume_from):
        import types
        cfg = Config()
        from mpisppy.utils.w_utils.wxbarreader import add_options_to_config
        add_options_to_config(cfg)
        cfg.W_and_xbar_reader = True
        cfg.init_W_fname = os.path.join(tempfile.gettempdir(),
                                        "mpisppy_no_such_W0.csv")
        cfg.init_Xbar_fname = os.path.join(tempfile.gettempdir(),
                                           "mpisppy_no_such_xbar0.csv")
        return types.SimpleNamespace(
            options={"cfg": cfg, "resume_from": resume_from})

    def test_a_resume_does_not_require_the_files_to_exist(self):
        """Resubmitting the original command after the init files were
        cleaned up must not fail in the constructor, before the skip in
        pre_iter0 is ever reached."""
        from mpisppy.utils.w_utils.wxbarreader import WXBarReader
        WXBarReader(self._ph_stub(resume_from="./ckpt"))

    def test_a_fresh_run_still_requires_them(self):
        from mpisppy.utils.w_utils.wxbarreader import WXBarReader
        with self.assertRaisesRegex(RuntimeError, "Cannot find"):
            WXBarReader(self._ph_stub(resume_from=None))


class TestCheckpointingWithoutAHub(unittest.TestCase):
    """--EF and the three write-only modes branch off ahead of do_decomp, so
    the hub-side guard never sees them: they accept the checkpoint flags and
    act on neither, exiting 0."""

    MODES = ("--EF", "--pickle-bundles-dir", "--pickle-scenarios-dir",
             "--write-scenario-lp-mps-files-dir")

    def _cfg(self, **overrides):
        cfg = Config()
        cfg.checkpoint_args()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_checkpoint_dir_is_refused(self):
        from mpisppy.generic.decomp import refuse_checkpointing_without_a_hub
        for mode in self.MODES:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "--checkpoint-dir"):
                    refuse_checkpointing_without_a_hub(
                        self._cfg(checkpoint_dir="/tmp/nope"), mode)

    def test_resume_from_is_refused(self):
        from mpisppy.generic.decomp import refuse_checkpointing_without_a_hub
        for mode in self.MODES:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "--resume-from"):
                    refuse_checkpointing_without_a_hub(
                        self._cfg(resume_from="/tmp/nope"), mode)

    def test_a_run_without_the_flags_is_left_alone(self):
        from mpisppy.generic.decomp import refuse_checkpointing_without_a_hub
        for mode in self.MODES:
            with self.subTest(mode=mode):
                refuse_checkpointing_without_a_hub(self._cfg(), mode)


class TestSpokeIdentitySurvivesADifferentCylinderSet(unittest.TestCase):
    """Which cylinders run is on the list a resume may change, so a spoke's
    file cannot be named by its position in the wheel.

    NON_STRUCTURAL_CFG_KEYS carries lagrangian, xhatshuffle, fwph and the
    rest deliberately: the hub's iterate does not depend on the spokes, so a
    checkpoint stays valid across a different spoke set. But dropping a
    cylinder renumbers every cylinder after it, and the spoke file used to be
    named by that number. One spoke of a class then looked for a file that
    was not there while its own sat beside it under the old number; two
    spokes of one class was worse, because the shifted one found the *other*
    one's file under its own new number and restored an incumbent that was
    never its. Same class and same models, so the values are feasible and
    nothing downstream notices.
    """

    class _Hub:
        pass

    class _Lagrangian:
        pass

    class _XhatShuffle:
        pass

    def _identity(self, classes, strata_rank):
        """(ordinal, count) for the cylinder at strata_rank in this wheel."""
        from mpisppy.extensions.checkpointer import Checkpointer
        ext = Checkpointer.__new__(Checkpointer)
        spoke = classes[strata_rank].__new__(classes[strata_rank])
        spoke.strata_rank = strata_rank
        spoke.communicators = [{"spcomm_class": c} for c in classes]
        ext.opt = types.SimpleNamespace(spcomm=spoke)
        return ext._class_ordinal_and_count()

    def test_dropping_an_earlier_cylinder_does_not_move_the_ordinal(self):
        """The reported failure: resume without --lagrangian and the xhat
        spoke could not find the incumbent it had written."""
        wrote = self._identity(
            [self._Hub, self._Lagrangian, self._XhatShuffle], strata_rank=2)
        resumed = self._identity(
            [self._Hub, self._XhatShuffle], strata_rank=1)
        self.assertEqual(wrote, resumed,
                         msg="the spoke's file name moved because an "
                             "unrelated cylinder was dropped")

    def test_two_spokes_of_one_class_stay_apart(self):
        classes = [self._Hub, self._XhatShuffle, self._XhatShuffle]
        self.assertEqual(self._identity(classes, strata_rank=1), (0, 2))
        self.assertEqual(self._identity(classes, strata_rank=2), (1, 2))

    def test_they_stay_apart_when_an_unrelated_cylinder_goes(self):
        """The cross-assignment: with the ordinal, neither of the two moves
        onto the other's file."""
        before = [self._Hub, self._Lagrangian,
                  self._XhatShuffle, self._XhatShuffle]
        after = [self._Hub, self._XhatShuffle, self._XhatShuffle]
        self.assertEqual(self._identity(before, strata_rank=2),
                         self._identity(after, strata_rank=1))
        self.assertEqual(self._identity(before, strata_rank=3),
                         self._identity(after, strata_rank=2))

    def test_a_spoke_outside_a_wheel_is_the_only_one_of_its_class(self):
        from mpisppy.extensions.checkpointer import Checkpointer
        ext = Checkpointer.__new__(Checkpointer)
        ext.opt = types.SimpleNamespace(spcomm=_SpokeStub(strata_rank=2))
        self.assertEqual(ext._class_ordinal_and_count(), (0, 1))

    def test_the_file_name_carries_the_ordinal_not_the_strata_rank(self):
        self.assertEqual(
            checkpointing._spoke_filename("XhatShuffleInnerBound", 1, 0),
            "spoke_XhatShuffleInnerBound_ordinal_01_rank_0000.pkl")


class TestDroppingOneOfTwoSameClassSpokesIsReported(unittest.TestCase):
    """The one change the ordinal cannot absorb, so it is said out loud.

    Removing one of two same-class spokes makes the survivor's ordinal the
    removed one's, and it would read that spoke's file without a word. The
    written file records how many cylinders of its class the wheel carried,
    which is what makes the difference visible.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, class_count):
        opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        _set_and_cache_solution(opt, 4.0)
        checkpointing.write_spoke_incumbent(
            opt, self.ckpt_dir, "_SpokeStub", 0, best_inner_bound=4.0,
            class_count=class_count)

    def _resume_with(self, class_count, tocs):
        from mpisppy.extensions import checkpointer as mod
        opt = _xhat_eval(resume_from=self.ckpt_dir)
        # SPBase.spcomm is a weakref, so the stub needs an owner that
        # outlives this call or it is collected before the extension reads it.
        self.spoke = _SpokeStub(strata_rank=1)
        opt.spcomm = self.spoke
        ext = Checkpointer(opt)
        ext._class_ordinal_and_count = lambda: (0, class_count)
        with mock.patch.object(mod, "global_toc",
                               side_effect=lambda msg, *a, **k: tocs.append(msg)):
            ext._restore_incumbent()
        return opt

    def test_a_changed_count_is_reported(self):
        self._write(class_count=2)
        tocs = []
        opt = self._resume_with(1, tocs)
        self.assertEqual(opt.best_solution_obj_val, 4.0)  # still restored
        self.assertTrue(
            any("may be restoring an incumbent that belonged to a different"
                in m for m in tocs),
            msg=f"the identity change was not reported: {tocs}")

    def test_an_unchanged_count_says_nothing(self):
        self._write(class_count=2)
        tocs = []
        self._resume_with(2, tocs)
        self.assertFalse(
            any("belonged to a different" in m for m in tocs),
            msg=f"reported an identity change that did not happen: {tocs}")


class TestTheStudyBoundStaysOffDualCylinders(unittest.TestCase):
    """--stop-at-iteration-number counts hub iterations.

    A dual cylinder (--ph-dual, --relaxed-ph) counts its own iterations from 1
    on every run and is meant to run until the hub is done. Handed the study
    bound, it would stop after that many of its own iterations -- on a
    resumed run, long before the hub finishes -- and the hub would get no new
    duals from then on. Built through the real cfg_vanilla builders, so this
    also pins that both dual builders say they are dual cylinders. No solver.
    """

    BOUND = 7

    def _cylinders(self):
        import mpisppy.utils.cfg_vanilla as vanilla

        cfg = Config()
        cfg.popular_args()
        cfg.ph_args()
        cfg.two_sided_args()
        cfg.relaxed_ph_args()
        cfg.ph_dual_args()
        cfg.checkpoint_args()
        farmer.inparser_adder(cfg)
        cfg.num_scens = 3
        cfg.default_rho = 1.0
        cfg.solver_name = "unused"
        cfg.max_iterations = 10
        cfg.stop_at_iteration_number = self.BOUND
        beans = (cfg, farmer.scenario_creator, farmer.scenario_denouement,
                 farmer.scenario_names_creator(3))
        kwargs = {"scenario_creator_kwargs": farmer.kw_creator(cfg)}
        return (vanilla.ph_hub(*beans, **kwargs),
                vanilla.ph_dual_spoke(*beans, **kwargs),
                vanilla.relaxed_ph_spoke(*beans, **kwargs))

    def test_the_dual_cylinders_do_not_get_it(self):
        _, ph_dual, relaxed_ph = self._cylinders()
        for name, cylinder in (("ph_dual", ph_dual),
                               ("relaxed_ph", relaxed_ph)):
            self.assertNotIn(
                "stop_at_iteration_number",
                cylinder["opt_kwargs"]["options"],
                msg=f"the {name} cylinder was given the study bound, so it "
                    f"would stop after that many of its own iterations")

    def test_the_hub_does(self):
        hub, _, _ = self._cylinders()
        self.assertEqual(
            hub["opt_kwargs"]["options"]["stop_at_iteration_number"],
            self.BOUND)


class TestChildProcessesImportTheCheckoutUnderTest(unittest.TestCase):
    """The mpiexec legs and fresh-process resumes must run this checkout.

    With an editable install, a child Python process imports ``mpisppy``
    from wherever it was installed from. Run from a second worktree, the
    tests would then compare that other checkout's code against itself and
    pass or fail on the wrong code.
    """

    def test_a_child_imports_this_checkout(self):
        # Importing mpisppy prints a banner, so the path is marked.
        result = subprocess.run(
            [sys.executable, "-c",
             "import mpisppy; print('MPISPPY_FILE=' + mpisppy.__file__)"],
            capture_output=True, text=True, timeout=120, check=True,
            env=subprocess_env(),
            cwd=tempfile.gettempdir(),
        )
        marked = [line for line in result.stdout.splitlines()
                  if line.startswith("MPISPPY_FILE=")]
        self.assertEqual(len(marked), 1, msg=result.stdout)
        child = os.path.realpath(marked[0][len("MPISPPY_FILE="):])
        self.assertTrue(
            child.startswith(os.path.realpath(REPO_ROOT) + os.sep),
            msg=f"the child imported {child}, not the checkout at {REPO_ROOT}")

    #: os functions that start a process or replace or fork this one. They
    #: are not checked for the environment they are given, since some take it
    #: positionally and some not at all; the checkpoint tests must use
    #: subprocess instead. Matched as name prefixes.
    OS_LAUNCHERS = ("system", "popen", "exec", "spawn", "posix_spawn", "fork",
                    "startfile")
    #: Launchers that take env=, by module, which must be given
    #: env=subprocess_env().
    ASYNCIO_LAUNCHERS = {"create_subprocess_exec", "create_subprocess_shell"}
    ENV_LAUNCHERS = {
        "subprocess": {"run", "Popen", "call", "check_call", "check_output"},
        "asyncio": ASYNCIO_LAUNCHERS,
        "asyncio.subprocess": ASYNCIO_LAUNCHERS,
    }
    #: subprocess functions that start a process but take no env=.
    NO_ENV_LAUNCHERS = {"getoutput", "getstatusoutput"}

    def test_every_launch_in_the_checkpoint_tests_passes_the_environment(self):
        """Every process the checkpoint tests or their drivers start is given
        env=subprocess_env(), however the launcher was imported.

        Resolves the module behind a call through ``import x``,
        ``import x as y``, ``import x.sub`` (which binds ``x``),
        ``from x import f [as g]`` and ``from x import sub [as s]``, and
        through dotted chains such as ``asyncio.subprocess.f``. Passing some
        env= is not enough: env=None or env=os.environ is the inherited
        environment again, so the value must be the call
        ``subprocess_env()`` itself, written as a keyword.

        Bindings are collected from the whole file regardless of scope, and
        a name bound more than one way (``import subprocess`` at the top and
        ``from asyncio import subprocess`` inside a function, say) is
        checked as every one of them, so neither hides the other.

        Not followed, so a launch written these ways is not checked:
        ``from x import *``; a module or launcher held in a variable or
        passed along (``f = os.system``, ``functools.partial``, getattr);
        a module reached as an attribute of another (``subprocess.os.system``,
        ``asyncio.subprocess.subprocess.Popen``); an event loop's
        ``subprocess_exec``/``subprocess_shell``, however the loop was
        obtained; launchers outside os, subprocess and asyncio, such as
        ``pty.spawn``, ``multiprocessing`` or mpi4py's ``Comm.Spawn``.

        Reported although correct: a launch whose env is built first and
        passed as a variable, or through ``**kwargs``; a local name that
        shadows ``os`` or ``subprocess``; and, since every binding of a name
        is checked, a name imported from two of these modules in different
        places -- ``from subprocess import run`` in one function and
        ``from asyncio import run`` in another makes ``run(main())`` read as
        a subprocess launch.
        """
        import ast
        import glob
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        paths = sorted(glob.glob(os.path.join(tests_dir, "test_checkpoint*.py"))
                       + glob.glob(os.path.join(tests_dir, "*_driver.py")))
        self.assertIn("test_checkpoint_multirank.py",
                      {os.path.basename(p) for p in paths})
        watched = ("os", "subprocess", "asyncio")
        problems = []
        for path in paths:
            with open(path) as f:
                tree = ast.parse(f.read())
            # A name in this file -> every dotted path it is bound to.
            bound = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top not in watched:
                            continue
                        if alias.asname is None:
                            bound.setdefault(top, set()).add(top)
                        else:
                            bound.setdefault(alias.asname, set()).add(
                                alias.name)
                elif (isinstance(node, ast.ImportFrom) and node.module
                        and node.module.split(".")[0] in watched):
                    for alias in node.names:
                        bound.setdefault(alias.asname or alias.name,
                                         set()).add(
                            f"{node.module}.{alias.name}")

            def dotted(expr):
                """Every dotted path expr can mean, given the bindings."""
                if isinstance(expr, ast.Name):
                    return bound.get(expr.id, set())
                if isinstance(expr, ast.Attribute):
                    return {f"{base}.{expr.attr}"
                            for base in dotted(expr.value)}
                return set()

            for node, target in ((node, target)
                                 for node in ast.walk(tree)
                                 if isinstance(node, ast.Call)
                                 for target in sorted(dotted(node.func))):
                if "." not in target:
                    continue
                module, name = target.rsplit(".", 1)
                where = f"{os.path.basename(path)}:{node.lineno}"
                if module == "os" and name.startswith(self.OS_LAUNCHERS):
                    problems.append(f"{where} starts a process through "
                                    f"os.{name}; use subprocess with "
                                    f"env=subprocess_env()")
                elif module == "subprocess" and name in self.NO_ENV_LAUNCHERS:
                    problems.append(f"{where} starts a process through "
                                    f"subprocess.{name}, which takes no env=")
                elif name in self.ENV_LAUNCHERS.get(module, ()):
                    env = next((k.value for k in node.keywords
                                if k.arg == "env"), None)
                    if not (isinstance(env, ast.Call)
                            and isinstance(env.func, ast.Name)
                            and env.func.id == "subprocess_env"):
                        problems.append(f"{where} does not pass "
                                        f"env=subprocess_env()")
        self.assertEqual(problems, [],
                         msg="these launches may start a child that imports "
                             "a different checkout's mpisppy")


if __name__ == "__main__":
    unittest.main()
