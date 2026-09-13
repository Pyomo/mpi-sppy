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
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import mpisppy.utils.checkpointing as checkpointing
import mpisppy.tests.examples.farmer as farmer
from mpisppy.extensions.checkpointer import Checkpointer
from mpisppy.extensions.extension import Extension
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
        options["checkpoint_backend"] = checkpointing.DILL_RELOAD_BACKEND
        options["checkpoint_every_iterations"] = 1
    if resume_from is not None:
        options["resume_from"] = resume_from
    options.update(overrides)
    return options


#: The iteration at which ClockRewinder trips the --time-limit break.
_LATE_STOP_ITERATION = 3


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


class PreIter0Tagger(Extension):
    """Tag the models pre_iter0 sees, so a test can tell whether the hook ran
    on the models the run actually iterates or on fresh ones a resume splice
    discarded."""

    def pre_iter0(self):
        for s in self.opt.local_scenarios.values():
            s._pre_iter0_saw_this_model = True


def _extension_class(name):
    if name is None:
        return None
    if name == "norm_rho":
        from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
        return NormRhoUpdater
    if name == "miditer_mutator":
        return MidIterMutator
    if name == "recorder":
        return StateRecorder
    if name == "coeff_rho":
        from mpisppy.extensions.coeff_rho import CoeffRho
        return CoeffRho
    if name == "pre_tagger":
        return PreIter0Tagger
    if name == "clock_rewinder":
        return ClockRewinder
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

    def test_multirank_is_refused_at_setup(self):
        with self.assertRaises(RuntimeError) as ctx:
            Checkpointer(self._stub(n_proc=2))
        self.assertIn("single rank", str(ctx.exception))

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
                     "checkpoint_every_iterations", "resume_from",
                     "stop_at_iteration_number"):
            self.assertIn(name, self.cfg)

    def test_cadence_defaults_to_every_iteration(self):
        self.assertEqual(self.cfg.checkpoint_every_iterations, 1)

    def test_checkpointing_is_off_by_default(self):
        self.assertIsNone(self.cfg.checkpoint_dir)
        self.assertIsNone(self.cfg.resume_from)

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

    def test_a_tracker_holding_nothing_reports_instead_of_raising(self):
        """The same window, with no sets at all.

        A WTracker built after the run -- the pattern in wtracker's own
        __main__ block -- has grabbed nothing, and asked for a window it
        indexed a dict that has no such keys.
        """
        from mpisppy.utils.w_utils.wtracker import WTracker
        ph = self._ph(self.STOP, "direct", wtracker=False)
        ph.ph_main()
        tracker = WTracker(ph)
        self.assertEqual(tracker.local_Ws, {})
        result = tracker.compute_moving_stats(self.WLEN)
        self.assertIsInstance(
            result, str,
            msg="a tracker holding nothing computed a window instead of "
                "saying it had too few iterations")
        self.assertIn("Not enough iterations tracked", result)


@unittest.skipIf(not solver_available,
                 "no solver is available for the wtracker window tests")
class TestTheOtherWindowReadersReadTheSameWindow(unittest.TestCase):
    """``check_cross_zero`` and ``check_w_stdev`` read the window the report
    reads.

    There are three readers of it, and the fix for a resumed run -- whose W
    sets do not start at iteration 1 -- reached only the report. The other
    two went on computing their window from 1, so on the very runs the
    report now handles they raised instead: KeyError from a window that
    starts before anything was tracked, and ValueError from unpacking the
    sentence that says so as though it were a pair of statistics.
    ``check_cross_zero`` also read iteration 0 on a fresh run, which is
    never tracked, and compared the same pair of sets on every pass of its
    loop whatever the window.

    Neither method is called from mpi-sppy today. Both are public, and
    ``doc/designs/w_oscillation_design.md`` names ``check_cross_zero`` as
    the thing that design succeeds.
    """

    WLEN = 3

    def _tracker(self, iterations):
        """A tracker holding one W set at each of ``iterations``."""
        from mpisppy.utils.w_utils.wtracker import WTracker
        ph = PH(_options(1), SCENARIO_NAMES, farmer.scenario_creator,
                farmer.scenario_denouement,
                scenario_creator_kwargs=CREATOR_KWARGS)
        ph.ph_main()
        tracker = WTracker(ph)
        for iteration in iterations:
            # What grab_local_Ws keys the set it grabs by.
            ph._PHIter = iteration
            tracker.grab_local_Ws()
        self.assertEqual(sorted(tracker.local_Ws), list(iterations))
        return tracker

    def test_a_window_shorter_than_asked_for_is_reported_not_raised(self):
        """The shape a run resumed at iteration 4 and stopped at 7 leaves."""
        tracker = self._tracker([5, 6, 7])
        # One subTest each: they fail differently -- KeyError from the one
        # that indexes the window, ValueError from the one that unpacks the
        # sentence -- and asking them in one loop hides the second.
        for reader in ("check_cross_zero", "check_w_stdev"):
            with self.subTest(reader=reader):
                answer = (tracker.check_cross_zero(self.WLEN)
                          if reader == "check_cross_zero"
                          else tracker.check_w_stdev(self.WLEN, 1e-3))
                self.assertIsInstance(answer, str)
                self.assertIn("Not enough iterations tracked", answer)
                self.assertTrue(answer.startswith("WTRACKER"), msg=answer)

    def test_the_first_window_of_a_fresh_run_does_not_read_iteration_zero(
            self):
        """Four sets and a window of three is the first window a fresh run
        can report, and iteration 0 is not one of the four."""
        tracker = self._tracker([1, 2, 3, 4])
        answer = tracker.check_cross_zero(self.WLEN)
        self.assertNotIn("Not enough iterations", answer)
        self.assertTrue(answer.startswith("WTRACKER"), msg=answer)

    def test_a_window_after_a_resume_is_read_where_the_sets_are(self):
        """And a resumed run with enough sets gets its answer, from the
        window that ends at the last iteration rather than one starting at
        an iteration nothing tracked."""
        tracker = self._tracker([5, 6, 7, 8, 9, 10])
        for reader in ("check_cross_zero", "check_w_stdev"):
            with self.subTest(reader=reader):
                answer = (tracker.check_cross_zero(self.WLEN)
                          if reader == "check_cross_zero"
                          else tracker.check_w_stdev(self.WLEN, 1e-3))
                self.assertNotIn("Not enough iterations", answer)
                self.assertTrue(answer.startswith("WTRACKER"), msg=answer)


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


if __name__ == "__main__":
    unittest.main()
