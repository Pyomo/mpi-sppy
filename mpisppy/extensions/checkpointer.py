###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Write checkpoints so a run can be stopped and resumed later.

Attached only when ``--checkpoint-dir`` or ``--resume-from`` is given, so a run
that does not ask for checkpointing pays nothing at all -- the extension is
never constructed and none of its hooks exist. This extension decides *when*
to write; ``mpisppy/utils/checkpointing.py`` owns the on-disk format.

One class serves three kinds of cylinder, because each holds a different
part of the answer:

* **On the PH hub** it writes the iterate -- the scenario models and the
  non-model state around them -- at completed iterations. The hub's *restore*
  is not here: it lives in ``PHBase.Iter0``, because splicing reloaded models
  in has to happen mid-startup, before solvers are created.
* **On an inner-bound spoke** (the xhat spokes, the L-shaped xhatter and
  the slammers) it writes that spoke's best incumbent, by variable name, with
  the spoke's loop position (xhatshuffle's cursor) and its extensions' state.
  It writes when the incumbent improves or the cursor moves, right after a
  restore, and when the spoke finalizes. It restores the incumbent in
  ``pre_iter0``, which the spoke's prep calls once before its loop starts.
  The hub's checkpoint does not carry the best solution -- that lives in
  ``best_solution_cache`` on the spoke -- so without this a resumed cylinders
  run would restore its iterate perfectly and still throw away the answer it
  had found.
* **On a dual cylinder** (``relaxed_ph``, ``ph_dual``) it writes the
  cylinder's own W, and restores it in ``post_iter0``, so the cylinder feeds
  the hub from where it stopped rather than from W = 0.

A run that gives ``--resume-from`` without ``--checkpoint-dir`` still gets the
extension, with writing switched off: reading is a spoke's whole job here.

**A checkpoint is only ever written at an iteration boundary.** That is the
whole design, and it is worth being explicit about why, because the obvious
alternative -- snapshot whatever state exists when the run ends -- does not
work and cannot be patched into working.

``iterk_loop`` runs Compute_Xbar, then Update_W, then ``miditer``, then *may
break* (the user converger, the convergence threshold, ``--time-limit``), and
only then solves. A run that ends through one of those breaks leaves the models
describing half an iteration: dual weights advanced to iteration k, and
nonanticipative values still those of k-1's solve. ``--time-limit`` -- the
planned-stop recipe this feature exists for -- exits that way every time.

Reconstructing a coherent iterate from that state means undoing everything the
first half of the iteration did, and that set is open-ended: ``miditer`` gives
every extension a chance to change rho, fix variables, relax domains, or add
cuts. Any list of things to rewind is a list of the extensions someone has
thought about so far.

Writing after the solve sidesteps all of it: a checkpoint written there always
describes a *completed* iteration, no matter which extensions are loaded or
what they touched. The invariant is one sentence and it holds by construction.

The cost is a model serialization per checkpoint rather than one per run. That
is the deliberate trade: correctness that needs no knowledge of any extension.
Retention is a single generation, so each write replaces the last, and the disk
footprint does not grow with the iteration count. Each write is bracketed by
``global_toc`` so the cost is visible in the log rather than guessed at.

``--checkpoint-every-iterations K`` buys that cost back on models whose solves
are cheap enough for serialization to dominate: writes happen at every K-th
completed iteration instead of every one, so an unplanned stop loses up to
K-1 iterations. It moves *which* boundaries are checkpoint points; it does not
move the write off an iteration boundary, so the coherence argument above is
untouched. The last iteration of an exhausted iteration limit is always
written, because raising ``--max-iterations`` and resuming is a supported
workflow and that iterate is known-good and already in memory.

``--checkpoint-before-seconds S`` covers the stop K does not: a run that ends
against a wall clock rather than at an iteration limit, at an iteration that
is not a multiple of K. It writes once, at the end of the first completed
iteration after which one more iteration, as long as the last, would take the
run past S seconds; if that iteration is a multiple of K, the write it gets
anyway is the one. See
``_deadline_is_near``.

A checkpoint therefore describes a *completed PH iteration*. A run that ends
before finishing iteration 1 publishes nothing: no iteration completed, so
there is no iterate to resume from. Iteration 0 is deliberately not a
checkpoint point -- ``Iter0`` splices the W and proximal terms into the
objective after the last extension hook available to us, so a checkpoint taken
during it would capture a model whose objective is not yet the one PH iterates
on.

**The write does not depend on extension order.** It happens in
``maybe_checkpoint``, a hook of this extension's own that ``iterk_loop`` calls
directly once the iteration is over -- after every extension's ``enditer``,
including those of user extensions supplied with
``--user-defined-extensions``. So whatever an extension changes on a scenario
model in its ``enditer`` (rho, nonant fixedness, domains, cuts) is part of the
checkpoint for that iteration, and a resume picks it up. Dispatching the write
from an ``enditer`` instead would have made that depend on the order
extensions were attached in, and lost any change made by a later one: a resume
starts at the next iteration, so the ``enditer`` that made the change never
runs again.

The same hook is what the xhatter spokes call once per pass through their main
loops, which have no ``enditer`` to borrow, and once more when they finalize:
one Checkpointer serves the hub and the spokes.

**Multi-rank cylinders.** A hub spread over several ranks holds its scenarios
in slices, so one checkpoint generation spans all of them and is published only
once every rank's files are on disk. The write is therefore collective (see
``mpisppy/utils/checkpointing.py``), and the iteration-count triggers below are
safe in it because they are pure functions of the absolute iteration number and
the iteration limit -- identical on every rank of a synchronous PH cylinder, so
the ranks reach the write together without being asked.

The deadline trigger is not: elapsed wall clock is rank-local, and a rank that
decided to write alone would hang the cylinder at the write barrier for the
rest of the job. So it is put through ``allreduce_or`` before it is believed,
and that collective is reached at every completed iteration, on every rank,
before the iteration-count tests -- so nothing that could differ between ranks
decides whether it is reached. Any trigger added later has to do the same.

The *spoke* incumbent write needs none of that. Each rank writes only its own
file, and the incumbent objective that gates the write comes from an
all-reduced objective evaluation, so the ranks are already in step; the design
deliberately keeps spokes uncoordinated with the hub and with each other
(section 9, item 6). A spoke has no use for the cadence or deadline triggers
either -- it already writes whenever it has something new to write -- so both
are hub-only and a spoke simply ignores them. Being in step about *when* to
write does not make the files one incumbent, though: a write can fail on one
rank. So the restore checks across the spoke's ranks that every file holds
the same incumbent, drops it on every rank if not, and otherwise gives every
rank rank 0's cursor.

See ``doc/designs/checkpointing_design.md``.
"""

import os
import shutil
import math
import time

from mpisppy import global_toc
from mpisppy.extensions.extension import Extension
import mpisppy.utils.checkpointing as ckpt


def _same_objective(a, b):
    """``a == b``, except that two NaNs are the same objective. The spoke
    write compares the incumbent's objective with the last one it wrote or
    failed to write, and NaN != NaN would make an unchanged NaN incumbent a
    new one on every pass of the loop."""
    return a == b or (a != a and b != b)


class Checkpointer(Extension):
    """Write a resumable checkpoint at each completed PH iteration."""

    # Nothing to carry across a resume. This is the extension doing the
    # checkpointing. Its attributes describe the file it last wrote and
    # what a resume handed it, both established fresh on every run; there
    # is nothing here for it to carry to itself.
    def checkpoint_state(self):
        return None

    def restore_state(self, state):
        pass

    def __init__(self, opt):
        super().__init__(opt)
        options = opt.options
        self.ckpt_dir = options.get("checkpoint_dir", None)
        self.backend = options.get("checkpoint_backend",
                                   ckpt.DILL_RELOAD_BACKEND)
        every = options.get("checkpoint_every_iterations", 1)
        self.every = 1 if every is None else int(every)
        if self.every < 1:
            raise RuntimeError(
                f"--checkpoint-every-iterations must be at least 1, got "
                f"{self.every}. It counts completed iterations between "
                f"writes; 1 writes at every iteration."
            )

        #: --checkpoint-before-seconds S, or None. See _deadline_is_near.
        before = options.get("checkpoint_before_seconds", None)
        self.before_seconds = None if before is None else float(before)
        # Finite as well as positive: NaN and +inf both pass "<= 0", and
        # either makes the deadline test false on every iteration, which
        # would switch off a safeguard the user asked for without a word.
        if self.before_seconds is not None and not (
                math.isfinite(self.before_seconds)
                and self.before_seconds > 0):
            raise RuntimeError(
                f"--checkpoint-before-seconds must be a finite positive "
                f"number, got "
                f"{self.before_seconds}. It is a wall-clock deadline measured "
                f"from the start of this run."
            )
        #: Set once the deadline trigger has fired, so it fires at most once.
        self._before_seconds_fired = False

        # Restore-only: --resume-from without --checkpoint-dir. The hub does
        # not need the extension for that (its resume branch is in Iter0), but
        # a spoke does -- restoring its incumbent is this extension's job --
        # and cfg_vanilla attaches it to every cylinder rather than reasoning
        # about which ones. So a run that only reads is a supported state, and
        # writing is what gets switched off.
        self.write_enabled = self.ckpt_dir is not None
        if not self.write_enabled and not options.get("resume_from", None):
            raise RuntimeError(
                "Checkpointer was attached without a checkpoint directory. "
                "It should only be attached when --checkpoint-dir or "
                "--resume-from is set."
            )
        # Everything below fails at setup rather than after a multi-hour run
        # reaches its first write and discovers it cannot finish one.
        ckpt.require_implemented_backend(self.backend)

        # One class, three jobs. On the hub it writes the PH iterate, models
        # and all. On an inner-bound spoke it writes that spoke's best
        # incumbent by variable name, with its loop position and extension
        # state; on a dual cylinder, that cylinder's W. Neither spoke kind
        # writes models, generations or dill (which is why the dill checks
        # below are hub-only).
        #
        # The invariant the hub write rests on -- the write happens after the
        # solve, so W and the nonants agree -- is a property of the
        # *synchronous* iterk_loop. APH inherits this wiring because aph_hub
        # is built by calling ph_hub, but its loop dispatches a fraction of
        # the scenarios per pass, keeps its own hardcoded iteration range that
        # no resume offset touches, and runs on a worker thread under the
        # listener. A checkpoint written there would not describe a completed
        # iteration and a resumed run would renumber from 1, overwriting the
        # checkpoint it resumed from.
        from mpisppy.opt.ph import PH
        from mpisppy.utils.xhat_eval import Xhat_Eval
        # A third kind of cylinder: one that runs PH itself without being the
        # hub -- relaxed_ph and ph_dual. Its opt is a PHBase, which is neither
        # the hub's PH nor an xhat spoke's Xhat_Eval, so the cylinder builder
        # says which it is rather than leaving it to be inferred. It writes W
        # rather than models (see checkpointing.dual_spoke_state) and restores
        # it in post_iter0.
        self.dual_spoke_mode = \
            options.get("checkpoint_role", "hub") == "dual_spoke"
        self.spoke_mode = not self.dual_spoke_mode and not isinstance(opt, PH)
        if self.spoke_mode and not isinstance(opt, Xhat_Eval):
            raise RuntimeError(
                f"Checkpointing supports the synchronous PH hub and the xhat "
                f"spokes, but this cylinder is {type(opt).__name__}. Remove "
                f"--checkpoint-dir and --resume-from, or run PH."
            )

        #: The iteration a dual cylinder's restored W was written at, or None
        #: if it started from W=0. Read by tests, which otherwise cannot tell
        #: a restored W from one the cylinder converged to on its own.
        self.restored_dual_generation = None
        #: The incumbent objective this spoke restored from disk, or None if
        #: it started without one. Read by tests, which otherwise cannot tell
        #: a restored incumbent from one the spoke happened to re-find.
        self.restored_incumbent_obj = None
        #: The loop cursor this spoke restored, held until the loop exists to
        #: receive it (the restore runs in pre_iter0, the loop is built after).
        #: Collected by XhatInnerBoundBase._checkpointed_loop_state.
        self.restored_loop_state = None
        #: Likewise this spoke's extension state, handed over at the end of
        #: xhat_prep so post_iter0 cannot overwrite it.
        self.restored_extension_state = None
        #: The incumbent objective the last write recorded, so an unchanged
        #: incumbent is not rewritten on every pass of a loop that spins.
        self._last_written_obj = None
        #: Likewise for the loop cursor, which moves independently of the
        #: incumbent -- most cursor moves do not improve on the best xhat.
        self._last_written_loop_progress = None
        #: The incumbent objective whose write last failed. Without it a
        #: failure retries on every pass of that same loop, rebuilding and
        #: pickling the whole incumbent and printing a warning each time.
        #: A cursor move alone does not retry either: the warning promises
        #: the next improvement, and the cursor moves far more often.
        self._last_failed_obj = None

        if not self.write_enabled:
            # Nothing below is about reading, and a restore-only run must not
            # inherit refusals that only protect a write.
            return

        # Everything the setup refusals below look at is per rank -- which
        # scenarios this rank owns, what this rank's node can write, whether
        # dill imports here -- so each of them can refuse on one rank and
        # pass on the others. The run's next collective would then be waiting
        # for a rank that has already raised, and a refusal meant to arrive
        # in the first second of the run becomes a job that hangs until its
        # wall-clock limit instead. Every rank raises or none does.
        ckpt.run_agreed(opt, self._refuse_a_run_that_cannot_checkpoint,
                        "be set up to checkpoint, so the run is refused")

    def _refuse_a_run_that_cannot_checkpoint(self):
        """The setup refusals that are this rank's own to make.

        Local by nature and agreed by the caller; see
        ``checkpointing.run_agreed``.
        """
        opt = self.opt
        if not self.spoke_mode and not self.dual_spoke_mode:
            ckpt.require_dill(self.backend)

        # Two scenario names that sanitize to the same file name would
        # silently overwrite each other's model files; refuse now rather than
        # at the first write. Per rank, which is the right scope: file names
        # carry the rank, so only names sharing a rank can collide.
        ckpt.check_filename_collisions(opt.local_scenarios)

        ckpt.probe_directory_is_writable(opt, self.ckpt_dir)

        # Not a refusal, but deleting can fail on one rank like the checks
        # above, so it is agreed with them.
        # The hub only: a dual cylinder is not a spoke_mode one either, and
        # two cylinders deleting the one directory at once race each other.
        if (not self.spoke_mode and not self.dual_spoke_mode
                and opt.cylinder_rank == 0):
            self._clear_other_studies_spoke_files()

    def _clear_other_studies_spoke_files(self):
        """Delete ``spokes/`` unless this run is resuming from this directory.

        A spoke overwrites only its own file, and only once it has an
        incumbent or dual weights to write, so a run started in a directory
        an earlier study used
        leaves that study's spoke files beside its own. A later resume of
        this run then restores them into whichever spoke has the same name:
        silently when the configuration matches, and with a refusal that
        kills the spoke when it does not. When this run resumes from the
        directory it writes to, those files are this study's and are kept.

        The hub does this, once: every cylinder shares the cfg that decides
        whether a Checkpointer is attached, and generic_cylinders refuses a
        --checkpoint-dir run whose hub did not get one. It runs before any
        spoke can write: the hub builds this extension while constructing its
        opt object, before make_windows; each spoke rank waits in make_windows
        for every hub rank on its window's communicator (the hub rank of the
        same number when every cylinder has the same number of ranks, every
        hub rank otherwise); and every hub rank is held in this setup's
        agreement until hub rank 0 has finished deleting.
        """
        if self._same_directory(self.opt.options.get("resume_from", None)):
            return
        spokes_dir = os.path.join(self.ckpt_dir, ckpt.SPOKES_SUBDIR)
        if not os.path.isdir(spokes_dir):
            return
        global_toc(f"Removing spoke files left in {spokes_dir} by "
                   f"an earlier run; this run is not resuming from that "
                   f"directory", True)
        shutil.rmtree(spokes_dir)

    def pre_iter0(self):
        if self.dual_spoke_mode:
            # Nothing to prove: this cylinder writes no models, so there is
            # no dill to probe. Its own restore waits for post_iter0, by
            # which time PH_Prep has attached the W it writes into.
            return
        if self.spoke_mode:
            # xhat_prep calls this once, before the spoke's loop starts, which
            # is the spoke's equivalent of the hub's resume branch in Iter0.
            # A spoke writes its extensions' state with its incumbent, so it
            # refuses one that cannot be written here, as the hub does at the
            # start of Iter0, rather than at its first write.
            ckpt.require_state_contract(self.opt)
            self._restore_incumbent()
            return
        if not self.write_enabled:
            return
        # Prove now that this run's models can actually be checkpointed. A run
        # that only found out at its first write would lose exactly the state
        # checkpointing exists to preserve.
        ckpt.probe_model_is_dillable(self.opt)

    def post_iter0(self):
        """Put a dual cylinder's W back, once there is a W to put it in.

        Iter0 has just solved this cylinder's subproblems from W = 0 and is
        about to hand the loop an iterate that owes nothing to the study.
        Overwriting it here rather than skipping the solve keeps the cylinder
        ordinary -- solvers created, bound reported, prox terms spliced -- at
        the cost of one solve round that a resumed run throws away.
        """
        if not self.dual_spoke_mode:
            if not self.spoke_mode:
                self._report_unclaimed_spoke_files()
            return
        resume_from = self.opt.options.get("resume_from", None)
        if not resume_from:
            return
        cylinder, ordinal = self._spoke_identity()
        rank0 = self.opt.cylinder_rank == 0
        # Collective, for the reason given at the xhat spoke's load.
        state = ckpt.run_agreed(
            self.opt,
            lambda: ckpt.load_dual_spoke_state(self.opt, resume_from,
                                               cylinder, ordinal),
            "read their checkpointed dual weights, so none of them restores "
            "any")
        # Collective, and reached whether or not this rank found a file: W is
        # per rank, but the iteration it belongs to is the cylinder's, and
        # ranks restoring different iterations blend them in Compute_Xbar.
        # See agree_dual_spoke_restore.
        state, disagreement = ckpt.agree_dual_spoke_restore(self.opt, state)
        if disagreement is not None:
            global_toc(f"WARNING: {disagreement}; {cylinder} starts from "
                       f"W=0", rank0)
            return
        if state is None:
            global_toc(f"No checkpointed dual weights for {cylinder} in "
                       f"{resume_from}; this cylinder starts from W=0", rank0)
            return
        was = state.get("class_count")
        now = self._class_ordinal_and_count()[1]
        if was is not None and was != now:
            global_toc(
                f"WARNING: the checkpoint was written by a wheel carrying "
                f"{was} {cylinder} cylinder(s) and this run has {now}, so "
                f"this cylinder may be restoring dual weights that belonged "
                f"to a different one.", rank0)
        # Agreed like the load: putting W back resolves the file's entries
        # against the nonants of this rank's own models, so it is a refusal
        # one rank can make while the others walk into Compute_Xbar's
        # allreduce. The agreement above has already settled that every rank
        # has a state, so all of them reach this.
        ckpt.run_agreed(
            self.opt,
            lambda: ckpt.restore_dual_spoke_state(self.opt, state),
            "put their checkpointed dual weights back on their models, so "
            "none of them restores any")
        # What every check up to here asks is whether the file describes this
        # model. This asks whether the weights now on the models reproduce
        # the E[W] the file recorded: they become another cylinder's
        # Lagrangian bound, which the hub keeps as best-so-far. The sums are
        # allreduces, so every rank computes them before the agreement; the
        # comparison is with this rank's own file, so it is agreed.
        from mpisppy.phbase import Wbar_by_node, W_magnitude_by_node
        bars = Wbar_by_node(self.opt)
        sizes = W_magnitude_by_node(self.opt)
        ckpt.run_agreed(
            self.opt,
            lambda: ckpt.require_restored_duals_match_their_file(
                self.opt, cylinder, state["generation"], state["Wbar"],
                bars, sizes),
            "confirm that their restored dual weights reproduce the E[W] "
            "their files recorded, so none of them resumes from those weights")
        self.restored_dual_generation = state["generation"]
        # Only W crosses the checkpoint for a dual cylinder. Its own
        # extensions -- --grad-rho on --ph-dual, say -- are rebuilt fresh,
        # and a stateful one then does not retrace the uninterrupted run.
        others = sorted(name for name, _ in ckpt._extension_objects(self.opt)
                        if name != type(self).__name__)
        if others:
            global_toc(
                f"WARNING: {cylinder} carries only its dual weights across a "
                f"checkpoint; its own extensions ({', '.join(others)}) start "
                f"fresh on this resumed run.", rank0)
        global_toc(f"Restored the checkpointed dual weights for {cylinder} "
                   f"(written at its iteration {state['generation']})", rank0)

    def _report_unclaimed_spoke_files(self):
        """On the hub of a resumed run: name every spoke file nobody reads.

        The spoke that would report its own file is the one that was dropped,
        so it falls to the hub. Agreed like every other restore step: the
        listing is this rank's own file handling, and the hub's next step is
        collective.
        """
        resume_from = self.opt.options.get("resume_from", None)
        if not resume_from:
            return
        communicators = getattr(getattr(self.opt, "spcomm", None),
                                "communicators", None) or []
        names = [d["spcomm_class"].__name__ for d in communicators]
        claimed = {(name, names[:i].count(name))
                   for i, name in enumerate(names) if i > 0}
        unclaimed = ckpt.run_agreed(
            self.opt,
            lambda: ckpt.unclaimed_spoke_files(self.opt, resume_from,
                                               claimed),
            "list the spoke files in the checkpoint, so none of them resumes")
        for cylinder, ordinal in unclaimed:
            global_toc(
                f"WARNING: the checkpoint in {resume_from} holds a file for "
                f"{cylinder} (ordinal {ordinal}), and no such cylinder runs "
                f"in this resume, so what it held -- an incumbent, or a dual "
                f"cylinder's weights -- is not restored.",
                self.opt.cylinder_rank == 0)

    def _dual_spoke_checkpoint(self):
        """Write W at the end of a completed iteration of this cylinder.

        Every iteration, with no cadence to divide it: the file is a couple
        of floats per nonanticipative variable, and the iteration that
        produced it was a round of subproblem solves. Failures warn for the
        same reason the other cylinders' do -- losing this file costs a
        resumed run a dual restart, which is not worth killing a running
        cylinder over.
        """
        if not self.write_enabled:
            return
        # Collective, so outside the try and before anything a rank can fail
        # at alone: every rank of the cylinder reaches this every iteration.
        # Recorded so the restore can check it gets these weights back.
        from mpisppy.phbase import Wbar_by_node
        wbar = Wbar_by_node(self.opt)
        try:
            cylinder, ordinal = self._spoke_identity()
            ckpt.write_dual_spoke_state(
                self.opt, self.ckpt_dir, cylinder, ordinal,
                generation=int(getattr(self.opt, "_PHIter", 0)),
                class_count=self._class_ordinal_and_count()[1],
                wbar=wbar)
        except Exception as exc:
            # By the rank that failed, as for the xhat spokes: each rank
            # writes its own file, so the failure is this rank's alone.
            global_toc(
                f"WARNING: rank {self.opt.cylinder_rank} of this cylinder "
                f"could not write its dual weights ({type(exc).__name__}); "
                f"the run continues and the end of the next iteration "
                f"will try again if the run gets that far.\n{exc}",
                True)

    def _spoke_identity(self):
        """(cylinder name, ordinal among cylinders of that class).

        The ordinal names this spoke's file. Its strata rank would be the
        obvious choice and is the wrong one: that is the cylinder's index in
        the whole wheel, and which cylinders run is on the list a resume may
        change, so dropping a lagrangian renumbers every cylinder after it.
        Counting only cylinders of this spoke's own class gives a number that
        an unrelated cylinder coming or going does not move.
        """
        spoke = self.opt.spcomm
        if spoke is None:
            raise RuntimeError(
                "The Checkpointer was attached to a spoke's opt object that "
                "has no spcomm, so there is no cylinder to write for. This "
                "extension is attached by cfg_vanilla to cylinders run by "
                "WheelSpinner."
            )
        return type(spoke).__name__, self._class_ordinal_and_count()[0]

    def _class_ordinal_and_count(self):
        """(this spoke's ordinal among its own class, how many of that class).

        Read from the cylinder list WheelSpinner hands every SPCommunicator.
        A spoke built without one -- a test stub, or a spoke driven outside
        the wheel -- is the only cylinder as far as it can tell, which is the
        right answer for the single-spoke case and the only one available.
        """
        spoke = self.opt.spcomm
        cylinder = type(spoke).__name__
        communicators = getattr(spoke, "communicators", None)
        strata_rank = getattr(spoke, "strata_rank", 0)
        if not communicators:
            return 0, 1
        names = [d["spcomm_class"].__name__ for d in communicators]
        return names[:strata_rank].count(cylinder), names.count(cylinder)

    def _restore_incumbent(self):
        """Load this spoke's checkpointed incumbent, if a resume asked for one.

        A missing file is normal and says so once: the run being resumed may
        have stopped before this spoke found anything. A file that exists but
        does not match this run raises, exactly as the hub's does.
        """
        resume_from = self.opt.options.get("resume_from", None)
        if not resume_from:
            return
        cylinder, ordinal = self._spoke_identity()
        rank0 = self.opt.cylinder_rank == 0
        # Collective: the load refuses per rank -- it reads the file named
        # after this rank and checks it against the scenarios this rank owns
        # -- and everything after it is collective, so a rank-local refusal
        # would strand the others there. Every rank refuses or none does.
        state = ckpt.run_agreed(
            self.opt,
            lambda: ckpt.load_spoke_incumbent(self.opt, resume_from,
                                              cylinder, ordinal),
            "read their checkpointed incumbent, so none of them restores one")
        # Collective, and reached whether or not this rank found a file: the
        # cursor and the bound in there belong to the cylinder, not to a
        # rank, and ranks that resume from different cursors go on to
        # broadcast from different roots. See agree_spoke_restore.
        state, disagreement = ckpt.agree_spoke_restore(self.opt, state)
        if disagreement is not None:
            global_toc(f"WARNING: {disagreement}", rank0)
            return
        if state is not None:
            # The ordinal is stable when an unrelated cylinder comes or goes,
            # but not when one of two same-class spokes does: the survivor's
            # ordinal becomes the removed one's, and it would read that
            # spoke's file without a word. The values are feasible for the
            # same model, so nothing downstream would notice.
            was = state.get("class_count")
            now = self._class_ordinal_and_count()[1]
            if was is not None and was != now:
                global_toc(
                    f"WARNING: the checkpoint was written by a wheel carrying "
                    f"{was} {cylinder} cylinder(s) and this run has {now}, so "
                    f"this spoke may be restoring an incumbent that belonged "
                    f"to a different one. It is a feasible solution for the "
                    f"same model either way.", rank0)
        # Agreed too: putting the values back is per rank -- it resolves the
        # file's variable names against this rank's own models -- so it is a
        # refusal one rank can make alone, and the ranks that did restore
        # would carry on into the loop without it. The agreement above has
        # already settled whether there is a state to restore, so every rank
        # reaches this with the same answer.
        obj = ckpt.run_agreed(
            self.opt,
            lambda: None if state is None
            else ckpt.restore_spoke_incumbent(self.opt, state),
            "put their checkpointed incumbent back on their models, so none "
            "of them restores one")
        if state is None:
            global_toc(f"No checkpointed incumbent for {cylinder} in "
                       f"{resume_from}; this spoke starts without one", rank0)
            return
        self.restored_incumbent_obj = obj
        self.opt.spcomm.best_inner_bound = state["best_inner_bound"]
        # Held rather than applied: the spoke's loop -- and the cursor this
        # describes -- is built after pre_iter0, so it collects this itself
        # once it exists. Older files have no such key.
        self.restored_loop_state = state.get("loop_state")
        # Held for the same reason, and handed over at the end of xhat_prep:
        # post_iter0 has not run yet, and it is where an extension rebuilds
        # its bookkeeping from the models.
        self.restored_extension_state = state.get("extension_state")
        # These two mean "what the file in self.ckpt_dir already holds", so
        # they may only be seeded when that is the file we just read. Resuming
        # into a *different* directory is the documented
        # stop-today-resume-tomorrow flow, and there this spoke has written
        # nothing yet: seeding them there makes the skip test below decline to
        # write until the spoke strictly improves on what it restored, so a
        # study whose xhat has stopped improving -- a converged one, the case
        # where the answer is worth the most -- publishes no incumbent at all
        # and the next resume starts without one, with exit code 0 throughout.
        if self.write_enabled and self._same_directory(resume_from):
            self._last_written_obj = obj
            self._last_written_loop_progress = \
                self.opt.spcomm.loop_state_progress(self.restored_loop_state)
        # The hub learns bounds only from what a spoke sends, so a restored
        # incumbent that is never published leaves the hub reporting an
        # infinite inner bound -- and its gap and convergence tests, and
        # Gapper's automatic mipgap, reading from it -- until this spoke
        # happens to improve on the answer it already has. Publish it now,
        # before this spoke has solved anything: WheelSpinner.run creates the
        # send buffers before spcomm.main(), which is where this runs from.
        # The bound and the values beside it are the restored pair, so the
        # BEST_XHAT buffer never carries an objective that belongs to other
        # values.
        bound = state["best_inner_bound"]
        if bound is not None:
            self.opt.spcomm.send_bound(bound)
            self.opt.spcomm.send_best_xhat()
        global_toc(f"Restored the checkpointed incumbent for {cylinder} "
                   f"(objective {obj})", rank0)
        # Write it to this run's directory now, not at the bottom of the
        # first loop pass: a short resume whose hub finishes before this
        # spoke starts its loop -- the spoke's prep can solve for a while --
        # never reaches that pass, and leaves a directory whose hub
        # checkpoint has no incumbent beside it. Resuming in place skips the
        # write, because _last_written_obj was seeded above. The loop state
        # and extension state are the ones just read, not the spoke's: it
        # has neither yet, and asking it would write a fresh cursor and fresh
        # extension state over the restored ones.
        self._spoke_checkpoint(
            restored=(self.restored_loop_state,
                      self.restored_extension_state))

    def _same_directory(self, other):
        """True when ``other`` names the directory this run writes to.

        Compared by resolved path rather than by string: the resume flow
        passes these on a command line, so the same directory routinely
        arrives spelled two ways.
        """
        if not other or not self.ckpt_dir:
            return False
        try:
            return os.path.realpath(other) == os.path.realpath(self.ckpt_dir)
        except OSError:
            return False

    def _is_final_iteration(self):
        """True when the loop bound says this completed iteration is the last.

        ``iterk_loop`` works out the absolute number of its last iteration
        from the two bounds -- ``--max-iterations`` over this run and
        ``--stop-at-iteration-number`` over the study -- and leaves it in
        ``_stop_iteration``, so at the end of that iteration there is no next
        pass. Reading the answer rather than ``PHIterLimit`` is what makes the
        rule hold on a resumed run, where the limit counts this run's
        iterations and ``_PHIter`` counts the study's.

        Only the iteration bounds are knowable here: convergence, the user
        converger and ``--time-limit`` are all decided in the *next*
        iteration's top half, and the cylinder-convergence test fires after
        this hook.
        """
        stop = getattr(self.opt, "_stop_iteration", None)
        if stop is None:
            # Called from outside iterk_loop, which is where that attribute is
            # set. Fall back to what the loop would have computed from the
            # per-run limit alone.
            limit = self.opt.options.get("PHIterLimit", None)
            if limit is None:
                return False
            stop = int(getattr(self.opt, "_resume_iteration", 0)) + int(limit)
        return int(getattr(self.opt, "_PHIter", 0)) >= int(stop)

    def _should_write(self):
        """Whether this completed iteration is a checkpoint point.

        Every K-th iteration by absolute number, so the cadence is unchanged
        by a resume (which picks the global counter up where it left off).
        The final iteration of an exhausted iteration limit is always written:
        raising ``--max-iterations`` and resuming is an explicitly supported
        workflow, and dropping the last K-1 iterations of a run that ended by
        finishing its budget would lose work that is known to be coherent and
        is sitting in memory.

        The deadline trigger is asked first, at every completed iteration,
        so that a write at a multiple of K that lands near the deadline is
        the deadline write: asked only off the K boundaries, it wrote again
        at the next iteration, just when the user had said time was short.
        Asking it every time also keeps its collective on every rank of a
        multi-rank cylinder at every iteration, since nothing before it can
        differ between ranks.
        """
        near = self._deadline_is_near()
        iteration = int(getattr(self.opt, "_PHIter", 0))
        return (near or iteration % self.every == 0
                or self._is_final_iteration())

    def _deadline_is_near(self):
        """``--checkpoint-before-seconds S``: is there time for another one?

        The gap this closes is the one ``--checkpoint-every-iterations K``
        opens. At K > 1 a run against a hard wall clock -- a scheduler slot, a
        ``--time-limit`` -- can stop at an iteration that is not a multiple of
        K, and then the newest checkpoint is up to K-1 iterations old. If K is
        larger than the number of iterations the run ever completes, there is
        no checkpoint at all. So at the end of each completed iteration this
        asks whether *another* iteration would carry the run past S seconds of
        elapsed wall clock, and writes now if it would.

        The estimate of "another iteration" is the last one measured
        (``PHBase._last_iteration_seconds``), seeded by iteration 0. That is
        the whole model: mpi-sppy does not pad it, and S is not adjusted for
        the write it triggers. The write's own cost is bracketed by ``toc`` in
        ``_write`` so it can be read off a log rather than guessed at, and
        leaving room for it is the user's to do when choosing S.

        **The test goes through allreduce_or**, because elapsed wall clock is
        rank-local and the hub write is a collective bracketed by barriers: a
        rank that decided to write while another decided not to would hang the
        cylinder for the rest of the job. Everything that could return early
        here is identical on every rank -- the option itself, and a latch set
        only from the all-reduced answer -- so the ranks arrive at the
        collective together.

        Latched afterwards because the deadline passes only once. Without it
        every subsequent iteration would also be past S and would write, which
        is the per-iteration cost K was set to avoid, at the point in the run
        where the user has said time is short.
        """
        if self.before_seconds is None or self._before_seconds_fired:
            return False
        last = getattr(self.opt, "_last_iteration_seconds", None)
        # Read back from a checkpoint on a resume, so not to be trusted to be
        # a number: a NaN would make the test below false for good.
        if last is not None and not math.isfinite(last):
            last = None
        elapsed = time.perf_counter() - self.opt.start_time
        near = self.opt.allreduce_or(
            elapsed + (0.0 if last is None else last) >= self.before_seconds)
        if near:
            self._before_seconds_fired = True
            global_toc(
                f"Within one iteration of --checkpoint-before-seconds "
                f"{self.before_seconds} ({elapsed:.1f} seconds elapsed, last "
                f"iteration {0.0 if last is None else last:.1f}); "
                f"checkpointing now",
                self.opt.cylinder_rank == 0)
        return near

    def maybe_checkpoint(self):
        """Write the checkpoint if this iteration is a checkpoint point.

        See the module docstring for why the write lives here.

        A mid-run write failure -- disk full, an NFS hiccup -- is warned
        about, not raised: the previously published generation is untouched
        and remains resumable, while the optimization progress that a raise
        would destroy lives only in memory. The next multiple of K, the
        ``--checkpoint-before-seconds`` write if it has not happened yet, or
        the last iteration of the iteration limit tries again, whichever
        comes first, if the run gets that far: a stop on convergence, the
        cylinders' gap, ``--max-stalled-iters`` or ``--time-limit`` writes
        nothing more. The deadline write fires once and is not retried, and
        a failure at the last iteration of the limit has nothing after it.
        Conditions detectable at setup (unwritable directory,
        undillable model, unknown backend) still fail loudly in ``__init__``
        and ``pre_iter0``.

        Every exception is caught, not just the ``RuntimeError`` that
        ``write_checkpoint`` raises for a failed model dump: only the model
        dump is wrapped, so the leaf write, the publishing renames and the
        manifest write all surface a bare ``OSError``, and ENOSPC between the
        last model file and the leaf would otherwise take the run down by the
        exact route this is here to prevent. Continuing is safe at every one
        of those points -- each is either pre-commit (the manifest still
        names the previous generation, which is intact) or the atomic
        manifest flip itself.

        On a multi-rank cylinder the ranks agree on failure inside
        ``write_checkpoint``, so either all of them raise here and warn, or
        none does. Catching independently per rank would be the deadlock this
        is written to avoid: the run continues on every rank or on none, and
        no rank is left waiting at the next write's barrier for one that
        already gave up.
        """
        if self.spoke_mode:
            self._spoke_checkpoint()
            return
        if self.dual_spoke_mode:
            self._dual_spoke_checkpoint()
            return
        if not self.write_enabled:
            return
        if not self._should_write():
            return
        try:
            self._write()
        except Exception as exc:
            # Rank 0 reports because a multi-rank cylinder should print one
            # summary, and the rank that actually failed reports because it
            # is the only one holding the cause -- gating on rank 0 alone
            # silences the diagnosis and leaves a bare "some rank failed".
            rank = int(self.opt.cylinder_rank)
            mine = getattr(exc, "mpisppy_failed_locally", True)
            global_toc(
                f"WARNING: checkpoint write failed at iteration "
                f"{int(getattr(self.opt, '_PHIter', 0))} on rank {rank} "
                f"({type(exc).__name__}); the previously published "
                f"checkpoint (if any) is intact, and "
                f"{self._what_retries()}.\n{exc}",
                rank == 0 or mine)

    def _what_retries(self):
        """The end of the failed-write warning: which write, if any, tries
        again. Named rather than left as "the next checkpoint point",
        because a user deciding whether to stop the job needs to know
        whether one is coming before the scheduler's kill."""
        if self._is_final_iteration():
            return ("this was the last iteration of the iteration limit, so "
                    "no later write will try again; a resume starts from "
                    "the previously published checkpoint")
        pending = (self.before_seconds is not None
                   and not self._before_seconds_fired)
        # "if the run gets that far": _is_final_iteration knows only the
        # iteration bounds, and a run that stops on convergence, the
        # cylinders' gap, --max-stalled-iters or --time-limit ends with no
        # further write.
        retry = ("the run continues; the next multiple of "
                 "--checkpoint-every-iterations"
                 + (", the --checkpoint-before-seconds write,"
                    if pending else "")
                 + " or the last iteration of the iteration limit, whichever "
                 "comes first, will try again if the run gets that far (a "
                 "stop on convergence, the cylinders' gap, "
                 "--max-stalled-iters or --time-limit writes nothing more)")
        if self.before_seconds is not None and not pending:
            retry += (" (--checkpoint-before-seconds has already fired and "
                      "does not retry)")
        return retry

    def _write(self):
        """Write one generation, bracketed by toc so the cost is legible.

        The pair of timestamps *is* the measured write duration, which is what
        a user needs in order to judge the per-iteration overhead on their own
        models -- mpi-sppy deliberately does not estimate it for them.
        """
        rank0 = self.opt.cylinder_rank == 0
        generation = int(getattr(self.opt, "_PHIter", 0))
        global_toc(f"Writing checkpoint at iteration {generation} "
                   f"to {self.ckpt_dir}", rank0)
        ckpt.write_checkpoint(self.opt, self.ckpt_dir, generation,
                              backend=self.backend)
        global_toc(f"Checkpoint written at iteration {generation}", rank0)

    def _spoke_checkpoint(self, restored=None):
        """Write if anything worth keeping moved.

        ``restored`` is (loop state, extension state) as a resume just read
        them, for the write straight after a restore, before the spoke holds
        either; otherwise both are asked of the spoke and its extensions.

        Called once per pass of a loop that spins while it waits on the hub
        (and also right after a restore and at finalize), so the common case
        has to be cheap: comparing a float and a small dict and returning.

        Two things can move. The incumbent improves rarely. The **loop cursor**
        moves whenever the spoke tries another scenario, which is more often --
        but every cursor move is the result of a subproblem solve, so writes
        are bounded by solves. A pass that solves nothing writes nothing,
        which is the case that has to stay cheap and does. A write is not
        free, though: it pickles the whole cached incumbent and fsyncs twice,
        which is small next to a MIP solve and comparable to a tiny LP's.
        (Keying the incumbent by variable name used to dominate; it is now
        done once per incumbent, see checkpointing._values_by_name.)

        Failures warn rather than raise, for the hub's reason and one more:
        this file is an optimization. Losing it costs a resumed run the
        answer it had found, which is worth a loud warning and not worth
        killing a running spoke over.
        """
        spoke = self.opt.spcomm
        if not self.write_enabled:
            # --resume-from with no --checkpoint-dir. The restore already
            # published the incumbent, which is this spoke's whole job; there
            # is nowhere to write. Without this the write below is attempted
            # with ckpt_dir=None on every improvement, and only the warning
            # path keeps that from ending the run.
            return

        obj = getattr(self.opt, "best_solution_obj_val", None)
        if obj is None or _same_objective(obj, self._last_failed_obj):
            # Nothing to write yet: the file carries a solution, and the
            # cursor rides along with it rather than on its own. Or the last
            # write of this incumbent failed; wait for a new one.
            return
        if restored is None:
            loop_state = spoke.checkpoint_loop_state()
            extension_state = ckpt.GATHER_EXTENSION_STATE
        else:
            loop_state, extension_state = restored
        # Compared on the progress projection rather than the whole state.
        # xhatshuffle's loop state carries xh_iter, which counts passes of a
        # loop that spins while it waits on the hub, so it differs on every
        # pass whether or not anything was solved -- and comparing it made a
        # pass that solves nothing write the incumbent, pickle and rename the
        # file and fsync the directory. Measured at 0.115 ms a pass on farmer
        # over tmpfs: about 8,700 writes a second, per spoke rank, silently,
        # since the toc below only speaks when the objective improved.
        progress = spoke.loop_state_progress(loop_state)
        if (_same_objective(obj, self._last_written_obj)
                and progress == self._last_written_loop_progress):
            return
        try:
            cylinder, ordinal = self._spoke_identity()
            path = ckpt.write_spoke_incumbent(
                self.opt, self.ckpt_dir, cylinder, ordinal,
                best_inner_bound=getattr(spoke, "best_inner_bound", None),
                loop_state=loop_state,
                class_count=self._class_ordinal_and_count()[1],
                extension_state=extension_state,
                progress=progress)
        except Exception as exc:
            self._last_failed_obj = obj
            # Printed by the rank that failed, whichever it is: each rank
            # writes its own file, so the failure is this rank's alone, and a
            # rank-0-only warning left every other rank's failures silent.
            global_toc(
                f"WARNING: rank {self.opt.cylinder_rank} of this spoke could "
                f"not write its incumbent ({type(exc).__name__}); the run "
                f"continues and the next improvement will try again.\n{exc}",
                True)
            return
        if path is None:
            return
        improved = not _same_objective(obj, self._last_written_obj)
        self._last_written_obj = obj
        self._last_written_loop_progress = progress
        if not improved:
            # The cursor moved but the answer did not. Worth writing, not
            # worth a line in the log every time the spoke tries a scenario.
            return
        # One line, not the pair the hub prints. The pair exists to measure a
        # write whose cost a user has to trade off against checkpoint
        # frequency; this write has no frequency knob and costs a rename.
        global_toc(f"Checkpointed incumbent (objective {obj})",
                   self.opt.cylinder_rank == 0)
