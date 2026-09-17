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

One class serves two cylinders, because they hold different halves of the
answer:

* **On the PH hub** it writes the iterate -- the scenario models and the
  non-model state around them -- at completed iterations. The hub's *restore*
  is not here: it lives in ``PHBase.Iter0``, because splicing reloaded models
  in has to happen mid-startup, before solvers are created.
* **On an xhat spoke** it writes that spoke's best incumbent, by variable
  name, whenever the incumbent improves, and restores it in ``pre_iter0``
  (which ``xhat_prep`` calls once before the spoke's loop starts). The hub's
  checkpoint does not carry the best solution -- that lives in
  ``best_solution_cache`` on the spoke -- so without this a resumed cylinders
  run would restore its iterate perfectly and still throw away the answer it
  had found.

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
loops, which have no ``enditer`` to borrow: one Checkpointer serves the hub
and the spokes.

See ``doc/designs/checkpointing_design.md``.
"""

import os

from mpisppy import global_toc
from mpisppy.extensions.extension import Extension
import mpisppy.utils.checkpointing as ckpt


class Checkpointer(Extension):
    """Write a resumable checkpoint at each completed PH iteration."""

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

        # One class, two jobs. On the hub it writes the PH iterate, models and
        # all. On an xhat spoke it writes only that spoke's best incumbent, by
        # variable name -- no models, no generations, no dill (which is why
        # the dill checks below are hub-only).
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
        self.spoke_mode = not isinstance(opt, PH)
        if self.spoke_mode and not isinstance(opt, Xhat_Eval):
            raise RuntimeError(
                f"Checkpointing supports the synchronous PH hub and the xhat "
                f"spokes, but this cylinder is {type(opt).__name__}. Remove "
                f"--checkpoint-dir, or run PH."
            )

        #: Set once a restored incumbent still needs publishing to the hub.
        self._publish_restored_bound = None
        #: The incumbent objective this spoke restored from disk, or None if
        #: it started without one. Read by tests, which otherwise cannot tell
        #: a restored incumbent from one the spoke happened to re-find.
        self.restored_incumbent_obj = None
        #: The incumbent objective the last write recorded, so an unchanged
        #: incumbent is not rewritten on every pass of a loop that spins.
        self._last_written_obj = None

        if not self.spoke_mode and self.write_enabled:
            ckpt.require_dill(self.backend)

        if not self.write_enabled:
            # Nothing below is about reading, and a restore-only run must not
            # inherit refusals that only protect a write.
            return

        # Multi-rank writing is not implemented: every rank would compute the
        # same staging and generation directory and race to create, replace and
        # delete it, so ranks destroy each other's files. Refuse rather than
        # abort the job at its very end with a half-published generation.
        n_proc = getattr(opt, "n_proc", 1)
        if n_proc > 1:
            what = "spoke" if self.spoke_mode else "hub"
            raise RuntimeError(
                f"Checkpointing currently supports a single rank per "
                f"cylinder, but this {what} has {n_proc}. Multi-rank "
                f"checkpointing is planned; until then, either drop "
                f"--checkpoint-dir or give every cylinder a single rank."
            )

        # Two scenario names that sanitize to the same file name would
        # silently overwrite each other's model files; refuse now rather than
        # at the first write.
        ckpt.check_filename_collisions(opt.local_scenarios)

        # Create and probe the directory now. Discovering only at write time
        # that the path is unwritable would mean the run never checkpoints.
        try:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            probe = os.path.join(self.ckpt_dir, ".mpisppy_write_probe")
            with open(probe, "w"):
                pass
            os.remove(probe)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot write to the checkpoint directory "
                f"'{self.ckpt_dir}' ({type(exc).__name__}: {exc})."
            ) from exc

    def pre_iter0(self):
        if self.spoke_mode:
            # xhat_prep calls this once, before the spoke's loop starts, which
            # is the spoke's equivalent of the hub's resume branch in Iter0.
            self._restore_incumbent()
            return
        if not self.write_enabled:
            return
        # Prove now that this run's models can actually be checkpointed. A run
        # that only found out at its first write would lose exactly the state
        # checkpointing exists to preserve.
        ckpt.probe_model_is_dillable(self.opt)

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
        state = ckpt.load_spoke_incumbent(self.opt, resume_from, cylinder,
                                          ordinal)
        if state is None:
            global_toc(f"No checkpointed incumbent for {cylinder} in "
                       f"{resume_from}; this spoke starts without one", rank0)
            return
        # The ordinal is stable when an unrelated cylinder comes or goes, but
        # not when one of two same-class spokes does: the survivor's ordinal
        # becomes the removed one's, and it would read that spoke's file
        # without a word. The values are feasible for the same model, so
        # nothing downstream would notice.
        was = state.get("class_count")
        now = self._class_ordinal_and_count()[1]
        if was is not None and was != now:
            global_toc(
                f"WARNING: the checkpoint was written by a wheel carrying "
                f"{was} {cylinder} cylinder(s) and this run has {now}, so "
                f"this spoke may be restoring an incumbent that belonged to a "
                f"different one. It is a feasible solution for the same "
                f"model either way.", rank0)
        obj = ckpt.restore_spoke_incumbent(self.opt, state)
        self.restored_incumbent_obj = obj
        self.opt.spcomm.best_inner_bound = state["best_inner_bound"]
        # _last_written_obj means "what the file in self.ckpt_dir already
        # holds", so it may only be seeded when that is the file we just read.
        # Resuming into a *different* directory is the documented
        # stop-today-resume-tomorrow flow, and there this spoke has written
        # nothing yet: seeding it there makes the skip test below decline to
        # write until the spoke strictly improves on what it restored, so a
        # study whose xhat has stopped improving -- a converged one, the case
        # where the answer is worth the most -- publishes no incumbent at all
        # and the next resume starts without one, with exit code 0 throughout.
        if self.write_enabled and self._same_directory(resume_from):
            self._last_written_obj = obj
        # The hub learns bounds only from what a spoke sends, so a restored
        # incumbent that is never published leaves the hub reporting an
        # infinite inner bound -- and its gap and convergence tests reading
        # from it -- until this spoke happens to improve on the answer it
        # already has. Publishing needs the send buffers, which exist by the
        # time the loop runs but not necessarily here, so it is deferred to
        # the first checkpoint point.
        self._publish_restored_bound = state["best_inner_bound"]
        global_toc(f"Restored the checkpointed incumbent for {cylinder} "
                   f"(objective {obj})", rank0)

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
        """
        iteration = int(getattr(self.opt, "_PHIter", 0))
        return iteration % self.every == 0 or self._is_final_iteration()

    def maybe_checkpoint(self):
        """Write the checkpoint if this iteration is a checkpoint point.

        See the module docstring for why the write lives here.

        A mid-run write failure -- disk full, an NFS hiccup -- is warned
        about, not raised: the previously published generation is untouched
        and remains resumable, while the optimization progress that a raise
        would destroy lives only in memory. The next checkpoint point tries
        again. Conditions detectable at setup (unwritable directory,
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
        """
        if self.spoke_mode:
            self._spoke_checkpoint()
            return
        if not self.write_enabled:
            return
        if not self._should_write():
            return
        try:
            self._write()
        except Exception as exc:
            global_toc(
                f"WARNING: checkpoint write failed at iteration "
                f"{int(getattr(self.opt, '_PHIter', 0))} "
                f"({type(exc).__name__}); the run continues, the previously "
                f"published checkpoint (if any) is intact, and the next "
                f"checkpoint point will try again.\n{exc}",
                self.opt.cylinder_rank == 0)

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

    def _spoke_checkpoint(self):
        """Publish a restored bound, then write the incumbent if it improved.

        Called once per pass of a loop that spins while it waits on the hub,
        so the common case has to be cheap: comparing two floats and
        returning. A write happens only when the incumbent objective differs
        from the one already on disk.

        Failures warn rather than raise, for the hub's reason and one more:
        this file is an optimization. Losing it costs a resumed run the
        answer it had found, which is worth a loud warning and not worth
        killing a running spoke over.
        """
        spoke = self.opt.spcomm
        if self._publish_restored_bound is not None:
            bound, self._publish_restored_bound = \
                self._publish_restored_bound, None
            if bound is not None:
                spoke.send_bound(bound)
                spoke.send_best_xhat()

        if not self.write_enabled:
            # --resume-from with no --checkpoint-dir. Publishing the restored
            # bound above is this spoke's whole job; there is nowhere to
            # write. Without this the write below is attempted with
            # ckpt_dir=None on every improvement, and only the warning path
            # keeps that from ending the run.
            return

        obj = getattr(self.opt, "best_solution_obj_val", None)
        if obj is None or obj == self._last_written_obj:
            return
        try:
            cylinder, ordinal = self._spoke_identity()
            path = ckpt.write_spoke_incumbent(
                self.opt, self.ckpt_dir, cylinder, ordinal,
                best_inner_bound=getattr(spoke, "best_inner_bound", None),
                class_count=self._class_ordinal_and_count()[1])
        except Exception as exc:
            global_toc(
                f"WARNING: this spoke could not write its incumbent "
                f"({type(exc).__name__}); the run continues and the next "
                f"improvement will try again.\n{exc}",
                self.opt.cylinder_rank == 0)
            return
        if path is None:
            return
        self._last_written_obj = obj
        # One line, not the pair the hub prints. The pair exists to measure a
        # write whose cost a user has to trade off against checkpoint
        # frequency; this write has no frequency knob and costs a rename.
        global_toc(f"Checkpointed incumbent (objective {obj})",
                   self.opt.cylinder_rank == 0)
