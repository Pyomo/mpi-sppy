###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Write checkpoints so a run can be stopped and resumed later.

Attached only when ``--checkpoint-dir`` is given, so a run that does not ask
for checkpointing pays nothing at all -- the extension is never constructed and
none of its hooks exist. This extension decides *when* to write;
``mpisppy/utils/checkpointing.py`` owns the on-disk format, and the resume
branch lives in ``PHBase.Iter0`` (restoring has to happen mid-startup, before
solvers are created).

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

Writing at ``enditer`` sidesteps all of it. ``enditer`` fires after the solve,
so a checkpoint written there always describes a *completed* iteration, no
matter which extensions are loaded or what they touched. The invariant is one
sentence and it holds by construction.

The cost is a model serialization per iteration rather than one per run. That
is the deliberate trade: correctness that needs no knowledge of any extension.
Retention is a single generation, so each write replaces the last, and the disk
footprint does not grow with the iteration count. Each write is bracketed by
``global_toc`` so the per-iteration cost is visible in the log rather than
guessed at.

A checkpoint therefore describes a *completed PH iteration*. A run that ends
before finishing iteration 1 publishes nothing: no iteration completed, so
there is no iterate to resume from. Iteration 0 is deliberately not a
checkpoint point -- ``Iter0`` splices the W and proximal terms into the
objective after the last extension hook available to us, so a checkpoint taken
during it would capture a model whose objective is not yet the one PH iterates
on.

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

        if self.ckpt_dir is None:
            raise RuntimeError(
                "Checkpointer was attached without a checkpoint directory. "
                "It should only be attached when --checkpoint-dir is set."
            )
        # Everything below fails at setup rather than after a multi-hour run
        # reaches its first write and discovers it cannot finish one.
        if self.backend != ckpt.DILL_RELOAD_BACKEND:
            raise RuntimeError(
                f"--checkpoint-backend '{self.backend}' is not implemented. "
                f"The only supported backend is "
                f"'{ckpt.DILL_RELOAD_BACKEND}'."
            )
        ckpt.require_dill(self.backend)

        # The invariant this design rests on -- enditer fires after the solve,
        # so W and the nonants agree -- is a property of the *synchronous*
        # iterk_loop. APH inherits this wiring because aph_hub is built by
        # calling ph_hub, but its loop dispatches a fraction of the scenarios
        # per pass, keeps its own hardcoded iteration range that no resume
        # offset touches, and runs on a worker thread under the listener. A
        # checkpoint written there would not describe a completed iteration
        # and a resumed run would renumber from 1, overwriting the checkpoint
        # it resumed from.
        from mpisppy.opt.ph import PH
        if not isinstance(opt, PH):
            raise RuntimeError(
                f"Checkpointing currently supports the synchronous PH hub "
                f"only, but this hub is {type(opt).__name__}. Remove "
                f"--checkpoint-dir, or run PH."
            )

        # Multi-rank writing is not implemented: every rank would compute the
        # same staging and generation directory and race to create, replace and
        # delete it, so ranks destroy each other's files. Refuse rather than
        # abort the job at its very end with a half-published generation.
        n_proc = getattr(opt, "n_proc", 1)
        if n_proc > 1:
            raise RuntimeError(
                f"Checkpointing currently supports a single rank per hub, but "
                f"this hub has {n_proc}. Multi-rank checkpointing is planned; "
                f"until then, either drop --checkpoint-dir or give the hub a "
                f"single rank."
            )

        # Create and probe the directory now. Discovering at write time that
        # the path is unwritable would raise out of the iteration loop and take
        # the run with it.
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
        # Prove now that this run's models can actually be checkpointed. A run
        # that only found out at its first write would lose exactly the state
        # checkpointing exists to preserve.
        ckpt.probe_model_is_dillable(self.opt)

    def enditer(self):
        """Write the checkpoint. See the module docstring for why it is here."""
        self._write()

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
