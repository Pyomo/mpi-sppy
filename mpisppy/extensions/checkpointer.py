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
for checkpointing pays nothing. This extension decides *when* to write;
``mpisppy/utils/checkpointing.py`` owns the on-disk format, and the resume
branch lives in ``PHBase.Iter0`` (restoring has to happen mid-startup, before
solvers are created).

The trigger implemented here is the terminal checkpoint
(``--checkpoint-at-termination``, on by default): one complete, resumable
checkpoint written when the run ends for any internal reason -- convergence,
the iteration limit, or hitting ``--time-limit``. That is the planned-stop path
for a multi-day study: set ``--time-limit`` to the day's budget and the run
stops itself and checkpoints, ready to resume the next morning.

See ``doc/designs/checkpointing_design.md``.
"""

import os

from mpisppy import global_toc
from mpisppy.extensions.extension import Extension
import mpisppy.utils.checkpointing as ckpt


class Checkpointer(Extension):
    """Write a resumable checkpoint on the run's own termination."""

    def __init__(self, opt):
        super().__init__(opt)
        options = opt.options
        self.ckpt_dir = options.get("checkpoint_dir", None)
        self.backend = options.get("checkpoint_backend",
                                   ckpt.DILL_RELOAD_BACKEND)
        self.at_termination = options.get("checkpoint_at_termination", True)

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

        # Multi-rank writing is not implemented: every rank would compute the
        # same staging and generation directory and race to create, replace and
        # delete it, so ranks destroy each other's files. Refuse rather than
        # abort the job at its very end with a half-published generation.
        if getattr(opt, "n_proc", 1) > 1:
            raise RuntimeError(
                f"Checkpointing currently supports a single rank, but this "
                f"run has {opt.n_proc}. Multi-rank checkpointing is planned; "
                f"until then, drop --checkpoint-dir or run on one rank."
            )

        # Create and probe the directory now. Discovering at the terminal
        # write that the path is unwritable would raise out of finalization
        # and take the run's other output (solution files) with it.
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

    def _write(self, why):
        """Write one generation, bracketed by toc so the cost is legible.

        The pair of timestamps *is* the measured write duration, which is what
        a user needs in order to choose a deadline for the anticipated
        trigger -- mpi-sppy deliberately does not estimate that cost for them.
        """
        rank0 = self.opt.cylinder_rank == 0
        generation = int(getattr(self.opt, "_PHIter", 0))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        global_toc(f"Writing checkpoint ({why}) at iteration {generation} "
                   f"to {self.ckpt_dir}", rank0)
        ckpt.write_checkpoint(self.opt, self.ckpt_dir, generation,
                              backend=self.backend)
        global_toc(f"Checkpoint written ({why}) at iteration {generation}",
                   rank0)

    def pre_iter0(self):
        # Prove now that this run's models can actually be checkpointed. A run
        # that only finds out at its terminal checkpoint would lose exactly the
        # state it was trying to preserve.
        ckpt.probe_model_is_dillable(self.opt)

    def enditer(self):
        """Snapshot at the one point in the iteration where state is coherent.

        ``iterk_loop`` runs Compute_Xbar, then Update_W, then *may break* --
        on the user converger, the convergence threshold, or ``--time-limit``
        -- and only then solves. An exit through any of those breaks leaves the
        dual weights advanced to iteration k while the nonanticipative values
        are still those of iteration k-1's solve. Checkpointing that state and
        resuming from it would apply the dual update to the same iterate twice
        and skip iteration k's solve entirely.

        ``enditer`` fires after the solve, so W and the nonants are both at
        iteration k. Writing here means the checkpoint always describes a
        completed iteration, whatever the run exits on -- which matters most
        for the planned-stop recipe, since ``--time-limit`` exits through
        exactly one of those early breaks.

        Retention is a single generation, so each write replaces the last.
        """
        if not self.at_termination:
            return
        self._write("iteration complete")
