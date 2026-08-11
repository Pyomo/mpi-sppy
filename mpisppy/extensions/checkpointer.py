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
        # Fail at setup rather than after a multi-hour run reaches its first
        # write and discovers the backend is unusable.
        ckpt.require_dill(self.backend)

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

    def post_everything(self):
        if not self.at_termination:
            return
        # post_everything runs after scenario_denouement. Standard denouements
        # only report, but one that re-solves or mutates a model would be
        # captured in this checkpoint.
        self._write("termination")
