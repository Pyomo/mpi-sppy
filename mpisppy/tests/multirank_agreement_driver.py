###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Run generic_cylinders with one rank's checkpoint setup or restore sabotaged.

Used by ``test_checkpoint_multirank.py`` to hold a property the whole
checkpointing protocol rests on: **no step of setting a checkpoint up or of
restoring one may raise on one rank alone.** Every one of those steps is per
rank by nature -- it reads the file named after this rank, checks it against
the scenarios this rank owns, or writes from the node this rank is on -- and
the next thing the run does is collective. A rank that raises on its own
leaves the rest of its cylinder waiting in that collective for a rank that has
already gone, and a refusal meant to arrive in the first second of a run
becomes a job that hangs until its wall-clock limit with nothing in the log.

``STEPS`` below names one step inside each agreement on those paths. The test
runs the job once per step with that step made to fail on one rank, and asks
for what an agreement promises and a rank-local raise cannot give: the job
ends, and the message says how many ranks could not do it.

Usage::

  mpiexec -np 6 python -m mpi4py multirank_agreement_driver.py \\
      --break-step load_checkpoint --on-cylinder-rank 1 [generic_cylinders args]

Not named ``test_*``: it is a helper, and pytest must not collect it.
"""

import sys

import mpisppy.utils.checkpointing as ckpt
from mpisppy import generic_cylinders

#: The injected failure, printed by the sabotaged step so the test can tell a
#: run that agreed about a failure from one where the step was never reached.
INJECTED = "INJECTED FAILURE"

#: One step inside each agreement on the setup and restore paths, named by the
#: module attribute to replace. Each takes the cylinder's ``opt`` as its first
#: argument, which is how the sabotage picks a single rank: the ranks of every
#: cylinder that reaches the step are numbered from zero, so this fails on one
#: rank of that cylinder and leaves the others to discover it.
STEPS = {
    # Setup, on every cylinder that checkpoints: Checkpointer.__init__.
    "probe_directory_is_writable": (ckpt, "probe_directory_is_writable"),
    # Restore, on the hub: PHBase._restore_from_checkpoint_if_resuming.
    "load_checkpoint": (ckpt, "load_checkpoint"),
    # Restore, on an xhat spoke: Checkpointer._restore_incumbent, which reads
    # the file and then puts the values back, agreeing on each.
    "load_spoke_incumbent": (ckpt, "load_spoke_incumbent"),
    "restore_spoke_incumbent": (ckpt, "restore_spoke_incumbent"),
    # Restore, at the end of the hub's Iter0:
    # PHBase._restore_extension_state_if_resuming.
    "restore_extension_state": (ckpt, "restore_extension_state"),
}


def _sabotage(step, cylinder_rank):
    """Make one rank's ``step`` raise, on every cylinder that runs it."""
    module, attribute = STEPS[step]
    real = getattr(module, attribute)

    def failing(opt, *args, **kwargs):
        if int(opt.cylinder_rank) == cylinder_rank:
            print(f"{INJECTED}: {step} on cylinder rank {cylinder_rank} of "
                  f"{type(opt).__name__}", flush=True)
            raise RuntimeError(f"{step} failed (injected by the test)")
        return real(opt, *args, **kwargs)

    setattr(module, attribute, failing)


def main():
    args = sys.argv[1:]
    if args[0] != "--break-step" or args[2] != "--on-cylinder-rank":
        raise RuntimeError(
            "usage: multirank_agreement_driver.py --break-step STEP "
            "--on-cylinder-rank R [generic_cylinders args]")
    _sabotage(args[1], int(args[3]))
    sys.argv = [sys.argv[0]] + args[4:]
    generic_cylinders.main()


if __name__ == "__main__":
    main()
