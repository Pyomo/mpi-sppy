###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Run generic_cylinders with one rank's checkpoint write sabotaged.

Used by ``test_checkpoint_multirank.py`` to pin the behavior a multi-rank write
has to have and cannot be talked out of: a write that fails on *one* rank must
not hang the others.

The design has a failed write warn and let the run continue, because the
previously published checkpoint is still resumable while the optimization
progress a raise would destroy lives only in memory. On one rank that is a
return. On several it is a deadlock unless the ranks agree about the failure --
the rank that gave up would skip the barrier the others are waiting at, and the
job would sit there until its wall-clock limit with no error anywhere.

Usage::

  mpiexec -np 2 python -m mpi4py multirank_failure_driver.py \\
      --fail-on-rank 1 --fail-at-generation 3 [generic_cylinders args]

Not named ``test_*``: it is a helper, and pytest must not collect it.
"""

import sys

import mpisppy.utils.checkpointing as ckpt
from mpisppy import generic_cylinders


def _sabotage(fail_on_rank, fail_at_generation):
    """Make one rank's model write raise, once, at one generation."""
    real_write_models = ckpt._write_models

    def failing_write_models(opt, staging_dir, rank, backend):
        if (rank == fail_on_rank
                and int(getattr(opt, "_PHIter", 0)) == fail_at_generation):
            raise OSError(28, "No space left on device (injected by the test)")
        return real_write_models(opt, staging_dir, rank, backend)

    ckpt._write_models = failing_write_models


def main():
    args = sys.argv[1:]
    if args[0] != "--fail-on-rank" or args[2] != "--fail-at-generation":
        raise RuntimeError(
            "usage: multirank_failure_driver.py --fail-on-rank R "
            "--fail-at-generation G [generic_cylinders args]")
    _sabotage(int(args[1]), int(args[3]))
    sys.argv = [sys.argv[0]] + args[4:]
    generic_cylinders.main()


if __name__ == "__main__":
    main()
