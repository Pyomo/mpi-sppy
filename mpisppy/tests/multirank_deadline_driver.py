###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Run generic_cylinders with ``--checkpoint-before-seconds`` reached on one rank.

Used by ``test_checkpoint_multirank.py`` to pin the thing the deadline trigger
is most able to get wrong. Every other trigger is a pure function of the
iteration number, so the ranks of a cylinder reach a write together without
being asked. Elapsed wall clock is not: it is rank-local, and a rank that
believed its own clock and started writing while another did not would sit in
the write's barrier until the job's wall-clock limit, with no error anywhere.

So the trigger puts its test through ``allreduce_or``, and this driver skews
the clock on exactly one rank to make sure that is what happens: the skewed
rank is far past the deadline and the others are nowhere near it. The run
either writes on every rank and finishes, or it hangs -- there is no third
outcome, which is why the test can assert on a timeout.

The skew is applied around ``maybe_checkpoint`` rather than inside it, so the
trigger itself -- the test, the collective, the latch -- is the shipped code.

Usage::

  mpiexec -np 2 python -m mpi4py multirank_deadline_driver.py \\
      --skew-on-rank 1 --skew-at-generation 2 [generic_cylinders args]

Not named ``test_*``: it is a helper, and pytest must not collect it.
"""

import sys

from mpisppy import generic_cylinders
from mpisppy.extensions.checkpointer import Checkpointer


def _skew_the_clock(skew_on_rank, skew_at_generation):
    """Put one rank far past its deadline, at one generation."""
    real_maybe_checkpoint = Checkpointer.maybe_checkpoint

    def skewing_maybe_checkpoint(self):
        if (int(self.opt.cylinder_rank) == skew_on_rank
                and int(getattr(self.opt, "_PHIter", 0)) == skew_at_generation):
            # A year of elapsed time clears any deadline the test sets, and
            # clears it on this rank alone.
            self.opt.start_time -= 365 * 24 * 3600.0
        return real_maybe_checkpoint(self)

    Checkpointer.maybe_checkpoint = skewing_maybe_checkpoint


def main():
    args = sys.argv[1:]
    if args[0] != "--skew-on-rank" or args[2] != "--skew-at-generation":
        raise RuntimeError(
            "usage: multirank_deadline_driver.py --skew-on-rank R "
            "--skew-at-generation G [generic_cylinders args]")
    _skew_the_clock(int(args[1]), int(args[3]))
    sys.argv = [sys.argv[0]] + args[4:]
    generic_cylinders.main()


if __name__ == "__main__":
    main()
