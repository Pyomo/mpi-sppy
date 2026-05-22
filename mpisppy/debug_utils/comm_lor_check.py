###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Collective sanity check for an MPI communicator.

Each rank contributes uint8 0 and the values are combined with MPI.LOR. Since
every contribution is 0, the reduced value is always 0 in normal operation,
so the call is effectively a barrier that announces itself. If MPI hangs or
the comm is broken, the missing STOP line is itself diagnostic.

The non-zero branch is a defensive guard: if any rank somehow sees a non-zero
reduced value, that rank prints the value, the comm name, and a stack trace.
Under real MPI it cannot fire (every contribution is hard-coded to 0); the
test exercises it via a stub communicator whose Allreduce writes 1 into the
recv buffer.
"""

import traceback

import numpy as np

import mpisppy.MPI as MPI


def comm_lor_check(comm, name):
    """Collective LOR sanity check on `comm`.

    Args:
        comm: an MPI communicator (or the single-rank mock from mpisppy.MPI).
        name (str): human-readable label for the comm, used in messages.
    """
    rank = comm.Get_rank()
    if rank == 0:
        print(f"[comm_lor_check] START comm={name!r}", flush=True)

    local = np.zeros(1, dtype=np.uint8)
    reduced = np.zeros(1, dtype=np.uint8)
    comm.Allreduce(local, reduced, op=MPI.LOR)

    if reduced[0] != 0:
        print(
            f"[comm_lor_check] NONZERO value={int(reduced[0])} "
            f"comm={name!r} rank={rank}",
            flush=True,
        )
        traceback.print_stack()

    if rank == 0:
        print(f"[comm_lor_check] STOP  comm={name!r}", flush=True)
