###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
'''Basic test script for the one-sided MPI protocol used by mpi-sppy.

    Runs on two processes. The publisher uses an exclusive self-lock, local
    stores, and Win.Sync. The reader uses a short Lock_all/Get/Flush/Unlock_all
    epoch. It raises an exception if either of two published values is not read
    correctly; otherwise it exits normally.

    Just because this test passes doesn't mean that MPI one-sided calls will
    work as expected.
'''
import mpi4py.MPI as mpi
import numpy as np
import sys


def _publish(win, exposed, values, rank):
    """Publish local stores using the same epoch as SPWindow.put()."""
    win.Lock(rank, mpi.LOCK_EXCLUSIVE)
    exposed[:] = values
    win.Sync()
    win.Unlock(rank)


def _fetch(win, target_rank, result):
    """Fetch a snapshot using the same epoch as SPWindow.get()."""
    win.Lock_all()
    win.Get((result, result.size, mpi.DOUBLE), target_rank=target_rank)
    win.Flush(target_rank)
    win.Unlock_all()


def main():
    if mpi.COMM_WORLD.Get_size() == 1:
        print("ERROR: This script must be run with multiple MPI processes using mpirun or mpiexec, e.g.:", file=sys.stderr)
        print("       mpirun -n 2 mpi-sppy-one-sided-test", file=sys.stderr)
        sys.exit(2)  # Exit status 2: command line usage error

    rank = mpi.COMM_WORLD.Get_rank()

    array_size = 10
    win = mpi.Win.Allocate(mpi.DOUBLE.size*array_size, mpi.DOUBLE.size,
                           comm=mpi.COMM_WORLD)
    exposed = np.ndarray(
        buffer=win.tomemory(), dtype='d', shape=(array_size,))

    # As in SPWindow.__init__, make direct initialization visible before the
    # window can be inspected. This remains valid on ranks whose data is not
    # fetched by this two-rank smoke test.
    _publish(win, exposed, np.full(array_size, np.nan), rank)

    if rank == 0:
        _publish(win, exposed, np.full(array_size, 3.0), rank)
        mpi.COMM_WORLD.send(None, dest=1, tag=0)
        mpi.COMM_WORLD.recv(source=1, tag=1)

        _publish(win, exposed, np.arange(array_size, dtype='d'), rank)
        mpi.COMM_WORLD.send(None, dest=1, tag=2)
        mpi.COMM_WORLD.recv(source=1, tag=3)

    elif rank == 1:
        result = np.empty(array_size, dtype='d')

        mpi.COMM_WORLD.recv(source=0, tag=0)
        _fetch(win, 0, result)
        np.testing.assert_array_equal(result, np.full(array_size, 3.0))
        mpi.COMM_WORLD.send(None, dest=0, tag=1)

        mpi.COMM_WORLD.recv(source=0, tag=2)
        _fetch(win, 0, result)
        np.testing.assert_array_equal(result, np.arange(array_size, dtype='d'))
        mpi.COMM_WORLD.send(None, dest=0, tag=3)

    del exposed  # Important: release the exported view before Win.Free().
    win.Free()
    if rank == 1:
        print("Test passed. You might have an MPI installation that will work.")


if __name__ == '__main__':
    main()
