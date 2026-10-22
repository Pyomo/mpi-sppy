###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' Basic test script to make sure that the one-sided MPI calls work as
    intended. Runs on two processes. Will raise an assertion error if the
    Lock/Get/Unlock combination blocks. Otherwise, exits normally and produces
    no output.

    Just because this test passes doesn't mean that MPI one-sided calls will
    work as expected.

    Takes about 4 seconds to execute, with all the sleep() calls.
'''
# Import through mpi-sppy so MPISPPY_MPI_PROGRESS_THREAD=1 selects its
# MPI_THREAD_MULTIPLE initialization and shared progress thread. Without that
# variable this remains the ordinary mpi4py-backed MPI interface.
from mpisppy import MPI as mpi
import numpy as np
import time
import sys

def main():
    if mpi.COMM_WORLD.Get_size() == 1:
        print("ERROR: This script must be run with multiple MPI processes using mpirun or mpiexec, e.g.:", file=sys.stderr)
        print("       mpirun -n 2 mpi-sppy-one-sided-test", file=sys.stderr)
        sys.exit(2) # Exit status 2: command line usage error
    
    rank = mpi.COMM_WORLD.Get_rank()

    array_size = 10
    win = mpi.Win.Allocate(mpi.DOUBLE.size*array_size, mpi.DOUBLE.size,
                           comm=mpi.COMM_WORLD)
    buff = np.ndarray(buffer=win.tomemory(), dtype='d', shape=(array_size,))

    if (rank == 0):
        buff[:] = 3. * np.ones(array_size, dtype='d')
        print("rank 0 published 3.0", flush=True)
        time.sleep(3)
        buff[:] = np.arange(array_size)
        print("rank 0 replaced buffer with arange", flush=True)

        win.Lock(1)
        win.Put((buff, array_size, mpi.DOUBLE), target_rank=1)
        win.Unlock(1)

    elif (rank == 1):
        buff = np.ones(array_size, dtype='d')
        time.sleep(1)

        start = time.monotonic()
        win.Lock(0)
        win.Get((buff, array_size, mpi.DOUBLE), target_rank=0)
        win.Unlock(0)
        elapsed = time.monotonic() - start

        model = win.Get_attr(mpi.WIN_MODEL)
        model_name = "unified" if model == mpi.WIN_UNIFIED else "separate"
        print(
            f"first Get: tail={buff[-1]!r}, elapsed={elapsed:.3f}s, "
            f"window_model={model_name}",
            flush=True,
        )

        assert(buff[-1] == 3.)

        time.sleep(3)

        win.Lock(1)
        win.Get((buff, array_size, mpi.DOUBLE), target_rank=1)
        win.Unlock(1)

        assert(buff[-1] == array_size-1)

    del buff # Important!
    win.Free()
    if (rank == 1):
        print("Test passed. You might have an MPI installation that will work.")

if __name__=='__main__':
    main()
