###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Abort the MPI job when one rank raises, instead of hanging the rest.

An exception that strikes some ranks and not others leaves the survivors
blocked in the next collective, so an mpiexec job that has already failed
sits there until someone kills it -- with no traceback, because the rank
that has one is waiting to be reaped.  ``python -m mpi4py`` solves this by
calling ``MPI_Abort`` on an uncaught exception, but it is only one of the
ways mpi-sppy is launched: the console entry points in
``mpisppy/entry_points.py`` bypass it, and so does a bare
``mpiexec -np 3 python my_driver.py``.

Keep this module free of heavy imports.  Its callers wrap code that has not
imported anything yet, precisely so that a failure *during* those imports
aborts too.
"""

import traceback

from mpisppy import MPI


def run_with_mpi_abort(real_main):
    """Call ``real_main()``, aborting the MPI job if it raises.

    Returns whatever ``real_main`` returns.  The exception is re-raised
    after the abort request, so a serial run (and the no-mpi4py mock comm,
    which has no ``Abort``) behaves exactly as it did before: same
    traceback, same exit code.
    """
    try:
        return real_main()
    except SystemExit:
        # argparse exits (--help, bad flags) happen identically on every
        # rank, so a plain exit cannot strand the others.
        raise
    except BaseException:
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1 and hasattr(comm, "Abort"):
            traceback.print_exc()
            comm.Abort(1)
        raise
