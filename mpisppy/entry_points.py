###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Console-script wrappers that add mpi4py-style abort on uncaught exceptions.

pip's console entry points bypass ``python -m mpi4py``, whose runner prints
the traceback and calls MPI_Abort when a rank dies; without that, the
surviving ranks block forever in a collective and the whole mpiexec job
hangs.  These wrappers give the entry points the same protection, so
``mpiexec -np 3 mpi-sppy-generic-cylinders ...`` is as safe as the
``python -m mpi4py -m mpisppy.generic_cylinders`` form.  Serial runs
(and the no-mpi4py mock in mpisppy.MPI) re-raise normally so tracebacks
and exit codes are unchanged.
"""

import traceback

from mpisppy import MPI


def _run_with_mpi_abort(real_main):
    try:
        real_main()
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


def generic_cylinders_main():
    from mpisppy.generic_cylinders import main
    _run_with_mpi_abort(main)


def mrp_generic_main():
    from mpisppy.mrp_generic import main
    _run_with_mpi_abort(main)


def one_sided_test_main():
    from mpi_one_sided_test import main
    _run_with_mpi_abort(main)
