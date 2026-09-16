###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Console scripts for the drivers that mpi-sppy installs on the PATH.

pip's console entry points bypass ``python -m mpi4py``, whose runner ends
the job when a rank dies; without that, the surviving ranks block forever
in a collective and the whole mpiexec job hangs.  Importing mpi-sppy is
what puts mpi4py's own mechanism in place instead (see
``mpisppy.utils.mpi_abort``), so ``mpiexec -np 3 mpi-sppy-generic-cylinders
...`` behaves as the ``python -m mpi4py -m mpisppy.generic_cylinders`` form
does.  Serial runs, and the no-mpi4py mock in mpisppy.MPI, are untouched:
same traceback, same exit code.

Loading this module loads the ``mpisppy`` package first, so the hook is
already in place when the functions below import their target module.  That
covers a failure *during* one of those imports -- which need not strike
every rank, e.g. a flaky shared filesystem -- rather than hanging the ranks
it missed.
"""


def generic_cylinders_main():
    from mpisppy.generic_cylinders import main
    main()


def mrp_generic_main():
    from mpisppy.mrp_generic import main
    main()


def one_sided_test_main():
    from mpi_one_sided_test import main
    main()
