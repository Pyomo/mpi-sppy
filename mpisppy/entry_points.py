###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Console-script wrappers that add mpi4py-style abort on uncaught exceptions.

pip's console entry points bypass ``python -m mpi4py``, whose runner ends
the job when a rank dies; without that, the surviving ranks block forever in
a collective and the whole mpiexec job hangs.  These wrappers install
mpi4py's own mechanism first (``mpisppy.utils.mpi_abort``), so
``mpiexec -np 3 mpi-sppy-generic-cylinders ...`` behaves as the
``python -m mpi4py -m mpisppy.generic_cylinders`` form does.  Serial runs,
and the no-mpi4py mock in mpisppy.MPI, are untouched: same traceback, same
exit code.

It is installed before the target module is imported, so that a failure
*during* that import -- which need not strike every rank, e.g. a flaky
shared filesystem -- ends the job too.
"""

from mpisppy.utils.mpi_abort import abort_on_uncaught_exception


def _run_with_mpi_abort(real_main):
    abort_on_uncaught_exception()
    return real_main()


def generic_cylinders_main():
    def _main():
        from mpisppy.generic_cylinders import main
        main()
    _run_with_mpi_abort(_main)


def mrp_generic_main():
    def _main():
        from mpisppy.mrp_generic import main
        main()
    _run_with_mpi_abort(_main)


def one_sided_test_main():
    def _main():
        from mpi_one_sided_test import main
        main()
    _run_with_mpi_abort(_main)
