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
hangs.  These wrappers hand the entry points to
``mpisppy.utils.mpi_abort.run_with_mpi_abort``, which gives them the same
protection, so
``mpiexec -np 3 mpi-sppy-generic-cylinders ...`` is as safe as the
``python -m mpi4py -m mpisppy.generic_cylinders`` form.  Serial runs
(and the no-mpi4py mock in mpisppy.MPI) re-raise normally so tracebacks
and exit codes are unchanged.
"""

from mpisppy.utils.mpi_abort import run_with_mpi_abort as _run_with_mpi_abort


# The target-module imports live inside the wrapped callables so that an
# exception raised while importing (which need not strike every rank,
# e.g. flaky shared filesystems) also aborts instead of hanging.

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
