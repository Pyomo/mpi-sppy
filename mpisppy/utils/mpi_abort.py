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

import sys
import traceback

from mpisppy import MPI


def _exit_status(exc):
    """What a ``SystemExit`` means as a status: 0 for a clean exit.

    ``sys.exit()`` and ``sys.exit(None)`` are 0; ``sys.exit(2)`` is 2;
    ``sys.exit("no scenario data")`` prints the message and exits 1.
    """
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _abort(comm):
    """Report and abort ``comm``, or return if there is nothing to abort."""
    if comm is None:
        comm = MPI.COMM_WORLD
    try:
        multi_rank = comm.Get_size() > 1
    except Exception:
        # Not a real communicator -- a test stub, say. Let the caller's own
        # exception be what is reported, rather than this one.
        return
    if not multi_rank or not hasattr(comm, "Abort"):
        return
    traceback.print_exc()
    # Abort does not unwind, so whatever is still buffered is lost with the
    # process. On a batch job stdout is a file and Python block-buffers it,
    # which is the whole run log.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    comm.Abort(1)


def run_with_mpi_abort(real_main, comm=None):
    """Call ``real_main()``, aborting the MPI job if it raises.

    Returns whatever ``real_main`` returns. The exception is re-raised after
    the abort request, so a serial run -- and the no-mpi4py mock comm, which
    has no ``Abort`` -- behaves exactly as it did before: same traceback,
    same exit code.

    ``comm`` is the communicator the job spans, defaulting to
    ``COMM_WORLD``. Callers that run on a sub-communicator must pass it, or
    one group's failure takes down groups that have nothing to do with it,
    and takes the caller's own ``except`` with it.

    Two exceptions are deliberately not aborted on:

    * ``KeyboardInterrupt``. The launcher delivers SIGINT to every rank, so
      an interrupt is already uniform and strands nobody -- the reason to
      abort does not apply, and propagating lets each rank's ``finally``
      and ``atexit`` write out what it has.
    * ``SystemExit`` with a status of 0. This is the rule mpi4py's own
      runner uses, and it is what separates "argparse printed --help on
      every rank" from "one rank's scenario_creator called
      ``sys.exit('no scenario data')``" -- which is not uniform, and which
      hangs the job exactly like any other one-rank failure.
    """
    try:
        return real_main()
    except KeyboardInterrupt:
        raise
    except SystemExit as exc:
        if _exit_status(exc):
            _abort(comm)
        raise
    except BaseException:
        _abort(comm)
        raise
