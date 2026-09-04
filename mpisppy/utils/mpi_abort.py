###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Abort the MPI job when a rank dies, instead of hanging the rest.

An exception that strikes some ranks and not others leaves the survivors
blocked in the next collective, so an mpiexec job that has already failed
sits there until someone kills it -- with no traceback, because the rank
that has one is waiting to be reaped.

``python -m mpi4py`` solves this, and it is the only launcher that does:
the console entry points in ``mpisppy/entry_points.py`` bypass it, and so
does a bare ``mpiexec -np 3 python my_driver.py``.  Rather than reimplement
what it does, this installs mpi4py's own mechanism for the launchers that
miss it, so a job behaves the same way however it was started.

What that mechanism is, exactly: an excepthook that hands the exception to
``mpi4py.run.set_abort_status``, which records a status and lets interpreter
exit call ``COMM_WORLD.Abort(status)``.  Three things follow, and each is a
property worth having rather than an accident:

* The abort is deferred to exit, so the failing rank's ``finally`` blocks,
  its ``atexit`` handlers and its buffered output all run first.
* Only *uncaught* exceptions reach an excepthook, so a caller that catches
  its own failure -- a driver retrying with another solver, a test asserting
  that a call raises -- is untouched.  This is the reason for an excepthook
  rather than a wrapper around ``run()``: a wrapper fires on exceptions the
  caller meant to handle.
* ``SystemExit`` never reaches an excepthook at all, so ``sys.exit`` and
  argparse pass through with their own status, and ``KeyboardInterrupt``
  does reach it, so Ctrl-C ends the job rather than stranding the ranks
  that were inside a collective when it arrived.

The abort is always on ``COMM_WORLD``.  MPI offers no way to end part of a
job -- ``MPI_Abort`` on a sub-communicator is permitted to take everything
down, and OpenMPI does -- so a caller that runs one wheel per group of ranks
and wants a failed group not to take the others with it needs process-level
isolation, one mpiexec job per group.

Keep this module free of heavy imports.  Its callers install the hook before
importing anything else, precisely so that a failure *during* those imports
aborts too.
"""

import sys

from mpisppy import MPI, haveMPI

#: Installed at most once per process, and never removed: an excepthook that
#: uninstalled itself would leave the window it was covering.
_installed = False


def abort_on_uncaught_exception():
    """Make an uncaught exception abort the job, as ``python -m mpi4py`` does.

    Idempotent, and a no-op where there is nothing to abort: without mpi4py
    (the mock comm in ``mpisppy.MPI``), and on a single-rank job, where a
    traceback and an exit code already say everything an abort would and
    say it more clearly.

    Returns True if the hook is in place afterwards, for callers that want
    to say so.
    """
    global _installed
    if _installed:
        return True
    if not haveMPI:
        return False
    try:
        if MPI.COMM_WORLD.Get_size() <= 1:
            return False
        from mpi4py.run import set_abort_status
    except Exception:
        # No usable communicator, or an mpi4py without the runner module.
        # The caller's own exception is what should be reported, not ours.
        return False

    previous = sys.excepthook

    def _abort_then_report(exc_type, exc, traceback):
        # Order matters: record the status first, so that if reporting the
        # exception itself fails the job still ends rather than hanging.
        try:
            set_abort_status(exc)
        finally:
            previous(exc_type, exc, traceback)

    sys.excepthook = _abort_then_report
    _installed = True
    return True
