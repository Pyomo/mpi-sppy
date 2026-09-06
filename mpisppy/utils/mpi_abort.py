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
  that were inside a collective when it arrived -- except where a live
  non-daemon thread blocks the exit, which is the APH case below.

The abort is always on ``COMM_WORLD``.  MPI offers no way to end part of a
job -- ``MPI_Abort`` on a sub-communicator is permitted to take everything
down, and OpenMPI does -- so a caller that runs one wheel per group of ranks
and wants a failed group not to take the others with it needs process-level
isolation, one mpiexec job per group.

``mpisppy/__init__.py`` calls this at import, so a driver is covered from
its first ``import mpisppy`` rather than from the moment it reaches a
``WheelSpinner``.  That matters because the failures that strike one rank
and not the others are mostly the early ones -- reading a scenario file off
a flaky mount, checking out a per-rank solver license, importing a model
module -- and they happen before any wheel is built.  Keep the module free
of heavy imports for the same reason: it runs on every import of mpi-sppy.

The hook is never uninstalled, since one that uninstalled itself would
leave open the window it was covering.  Setting ``MPISPPY_NO_ABORT_HOOK``
declines to install it in the first place -- it has to be set before the
process starts, since that install happens at import -- for a job whose
ranks are doing independent work.  ``mpiexec -np 40 python sweep.py`` with
a case per rank has nothing to hang, so a rank that raises should take only
itself down and leave the other 39 to finish and write their results.  That
means genuinely independent: anything that builds an mpi-sppy object across
ranks is collective, ``--EF`` included, since ``SPBase.__init__`` gathers
and allreduces as soon as there is more than one rank.  Turning it off is
announced, because a job that hangs later should not leave anyone guessing
why.

Installing at import means importing mpi-sppy claims the process's
``sys.excepthook``, whether or not a wheel is ever spun.  An application
that embeds mpi-sppy for part of its work therefore has *its* uncaught
exceptions end the job too.  That is the intended trade -- a hang with no
traceback is the worse outcome, and an exception the application catches is
never seen here -- but it is a process-wide effect of an import and is
worth knowing.

Two things defeat the deferral itself.  A live non-daemon thread is one:
CPython joins those before it runs ``atexit``, so an uncaught exception on
the main thread while such a thread is still going blocks the interpreter
short of the abort.  ``Synchronizer.run`` in ``mpisppy/utils/listener_util``
starts two such threads and joins only one of them, so APH is in that
position for its whole run -- Ctrl-C during an APH job raises out of that
join with both threads alive, and the interpreter blocks before the abort
status is ever acted on.  This is the one way the deferred abort is weaker
than the immediate ``comm.Abort(1)`` it replaced, and making those threads
daemons is what would close it.

The other is an exception on a worker thread.  Those go to
``threading.excepthook``, and recording an abort status from there would
not help: the status is acted on at interpreter exit, and a main thread
waiting on that worker never reaches it.  ``APH`` is that shape -- it runs
its iterations on a thread while the main thread blocks in the listener's
``join`` (``mpisppy/utils/listener_util``), and nothing there notices a
worker that died -- so an uncaught exception inside an APH iteration hangs
the job, as it did before this module existed.  Ending that one needs the
synchronizer to see the dead worker and set ``quitting``, which is a change
to APH rather than to an excepthook.
"""

import os
import sys

# Imported from the mpisppy package rather than through it: mpisppy/__init__
# imports this module while it is still initializing, so reaching back for
# names it has bound would make this sensitive to the order of the lines
# there. Going straight to the submodule has no such dependency.
import mpisppy.MPI as MPI
from mpisppy.MPI import haveMPI

#: The hook this module installed, or None. Held as the object rather than a
#: bool so that a later call can tell "ours is still in place" from "someone
#: has replaced sys.excepthook since", and reinstall in the second case.
_installed_excepthook = None

#: Set to decline the hook. Values that read as off are treated as off, so
#: MPISPPY_NO_ABORT_HOOK=false does not disarm a job. Consulted where the
#: hook would be installed, which is the import of mpi-sppy, so it has to be
#: in the environment before the process starts: MPISPPY_NO_ABORT_HOOK=1
#: mpiexec ... Setting it from inside a running driver is too late, and
#: nothing is uninstalled -- see the note above on why.
OPT_OUT_ENVVAR = "MPISPPY_NO_ABORT_HOOK"

#: Anything else, including "1", "yes" and "please", turns the hook off.
_OFF_VALUES = ("", "0", "false", "no", "off")

#: Said once, not once per call site.
_announced_opt_out = False


def _announce_opt_out():
    """Say that the abort is off, once, so a later hang is not a mystery."""
    global _announced_opt_out
    if _announced_opt_out:
        return
    try:
        from mpisppy import global_toc
        if MPI.COMM_WORLD.Get_size() > 1:
            global_toc(f"{OPT_OUT_ENVVAR} is set: a rank that raises an "
                       "uncaught exception will not end the job, and the "
                       "other ranks can hang in a collective")
            # Latched only now. Setting it up front would mean a call that
            # failed to say anything -- an unusable comm, an import that
            # raised -- silenced every later attempt, and a job that opted
            # out would run with no message at all.
            _announced_opt_out = True
    except Exception:
        pass  # saying so is a courtesy; failing to say so must not raise


def abort_on_uncaught_exception():
    """Make an uncaught exception abort the job, as ``python -m mpi4py`` does.

    Called at import by ``mpisppy/__init__.py``; public because a driver
    that replaces ``sys.excepthook`` of its own accord can call it again to
    put the abort back in front of the new hook.

    A no-op where there is nothing to abort: without mpi4py (the mock comm
    in ``mpisppy.MPI``), and on a single-rank job, where a traceback and an
    exit code already say everything an abort would and say it more clearly.
    Also a no-op, with a message, when ``MPISPPY_NO_ABORT_HOOK`` is set.

    Returns True if the hook is in place afterwards, for callers that want
    to say so.
    """
    global _installed_excepthook
    if _installed_excepthook is not None \
            and sys.excepthook is _installed_excepthook:
        return True
    if os.environ.get(OPT_OUT_ENVVAR, "").strip().lower() not in _OFF_VALUES:
        _announce_opt_out()
        return False
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
    _installed_excepthook = _abort_then_report
    return True
