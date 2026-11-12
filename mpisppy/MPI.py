###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import os as _os
import re as _re

_enable_progress_thread = _os.environ.get("MPISPPY_MPI_PROGRESS_THREAD") == "1"
_native_async_progress = False


def _has_mvapich_async_progress(mpi):
    if _os.environ.get("MVAPICH_ASYNC_PROGRESS") != "1":
        return False

    vendor, version = mpi.get_vendor()
    if vendor.upper().startswith("MVAPICH"):
        return tuple(version[:3]) > (2, 3, 7)

    # Some mpi4py builds identify the vendor only in the library version.
    library_version = mpi.Get_library_version().lstrip()
    match = _re.match(
        r"MVAPICH(?:2|-GDR)?\b[^\d\n]*(\d+)\.(\d+)(?:\.(\d+))?",
        library_version,
        flags=_re.IGNORECASE,
    )
    if match is None:
        return False
    return tuple(int(part or 0) for part in match.groups()) > (2, 3, 7)

try:
    if _enable_progress_thread:
        # Set the thread level before mpi4py initializes MPI. Let mpi4py
        # finalize MPI at exit when it owns initialization. Open MPI's OSC
        # components otherwise make no RMA progress during solver work.
        import mpi4py as _mpi4py

        _mpi4py.rc.initialize = True
        _mpi4py.rc.threads = True
        _mpi4py.rc.thread_level = "multiple"

    from mpi4py import MPI as _mpi

    _native_async_progress = _has_mvapich_async_progress(_mpi)
    if _enable_progress_thread:
        if not _mpi.Is_initialized():
            raise RuntimeError(
                "MPISPPY_MPI_PROGRESS_THREAD=1 requires initialized MPI; "
                "import mpi-sppy before disabling mpi4py's auto-initialization"
            )
        _provided_thread_level = _mpi.Query_thread()
        # Let MVAPICH supply progress itself when its async worker is enabled.
        if _native_async_progress:
            _enable_progress_thread = False
            if _mpi.COMM_WORLD.Get_rank() == 0:
                print(
                    "mpi-sppy MPI progress: MVAPICH native async progress is active; "
                    "using the native runtime",
                    flush=True,
                )
        elif _provided_thread_level < _mpi.THREAD_MULTIPLE:
            raise RuntimeError(
                "MPISPPY_MPI_PROGRESS_THREAD=1 requires MPI_THREAD_MULTIPLE; "
                f"MPI provided level {_provided_thread_level}"
            )

    from mpi4py.MPI import *  # noqa: F403

    haveMPI = True

except ImportError as _mpi_import_error:
    if _enable_progress_thread:
        raise RuntimeError(
            "MPISPPY_MPI_PROGRESS_THREAD=1 requires mpi4py and the native MPI "
            "progress extension"
        ) from _mpi_import_error

    import copy as _cp

    import numpy as _np

    UNDEFINED = -1
    SUM = _np.sum
    MAX = _np.max 
    MIN = _np.min
    LOR = _np.logical_or
    DOUBLE = _np.double
    INT = _np.intc
    haveMPI = False

    class _MockMPIComm:

        @property
        def rank(self):
            return 0

        @property
        def size(self):
            return 1

        def allreduce(self, sendobj, op=SUM):
            return _cp.deepcopy(sendobj)

        def barrier(self):
            pass

        def bcast(self, data, root=0):
            return data

        def gather(self, obj, root=0):
            if root != self.rank:
                return
            else:
                return [obj]

        def Gatherv(self, sendbuf, recvbuf, root=0):
            # TBD: check this
            _set_data(sendbuf, recvbuf)            

        def Allreduce(self, sendbuf, recvbuf, op=SUM):
            _set_data(sendbuf, recvbuf)

        def Barrier(self):
            pass

        def Bcast(self, data, root=0):
            pass

        def Get_rank(self):
            return self.rank

        def Get_size(self):
            return self.size

        def Split(self, color=0, key=0):
            return _MockMPIComm()

    def _process_BufSpec(bufspec):
        if isinstance(bufspec, (list, tuple)):
            bufspec = bufspec[0]
        return bufspec, bufspec.size, bufspec.dtype

    def _set_data(sendbuf, recvbuf):
        send_data, send_size, send_type = _process_BufSpec(sendbuf)
        recv_data, recv_size, recv_type = _process_BufSpec(recvbuf)

        if send_size != recv_size:
            raise RuntimeError("Send and receive buffers should be of the same size")
        if send_type != recv_type:
            raise RuntimeError("Send and receive buffers should be of the same type")

        recv_data[:] = send_data

    COMM_WORLD = _MockMPIComm()


if haveMPI and _enable_progress_thread:
    # MPI thread-level configuration above must precede worker selection.
    from mpisppy.mpi_progress import create_controller as _create_progress_controller

    _progress_controller = _create_progress_controller(_mpi, COMM_WORLD, __file__)
    progress_active = _progress_controller.progress_active
else:
    from contextlib import nullcontext as progress_active  # noqa: F401

# A multi-node wheel needs progress on every rank. Keep the selected mode
# available to its startup guard, including MVAPICH's own async worker.
progress_available = haveMPI and (_enable_progress_thread or _native_async_progress)
