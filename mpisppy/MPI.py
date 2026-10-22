###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import os as _os


_enable_progress_thread = _os.environ.get("MPISPPY_MPI_PROGRESS_THREAD") == "1"

try:
    if _enable_progress_thread:
        # mpi4py normally initializes MPI when mpi4py.MPI is imported.  Disable
        # that behavior so this opt-in path can require THREAD_MULTIPLE before
        # any MPI object is used.  Open MPI's OSC components otherwise make no
        # RMA progress while a target rank is in Python/solver work.
        import mpi4py as _mpi4py
        _mpi4py.rc.initialize = False
        _mpi4py.rc.threads = True
        _mpi4py.rc.thread_level = "multiple"

        from mpi4py import MPI as _mpi
        if _mpi.Is_initialized():
            _provided_thread_level = _mpi.Query_thread()
        else:
            _provided_thread_level = _mpi.Init_thread(_mpi.THREAD_MULTIPLE)
        if _provided_thread_level < _mpi.THREAD_MULTIPLE:
            raise RuntimeError(
                "MPISPPY_MPI_PROGRESS_THREAD=1 requires MPI_THREAD_MULTIPLE; "
                f"MPI provided level {_provided_thread_level}"
            )

    from mpi4py.MPI import *  # noqa: F403
    haveMPI = True

except ImportError:
    import numpy as _np
    import copy as _cp

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
    # Keep Open MPI making progress while the main application thread is in a
    # solver or other non-MPI Python code.  Iprobe is non-destructive: it does
    # not receive or reorder application messages.  The Event wait avoids a
    # pure busy spin; batch scripts reserve one CPU in addition to solver
    # threads for this thread.
    import atexit as _atexit
    import threading as _threading

    _progress_stop = _threading.Event()

    def _mpi_progress_loop():
        while not _progress_stop.is_set():
            COMM_WORLD.Iprobe(ANY_SOURCE, ANY_TAG)
            _progress_stop.wait(0.0005)

    _progress_thread = _threading.Thread(
        target=_mpi_progress_loop,
        name="mpisppy-mpi-progress",
        daemon=True,
    )
    _progress_thread.start()

    def _stop_mpi_progress_thread():
        _progress_stop.set()
        _progress_thread.join(timeout=5)

    # mpi4py registers its MPI finalizer while mpi4py.MPI is imported.  atexit
    # runs handlers in reverse registration order, so this stops the thread
    # before mpi4py finalizes MPI.
    _atexit.register(_stop_mpi_progress_thread)
