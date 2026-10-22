###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import os as _os
import math as _math
import time as _time
from contextlib import contextmanager as _contextmanager


_enable_progress_thread = _os.environ.get("MPISPPY_MPI_PROGRESS_THREAD") == "1"
_progress_diagnostics = _os.environ.get("MPISPPY_MPI_PROGRESS_DIAGNOSTICS") == "1"

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
    # solver or other non-MPI Python code. Iprobe is non-destructive: it does
    # not receive or reorder application messages. The worker is activated only
    # around long compute regions: polling continuously would contend with the
    # MPI_THREAD_MULTIPLE locks during normal MPI-heavy execution.
    import atexit as _atexit
    import threading as _threading

    try:
        _progress_max_interval = (
            float(_os.environ.get("MPISPPY_MPI_PROGRESS_INTERVAL_US", "500"))
            * 1e-6
        )
    except ValueError as _exc:
        raise RuntimeError(
            "MPISPPY_MPI_PROGRESS_INTERVAL_US must be a positive number"
        ) from _exc
    if not _math.isfinite(_progress_max_interval) or _progress_max_interval <= 0:
        raise RuntimeError("MPISPPY_MPI_PROGRESS_INTERVAL_US must be positive")

    _progress_stop = _threading.Event()
    _progress_active = _threading.Event()
    _progress_state_lock = _threading.Lock()
    _progress_active_count = 0
    _progress_probes = 0
    _progress_probe_seconds = 0.0
    _progress_max_probe_seconds = 0.0
    _progress_wakeups = 0
    _progress_active_seconds = 0.0
    _progress_worker_thread_cpu_seconds = 0.0
    _progress_worker_native_id = None
    _progress_wait_late_seconds = 0.0
    _progress_max_wait_late_seconds = 0.0
    _progress_compute_regions = 0
    _progress_compute_seconds = 0.0
    _progress_compute_process_cpu_seconds = 0.0
    _progress_compute_thread_cpu_seconds = 0.0

    def _mpi_progress_loop():
        global _progress_probes, _progress_probe_seconds
        global _progress_max_probe_seconds, _progress_wakeups
        global _progress_active_seconds, _progress_worker_thread_cpu_seconds
        global _progress_worker_native_id
        global _progress_wait_late_seconds, _progress_max_wait_late_seconds
        _progress_worker_native_id = _threading.get_native_id()
        if _progress_diagnostics:
            print(
                "mpi-sppy MPI progress: "
                f"rank={COMM_WORLD.Get_rank()}, "
                f"worker_native_id={_progress_worker_native_id}",
                flush=True,
            )
        worker_cpu_start = _time.thread_time()
        while not _progress_stop.is_set():
            _progress_active.wait()
            if _progress_stop.is_set():
                break

            active_start = _time.monotonic()
            with _progress_state_lock:
                _progress_wakeups += 1
            while _progress_active.is_set() and not _progress_stop.is_set():
                probe_start = _time.monotonic()
                COMM_WORLD.Iprobe(ANY_SOURCE, ANY_TAG)
                probe_seconds = _time.monotonic() - probe_start
                with _progress_state_lock:
                    _progress_probes += 1
                    _progress_probe_seconds += probe_seconds
                    _progress_max_probe_seconds = max(
                        _progress_max_probe_seconds, probe_seconds
                    )

                # A short initial burst minimizes the latency of a new RMA
                # operation. Back off for long solves to bound lock contention
                # and CPU use, while retaining the configured worst-case delay.
                elapsed = _time.monotonic() - active_start
                if elapsed < 0.010:
                    interval = min(50e-6, _progress_max_interval)
                elif elapsed < 0.100:
                    interval = min(250e-6, _progress_max_interval)
                else:
                    interval = _progress_max_interval
                wake_deadline = _time.monotonic() + interval
                _progress_stop.wait(interval)
                if not _progress_stop.is_set():
                    wait_late_seconds = max(0.0, _time.monotonic() - wake_deadline)
                    with _progress_state_lock:
                        _progress_wait_late_seconds += wait_late_seconds
                        _progress_max_wait_late_seconds = max(
                            _progress_max_wait_late_seconds, wait_late_seconds
                        )
            with _progress_state_lock:
                _progress_active_seconds += _time.monotonic() - active_start
        with _progress_state_lock:
            _progress_worker_thread_cpu_seconds += (
                _time.thread_time() - worker_cpu_start
            )

    _progress_thread = _threading.Thread(
        target=_mpi_progress_loop,
        name="mpisppy-mpi-progress",
        daemon=True,
    )
    _progress_thread.start()

    def _activate_mpi_progress():
        global _progress_active_count
        with _progress_state_lock:
            _progress_active_count += 1
            _progress_active.set()

    def _deactivate_mpi_progress():
        global _progress_active_count
        with _progress_state_lock:
            if _progress_active_count <= 0:
                raise RuntimeError("unbalanced mpi-sppy MPI progress deactivation")
            _progress_active_count -= 1
            if _progress_active_count == 0:
                _progress_active.clear()

    def _stop_mpi_progress_thread():
        _progress_stop.set()
        _progress_active.set()
        _progress_thread.join(timeout=5)
        if _progress_diagnostics:
            with _progress_state_lock:
                print(
                    "mpi-sppy MPI progress: "
                    f"rank={COMM_WORLD.Get_rank()}, "
                    f"compute_regions={_progress_compute_regions}, "
                    f"compute_seconds={_progress_compute_seconds:.3f}, "
                    "compute_process_cpu_seconds="
                    f"{_progress_compute_process_cpu_seconds:.3f}, "
                    "compute_thread_cpu_seconds="
                    f"{_progress_compute_thread_cpu_seconds:.3f}",
                    flush=True,
                )
                print(
                    "mpi-sppy MPI progress: "
                    f"rank={COMM_WORLD.Get_rank()}, probes={_progress_probes}, "
                    f"probe_seconds={_progress_probe_seconds:.3f}, "
                    f"max_probe_ms={1000 * _progress_max_probe_seconds:.3f}, "
                    f"wakeups={_progress_wakeups}, "
                    f"active_seconds={_progress_active_seconds:.3f}, "
                    "worker_thread_cpu_seconds="
                    f"{_progress_worker_thread_cpu_seconds:.3f}, "
                    f"worker_native_id={_progress_worker_native_id}, "
                    f"wait_late_seconds={_progress_wait_late_seconds:.3f}, "
                    "max_wait_late_ms="
                    f"{1000 * _progress_max_wait_late_seconds:.3f}",
                    flush=True,
                )

    # mpi4py registers its MPI finalizer while mpi4py.MPI is imported.  atexit
    # runs handlers in reverse registration order, so this stops the thread
    # before mpi4py finalizes MPI.
    _atexit.register(_stop_mpi_progress_thread)


@_contextmanager
def progress_active():
    """Request MPI progress while executing a long non-MPI compute region.

    This is a no-op unless ``MPISPPY_MPI_PROGRESS_THREAD=1`` was selected.
    Nested uses are reference counted so an inner compute region cannot stop
    progress requested by its caller.
    """
    global _progress_compute_regions, _progress_compute_seconds
    global _progress_compute_process_cpu_seconds
    global _progress_compute_thread_cpu_seconds
    if haveMPI and _enable_progress_thread:
        compute_start = _time.monotonic()
        compute_process_cpu_start = _time.process_time()
        compute_cpu_start = _time.thread_time()
        _activate_mpi_progress()
        try:
            yield
        finally:
            _deactivate_mpi_progress()
            with _progress_state_lock:
                _progress_compute_regions += 1
                _progress_compute_seconds += _time.monotonic() - compute_start
                _progress_compute_process_cpu_seconds += (
                    _time.process_time() - compute_process_cpu_start
                )
                _progress_compute_thread_cpu_seconds += (
                    _time.thread_time() - compute_cpu_start
                )
    else:
        yield
