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
_progress_rank_context = None

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
        _progress_interval_us = float(
            _os.environ.get("MPISPPY_MPI_PROGRESS_INTERVAL_US", "500")
        )
        _progress_max_interval = _progress_interval_us * 1e-6
    except ValueError as _exc:
        raise RuntimeError(
            "MPISPPY_MPI_PROGRESS_INTERVAL_US must be a positive number"
        ) from _exc
    if not _math.isfinite(_progress_max_interval) or _progress_max_interval <= 0:
        raise RuntimeError("MPISPPY_MPI_PROGRESS_INTERVAL_US must be positive")
    if COMM_WORLD.Get_rank() == 0:
        _interval_source = (
            "environment" if "MPISPPY_MPI_PROGRESS_INTERVAL_US" in _os.environ
            else "default"
        )
        print(
            "mpi-sppy MPI progress: "
            f"MPISPPY_MPI_PROGRESS_INTERVAL_US={_progress_interval_us:g} "
            f"({_interval_source})",
            flush=True,
        )

    _progress_stop = _threading.Event()
    _progress_active = _threading.Event()
    _progress_state_lock = _threading.Lock()
    _progress_caller_state = _threading.local()
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
    _progress_compute_start = 0.0
    _progress_compute_process_cpu_start = 0.0
    _progress_mpi_call_seconds = 0.0
    _progress_gil_probe_seconds = 0.0
    _progress_gil_wait_seconds = 0.0
    _progress_wait_syscall_late_seconds = 0.0
    _progress_max_gil_probe_seconds = 0.0
    _progress_max_gil_wait_seconds = 0.0

    _native_worker_enabled = (
        _os.environ.get("MPISPPY_MPI_PROGRESS_NATIVE_WORKER") == "1"
    )
    if _native_worker_enabled:
        _worker_module_path = _os.environ.get(
            "MPISPPY_MPI_PROGRESS_WORKER_MODULE"
        )
        if _worker_module_path:
            import importlib.util as _import_util
            import sys as _sys

            _worker_spec = _import_util.spec_from_file_location(
                "mpisppy._mpi_progress_worker", _worker_module_path
            )
            if _worker_spec is None or _worker_spec.loader is None:
                raise RuntimeError(
                    "Could not load native MPI progress worker at "
                    f"{_worker_module_path}"
                )
            _mpi_progress_worker = _import_util.module_from_spec(_worker_spec)
            _sys.modules[_worker_spec.name] = _mpi_progress_worker
            try:
                _worker_spec.loader.exec_module(_mpi_progress_worker)
            except Exception:
                del _sys.modules[_worker_spec.name]
                raise
        else:
            try:
                from mpisppy import _mpi_progress_worker
            except ImportError as _exc:
                raise RuntimeError(
                    "MPISPPY_MPI_PROGRESS_NATIVE_WORKER=1 requires the optional "
                    "mpisppy._mpi_progress_worker extension; build it with "
                    "MPISPPY_BUILD_MPI_PROGRESS_WORKER=1 and the MPI compiler "
                    "used by mpi4py"
                ) from _exc
        _mpi_progress_worker.configure(
            COMM_WORLD, _progress_max_interval, _progress_diagnostics
        )

    _progress_native_diagnostics = (
        _progress_diagnostics
        and _os.environ.get("MPISPPY_MPI_PROGRESS_NATIVE_DIAGNOSTICS") == "1"
    )
    if _native_worker_enabled and _progress_native_diagnostics:
        raise RuntimeError(
            "MPISPPY_MPI_PROGRESS_NATIVE_DIAGNOSTICS times the Python worker; "
            "disable it when MPISPPY_MPI_PROGRESS_NATIVE_WORKER=1"
        )
    if _progress_native_diagnostics:
        _native_module_path = _os.environ.get(
            "MPISPPY_MPI_PROGRESS_NATIVE_MODULE"
        )
        if _native_module_path:
            import importlib.util as _import_util
            import sys as _sys

            _native_spec = _import_util.spec_from_file_location(
                "mpisppy._mpi_progress_native", _native_module_path
            )
            if _native_spec is None or _native_spec.loader is None:
                raise RuntimeError(
                    "Could not load native MPI progress helper at "
                    f"{_native_module_path}"
                )
            _mpi_progress_native = _import_util.module_from_spec(_native_spec)
            _sys.modules[_native_spec.name] = _mpi_progress_native
            try:
                _native_spec.loader.exec_module(_mpi_progress_native)
            except Exception:
                del _sys.modules[_native_spec.name]
                raise
        else:
            try:
                from mpisppy import _mpi_progress_native
            except ImportError as _exc:
                raise RuntimeError(
                    "Native MPI progress diagnostics requested but the optional "
                    "module is unavailable; set "
                    "MPISPPY_MPI_PROGRESS_NATIVE_MODULE to its built path"
                ) from _exc
        _progress_stop_read_fd, _progress_stop_write_fd = _os.pipe()

    def _mpi_progress_loop():
        global _progress_probes, _progress_probe_seconds, _progress_max_probe_seconds
        global _progress_wakeups, _progress_active_seconds
        global _progress_worker_thread_cpu_seconds, _progress_worker_native_id
        global _progress_wait_late_seconds, _progress_max_wait_late_seconds
        global _progress_mpi_call_seconds, _progress_gil_probe_seconds
        global _progress_gil_wait_seconds
        global _progress_wait_syscall_late_seconds
        global _progress_max_gil_probe_seconds, _progress_max_gil_wait_seconds
        probes = wakeups = 0
        probe_seconds = max_probe_seconds = 0.0
        active_seconds = wait_late_seconds = max_wait_late_seconds = 0.0
        mpi_call_seconds = gil_probe_seconds = gil_wait_seconds = 0.0
        wait_syscall_late_seconds = 0.0
        max_gil_probe_seconds = max_gil_wait_seconds = 0.0
        if _progress_diagnostics:
            _progress_worker_native_id = _threading.get_native_id()
            print(
                "mpi-sppy MPI progress: "
                f"rank={COMM_WORLD.Get_rank()}, "
                f"worker_native_id={_progress_worker_native_id}, "
                f"native_diagnostics={int(_progress_native_diagnostics)}",
                flush=True,
            )
            worker_cpu_start = _time.thread_time()
        while not _progress_stop.is_set():
            _progress_active.wait()
            if _progress_stop.is_set():
                break

            active_start = _time.monotonic()  # Needed for the polling backoff.
            if _progress_diagnostics:
                wakeups += 1
            while _progress_active.is_set() and not _progress_stop.is_set():
                if _progress_native_diagnostics:
                    probe_start = _time.monotonic()
                    _, mpi_duration, gil_duration = (
                        _mpi_progress_native.timed_iprobe(
                            COMM_WORLD, ANY_SOURCE, ANY_TAG
                        )
                    )
                    probe_duration = _time.monotonic() - probe_start
                    mpi_call_seconds += mpi_duration
                    gil_probe_seconds += gil_duration
                    max_gil_probe_seconds = max(
                        max_gil_probe_seconds, gil_duration
                    )
                elif _progress_diagnostics:
                    probe_start = _time.monotonic()
                    COMM_WORLD.Iprobe(ANY_SOURCE, ANY_TAG)
                    probe_duration = _time.monotonic() - probe_start
                else:
                    COMM_WORLD.Iprobe(ANY_SOURCE, ANY_TAG)
                if _progress_diagnostics:
                    probes += 1
                    probe_seconds += probe_duration
                    max_probe_seconds = max(max_probe_seconds, probe_duration)

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
                if _progress_native_diagnostics:
                    stopped, waited, gil_duration = (
                        _mpi_progress_native.timed_wait(
                            _progress_stop_read_fd, interval
                        )
                    )
                    if not stopped and not _progress_stop.is_set():
                        gil_wait_seconds += gil_duration
                        max_gil_wait_seconds = max(
                            max_gil_wait_seconds, gil_duration
                        )
                        wait_syscall_late_seconds += max(0.0, waited - interval)
                        lateness = max(0.0, waited + gil_duration - interval)
                    else:
                        continue
                elif _progress_diagnostics:
                    wake_deadline = _time.monotonic() + interval
                    _progress_stop.wait(interval)
                    if _progress_stop.is_set():
                        continue
                    lateness = max(0.0, _time.monotonic() - wake_deadline)
                else:
                    _progress_stop.wait(interval)
                    continue
                wait_late_seconds += lateness
                max_wait_late_seconds = max(max_wait_late_seconds, lateness)
            if _progress_diagnostics:
                active_seconds += _time.monotonic() - active_start
        if _progress_diagnostics:
            # Only this worker writes these fields; shutdown reads after join.
            _progress_probes = probes
            _progress_probe_seconds = probe_seconds
            _progress_max_probe_seconds = max_probe_seconds
            _progress_wakeups = wakeups
            _progress_active_seconds = active_seconds
            _progress_wait_late_seconds = wait_late_seconds
            _progress_max_wait_late_seconds = max_wait_late_seconds
            _progress_mpi_call_seconds = mpi_call_seconds
            _progress_gil_probe_seconds = gil_probe_seconds
            _progress_gil_wait_seconds = gil_wait_seconds
            _progress_wait_syscall_late_seconds = wait_syscall_late_seconds
            _progress_max_gil_probe_seconds = max_gil_probe_seconds
            _progress_max_gil_wait_seconds = max_gil_wait_seconds
            _progress_worker_thread_cpu_seconds = (
                _time.thread_time() - worker_cpu_start
            )

    def _native_progress_loop():
        global _progress_worker_native_id
        if _progress_diagnostics:
            _progress_worker_native_id = _threading.get_native_id()
            print(
                "mpi-sppy MPI progress: "
                f"rank={COMM_WORLD.Get_rank()}, "
                f"worker_native_id={_progress_worker_native_id}, "
                "native_diagnostics=0, native_worker=1",
                flush=True,
            )
        _mpi_progress_worker.run()

    _progress_thread = _threading.Thread(
        target=(_native_progress_loop if _native_worker_enabled
                else _mpi_progress_loop),
        name="mpisppy-mpi-progress",
        daemon=True,
    )
    _progress_thread.start()

    def _activate_mpi_progress():
        global _progress_active_count, _progress_compute_start
        global _progress_compute_process_cpu_start
        with _progress_state_lock:
            if _progress_diagnostics and _progress_active_count == 0:
                _progress_compute_start = _time.monotonic()
                _progress_compute_process_cpu_start = _time.process_time()
            if _progress_active_count == 0:
                if _native_worker_enabled:
                    _mpi_progress_worker.set_active(True)
                else:
                    _progress_active.set()
            _progress_active_count += 1

    def _deactivate_mpi_progress():
        global _progress_active_count, _progress_compute_regions
        global _progress_compute_seconds, _progress_compute_process_cpu_seconds
        with _progress_state_lock:
            if _progress_active_count <= 0:
                raise RuntimeError("unbalanced mpi-sppy MPI progress deactivation")
            _progress_active_count -= 1
            if _progress_active_count == 0:
                if _progress_diagnostics:
                    _progress_compute_regions += 1
                    _progress_compute_seconds += (
                        _time.monotonic() - _progress_compute_start
                    )
                    _progress_compute_process_cpu_seconds += (
                        _time.process_time() - _progress_compute_process_cpu_start
                    )
                if _native_worker_enabled:
                    _mpi_progress_worker.set_active(False)
                else:
                    _progress_active.clear()

    def _stop_mpi_progress_thread():
        global _progress_probes, _progress_probe_seconds, _progress_max_probe_seconds
        global _progress_wakeups, _progress_active_seconds
        global _progress_worker_thread_cpu_seconds, _progress_worker_native_id
        global _progress_wait_late_seconds, _progress_max_wait_late_seconds
        global _progress_mpi_call_seconds, _progress_wait_syscall_late_seconds
        if _native_worker_enabled:
            _mpi_progress_worker.stop()
        else:
            _progress_stop.set()
            _progress_active.set()
            if _progress_native_diagnostics:
                _os.write(_progress_stop_write_fd, b"\0")
        _progress_thread.join(timeout=None if _native_worker_enabled else 5)
        worker_stopped = not _progress_thread.is_alive()
        if _native_worker_enabled and worker_stopped:
            _native_counts = _mpi_progress_worker.snapshot()
            _progress_probes = _native_counts["probes"]
            _progress_wakeups = _native_counts["wakeups"]
            _progress_active_seconds = _native_counts["active_seconds"]
            _progress_probe_seconds = _native_counts["mpi_call_seconds"]
            _progress_mpi_call_seconds = _native_counts["mpi_call_seconds"]
            _progress_max_probe_seconds = _native_counts["max_probe_seconds"]
            _progress_wait_late_seconds = _native_counts["wait_return_late_seconds"]
            _progress_wait_syscall_late_seconds = _progress_wait_late_seconds
            _progress_max_wait_late_seconds = (
                _native_counts["max_wait_return_late_seconds"]
            )
            _progress_worker_thread_cpu_seconds = (
                _native_counts["worker_thread_cpu_seconds"]
            )
            _progress_worker_native_id = _progress_thread.native_id
        if _progress_native_diagnostics and not _progress_thread.is_alive():
            _os.close(_progress_stop_read_fd)
            _os.close(_progress_stop_write_fd)
        if _progress_diagnostics:
            with _progress_state_lock:
                if not worker_stopped:
                    print(
                        "mpi-sppy MPI progress: worker did not stop; "
                        "worker counters are incomplete",
                        flush=True,
                    )
                print(
                    "mpi-sppy MPI progress: "
                    f"rank={COMM_WORLD.Get_rank()}, "
                    f"{_format_progress_rank_context()}, "
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
                    f"{_format_progress_rank_context()}, "
                    f"iprobe_round_trip_seconds={_progress_probe_seconds:.3f}, "
                    f"max_probe_ms={1000 * _progress_max_probe_seconds:.3f}, "
                    f"wakeups={_progress_wakeups}, "
                    f"active_seconds={_progress_active_seconds:.3f}, "
                    "worker_thread_cpu_seconds="
                    f"{_progress_worker_thread_cpu_seconds:.3f}, "
                    f"worker_native_id={_progress_worker_native_id}, "
                    f"wait_return_late_seconds="
                    f"{_progress_wait_late_seconds:.3f}, "
                    "max_wait_return_late_ms="
                    f"{1000 * _progress_max_wait_late_seconds:.3f}",
                    flush=True,
                )
                if _progress_native_diagnostics:
                    print(
                        "mpi-sppy MPI progress: "
                        f"rank={COMM_WORLD.Get_rank()}, "
                        f"{_format_progress_rank_context()}, "
                        f"mpi_call_seconds={_progress_mpi_call_seconds:.3f}, "
                        f"gil_probe_reacquire_seconds="
                        f"{_progress_gil_probe_seconds:.3f}, "
                        f"max_gil_probe_reacquire_ms="
                        f"{1000 * _progress_max_gil_probe_seconds:.3f}, "
                        f"gil_wait_reacquire_seconds="
                        f"{_progress_gil_wait_seconds:.3f}, "
                        f"max_gil_wait_reacquire_ms="
                        f"{1000 * _progress_max_gil_wait_seconds:.3f}, "
                        f"wait_syscall_late_seconds="
                        f"{_progress_wait_syscall_late_seconds:.3f}",
                        flush=True,
                    )
                if _native_worker_enabled and worker_stopped:
                    print(
                        "mpi-sppy MPI progress: "
                        f"rank={COMM_WORLD.Get_rank()}, "
                        f"{_format_progress_rank_context()}, "
                        "native_worker=1, "
                        f"mpi_call_seconds={_progress_mpi_call_seconds:.3f}, "
                        f"wait_syscall_late_seconds="
                        f"{_progress_wait_syscall_late_seconds:.3f}, "
                        "gil_probe_reacquire_seconds=0, "
                        "max_gil_probe_reacquire_ms=0, "
                        "gil_wait_reacquire_seconds=0, "
                        "max_gil_wait_reacquire_ms=0, "
                        f"mpi_error={_native_counts['mpi_error']}, "
                        f"wait_error={_native_counts['wait_error']}",
                        flush=True,
                    )

    # mpi4py registers its MPI finalizer while mpi4py.MPI is imported.  atexit
    # runs handlers in reverse registration order, so this stops the thread
    # before mpi4py finalizes MPI.
    _atexit.register(_stop_mpi_progress_thread)


def _format_progress_rank_context():
    """Format cylinder identity for the opt-in MPI progress diagnostics."""
    context = _progress_rank_context
    if context is None:
        return "role=unknown, spcomm_type=unknown, optimizer_type=unknown"
    return (
        f"role={context['role']}, "
        f"spcomm_type={context['spcomm_type']}, "
        f"optimizer_type={context['optimizer_type']}, "
        f"strata_rank={context['strata_rank']}, "
        f"cylinder_rank={context['cylinder_rank']}"
    )


def _set_progress_rank_context(role, spcomm_type, optimizer_type,
                               strata_rank, cylinder_rank):
    """Attach the hub/spoke identity once the wheel assigns this rank."""
    global _progress_rank_context
    if not (haveMPI and _enable_progress_thread and _progress_diagnostics):
        return
    with _progress_state_lock:
        _progress_rank_context = {
            "role": role,
            "spcomm_type": spcomm_type,
            "optimizer_type": optimizer_type,
            "strata_rank": strata_rank,
            "cylinder_rank": cylinder_rank,
        }
    print(
        "mpi-sppy MPI progress context: "
        f"rank={COMM_WORLD.Get_rank()}, {_format_progress_rank_context()}",
        flush=True,
    )


@_contextmanager
def progress_active():
    """Request MPI progress while executing a long non-MPI compute region.

    This is a no-op unless ``MPISPPY_MPI_PROGRESS_THREAD=1`` was selected.
    Nested uses are reference counted so an inner compute region cannot stop
    progress requested by its caller. Diagnostic wall and process CPU totals
    measure the union of active regions; caller CPU is counted once per thread.
    """
    global _progress_compute_thread_cpu_seconds
    if haveMPI and _enable_progress_thread:
        if _progress_diagnostics:
            depth = getattr(_progress_caller_state, "depth", 0)
            if depth == 0:
                compute_cpu_start = _time.thread_time()
            _progress_caller_state.depth = depth + 1
        _activate_mpi_progress()
        try:
            yield
        finally:
            _deactivate_mpi_progress()
            if _progress_diagnostics:
                _progress_caller_state.depth = depth
                if depth == 0:
                    with _progress_state_lock:
                        _progress_compute_thread_cpu_seconds += (
                            _time.thread_time() - compute_cpu_start
                        )
    else:
        yield
