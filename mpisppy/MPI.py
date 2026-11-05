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
        # Set the thread level before mpi4py initializes MPI. Let mpi4py
        # finalize MPI at exit when it owns initialization. Open MPI's OSC
        # components otherwise make no RMA progress during solver work.
        import mpi4py as _mpi4py

        _mpi4py.rc.initialize = True
        _mpi4py.rc.threads = True
        _mpi4py.rc.thread_level = "multiple"

        from mpi4py import MPI as _mpi

        if not _mpi.Is_initialized():
            raise RuntimeError(
                "MPISPPY_MPI_PROGRESS_THREAD=1 requires initialized MPI; "
                "import mpi-sppy before disabling mpi4py's auto-initialization"
            )
        _provided_thread_level = _mpi.Query_thread()
        # Prefer the MPI runtime's own async progress when it is enabled.
        if _os.environ.get("MVAPICH_ASYNC_PROGRESS") == "1" and (
            _mpi.get_vendor()[0].upper().startswith("MVAPICH")
            or _mpi.Get_library_version().lstrip().upper().startswith("MVAPICH")
        ):
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

except ImportError:
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
    # Keep Open MPI making progress while the main application thread is in a
    # solver or other non-MPI Python code. Iprobe is non-destructive: it does
    # not receive or reorder application messages. The worker is activated only
    # around long compute regions: polling continuously would contend with the
    # MPI_THREAD_MULTIPLE locks during normal MPI-heavy execution.
    import atexit as _atexit
    import math as _math
    import threading as _threading
    import time as _time
    from contextlib import contextmanager as _contextmanager

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
            "environment"
            if "MPISPPY_MPI_PROGRESS_INTERVAL_US" in _os.environ
            else "default"
        )
        print(
            "mpi-sppy MPI progress: "
            f"MPISPPY_MPI_PROGRESS_INTERVAL_US={_progress_interval_us:g} "
            f"({_interval_source})",
            flush=True,
        )

    _progress_state_changed = _threading.Condition()
    _progress_active_count = 0
    _progress_stopping = False
    _progress_polling = False
    import importlib as _importlib
    import importlib.metadata as _metadata
    import importlib.util as _import_util
    import json as _json
    import sys as _sys
    from pathlib import Path as _Path

    from mpisppy.utils.mpi_progress_compat import (
        check_native_mpi_compatibility as _check_native_mpi_compatibility,
    )

    def _is_editable_install():
        try:
            direct_url = _metadata.distribution("mpi-sppy").read_text("direct_url.json")
        except _metadata.PackageNotFoundError:
            return False
        if not direct_url:
            return False
        try:
            origin = _json.loads(direct_url)
        except ValueError:
            return False
        if not isinstance(origin, dict):
            return False
        dir_info = origin.get("dir_info")
        return isinstance(dir_info, dict) and dir_info.get("editable") is True

    _package_file = _Path(__file__).resolve()
    _source_root = _package_file.parent.parent
    _running_from_checkout = (_source_root / "setup.py").is_file() and (
        _source_root / "pyproject.toml"
    ).is_file()
    _extension_package = (
        None if (_is_editable_install() or _running_from_checkout) else "mpisppy"
    )
    # Editable wheels install their binaries in the active environment. A
    # matching source file identifies this checkout without assuming an
    # environment directory name or setuptools' editable link-tree location.
    try:
        import mpisppy_editable_native as _editable_native
    except ImportError:
        pass
    else:
        _editable_source = getattr(_editable_native, "SOURCE_ROOT", None)
        _same_source = False
        if isinstance(_editable_source, str):
            _expected_source_file = _Path(_editable_source) / "mpisppy" / "MPI.py"
            try:
                _same_source = _os.path.samefile(_package_file, _expected_source_file)
            except OSError:
                pass
        if _same_source and getattr(_editable_native, "BUILD_PREFIX", None) == str(
            _Path(_sys.prefix).resolve()
        ):
            _extension_package = "mpisppy_editable_native"

    def _progress_extension_name(name):
        if _extension_package is None:
            return None
        full_name = f"{_extension_package}.{name}"
        return full_name if _import_util.find_spec(full_name) is not None else None

    _worker_module_name = _progress_extension_name("_mpi_progress_worker")
    _native_worker_enabled = _worker_module_name is not None
    if _native_worker_enabled:
        try:
            _mpi_progress_worker = _importlib.import_module(_worker_module_name)
        except ImportError as _exc:
            raise RuntimeError(
                "Could not load the installed MPI progress worker; rebuild "
                "mpi-sppy with the MPI compiler used by mpi4py"
            ) from _exc
        _check_native_mpi_compatibility(_mpi_progress_worker, _mpi)
        _mpi_progress_worker.configure(COMM_WORLD, _progress_max_interval)
    if COMM_WORLD.Get_rank() == 0:
        print(
            "mpi-sppy MPI progress: "
            f"worker={'native' if _native_worker_enabled else 'python'}",
            flush=True,
        )

    def _mpi_progress_loop():
        global _progress_polling
        while True:
            with _progress_state_changed:
                while _progress_active_count == 0 and not _progress_stopping:
                    _progress_state_changed.wait()
                if _progress_stopping:
                    return
                active_start = _time.monotonic()

            while True:
                with _progress_state_changed:
                    if _progress_active_count == 0 or _progress_stopping:
                        break
                    # Mark the call in flight before releasing the lock so
                    # deactivation cannot miss it.
                    _progress_polling = True
                try:
                    COMM_WORLD.Iprobe(_mpi.ANY_SOURCE, _mpi.ANY_TAG)
                finally:
                    with _progress_state_changed:
                        _progress_polling = False
                        _progress_state_changed.notify_all()

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
                with _progress_state_changed:
                    if _progress_active_count > 0 and not _progress_stopping:
                        _progress_state_changed.wait(timeout=interval)

    def _progress_thread_main():
        try:
            if _native_worker_enabled:
                _mpi_progress_worker.run()
            else:
                _mpi_progress_loop()
        except BaseException:
            # A dead worker cannot make progress for a rank inside a solver.
            try:
                import traceback as _traceback

                _traceback.print_exc()
            finally:
                try:
                    COMM_WORLD.Abort(1)
                finally:
                    _os._exit(1)

    _progress_thread = _threading.Thread(
        target=_progress_thread_main,
        name="mpisppy-mpi-progress",
        daemon=True,
    )
    _progress_thread.start()

    def _activate_mpi_progress():
        global _progress_active_count
        with _progress_state_changed:
            if _progress_active_count == 0:
                if _native_worker_enabled:
                    _mpi_progress_worker.set_active(True)
            _progress_active_count += 1
            if not _native_worker_enabled and _progress_active_count == 1:
                _progress_state_changed.notify_all()

    def _deactivate_mpi_progress():
        global _progress_active_count
        with _progress_state_changed:
            if _progress_active_count <= 0:
                raise RuntimeError("unbalanced mpi-sppy MPI progress deactivation")
            _progress_active_count -= 1
            if _progress_active_count == 0:
                if _native_worker_enabled:
                    _mpi_progress_worker.set_active(False)
                else:
                    _progress_state_changed.notify_all()
                    # The context must not return while its final Iprobe is
                    # still using MPI (for example, before MPI.Finalize()).
                    stopped_polling = _progress_state_changed.wait_for(
                        lambda: not _progress_polling or _progress_active_count > 0,
                        timeout=30,
                    )
                    if not stopped_polling:
                        _os.write(2, b"mpi-sppy MPI progress: deactivation timed out\n")
                        _os._exit(1)

    def _stop_mpi_progress_thread():
        global _progress_stopping
        if _native_worker_enabled:
            _mpi_progress_worker.stop()
        else:
            with _progress_state_changed:
                _progress_stopping = True
                _progress_state_changed.notify_all()
        _progress_thread.join(
            timeout=(
                getattr(_mpi_progress_worker, "STOP_TIMEOUT_SECONDS", 30)
                if _native_worker_enabled
                else 5
            )
        )
        if _progress_thread.is_alive():
            try:
                _os.write(
                    2,
                    b"mpi-sppy MPI progress: worker did not stop before timeout; "
                    b"exiting rank before MPI finalization\n",
                )
            finally:
                _os._exit(1)

    # Python atexit handlers run before mpi4py's C-level exit callback, so
    # this stops the thread before mpi4py finalizes MPI.
    _atexit.register(_stop_mpi_progress_thread)

    @_contextmanager
    def progress_active():
        """Request MPI progress while executing a long non-MPI compute region.

        Nested uses are reference counted so an inner compute region cannot
        stop progress requested by its caller.
        """
        _activate_mpi_progress()
        try:
            yield
        finally:
            _deactivate_mpi_progress()
else:
    from contextlib import nullcontext as progress_active  # noqa: F401
