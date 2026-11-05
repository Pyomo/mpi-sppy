###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

"""Manage opt-in MPI progress during long compute regions.

Set MPISPPY_MPI_PROGRESS_THREAD=1 to start a worker on runtimes that need
target-side MPI calls for passive-target RMA progress. Some runtimes provide
their own async progress. Newer MVAPICH releases (after 2.3.7) can do so
with MVAPICH_ASYNC_PROGRESS=1, in which case mpi-sppy does not start a worker.
MPISPPY_MPI_PROGRESS_INTERVAL_US sets the maximum requested polling delay;
its default is 500 microseconds.
"""

import atexit
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from mpisppy.utils.mpi_progress_compat import check_native_mpi_compatibility

# Both backends wait at most 50 us between probes for the first 10 ms, at
# most 250 us until 100 ms, then use the configured maximum interval.
DEFAULT_POLL_INTERVAL_US = 500
FAST_PHASE_END_SECONDS = 0.010
MID_PHASE_END_SECONDS = 0.100
FAST_POLL_INTERVAL_SECONDS = 50e-6
MID_POLL_INTERVAL_SECONDS = 250e-6

# Deactivation drains an in-flight MPI call. Native shutdown uses the same
# 30 s limit; the Python worker has a shorter 5 s shutdown limit.
DEACTIVATION_TIMEOUT_SECONDS = 30
PYTHON_STOP_TIMEOUT_SECONDS = 5
NATIVE_STOP_TIMEOUT_SECONDS = 30


# Editable installs put the native worker in the active environment rather
# than the source package. Detect them before selecting where to load it.
def _is_editable_install():
    try:
        direct_url = importlib.metadata.distribution("mpi-sppy").read_text(
            "direct_url.json"
        )
    except importlib.metadata.PackageNotFoundError:
        return False
    if not direct_url:
        return False
    try:
        origin = json.loads(direct_url)
    except ValueError:
        return False
    if not isinstance(origin, dict):
        return False
    dir_info = origin.get("dir_info")
    return isinstance(dir_info, dict) and dir_info.get("editable") is True


def _native_module_name(package_file):
    package_file = Path(package_file).resolve()
    source_root = package_file.parent.parent
    running_from_checkout = (source_root / "setup.py").is_file() and (
        source_root / "pyproject.toml"
    ).is_file()
    package = None if (_is_editable_install() or running_from_checkout) else "mpisppy"

    # Editable wheels keep their binaries in the active environment. Match
    # both the checkout and environment before loading one of those binaries.
    try:
        import mpisppy_editable_native as editable_native
    except ImportError:
        pass
    else:
        editable_source = getattr(editable_native, "SOURCE_ROOT", None)
        same_source = False
        if isinstance(editable_source, str):
            expected_file = Path(editable_source) / "mpisppy" / "MPI.py"
            try:
                same_source = os.path.samefile(package_file, expected_file)
            except OSError:
                pass
        if same_source and getattr(editable_native, "BUILD_PREFIX", None) == str(
            Path(sys.prefix).resolve()
        ):
            package = "mpisppy_editable_native"

    if package is None:
        return None
    name = f"{package}._mpi_progress_worker"
    return name if importlib.util.find_spec(name) is not None else None


class _PythonBackend:
    """Poll MPI through mpi4py while a solver computes.

    Selected when progress is enabled and no native C worker is installed.
    Its Iprobe calls help advance passive-target RMA on runtimes that need
    target-side MPI calls, but require the solver to release the GIL.
    The polling delay follows MPISPPY_MPI_PROGRESS_INTERVAL_US.
    """

    kind = "python"
    stop_timeout = PYTHON_STOP_TIMEOUT_SECONDS

    def __init__(self, mpi, comm, max_interval):
        self.mpi = mpi
        self.comm = comm
        self.max_interval = max_interval
        self.changed = threading.Condition()
        self.active = False
        self.polling = False
        self.stopping = False

    def run(self):
        while True:
            with self.changed:
                while not self.active and not self.stopping:
                    self.changed.wait()
                if self.stopping:
                    return
                active_start = time.monotonic()

            while True:
                with self.changed:
                    if not self.active or self.stopping:
                        break
                    # Mark the call in flight before releasing the lock so
                    # deactivation cannot miss it.
                    self.polling = True
                try:
                    self.comm.Iprobe(self.mpi.ANY_SOURCE, self.mpi.ANY_TAG)
                finally:
                    with self.changed:
                        self.polling = False
                        self.changed.notify_all()

                # Probe quickly at the start of a solve, then back off to
                # bound contention with application MPI calls.
                elapsed = time.monotonic() - active_start
                if elapsed < FAST_PHASE_END_SECONDS:
                    interval = min(FAST_POLL_INTERVAL_SECONDS, self.max_interval)
                elif elapsed < MID_PHASE_END_SECONDS:
                    interval = min(MID_POLL_INTERVAL_SECONDS, self.max_interval)
                else:
                    interval = self.max_interval
                with self.changed:
                    if self.active and not self.stopping:
                        self.changed.wait(timeout=interval)

    def activate(self):
        with self.changed:
            self.active = True
            self.changed.notify_all()

    def deactivate(self):
        with self.changed:
            self.active = False
            self.changed.notify_all()
            # Do not leave the context while the last Iprobe is using MPI.
            if not self.changed.wait_for(
                lambda: not self.polling, timeout=DEACTIVATION_TIMEOUT_SECONDS
            ):
                os.write(2, b"mpi-sppy MPI progress: deactivation timed out\n")
                os._exit(1)

    def stop(self):
        with self.changed:
            self.stopping = True
            self.changed.notify_all()


class _NativeBackend:
    """Poll MPI through a C worker without holding Python's GIL.

    Selected when progress is enabled and a compatible native extension is
    installed. Its MPI_Iprobe calls can help advance passive-target RMA even
    while solver code holds the GIL. The polling delay follows
    MPISPPY_MPI_PROGRESS_INTERVAL_US.
    """

    kind = "native"

    def __init__(self, worker, comm, max_interval):
        self.worker = worker
        self.worker.configure(comm, max_interval)
        self.stop_timeout = getattr(
            worker, "STOP_TIMEOUT_SECONDS", NATIVE_STOP_TIMEOUT_SECONDS
        )

    def run(self):
        self.worker.run()

    def activate(self):
        self.worker.set_active(True)

    def deactivate(self):
        self.worker.set_active(False)

    def stop(self):
        self.worker.stop()


def _select_backend(mpi, comm, max_interval, package_file):
    module_name = _native_module_name(package_file)
    if module_name is None:
        return _PythonBackend(mpi, comm, max_interval)
    try:
        worker = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            "Could not load the installed MPI progress worker; rebuild "
            "mpi-sppy with the MPI compiler used by mpi4py"
        ) from exc
    check_native_mpi_compatibility(worker, mpi)
    return _NativeBackend(worker, comm, max_interval)


class ProgressController:
    """Reference count progress requests and own the backend's thread."""

    def __init__(self, backend, comm):
        self.backend = backend
        self.comm = comm
        self.active_count = 0
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run_worker,
            name="mpisppy-mpi-progress",
            daemon=True,
        )
        self.thread.start()
        # Python atexit handlers run before mpi4py's C-level exit callback.
        atexit.register(self.stop)

    def _run_worker(self):
        try:
            self.backend.run()
        except BaseException:
            # A dead worker cannot make progress for a rank inside a solver.
            try:
                traceback.print_exc()
            finally:
                try:
                    self.comm.Abort(1)
                finally:
                    os._exit(1)

    def activate(self):
        with self.lock:
            if self.active_count == 0:
                self.backend.activate()
            self.active_count += 1

    def deactivate(self):
        with self.lock:
            if self.active_count <= 0:
                raise RuntimeError("unbalanced mpi-sppy MPI progress deactivation")
            self.active_count -= 1
            if self.active_count == 0:
                self.backend.deactivate()

    def stop(self):
        self.backend.stop()
        self.thread.join(timeout=self.backend.stop_timeout)
        if self.thread.is_alive():
            try:
                os.write(
                    2,
                    b"mpi-sppy MPI progress: worker did not stop before timeout; "
                    b"exiting rank before MPI finalization\n",
                )
            finally:
                os._exit(1)

    @contextmanager
    def progress_active(self):
        """Request progress while executing a long non-MPI compute region."""
        self.activate()
        try:
            yield
        finally:
            self.deactivate()


def create_controller(mpi, comm, package_file):
    """Select an installed worker and start its thread after MPI initialization."""
    try:
        interval_us = float(
            os.environ.get(
                "MPISPPY_MPI_PROGRESS_INTERVAL_US", str(DEFAULT_POLL_INTERVAL_US)
            )
        )
        max_interval = interval_us * 1e-6
    except ValueError as exc:
        raise RuntimeError(
            "MPISPPY_MPI_PROGRESS_INTERVAL_US must be a positive number"
        ) from exc
    if not math.isfinite(max_interval) or max_interval <= 0:
        raise RuntimeError("MPISPPY_MPI_PROGRESS_INTERVAL_US must be positive")
    if comm.Get_rank() == 0:
        source = (
            "environment"
            if "MPISPPY_MPI_PROGRESS_INTERVAL_US" in os.environ
            else "default"
        )
        print(
            "mpi-sppy MPI progress: "
            f"MPISPPY_MPI_PROGRESS_INTERVAL_US={interval_us:g} ({source})",
            flush=True,
        )

    backend = _select_backend(mpi, comm, max_interval, package_file)
    if comm.Get_rank() == 0:
        print(f"mpi-sppy MPI progress: worker={backend.kind}", flush=True)
    return ProgressController(backend, comm)
