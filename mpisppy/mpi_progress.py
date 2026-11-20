###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

"""Manage opt-in MPI progress during long compute regions.

Set MPISPPY_MPI_PROGRESS_THREAD=1 to start the C worker on runtimes that need
target-side MPI calls for passive-target RMA progress. Newer MVAPICH releases
(after 2.3.7) can provide their own async progress with
MVAPICH_ASYNC_PROGRESS=1; when enabled, mpi-sppy uses that runtime progress
instead of starting its worker.
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
import traceback
from contextlib import contextmanager
from pathlib import Path

from mpisppy.utils.mpi_progress_compat import check_native_mpi_compatibility

# The native worker waits at most 50 us between probes for the first 10 ms,
# at most 250 us until 100 ms, then uses the configured maximum interval.
DEFAULT_POLL_INTERVAL_US = 500
# Maximum value allowed for polling interval
MAX_POLL_INTERVAL_SECONDS = 600

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


def _load_native_worker(mpi, comm, max_interval, package_file):
    """Load, verify, and configure the native MPI progress extension."""
    module_name = _native_module_name(package_file)
    if module_name is None:
        raise RuntimeError(
            "MPISPPY_MPI_PROGRESS_THREAD=1 requires the native MPI progress "
            "extension; install mpi-sppy from source with mpi4py and a matching "
            "MPI compiler"
        )
    try:
        worker = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            "Could not load the installed MPI progress worker; rebuild "
            "mpi-sppy with the MPI compiler used by mpi4py"
        ) from exc
    check_native_mpi_compatibility(worker, mpi)
    worker.configure(comm, max_interval)
    return worker


class ProgressController:
    """Reference count progress requests and own the C worker's thread."""

    def __init__(self, worker, comm):
        self.worker = worker
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
            self.worker.run()
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
                self.worker.set_active(True)
            self.active_count += 1

    def deactivate(self):
        with self.lock:
            if self.active_count <= 0:
                raise RuntimeError("unbalanced mpi-sppy MPI progress deactivation")
            self.active_count -= 1
            if self.active_count == 0:
                self.worker.set_active(False)

    def stop(self):
        self.worker.stop()
        self.thread.join(timeout=self.worker.STOP_TIMEOUT_SECONDS)
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
    """Start the native progress worker after MPI initialization."""
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
    if max_interval > MAX_POLL_INTERVAL_SECONDS:
        raise RuntimeError(
            "MPISPPY_MPI_PROGRESS_INTERVAL_US must not exceed 600 seconds"
        )
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

    worker = _load_native_worker(mpi, comm, max_interval, package_file)
    if comm.Get_rank() == 0:
        compiled_mpi = " ".join(worker.BUILD_MPI_LIBRARY_VERSION.split())
        print(
            f"mpi-sppy MPI progress: enabled (compiled MPI: {compiled_mpi})",
            flush=True,
        )
    return ProgressController(worker, comm)
