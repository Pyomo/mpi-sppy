###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

"""Checks for progress worker selection and its Python-facing guardrails."""

import importlib.util
import json
import os
import runpy
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from mpisppy import mpi_progress
from mpisppy.utils.mpi_progress_compat import check_native_mpi_compatibility


class TestMPIProgressNativeWorker(unittest.TestCase):
    def test_editable_install_metadata_variants(self):
        cases = (
            (None, False),
            ("", False),
            ("not json", False),
            ("[]", False),
            (json.dumps({"dir_info": []}), False),
            (json.dumps({"dir_info": {"editable": False}}), False),
            (json.dumps({"dir_info": {"editable": True}}), True),
        )
        for direct_url, expected in cases:
            with self.subTest(direct_url=direct_url):
                distribution = mock.Mock()
                distribution.read_text.return_value = direct_url
                with mock.patch.object(
                    mpi_progress.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ):
                    self.assertEqual(mpi_progress._is_editable_install(), expected)

        with mock.patch.object(
            mpi_progress.importlib.metadata,
            "distribution",
            side_effect=mpi_progress.importlib.metadata.PackageNotFoundError,
        ):
            self.assertFalse(mpi_progress._is_editable_install())

    def test_native_module_name_for_editable_install(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir)
            package_file = source_root / "mpisppy" / "MPI.py"
            package_file.parent.mkdir()
            package_file.touch()
            editable_native = ModuleType("mpisppy_editable_native")
            editable_native.SOURCE_ROOT = str(source_root)
            editable_native.BUILD_PREFIX = str(Path(sys.prefix).resolve())
            distribution = mock.Mock()
            distribution.read_text.return_value = json.dumps(
                {"dir_info": {"editable": True}}
            )
            with (
                mock.patch.object(
                    mpi_progress.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
                mock.patch.object(
                    mpi_progress.importlib.util, "find_spec", return_value=object()
                ) as find_spec,
                mock.patch.dict(sys.modules, {editable_native.__name__: editable_native}),
            ):
                module_name = mpi_progress._native_module_name(package_file)

        self.assertEqual(module_name, "mpisppy_editable_native._mpi_progress_worker")
        find_spec.assert_called_once_with(module_name)

    def test_installed_native_module_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_file = Path(temp_dir) / "mpisppy" / "MPI.py"
            package_file.parent.mkdir()
            package_file.touch()
            with (
                mock.patch.object(mpi_progress, "_is_editable_install", return_value=False),
                mock.patch.object(
                    mpi_progress.importlib.util, "find_spec", return_value=object()
                ) as find_spec,
                mock.patch.dict(
                    sys.modules, {"mpisppy_editable_native": None}, clear=False
                ),
            ):
                module_name = mpi_progress._native_module_name(package_file)

        self.assertEqual(module_name, "mpisppy._mpi_progress_worker")
        find_spec.assert_called_once_with(module_name)

    def test_native_worker_load_and_configuration(self):
        worker = SimpleNamespace(
            STOP_TIMEOUT_SECONDS=30,
            configure=mock.Mock(),
            run=mock.Mock(),
            set_active=mock.Mock(),
            stop=mock.Mock(),
        )
        mpi = SimpleNamespace()
        comm = SimpleNamespace()
        with (
            mock.patch.object(
                mpi_progress, "_native_module_name", return_value="fake_worker"
            ),
            mock.patch.object(
                mpi_progress.importlib, "import_module", return_value=worker
            ),
            mock.patch.object(mpi_progress, "check_native_mpi_compatibility") as check,
        ):
            loaded_worker = mpi_progress._load_native_worker(
                mpi, comm, 0.001, "MPI.py"
            )

        self.assertIs(loaded_worker, worker)
        worker.configure.assert_called_once_with(comm, 0.001)
        check.assert_called_once_with(worker, mpi)

    def test_controller_calls_native_worker_directly(self):
        worker = SimpleNamespace(
            STOP_TIMEOUT_SECONDS=1,
            run=mock.Mock(),
            set_active=mock.Mock(),
            stop=mock.Mock(),
        )
        comm = SimpleNamespace(Abort=mock.Mock())
        with mock.patch.object(mpi_progress.atexit, "register") as register:
            controller = mpi_progress.ProgressController(worker, comm)
            controller.activate()
            controller.deactivate()
            controller.stop()

        self.assertIs(controller.worker, worker)
        worker.run.assert_called_once_with()
        worker.set_active.assert_has_calls([mock.call(True), mock.call(False)])
        worker.stop.assert_called_once_with()
        register.assert_called_once_with(controller.stop)

    def test_native_extension_is_required(self):
        mpi = SimpleNamespace()
        comm = SimpleNamespace()
        with mock.patch.object(mpi_progress, "_native_module_name", return_value=None):
            with self.assertRaisesRegex(
                RuntimeError, "requires the native MPI progress extension"
            ):
                mpi_progress._load_native_worker(mpi, comm, 0.001, "MPI.py")

    def test_native_worker_import_failure_is_explained(self):
        with (
            mock.patch.object(
                mpi_progress, "_native_module_name", return_value="fake_worker"
            ),
            mock.patch.object(
                mpi_progress.importlib,
                "import_module",
                side_effect=ImportError("missing extension"),
            ),
            self.assertRaisesRegex(RuntimeError, "rebuild mpi-sppy"),
        ):
            mpi_progress._load_native_worker(
                SimpleNamespace(), SimpleNamespace(), 0.001, "MPI.py"
            )

    def test_create_controller_rejects_invalid_poll_intervals(self):
        comm = SimpleNamespace(Get_rank=lambda: 0)
        for value, message in (
            ("invalid", "positive number"),
            ("nan", "must be positive"),
            ("0", "must be positive"),
        ):
            with self.subTest(value=value):
                with (
                    mock.patch.dict(
                        os.environ, {"MPISPPY_MPI_PROGRESS_INTERVAL_US": value}
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    mpi_progress.create_controller(SimpleNamespace(), comm, "MPI.py")

    def test_controller_rejects_unbalanced_deactivation(self):
        controller = object.__new__(mpi_progress.ProgressController)
        controller.active_count = 0
        controller.lock = threading.Lock()
        with self.assertRaisesRegex(RuntimeError, "unbalanced"):
            controller.deactivate()


class TestMPIProgressStartup(unittest.TestCase):
    def _load_with_fake_mpi(
        self, *, initialized=True, thread_level=3, vendor="Open MPI",
        vendor_version=(4, 1, 0), library_version=None,
        async_progress="0", progress_thread="1"
    ):
        fake_mpi = ModuleType("mpi4py.MPI")
        fake_mpi.COMM_WORLD = SimpleNamespace(Get_rank=lambda: 0)
        fake_mpi.Is_initialized = lambda: initialized
        fake_mpi.Query_thread = lambda: thread_level
        fake_mpi.THREAD_MULTIPLE = 3
        fake_mpi.get_vendor = lambda: (vendor, vendor_version)
        fake_mpi.Get_library_version = lambda: library_version or vendor
        fake_mpi.__all__ = ["COMM_WORLD"]
        fake_mpi4py = ModuleType("mpi4py")
        fake_mpi4py.__path__ = []
        fake_mpi4py.MPI = fake_mpi
        fake_mpi4py.rc = SimpleNamespace()
        mpi_path = Path(__file__).resolve().parents[1] / "MPI.py"
        with (
            mock.patch.dict(
                os.environ,
                {
                    "MPISPPY_MPI_PROGRESS_THREAD": progress_thread,
                    "MVAPICH_ASYNC_PROGRESS": async_progress,
                },
            ),
            mock.patch.dict(
                sys.modules,
                {"mpi4py": fake_mpi4py, "mpi4py.MPI": fake_mpi},
            ),
        ):
            return runpy.run_path(str(mpi_path))

    def test_progress_worker_starts_with_thread_multiple(self):
        controller = SimpleNamespace(progress_active=mock.Mock())
        with mock.patch.object(
            mpi_progress, "create_controller", return_value=controller
        ) as create_controller:
            module = self._load_with_fake_mpi()

        rc = module["_mpi4py"].rc
        self.assertTrue(rc.initialize)
        self.assertTrue(rc.threads)
        self.assertEqual(rc.thread_level, "multiple")
        self.assertIs(module["_progress_controller"], controller)
        self.assertIs(module["progress_active"], controller.progress_active)
        self.assertTrue(module["progress_available"])
        create_controller.assert_called_once_with(
            module["_mpi"], module["COMM_WORLD"], module["__file__"]
        )

    def test_progress_worker_requires_initialized_mpi(self):
        with self.assertRaisesRegex(RuntimeError, "requires initialized MPI"):
            self._load_with_fake_mpi(initialized=False)

    def test_progress_worker_requires_thread_multiple(self):
        with self.assertRaisesRegex(RuntimeError, "requires MPI_THREAD_MULTIPLE"):
            self._load_with_fake_mpi(thread_level=2)

    def test_mvapich_async_progress_uses_runtime_worker(self):
        module = self._load_with_fake_mpi(
            thread_level=1, vendor="MVAPICH2", async_progress="1"
        )
        self.assertFalse(module["_enable_progress_thread"])
        self.assertNotIn("_progress_controller", module)
        self.assertTrue(module["progress_available"])

    def test_mvapich_async_progress_without_mpisppy_worker(self):
        module = self._load_with_fake_mpi(
            vendor="MVAPICH2", async_progress="1", progress_thread="0"
        )
        self.assertTrue(module["progress_available"])
        self.assertNotIn("_progress_controller", module)

    def test_mvapich_async_progress_requires_version_newer_than_237(self):
        for version in ((2, 3, 6), (2, 3, 7)):
            with self.subTest(version=version):
                module = self._load_with_fake_mpi(
                    vendor="MVAPICH2", vendor_version=version,
                    async_progress="1", progress_thread="0"
                )
                self.assertFalse(module["progress_available"])

        module = self._load_with_fake_mpi(
            vendor="MVAPICH2", vendor_version=(2, 3, 8),
            async_progress="1", progress_thread="0"
        )
        self.assertTrue(module["progress_available"])

    def test_old_mvapich_still_requires_its_own_progress_worker(self):
        with self.assertRaisesRegex(RuntimeError, "requires MPI_THREAD_MULTIPLE"):
            self._load_with_fake_mpi(
                thread_level=1, vendor="MVAPICH2", vendor_version=(2, 3, 7),
                async_progress="1"
            )

    def test_mvapich_library_version_fallback_checks_version(self):
        for library_version, expected in (
            ("MVAPICH2 Version: 2.3.7", False),
            ("MVAPICH2 Version: 2.3.8", True),
            ("MVAPICH2 Version: 4.1", True),
            ("MVAPICH2 (version unknown)", False),
        ):
            with self.subTest(library_version=library_version):
                module = self._load_with_fake_mpi(
                    vendor="MPICH", library_version=library_version,
                    async_progress="1", progress_thread="0"
                )
                self.assertEqual(module["progress_available"], expected)

    def test_async_progress_flag_on_other_mpi_is_not_sufficient(self):
        module = self._load_with_fake_mpi(
            vendor="Open MPI", async_progress="1", progress_thread="0"
        )
        self.assertFalse(module["progress_available"])

    def test_progress_setting_requires_mpi4py(self):
        mpi_path = Path(__file__).resolve().parents[1] / "MPI.py"
        with (
            mock.patch.dict(
                os.environ, {"MPISPPY_MPI_PROGRESS_THREAD": "1"}
            ),
            mock.patch.dict(sys.modules, {"mpi4py": None, "mpi4py.MPI": None}),
            self.assertRaisesRegex(RuntimeError, "requires mpi4py"),
        ):
            runpy.run_path(str(mpi_path))


class TestMPIProgressCompatibility(unittest.TestCase):
    def setUp(self):
        self.mpi = SimpleNamespace(
            Comm=object,
            ANY_SOURCE=-1,
            ANY_TAG=-2,
            _sizeof=lambda cls: 8,
            Get_library_version=lambda: "test MPI",
        )
        self.extension = SimpleNamespace(
            __name__="fake_progress_extension",
            BUILD_MPI_COMM_SIZE=8,
            BUILD_MPI_ANY_SOURCE=-1,
            BUILD_MPI_ANY_TAG=-2,
            BUILD_MPI_LIBRARY_VERSION="test MPI",
        )

    def test_matching_signature(self):
        check_native_mpi_compatibility(self.extension, self.mpi)

    def test_each_mismatch_is_rejected(self):
        for field, wrong in (
            ("BUILD_MPI_COMM_SIZE", 4),
            ("BUILD_MPI_ANY_SOURCE", -3),
            ("BUILD_MPI_ANY_TAG", -4),
            ("BUILD_MPI_LIBRARY_VERSION", "other MPI"),
        ):
            with self.subTest(field=field):
                extension = SimpleNamespace(**vars(self.extension))
                setattr(extension, field, wrong)
                with self.assertRaisesRegex(RuntimeError, field):
                    check_native_mpi_compatibility(extension, self.mpi)

    def test_missing_signature_field_is_rejected(self):
        del self.extension.BUILD_MPI_COMM_SIZE
        with self.assertRaisesRegex(RuntimeError, "BUILD_MPI_COMM_SIZE"):
            check_native_mpi_compatibility(self.extension, self.mpi)

    def test_mpi4py_without_comm_size_is_rejected(self):
        del self.mpi._sizeof
        with self.assertRaisesRegex(RuntimeError, "MPI_Comm size"):
            check_native_mpi_compatibility(self.extension, self.mpi)


@unittest.skipUnless(
    os.environ.get("MPISPPY_MPI_PROGRESS_THREAD") == "1",
    "requires an enabled progress worker",
)
class TestEnabledMPIProgress(unittest.TestCase):
    def test_nested_context_and_exception_cleanup(self):
        from mpisppy import MPI

        controller = MPI._progress_controller
        before = controller.active_count
        try:
            with MPI.progress_active():
                self.assertEqual(controller.active_count, before + 1)
                with MPI.progress_active():
                    self.assertEqual(controller.active_count, before + 2)
                    time.sleep(0.03)
                self.assertEqual(controller.active_count, before + 1)
                raise ValueError("compute failed")
        except ValueError:
            pass
        self.assertEqual(controller.active_count, before)
        self.assertTrue(controller.thread.is_alive())


@unittest.skipUnless(
    os.environ.get("MPISPPY_CHECKOUT"),
    "requires an installed progress extension",
)
class TestInstalledMPIProgress(unittest.TestCase):
    def test_installed_worker_and_thread_level(self):
        # mpi-sppy must configure MPI_THREAD_MULTIPLE before mpi4py initializes MPI.
        # isort: off
        import mpisppy
        from mpisppy import MPI
        from mpi4py import MPI as mpi4py_mpi
        # isort: on

        self.assertTrue(MPI.haveMPI)
        self.assertGreaterEqual(mpi4py_mpi.Query_thread(), mpi4py_mpi.THREAD_MULTIPLE)
        self.assertEqual(
            MPI._progress_controller.worker.__name__, "mpisppy._mpi_progress_worker"
        )
        self.assertIsNotNone(importlib.util.find_spec("mpisppy._mpi_progress_worker"))
        checkout = Path(os.environ["MPISPPY_CHECKOUT"]).resolve()
        self.assertFalse(Path(mpisppy.__file__).resolve().is_relative_to(checkout))


if __name__ == "__main__":
    unittest.main()
