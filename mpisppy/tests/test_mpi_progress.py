"""Checks for progress worker selection and its Python-facing guardrails."""

import importlib.util
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

from mpisppy.utils.mpi_progress_compat import check_native_mpi_compatibility


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


@unittest.skipUnless(os.environ.get("MPISPPY_EXPECT_PROGRESS_WORKER"),
                     "requires a progress worker installation")
class TestInstalledMPIProgress(unittest.TestCase):
    def test_installed_worker_and_thread_level(self):
        import mpisppy
        from mpisppy import MPI
        from mpi4py import MPI as mpi4py_mpi

        expected = os.environ["MPISPPY_EXPECT_PROGRESS_WORKER"]
        self.assertIn(expected, ("python", "native"))
        self.assertTrue(MPI.haveMPI)
        self.assertGreaterEqual(mpi4py_mpi.Query_thread(), mpi4py_mpi.THREAD_MULTIPLE)
        self.assertEqual(MPI._native_worker_enabled, expected == "native")
        self.assertEqual(
            importlib.util.find_spec("mpisppy._mpi_progress_worker") is not None,
            expected == "native",
        )
        checkout = Path(os.environ["MPISPPY_CHECKOUT"]).resolve()
        self.assertFalse(Path(mpisppy.__file__).resolve().is_relative_to(checkout))

    def test_nested_context_and_exception_cleanup(self):
        from mpisppy import MPI

        before = MPI._progress_active_count
        try:
            with MPI.progress_active():
                self.assertEqual(MPI._progress_active_count, before + 1)
                with MPI.progress_active():
                    self.assertEqual(MPI._progress_active_count, before + 2)
                    time.sleep(0.03)
                self.assertEqual(MPI._progress_active_count, before + 1)
                raise ValueError("compute failed")
        except ValueError:
            pass
        self.assertEqual(MPI._progress_active_count, before)
        self.assertTrue(MPI._progress_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
