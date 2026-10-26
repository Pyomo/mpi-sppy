"""Build optional MPI progress extensions only when explicitly requested."""

import os

from setuptools import Extension, setup


extensions = []
build_profiler = os.environ.get("MPISPPY_BUILD_MPI_PROGRESS_PROFILER") == "1"
build_worker = os.environ.get("MPISPPY_BUILD_MPI_PROGRESS_WORKER") == "1"
if build_profiler or build_worker:
    import mpi4py

if build_profiler:
    extensions.append(
        Extension(
            "mpisppy._mpi_progress_native",
            ["mpisppy/_mpi_progress_native.c"],
            include_dirs=[mpi4py.get_include()],
            extra_compile_args=["-std=c11"],
        )
    )

if build_worker:
    extensions.append(
        Extension(
            "mpisppy._mpi_progress_worker",
            ["mpisppy/_mpi_progress_worker.c"],
            include_dirs=[mpi4py.get_include()],
            extra_compile_args=["-std=c11", "-pthread"],
            extra_link_args=["-pthread"],
        )
    )

setup(ext_modules=extensions)
