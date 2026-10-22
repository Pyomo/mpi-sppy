"""Build the optional MPI progress profiler only when explicitly requested."""

import os

from setuptools import Extension, setup


extensions = []
if os.environ.get("MPISPPY_BUILD_MPI_PROGRESS_PROFILER") == "1":
    import mpi4py

    extensions.append(
        Extension(
            "mpisppy._mpi_progress_native",
            ["mpisppy/_mpi_progress_native.c"],
            include_dirs=[mpi4py.get_include()],
            extra_compile_args=["-std=c11"],
        )
    )

setup(ext_modules=extensions)
