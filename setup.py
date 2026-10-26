"""Build the MPI progress worker when mpi4py is in the build environment."""

import importlib.util
import os
import shutil
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel


class _EnvironmentBuildExt(_build_ext):
    """Keep editable extension builds out of the shared source checkout."""

    def finalize_options(self):
        super().finalize_options()
        if self.editable_mode:
            self.inplace = False

    def run(self):
        if self.editable_mode:
            self.inplace = False
        super().run()


class _EnvironmentEditableWheel(_editable_wheel):
    """Install editable extensions in the virtualenv, beside the .pth file."""

    def _run_build_commands(self, dist_name, unpacked_wheel, build_lib, tmp_dir):
        files, mapping = super()._run_build_commands(
            dist_name, unpacked_wheel, build_lib, tmp_dir
        )
        if not self.distribution.ext_modules:
            return files, mapping

        build_ext = self.get_finalized_command("build_ext")
        native_dir = Path(unpacked_wheel) / "mpisppy_editable_native"
        native_dir.mkdir()
        (native_dir / "__init__.py").write_text(
            f"SOURCE_ROOT = {str(Path(self.project_dir).resolve())!r}\n"
            f"BUILD_PREFIX = {str(Path(sys.prefix).resolve())!r}\n",
            encoding="utf-8",
        )
        extension_outputs = set()
        for extension in self.distribution.ext_modules:
            built = Path(build_ext.get_ext_fullpath(extension.name))
            shutil.copy2(built, native_dir / built.name)
            extension_outputs.add(built.resolve())

        # In strict editable mode setuptools also creates a shared link tree.
        # It must not contain a second copy of either MPI extension.
        files = [path for path in files
                 if Path(path).resolve() not in extension_outputs]
        mapping = {
            path: source for path, source in mapping.items()
            if Path(path).resolve() not in extension_outputs
        }
        return files, mapping


extensions = []
build_profiler = os.environ.get("MPISPPY_BUILD_MPI_PROGRESS_PROFILER") == "1"
mpi4py_available = importlib.util.find_spec("mpi4py") is not None
if build_profiler and not mpi4py_available:
    raise RuntimeError("MPI progress profiling requires mpi4py in the build environment")

if mpi4py_available:
    import mpi4py

    # Use the MPI compiler wrapper by default; an explicit CC may select the
    # wrapper for the MPI installation against which mpi4py was built.
    os.environ.setdefault("CC", "mpicc")
    extensions.append(
        Extension(
            "mpisppy._mpi_progress_worker",
            ["mpisppy/_mpi_progress_worker.c"],
            include_dirs=[mpi4py.get_include()],
            extra_compile_args=["-std=c11", "-pthread"],
            extra_link_args=["-pthread"],
        )
    )

if build_profiler:
    extensions.append(
        Extension(
            "mpisppy._mpi_progress_native",
            ["mpisppy/_mpi_progress_native.c"],
            include_dirs=[mpi4py.get_include()],
            extra_compile_args=["-std=c11"],
        )
    )

setup(
    ext_modules=extensions,
    cmdclass={
        "build_ext": _EnvironmentBuildExt,
        "editable_wheel": _EnvironmentEditableWheel,
    },
)
