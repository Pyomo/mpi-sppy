###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

"""Build the MPI progress worker when its MPI/POSIX toolchain is available."""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import warnings
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel


def _write_mpi_signature_header(directory, version, comm_size, any_source, any_tag):
    """Embed the compiler probe's MPI values in a C header."""
    chunks = [version[index : index + 32] for index in range(0, len(version), 32)]
    quoted = (
        "\n".join(
            '    "' + "".join(f"\\x{byte:02x}" for byte in chunk) + '"'
            for chunk in chunks
        )
        or '    ""'
    )
    (directory / "mpi_progress_build.h").write_text(
        "/* Generated from the MPI compiler selected for this build. */\n"
        f"#define MPISPPY_BUILD_MPI_COMM_SIZE {comm_size}\n"
        f"#define MPISPPY_BUILD_MPI_ANY_SOURCE {any_source}\n"
        f"#define MPISPPY_BUILD_MPI_ANY_TAG {any_tag}\n"
        "static const char mpisppy_build_mpi_library_version[] =\n"
        f"{quoted};\n",
        encoding="ascii",
    )


_MPI_SIGNATURE_RUNNER = """import ctypes
import sys

library = ctypes.CDLL(sys.argv[1])
signature = library.mpisppy_mpi_build_signature
signature.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
signature.restype = ctypes.c_char_p
comm_size = ctypes.c_size_t()
any_source = ctypes.c_int()
any_tag = ctypes.c_int()
version = signature(ctypes.byref(comm_size),
                    ctypes.byref(any_source), ctypes.byref(any_tag))
if version is None or comm_size.value == 0:
    sys.exit(1)
header = f"{comm_size.value} {any_source.value} {any_tag.value} {len(version)}\\n"
sys.stdout.buffer.write(header.encode("ascii") + version)
"""


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

    def build_extensions(self):
        # setuptools has now configured the actual compiler and linker, including
        # CFLAGS, CPPFLAGS, LDFLAGS, LDSHARED, and build_ext command options.
        extension = self.extensions[0]
        with tempfile.TemporaryDirectory(prefix="mpisppy-mpi-build-") as name:
            directory = Path(name)
            source = directory / "signature.c"
            probe = directory / ("signature" + sysconfig.get_config_var("EXT_SUFFIX"))
            source.write_text(
                """#include <mpi.h>
#include <stddef.h>
const char *mpisppy_mpi_build_signature(size_t *comm_size,
                                        int *any_source, int *any_tag)
{
    static char version[MPI_MAX_LIBRARY_VERSION_STRING];
    int length = 0;
    if (MPI_Get_library_version(version, &length) != MPI_SUCCESS ||
        length < 0 || length >= MPI_MAX_LIBRARY_VERSION_STRING)
        return NULL;
    version[length] = '\\0';
    *comm_size = sizeof(MPI_Comm);
    *any_source = MPI_ANY_SOURCE;
    *any_tag = MPI_ANY_TAG;
    return version;
}
""",
                encoding="utf-8",
            )
            try:
                # A fresh signature must never accompany an older extension.
                self.force = self.compiler.force = True
                objects = self.compiler.compile(
                    [str(source)],
                    output_dir=name,
                    macros=extension.define_macros,
                    include_dirs=extension.include_dirs,
                    debug=self.debug,
                    extra_postargs=extension.extra_compile_args,
                )
                self.compiler.link_shared_object(
                    objects + extension.extra_objects,
                    str(probe),
                    libraries=self.get_libraries(extension),
                    library_dirs=extension.library_dirs,
                    runtime_library_dirs=extension.runtime_library_dirs,
                    extra_postargs=extension.extra_link_args,
                    export_symbols=["mpisppy_mpi_build_signature"],
                    debug=self.debug,
                    build_temp=self.build_temp,
                    target_lang=extension.language or "c",
                )
                output = subprocess.run(
                    [sys.executable, "-c", _MPI_SIGNATURE_RUNNER, str(probe)],
                    check=True,
                    capture_output=True,
                    timeout=30,
                ).stdout
                header, separator, version = output.partition(b"\n")
                if not separator:
                    raise ValueError("missing MPI compiler signature")
                comm_size, any_source, any_tag, length = map(int, header.split())
                if comm_size <= 0 or len(version) != length:
                    raise ValueError("invalid MPI compiler signature")
                version.decode("utf-8")
                _write_mpi_signature_header(
                    directory,
                    version,
                    comm_size,
                    any_source,
                    any_tag,
                )
            except Exception as exc:
                warnings.warn(
                    "MPI compiler probe failed; using the Python MPI progress "
                    f"worker ({type(exc).__name__}: {exc})"
                )
                # clear any previous build that was left behind
                Path(self.get_ext_fullpath(extension.name)).unlink(missing_ok=True)
                self.extensions.clear()
                self.distribution.ext_modules.clear()
                return

            for extension in self.extensions:
                extension.include_dirs.insert(0, name)
            super().build_extensions()


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
        # It must not contain a second copy of the MPI extension.
        files = [
            path for path in files if Path(path).resolve() not in extension_outputs
        ]
        mapping = {
            path: source
            for path, source in mapping.items()
            if Path(path).resolve() not in extension_outputs
        }
        return files, mapping


extensions = []
mpi4py_available = importlib.util.find_spec("mpi4py") is not None

native_platform = sys.platform.startswith("linux") or sys.platform == "darwin"
compiler_command = shlex.split(os.environ.get("CC", "mpicc"), posix=os.name != "nt")
compiler_executables = compiler_command[:1]
compiler_uses_launcher = bool(compiler_executables) and os.path.basename(
    compiler_executables[0]
) in {"ccache", "sccache"}
if compiler_uses_launcher:
    compiler_executables = compiler_command[:2]
compiler_available = (
    bool(compiler_executables)
    and (not compiler_uses_launcher or len(compiler_executables) == 2)
    and all(shutil.which(executable) is not None for executable in compiler_executables)
)
if mpi4py_available and not native_platform:
    warnings.warn(
        "The MPI progress extension requires Linux or macOS; using the Python worker"
    )
elif mpi4py_available and not compiler_available:
    if "CC" in os.environ:
        raise RuntimeError(f"MPI compiler command {os.environ['CC']!r} was not found")
    warnings.warn("mpicc was not found; using the Python MPI progress worker")

if mpi4py_available and native_platform and compiler_available:
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

setup(
    ext_modules=extensions,
    cmdclass={
        "build_ext": _EnvironmentBuildExt,
        "editable_wheel": _EnvironmentEditableWheel,
    },
)
