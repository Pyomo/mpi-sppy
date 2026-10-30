###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Build and install aphfw_cpp, the compiled FW-APH hub for mpi-sppy.

Run as ``python -m mpisppy.aphfw_cpp.build``.  This wraps the raw build,
``pip install`` of this directory (see pyproject.toml), with the checks that
make it fail early and clearly: one solver found and complete, an MPI C++
compiler chosen to match the MPI that mpi4py uses, and, after the install, an
import of the module and a check that it and mpi4py load the same libmpi.  An
MPI mismatch is otherwise a crash at run time, not a build error.

The directory deliberately has no __init__.py: in aph-fw it sits at the
repository root, where a regular package named aphfw_cpp would shadow the
compiled module.
"""

import argparse
import os
import shutil
import subprocess
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# solver -> (environment variables that may name its home, CMake option)
SOLVERS = {
    "gurobi": (["GUROBI_HOME"], "USE_GUROBI"),
    "cplex": (["CPLEX_ROOT"], "USE_CPLEX"),
    "xpress": (["XPRESS_HOME", "XPRESSDIR"], "USE_XPRESS"),
    "highs": (["HIGHS_HOME"], "USE_HIGHS"),
}
# CMake variable that takes each solver's home
HOME_VAR = {"gurobi": "GUROBI_HOME", "cplex": "CPLEX_ROOT", "xpress": "XPRESS_HOME",
            "highs": "HIGHS_HOME"}


class BuildError(RuntimeError):
    pass


def check_sources():
    """The C++ sources are in a clone of mpi-sppy, not in its wheel.

    The wheel carries this file (setuptools packs .py files) but none of the
    sources beside it, so from a pip-installed mpi-sppy there is nothing to
    build.
    """
    if not os.path.isfile(os.path.join(SRC_DIR, "pyproject.toml")):
        raise BuildError(
            f"No aphfw_cpp sources in {SRC_DIR}.  They come only with a clone "
            "of the mpi-sppy repository from GitHub, not with the package from "
            "PyPI: install mpi-sppy from a clone and run this command there.")


def _env_home(solver):
    for var in SOLVERS[solver][0]:
        if os.environ.get(var):
            return os.environ[var], var
    return None, None


def choose_solver(solver, home):
    """Return (solver, home), from the arguments or else the environment."""
    if solver is None:
        found = [s for s in SOLVERS if _env_home(s)[0]]
        if not found:
            names = ", ".join(v for s in SOLVERS for v in SOLVERS[s][0])
            raise BuildError(
                "No solver found.  Set one of " + names + " to the solver's "
                "installation directory, or pass --solver and --solver-home.")
        if len(found) > 1 and home is None:
            raise BuildError(
                "More than one solver is set in the environment (" +
                ", ".join(found) + "); the module is built for exactly one.  "
                "Choose with --solver.")
        solver = found[0]
    if home is None:
        home, _ = _env_home(solver)
        if home is None:
            raise BuildError(
                f"--solver {solver} needs its installation directory: pass "
                f"--solver-home or set {SOLVERS[solver][0][0]}.")
    return solver, os.path.abspath(os.path.expanduser(home))


def _find_lib(dirs, name):
    """True if lib<name>.a or lib<name>.so* is in one of dirs."""
    for d in dirs:
        if os.path.isdir(d) and any(
                f.startswith(f"lib{name}.") for f in os.listdir(d)):
            return True
    return False


def check_solver_home(solver, home):
    """Fail now, with the path, rather than in the middle of CMake.

    The paths are the ones solver_interfaces/CMakeLists.txt searches.
    """
    if not os.path.isdir(home):
        raise BuildError(f"{solver} home {home} is not a directory.")
    if solver == "gurobi":
        # GUROBI_HOME is the platform directory (e.g. .../linux64)
        if not os.path.isfile(os.path.join(home, "include", "gurobi_c++.h")):
            raise BuildError(
                f"No include/gurobi_c++.h under {home}.  GUROBI_HOME must be "
                "the platform directory of a full Gurobi install (e.g. "
                ".../gurobi1302/linux64); the gurobipy wheel does not carry "
                "the C++ library.")
        libdir = os.path.join(home, "lib")
        if not _find_lib([libdir], "gurobi_c++"):
            raise BuildError(f"No libgurobi_c++ library in {libdir}.")
    elif solver == "cplex":
        # CPLEX_ROOT is the CPLEX_Studio directory, holding cplex/ and concert/
        missing = [p for p in ("cplex/include/ilcplex/cplex.h",
                               "concert/include/ilconcert")
                   if not os.path.exists(os.path.join(home, p))]
        if missing:
            raise BuildError(
                f"Missing {' and '.join(missing)} under {home}.  CPLEX_ROOT must be "
                "the CPLEX_Studio directory (e.g. .../CPLEX_Studio2211), the "
                "one holding cplex/ and concert/.")
        libdirs = [os.path.join(home, top, "lib", sub)
                   for top in ("cplex", "concert")
                   for sub in ("x86-64_linux/static_pic", "x86-64_linux/static_mt",
                               "x86-64_linux", "")]
        absent = [lib for lib in ("cplex", "ilocplex", "concert")
                  if not _find_lib(libdirs, lib)]
        if absent:
            raise BuildError(
                f"No {', '.join('lib' + lib for lib in absent)} under "
                f"{home}/cplex/lib or {home}/concert/lib.")
    elif solver == "xpress":
        if not os.path.isfile(os.path.join(home, "include", "xprs.h")):
            raise BuildError(
                f"No include/xprs.h under {home}.  XPRESS_HOME must be the "
                "Xpress installation directory, the one holding include/ and "
                "lib/.")
        libdirs = [os.path.join(home, d) for d in ("lib", "lib64")]
        if not _find_lib(libdirs, "xprs"):
            raise BuildError(f"No libxprs library in {home}/lib or {home}/lib64.")
    elif solver == "highs":
        # HIGHS_HOME is an installation prefix, e.g. a conda environment
        # with conda-forge's highs package
        missing = [p for p in ("include/highs/Highs.h",
                               "lib/cmake/highs/highs-config.cmake")
                   if not os.path.isfile(os.path.join(home, p))]
        if missing:
            raise BuildError(
                f"Missing {' and '.join(missing)} under {home}.  HIGHS_HOME must "
                "be the prefix HiGHS was installed into (e.g. a conda "
                "environment with conda-forge's highs package); the highspy "
                "wheel has no C++ headers.")
        if not _find_lib([os.path.join(home, "lib")], "highs"):
            raise BuildError(f"No libhighs library in {home}/lib.")


def _mpi_compilers_near_python():
    """mpicxx in the same bin directory as this Python (conda, venvs on it)."""
    bindir = os.path.dirname(os.path.realpath(sys.executable))
    for name in ("mpicxx", "mpic++", "mpiCC"):
        path = os.path.join(bindir, name)
        if os.access(path, os.X_OK):
            yield path


def choose_cxx(cxx):
    """Pick the MPI C++ compiler: argument, mpi4py's, beside python, PATH."""
    if cxx:
        found = shutil.which(cxx)
        if not found:
            raise BuildError(f"--cxx {cxx} is not an executable.")
        return found
    try:
        import mpi4py
        config = mpi4py.get_config()
    except ImportError:
        raise BuildError("mpi4py is not importable; mpi-sppy's hub needs it, "
                         "and the build has to match its MPI.")
    for key in ("mpicxx", "mpic++"):
        if config.get(key) and shutil.which(config[key]):
            return shutil.which(config[key])
    for path in _mpi_compilers_near_python():
        return path
    for name in ("mpicxx", "mpic++"):
        if shutil.which(name):
            return shutil.which(name)
    raise BuildError("No MPI C++ compiler (mpicxx) found.  Pass --cxx with "
                     "the one belonging to the MPI that mpi4py uses.")


def pip_command(solver, home, pip_args):
    cmd = [sys.executable, "-m", "pip", "install", SRC_DIR]
    for s in SOLVERS:
        on = "ON" if s == solver else "OFF"
        cmd += ["-C", f"cmake.define.{SOLVERS[s][1]}={on}"]
    cmd += ["-C", f"cmake.define.{HOME_VAR[solver]}={home}"]
    return cmd + list(pip_args)


def _loaded_libmpi(module):
    """The libmpi files a fresh interpreter has mapped after importing module.

    Read from /proc/self/maps, so it is what the dynamic loader actually
    chose (ldd cannot see a libmpi that a wheel loads by itself, as MPICH's
    mpi4py wheel does).  None where there is no /proc (not Linux).
    """
    code = ("import os, " + module + "\n"
            "if not os.path.exists('/proc/self/maps'):\n"
            "    print('?')\n"
            "else:\n"
            "    for line in open('/proc/self/maps'):\n"
            "        path = line.split()[-1]\n"
            "        if os.path.basename(path).startswith('libmpi.so'):\n"
            "            print(os.path.realpath(path))\n")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=os.path.expanduser("~"))
    if res.returncode != 0:
        raise BuildError(f"import {module} failed:\n" + res.stderr.strip())
    found = set(res.stdout.split())
    return None if "?" in found else found


def check_install():
    """Import the new module in a fresh interpreter and compare its libmpi."""
    res = subprocess.run([sys.executable, "-c",
                          "import aphfw_cpp; print(aphfw_cpp.__file__)"],
                         capture_output=True, text=True,
                         cwd=os.path.expanduser("~"))
    if res.returncode != 0:
        raise BuildError("The build finished but aphfw_cpp does not import:\n"
                         + res.stderr.strip())
    print(f"aphfw_cpp installed at {res.stdout.strip()}")
    ours, theirs = _loaded_libmpi("aphfw_cpp"), _loaded_libmpi("mpi4py.MPI")
    if ours is None or theirs is None:
        print("(not Linux: did not check that aphfw_cpp and mpi4py share "
              "one MPI library)")
    elif ours != theirs:
        raise BuildError(
            f"aphfw_cpp loads {', '.join(sorted(ours)) or 'no libmpi'} but "
            f"mpi4py loads {', '.join(sorted(theirs)) or 'no libmpi'}.  The "
            "two must share one MPI; rebuild with --cxx set to the mpicxx of "
            "the MPI that mpi4py uses.")
    else:
        print(f"aphfw_cpp and mpi4py both load {', '.join(sorted(ours))}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m mpisppy.aphfw_cpp.build",
        description="Build and install aphfw_cpp, the compiled hub behind "
                    "--aphfw-hub.  Needs a C++23 compiler, MPI (the one "
                    "mpi4py uses) and one of Gurobi, CPLEX, Xpress or "
                    "HiGHS.",
        epilog="Arguments after -- go to pip, e.g. -- --force-reinstall.")
    parser.add_argument("--solver", choices=sorted(SOLVERS),
                        help="solver backend (default: the one whose "
                             "environment variable is set)")
    parser.add_argument("--solver-home",
                        help="solver installation directory (default: from "
                             "GUROBI_HOME, CPLEX_ROOT, XPRESS_HOME or "
                             "HIGHS_HOME)")
    parser.add_argument("--cxx", help="MPI C++ compiler (default: mpi4py's, "
                                      "else the mpicxx beside this python)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the pip command and stop")
    args, pip_args = parser.parse_known_args(argv)
    if pip_args and pip_args[0] == "--":
        pip_args = pip_args[1:]

    try:
        check_sources()
        solver, home = choose_solver(args.solver, args.solver_home)
        check_solver_home(solver, home)
        cxx = choose_cxx(args.cxx)
        cmd = pip_command(solver, home, pip_args)
        print(f"solver: {solver} at {home}")
        print(f"CXX:    {cxx}")
        print("running: " + " ".join(cmd))
        sys.stdout.flush()  # before pip's output, which goes straight out
        if args.dry_run:
            return 0
        # An MPI wrapper finds the compiler it wraps on PATH; conda's mpicxx
        # fails when the environment's bin directory is not there.
        path = os.pathsep.join([os.path.dirname(cxx), os.environ.get("PATH", "")])
        env = dict(os.environ, CXX=cxx, PATH=path)
        if subprocess.run(cmd, env=env).returncode != 0:
            raise BuildError(
                "pip install failed; the CMake or compiler error is above.  "
                "Common causes: a compiler without C++23 support, or a "
                "solver version the build does not look for.")
        check_install()
    except BuildError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
