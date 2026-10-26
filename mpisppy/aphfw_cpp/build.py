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
}
# CMake variable that takes each solver's home
HOME_VAR = {"gurobi": "GUROBI_HOME", "cplex": "CPLEX_ROOT", "xpress": "XPRESS_HOME"}


class BuildError(RuntimeError):
    pass


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


def check_solver_home(solver, home):
    """Fail now, with the path, rather than in the middle of CMake."""
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
        libs = os.listdir(libdir) if os.path.isdir(libdir) else []
        if not any(f.startswith("libgurobi_c++") for f in libs):
            raise BuildError(f"No libgurobi_c++ library in {libdir}.")


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


def _libmpi(path):
    """The libmpi that ``path`` (a shared object) resolves to, via ldd."""
    if not shutil.which("ldd"):
        return None
    out = subprocess.run(["ldd", path], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split("=>")
        if "libmpi" in parts[0] and len(parts) == 2 and parts[1].split():
            return os.path.realpath(parts[1].split()[0])
    return None


def check_install():
    """Import the new module in a fresh interpreter and compare its libmpi."""
    code = ("import aphfw_cpp, mpi4py.MPI as m; "
            "print(aphfw_cpp.__file__); print(m.__file__)")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=os.path.expanduser("~"))
    if res.returncode != 0:
        raise BuildError("The build finished but aphfw_cpp does not import:\n"
                         + res.stderr.strip())
    module, mpi4py_ext = res.stdout.split()
    print(f"aphfw_cpp installed at {module}")
    ours, theirs = _libmpi(module), _libmpi(mpi4py_ext)
    if ours and theirs and ours != theirs:
        raise BuildError(
            f"aphfw_cpp loads {ours} but mpi4py loads {theirs}.  The two must "
            "share one MPI; rebuild with --cxx set to that MPI's mpicxx.")
    if ours:
        print(f"aphfw_cpp and mpi4py both load {ours}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m mpisppy.aphfw_cpp.build",
        description="Build and install aphfw_cpp, the compiled hub behind "
                    "--aphfw-hub.  Needs a C++23 compiler, MPI (the one "
                    "mpi4py uses) and one of Gurobi, CPLEX or Xpress.",
        epilog="Arguments after -- go to pip, e.g. -- --force-reinstall.")
    parser.add_argument("--solver", choices=sorted(SOLVERS),
                        help="solver backend (default: the one whose "
                             "environment variable is set)")
    parser.add_argument("--solver-home",
                        help="solver installation directory (default: from "
                             "GUROBI_HOME, CPLEX_ROOT or XPRESS_HOME)")
    parser.add_argument("--cxx", help="MPI C++ compiler (default: mpi4py's, "
                                      "else the mpicxx beside this python)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the pip command and stop")
    args, pip_args = parser.parse_known_args(argv)
    if pip_args and pip_args[0] == "--":
        pip_args = pip_args[1:]

    try:
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
        env = dict(os.environ, CXX=cxx)
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
