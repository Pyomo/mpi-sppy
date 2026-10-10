###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################


import os
import tempfile

import pyomo.environ as pyo
from math import log10, floor

from mpisppy.utils import sputils


def limit_solver_threads(solver, solver_name, threads=1):
    """Cap thread count on a directly-constructed Pyomo solver so test
    solves do not fan out across every core. Reuses the canonical->native
    option translator so we do not hardcode per-solver key names. Safe to
    call before or after set_instance for persistent solvers (thread
    options are applied at solve time)."""
    solver.options.update(
        sputils.translate_solver_options({"threads": threads}, solver_name))


def get_solver(persistent_OK=True):
    solvers = ["cplex","gurobi","xpress"]
    if persistent_OK:
        solvers = [n+e for e in ('_persistent', '') for n in solvers]
    
    for solver_name in solvers:
        try:
            solver_available = pyo.SolverFactory(solver_name).available()
        except Exception:
            solver_available = False
        if solver_available:
            break
    
    if '_persistent' in solver_name:
        persistent_solver_name = solver_name
    else:
        persistent_solver_name = solver_name+"_persistent"
    try:
        persistent_available = pyo.SolverFactory(persistent_solver_name).available()
    except Exception:
        persistent_available = False
    
    return solver_available, solver_name, persistent_available, persistent_solver_name

def round_pos_sig(x, sig=1):
    return round(x, sig-int(floor(log10(abs(x))))-1)


# --- HSL acknowledgement -----------------------------------------------------
#
# Ipopt builds that link the Harwell Subroutine Library print, in their own
# banner, that "any publicity material resulting from use of the HSL codes
# within IPOPT must contain the acknowledgement: HSL, a collection of Fortran
# codes for large-scale scientific computation."  Our test solves run with
# tee=False, so that banner never reaches the screen.  These helpers put the
# acknowledgement back, and say which linear solver is actually in use --
# worth knowing anyway, since the idaes-ext build defaults to ma27 rather than
# to MUMPS, and results can differ between the two.

_HSL_ACK = (
    "HSL, a collection of Fortran codes for large-scale scientific "
    "computation. See https://www.hsl.rl.ac.uk/"
)

_hsl_probe_result = None       # cache: (linear_solver_name, uses_hsl)
_hsl_announced = False


def ipopt_linear_solver():
    """Return (linear_solver_name, uses_hsl) for the ipopt on PATH.

    Ipopt names its linear solver in the banner it writes at the start of every
    solve, so one trivial solve into a logfile is enough. Returns (None, False)
    when ipopt is unavailable or the banner cannot be read.
    """
    global _hsl_probe_result
    if _hsl_probe_result is not None:
        return _hsl_probe_result

    result = (None, False)
    try:
        if pyo.SolverFactory("ipopt").available(exception_flag=False):
            m = pyo.ConcreteModel()
            m.x = pyo.Var(bounds=(-10, 10), initialize=0.0)
            m.o = pyo.Objective(expr=(m.x - 3) ** 2)
            m.c = pyo.Constraint(expr=m.x <= 1)
            fd, path = tempfile.mkstemp(suffix=".log")
            os.close(fd)
            try:
                pyo.SolverFactory("ipopt").solve(m, logfile=path)
                with open(path) as f:
                    text = f.read()
            finally:
                if os.path.exists(path):
                    os.remove(path)
            name = None
            for line in text.splitlines():
                if "running with linear solver" in line:
                    name = line.split("running with linear solver")[1]
                    name = name.strip().rstrip(".").split()[0]
                    break
            result = (name, "compiled using HSL" in text)
    except Exception:
        # Never let a courtesy message break a test run.
        result = (None, False)

    _hsl_probe_result = result
    return result


def announce_hsl_if_used():
    """Print the HSL acknowledgement, once per run, if ipopt links HSL.

    Gated on MPI rank: these tests also run under mpiexec, and the project
    convention is that such output comes from rank 0 only -- otherwise the
    banner is emitted once per rank and interleaves with itself.
    """
    global _hsl_announced
    if _hsl_announced:
        return
    _hsl_announced = True
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
    except ImportError:
        pass
    name, uses_hsl = ipopt_linear_solver()
    if not uses_hsl:
        return
    bar = "=" * 78
    print(
        f"\n{bar}\n"
        f"These tests solve with Ipopt built against HSL"
        + (f" (linear solver: {name})" if name else "")
        + f".\n{_HSL_ACK}\n{bar}",
        flush=True,
    )
