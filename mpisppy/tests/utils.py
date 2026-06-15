###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################


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
