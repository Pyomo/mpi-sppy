# Copyright 2021 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.


import pyomo.environ as pyo
from math import log10, floor

def get_solver():
    solvers = [n+e for e in ('_persistent', '') for n in ("cplex","gurobi","xpress")]
    
    for solvername in solvers:
        try:
            solver_available = pyo.SolverFactory(solvername).available()
        except:
            solver_available = False
        if solver_available:
            break
    
    if '_persistent' in solvername:
        persistentsolvername = solvername
    else:
        persistentsolvername = solvername+"_persistent"
    try:
        persistent_available = pyo.SolverFactory(persistentsolvername).available()
    except:
        persistent_available = False
    
    return solver_available, solvername, persistent_available, persistentsolvername

def round_pos_sig(x, sig=1):
    return round(x, sig-int(floor(log10(abs(x))))-1)
