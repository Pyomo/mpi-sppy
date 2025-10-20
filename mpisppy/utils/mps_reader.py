###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# IMPORTANT: parens in variable names will become underscore (_)
import mip   # from coin-or (pip install mip)
from mip.exceptions import ParameterNotAvailable
import pyomo.environ as pyo

# the following giant function is provided because CBC seems to have
#  trouble parsing free format MPS files.
def _read_obj_terms_from_mps(mps_path: str):
    """Return list of (var_name, coeff) tuples by parsing the MPS file directly."""
    obj_row = None
    obj_terms = []
    section = None

    with open(mps_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            tok0 = line.split()[0]

            if tok0 in ("NAME",):
                continue
            if tok0 == "ROWS":
                section = "ROWS"
                continue
            if tok0 == "COLUMNS":
                section = "COLUMNS"
                continue
            if tok0 in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
                section = None
                if tok0 != "COLUMNS":
                    # Once we reach RHS, we’re done collecting objective terms
                    if tok0 in ("RHS", "ENDATA"):
                        break
                continue

            if section == "ROWS":
                parts = line.split()
                # Row type N marks the objective row
                if parts[0] == "N":
                    obj_row = parts[1]
            elif section == "COLUMNS":
                parts = line.split()
                # Skip integer markers if they appear
                if parts[0] == "'MARKER'":
                    continue
                col = parts[0]
                rest = parts[1:]
                # Free MPS permits one or two (row, val) pairs per line
                # i.e., col row1 val1 [row2 val2]
                if len(rest) < 2:
                    continue
                # Walk pairs
                for i in range(0, len(rest), 2):
                    if i + 1 >= len(rest):
                        break
                    row, val = rest[i], rest[i + 1]
                    if obj_row is not None and row == obj_row:
                        obj_terms.append((col, float(val)))
    return obj_terms



def read_mps_and_create_pyomo_model(mps_path):
    """
    Reads an MPS file using mip and converts it into a Pyomo ConcreteModel.
    
    :param mps_path: Path to the MPS file.
    :return: Pyomo ConcreteModel.

    aside: Chatgpt was almost negative help with this function...
    """

    def _domain_lookup(v):
        # given a mip var, return its Pyomo domain
        if v.var_type == 'C':
            if v.lb == 0.0:
                return pyo.NonNegativeReals
            else:
                return pyo.Reals
        elif v.var_type == 'B':
            return pyo.Binary
        elif v.var_type == 'I':
            return pyo.Integers
        else:
            raise RuntimeError(f"Unknown type from coin mip {v.var_type=}")
        # BTW: I don't know how to query the mip object for SOS sets
        #      (maybe it transforms them?)
    
    # Read the MPS file and call the coin-mip model m
    m = mip.Model(solver_name="cbc")
    m.read(mps_path)

    # Create a Pyomo model
    model = pyo.ConcreteModel()

    varDict = dict()  # coin mip var to pyomo var
    # Add variables to Pyomo model with their bounds and domains
    for v in m.vars:
        vname = v.name.replace("(","_").replace(")","_")
        varDict[v] = pyo.Var(domain=_domain_lookup(v), bounds=(v.lb, v.ub))
        setattr(model, vname, varDict[v])
    #print(f"{dir(v)=}")

    # Add constraints
    for c in m.constrs:
        # Extract terms correctly from LinExpr
        body = sum(coeff * varDict[var] for var, coeff in c.expr.expr.items())
        
        if c.expr.sense == "=":
            pyomoC = pyo.Constraint(expr=(c.rhs, body, c.rhs))
        elif c.expr.sense == ">":
            pyomoC = pyo.Constraint(expr=(c.rhs, body, None))
        elif c.expr.sense == "<":
            pyomoC = pyo.Constraint(expr=(None, body, c.rhs))
        elif c.expr.sense == "":
            raise RuntimeError(f"Unexpected empty sense for constraint {c.name}"
                               f" from file {mps_path}")
        else:
            raise RuntimeError(f"Unexpected sense {c.expr.sense=}"
                               f" for constraint {c.name} from file {mps_path}")
        setattr(model, c.name, pyomoC)

    # objective function
    try:
        obj_items = list(m.objective.expr.items())  # usual path
        obj_expr = sum(coeff * varDict[vname] for vname, coeff in obj_items if vname in varDict)
    except ParameterNotAvailable:
        # CBC didn’t expose objective coefficients — fall back to parsing the file
        obj_items = _read_obj_terms_from_mps(mps_path)
        if not obj_items:
            raise RuntimeError("Could not retrieve objective coefficients from CBC or MPS file.")
        obj_expr = sum(coeff * varDict[vname] for vname, coeff in obj_items if vname in varDict)    
    if m.sense == mip.MINIMIZE:
        model.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
    else:
        model.objective = pyo.Objective(expr=obj_expr, sense=pyo.maximize)

    return model


if __name__ == "__main__":
    # for testing
    solver_name = "cplex"
    fname = "delme.mps"
    #fname = "test1.mps"
    pyomo_model = read_mps_and_create_pyomo_model(fname)
    pyomo_model.pprint()

    opt = pyo.SolverFactory(solver_name)
    opt.solve(pyomo_model)
    pyomo_obj = pyo.value(pyomo_model.objective)

    m = mip.Model(solver_name=solver_name)
    m.read(fname)
    coinstatus = m.optimize()
    coin_obj = m.objective_value
    print(f"{coinstatus=}, {coin_obj=}, {pyomo_obj=}")

    print("\ncoin var values")
    for v in m.vars:
        print(f"{v.name}: {v.x}")

    print("\npyomo var values")
    for v in pyomo_model.component_objects(pyo.Var, active=True):
        print(f"Variable: {v.name}")
        for index in v:
            print(f"  Index: {index}, Value: {v[index].value}")
        
    
