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
import pyomo.environ as pyo

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
    obj_expr = sum(coeff * varDict[v] for v, coeff in m.objective.expr.items())
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
        
    
