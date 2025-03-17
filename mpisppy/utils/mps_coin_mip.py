import mip   # from coin-or
import pyomo.environ as pyo
from pyomo.environ import  NonNegativeReals, minimize, maximize

def mip_to_pyomo(mps_path):
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
    
    # Read the MPS file and call the coin-mip model m
    m = mip.Model(solver_name="cbc")
    m.read(mps_path)

    # Create a Pyomo model
    model = pyo.ConcreteModel()

    varDict = dict()  # coin mip var to pyomo var
    # Add variables to Pyomo model with their bounds and domains
    for v in m.vars:
        vname = v.name
        print(f"Processing variable with name {vname} and type {v.var_type=}")
        varDict[v] = pyo.Var(domain=_domain_lookup(v), bounds=(v.lb, v.ub))
        setattr(model, vname, varDict[v])
    #print(f"{dir(v)=}")

    # Add constraints
    for c in m.constrs:
        cname = c.name
        # Extract terms correctly from LinExpr
        body = sum(coeff * varDict[var] for var, coeff in c.expr.expr.items())
        
        # Add the constant term if it exists
        body += c.expr.const if hasattr(c.expr, "const") else 0
        if c.expr.sense == "=":
            pyomoC = pyo.Constraint(expr=body == c.rhs)
        elif c.expr.sense == ">":
            pyomoC = pyo.Constraint(expr=body >= c.rhs)
        elif c.expr.sense == "<":
            pyomoC = pyo.Constraint(expr=body <= c.rhs)
        elif c.expr.sense == "":
            raise RuntimeError(f"Unexpected empty sense for constraint {c.name}"
                               f" from file {mps_path}")
        else:
            raise RuntimeError(f"Unexpected sense {c.expr.sense=}"
                               f" for constraint {c.name} from file {mps_path}")
        setattr(model, c.name, pyomoC)

    # objective function
    obj_expr = sum(coeff * varDict[v] for v, coeff in m.objective.expr.items())

    # Add the constant term if it exists
    obj_expr += m.objective.expr.get("const", 0)  # Ensure it doesn't crash

    # Set objective function (minimize or maximize)
    if m.sense == mip.MINIMIZE:
        model.objective = pyo.Objective(expr=obj_expr, sense=minimize)
    else:
        model.objective = pyo.Objective(expr=obj_expr, sense=maximize)

    return model

fname = "delme.mps"
#fname = "test1.mps"
pyomo_model = mip_to_pyomo(fname)
pyomo_model.pprint()

opt = pyo.SolverFactory('cplex')
opt.solve(pyomo_model)
pyomo_obj = pyo.value(pyomo_model.objective)

m = mip.Model(solver_name="cbc")
m.read(fname)
cbcstatus = m.optimize()
cbc_obj = m.objective_value
print(f"{cbcstatus=}, {cbc_obj=}, {pyomo_obj=}")
