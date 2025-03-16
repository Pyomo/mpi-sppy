import mip   # from coin-or
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, minimize, maximize

def mip_to_pyomo(mps_path):
    """
    Reads an MPS file using mip and converts it into a Pyomo ConcreteModel.
    This function extracts the model structure without solving it.
    
    :param mps_path: Path to the MPS file.
    :return: Pyomo ConcreteModel.
    """
    # Read the MPS file
    m = mip.Model(solver_name="cbc")
    m.read(mps_path)

    # Create a Pyomo model
    model = ConcreteModel()

    # Add variables to Pyomo model with their bounds
    model.variables = Var(
        range(len(m.vars)), 
        domain=NonNegativeReals  # Assuming all vars are non-negative unless modified below
    )

    # Map mip variables to Pyomo variables
    var_map = {v: model.variables[i] for i, v in enumerate(m.vars)}

    # Adjust variable bounds
    for i, v in enumerate(m.vars):
        if v.lb is not None:
            model.variables[i].setlb(v.lb)
        if v.ub is not None:
            model.variables[i].setub(v.ub)

    # Add constraints
    def constraint_rule(model, i):
        c = m.constrs[i]
        
        # Extract terms correctly from LinExpr
        expr = sum(coeff * var_map[var] for var, coeff in c.expr.expr.items())  # FIXED
        
        # Add the constant term if it exists
        expr += c.expr.const if hasattr(c.expr, "const") else 0

        # Return Pyomo constraint expression directly
        return expr == c.rhs  # Pyomo needs a symbolic equation, not an if-statement

    model.constraints = Constraint(range(len(m.constrs)), rule=constraint_rule)

    # **Fixed Objective Function Extraction**
    obj_expr = sum(coeff * var_map[var] for var, coeff in m.objective.expr.items())  # FIXED

    # Add the constant term if it exists
    obj_expr += m.objective.expr.get("const", 0)  # Ensure it doesn't crash

    # Set objective function (minimize or maximize)
    if m.sense == mip.MINIMIZE:
        model.objective = Objective(expr=obj_expr, sense=minimize)
    else:
        model.objective = Objective(expr=obj_expr, sense=maximize)

    return model

# Example usage:
# pyomo_model = mip_to_pyomo("delme.mps")
pyomo_model = mip_to_pyomo("delme.mps")
pyomo_model.pprint()
