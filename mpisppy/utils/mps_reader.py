###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# written using many iterations with chatpgt

import pulp
import re
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals

def sanitize_name(name):
    """Ensure variable names and constraint names are valid Pyomo indices."""
    if not isinstance(name, str):
        name = str(name)  # Convert to string if not already
    return re.sub(r"[^\w]", "_", name.strip())  # Strip spaces and replace special characters with underscores

def read_mps_and_create_pyomo_model(mps_file):
    """
    Reads an MPS file using PuLP and converts it into a Pyomo model.

    Parameters:
        mps_file (str): Path to the MPS file.

    Returns:
        Pyomo ConcreteModel: A Pyomo model equivalent to the one described in the MPS file.
    """
    
    # Read the MPS file using PuLP
    varNames, lp_problem = pulp.LpProblem.fromMPS(mps_file)  # Ensure we only use the problem object
    
    # Create a Pyomo model
    model = ConcreteModel()
    
    # Define variables in Pyomo with sanitized names
    var_names = [sanitize_name(v.name) for v in lp_problem.variables()]
    model.variables = Var(var_names, domain=NonNegativeReals)  # Indexed by sanitized names
    
    # Define the objective function correctly
    if lp_problem.sense == pulp.LpMinimize:
        model.obj = Objective(
            expr=sum(coeff * model.variables[sanitize_name(var.name)] for var, coeff in lp_problem.objective.items()),
            sense=1  # Minimize
        )
    else:
        model.obj = Objective(
            expr=sum(coeff * model.variables[sanitize_name(var.name)] for var, coeff in lp_problem.objective.items()),
            sense=-1  # Maximize
        )
    
    # Define constraints
    def get_constraint_expr(constraint):
        return sum(model.variables[sanitize_name(var)] * coeff for var, coeff in constraint)
    
    def constraint_rule(m, cname):
        """Use the original constraint name for lookup but the sanitized name for Pyomo indexing."""
        constraint = lp_problem.constraints[cname]  # Keep original constraint name for lookup
        lhs_expr = get_constraint_expr(constraint.items()) 
        
        if constraint.sense == pulp.LpConstraintEQ:
            return lhs_expr == constraint.constant
        elif constraint.sense == pulp.LpConstraintLE:
            return lhs_expr <= constraint.constant
        elif constraint.sense == pulp.LpConstraintGE:
            return lhs_expr >= constraint.constant

    # Create constraints with sanitized names, but keep original names for lookup
    constraint_mapping = {sanitize_name(cname): cname for cname in lp_problem.constraints.keys()}
    model.constraints = Constraint(
        constraint_mapping.keys(),
        rule=lambda m, sanitized_cname: constraint_rule(m, constraint_mapping[sanitized_cname])
    )
    
    return model



########### main for debugging ##########
if __name__ == "__main__":
    fname = "delme.mps"
    print(f"about to read {fname}")
    model = read_mps_and_create_pyomo_model(fname)

    for cname in model.constraints:
        print(f"Constraint: {cname}")
        print(model.constraints[cname].expr)
        print("-" * 80)
        model.constraints[cname].pprint()
        print("+" * 80)
    for v in model.variables:
        print(f"Variable Name: '{v}'")  # Quoting ensures visibility of leading/trailing spaces
    model.pprint()
    
