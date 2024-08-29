#
The following is the gurobipy dilemma

Unlike other mathematical modeling languages like Pyomo, AMPL, and GAMS, Gurobipy serves as an API to the Gurobi solver and does not retain symbols. Symbols are essential for updating and modifying the model when new data is added. In Gurobipy, if changed need to be made to a constraint or an objective functio, we workaround with two approaches.

**Recompute the Model**: Recomputing the model each time a modification occurs. This method is inefficient due to the repeated recomputation of the model.

**Maintain Symbols Manually**: Since symbols are not retained in Gurobipy, we must manually maintain our version of the symbols by expanding the objective function expression and calculating the coefficients for each term (e.g., xx and x2x2). The useful feature in Gurobipy here is the ability to modify coefficients in linear constraints using the `changeCoeff(constraint, var, newVal)` function. This allows us to update the coefficients in linear constraints accordingly.

The callout functions in `farmer_yyyy_agnostic.py` will behave diffrently. Instead of attaching elements to the model and changing those Param values, functions must now operate on an expanded version of the objective function. It's important to note that `changeCoeff` only modifies linear expressions, which poses a challenge since quadriatic expresssions need to be related to linear expressions; so we will need a new variable xs2 which is constrained to be equal to $x^2$

The diffrence between the callout functions:
    - `attach_Ws_and_prox(Ag, sname, scenario)` will retrieve the current nonants coefficients from the objective function, and if the nonants aren't in the obj they are noted with 0 and put in a seperate list 
    - `attach_PH_to_obj()` - create list of x^2 vars and attach coefs to obj function, make sure all nonants are in the objective function, call `_copy_Ws_xbars_rho_from_host(scenario)`` 
    - `_copy_Ws_xbars_rho_from_host(scenario)` function that will be called to calculate the necessary coefficients for the objective function
    - we should probably call the functions something else now

    
