.. _Examples:

Examples
========

If you installed directly from github, the top
level directory  ``examples`` 
contains some sub-directories with examples.

If you did not get the code from github (e.g., if
you installed with pip), you will need to
get the examples from:
https://github.com/Pyomo/mpi-sppy/tree/master/examples


Tutorial: Farmer Example
------------------------

In this section, we step through a simple example---the farmer example of
[birge2011]_ (Section 1.1). The

.. math::
    :nowrap:

    \begin{array}{rl}
    \min & (150x_1 + 230x_2 + 260x_3) + (238y_1+210y_2) - (170w_1 + 150w_2 + 36w_3 + 10 w_4) \\
    \mathrm{s.t.} & x_1 + x_2 + x_3 \leq 500 \\
    & (2.5)x_1 + y_1 - w_1 \geq 200 \\
    & (3)x_2 + y_2 - w_2 \geq 240 \\
    & (20)x_3 - w_3 - w_4 \geq 0 \\
    & w_3 \leq 6000 \\
    & x,y,w\geq0
    \end{array}

The decision variables are as follows:

- :math:`x_i` = number of acres to devote to crop i (1=wheat, 2=corn, 3=sugar
  beets)
- :math:`y_i` = tons of crop i to purchase from a wholesaler (i=1,2 but not 3)
- :math:`w_i` = tons of crop i sold (i=1,2)
- :math:`w_i` = tons of beets sold at favorable (i=3) or unfavorable (i=4)
  price

The following code creates an instance of the farmer's model:

.. testcode::

    import pyomo.environ as pyo

    def build_model():
        model = pyo.ConcreteModel()

        # Variables
        model.X = pyo.Var(["WHEAT", "CORN", "BEETS"], within=pyo.NonNegativeReals)
        model.Y = pyo.Var(["WHEAT", "CORN"], within=pyo.NonNegativeReals)
        model.W = pyo.Var(
            ["WHEAT", "CORN", "BEETS_FAVORABLE", "BEETS_UNFAVORABLE"],
            within=pyo.NonNegativeReals,
        )

        # Objective function
        obj_expression = (
            150 * model.X["WHEAT"] + 230 * model.X["CORN"] + 260 * model.X["BEETS"]
            + 238 * model.Y["WHEAT"] + 210 * model.Y["CORN"]
            - 170 * model.W["WHEAT"] - 150 * model.W["CORN"]
            - 36 * model.W["BEETS_FAVORABLE"] - 10 * model.W["BEETS_UNFAVORABLE"]
        )
        model.OBJ = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        # Constraints
        model.CONSTR= pyo.ConstraintList()

        model.CONSTR.add(pyo.summation(model.X) <= 500)
        model.CONSTR.add(
            2.5 * model.X["WHEAT"] + model.Y["WHEAT"] - model.W["WHEAT"] >= 200
        )
        model.CONSTR.add(
            3. * model.X["CORN"] + model.Y["CORN"] - model.W["CORN"] >= 240
        )
        model.CONSTR.add(
            20. * model.X["BEETS"] - model.W["BEETS_FAVORABLE"] - model.W["BEETS_UNFAVORABLE"] >= 240
        )
        model.W["BEETS_FAVORABLE"].setub(6000)

        return model

We can easily solve this model:

.. testcode::

    model = build_model()
    solver = pyo.SolverFactory("gurobi") 
    solver.solve(model)

    # Display the objective value to one decimal place
    print(f"{pyo.value(model.OBJ):.1f}")
    
The optimal objective value is:

.. testoutput::

    -112180.0
