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
[birge2011]_ (Section 1.1). This model can be expressed as the following linear
program (LP):

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

The coefficients of the :math:`x_i` variables in the second, third and fourth
constraints are the number of tons per acre that each crop will yield (2.5 for
wheat, 3 for corn, and 20 for sugar beets).


The following code creates an instance of the farmer's model:

.. testcode::

    import pyomo.environ as pyo

    def build_model(yields):
        model = pyo.ConcreteModel()

        # Variables
        model.X = pyo.Var(["WHEAT", "CORN", "BEETS"], within=pyo.NonNegativeReals)
        model.Y = pyo.Var(["WHEAT", "CORN"], within=pyo.NonNegativeReals)
        model.W = pyo.Var(
            ["WHEAT", "CORN", "BEETS_FAVORABLE", "BEETS_UNFAVORABLE"],
            within=pyo.NonNegativeReals,
        )

        # Objective function
        model.PLANTING_COST = 150 * model.X["WHEAT"] + 230 * model.X["CORN"] + 260 * model.X["BEETS"]
        model.PURCHASE_COST = 238 * model.Y["WHEAT"] + 210 * model.Y["CORN"]
        model.SALES_REVENUE = (
            170 * model.W["WHEAT"] + 150 * model.W["CORN"]
            + 36 * model.W["BEETS_FAVORABLE"] + 10 * model.W["BEETS_UNFAVORABLE"]
        )
        model.OBJ = pyo.Objective(
            expr=model.PLANTING_COST + model.PURCHASE_COST - model.SALES_REVENUE,
            sense=pyo.minimize
        )

        # Constraints
        model.CONSTR= pyo.ConstraintList()

        model.CONSTR.add(pyo.summation(model.X) <= 500)
        model.CONSTR.add(
            yields[0] * model.X["WHEAT"] + model.Y["WHEAT"] - model.W["WHEAT"] >= 200
        )
        model.CONSTR.add(
            yields[1] * model.X["CORN"] + model.Y["CORN"] - model.W["CORN"] >= 240
        )
        model.CONSTR.add(
            yields[2] * model.X["BEETS"] - model.W["BEETS_FAVORABLE"] - model.W["BEETS_UNFAVORABLE"] >= 0
        )
        model.W["BEETS_FAVORABLE"].setub(6000)

        return model

Note that the ``build_model`` function takes a list of values, containing the
yields for each crop. We can solve the model:

.. testcode::

    yields = [2.5, 3, 20]
    model = build_model(yields)
    solver = pyo.SolverFactory("cplex_direct")
    solver.solve(model)

    # Display the objective value to one decimal place
    print(f"{pyo.value(model.OBJ):.1f}")
    
The optimal objective value is:

.. testoutput::

    -118600.0

In practice, the farmer does not know the number of tons that each crop will
yield per acre planted--the yield depends on the weather, the quality of the
seeds, and other stochastic factors. Consequently, we replace the deterministic
model above with the stochastic LP:

.. math::
    :nowrap:

    \begin{array}{rl}
    \min & (150x_1 + 230x_2 + 260x_3) \\
    & \quad+\sum_{\omega\in\Omega}Pr[\omega]\big[(238y_1^\omega+210y_2^\omega) - (170w_1^\omega + 150w_2^\omega + 36w_3^\omega + 10 w_4^\omega)\big] \\
    \mathrm{s.t.} & x_1 + x_2 + x_3 \leq 500 \\
    & \xi^\omega_1 x_1 + y^\omega_1 - w^\omega_1 \geq 200\;\forall\;\omega\in\Omega\\
    & \xi^\omega_2 x_2 + y^\omega_2 - w^\omega_2 \geq 240\;\forall\;\omega\in\Omega\\
    & \xi^\omega_3 x_3 - w^\omega_3 - w^\omega_4 \geq 0\;\forall\;\omega\in\Omega\\
    & w^\omega_3 \leq 6000 \\
    & x,y^\omega,w^\omega\geq0\;\forall\;\omega\in\Omega
    \end{array}

The variables :math:`y_i` and :math:`w_i` have been replaced with copies
:math:`y_i^\omega` and :math:`w_i^\omega`, corresponding to the values of each
variable chosen under scenario :math:`\omega\in\Omega`, where :math:`\Omega` is
a finite set of scenarios. The parameter :math:`\xi^\omega_i` is the number of
tons of crop :math:`i` yielded per acre under scenario :math:`\omega`.

We assume that there are three scenarios: "good", "bad", and "average". We
assume that each scenario is equally likely to occur. The yield values
(:math:`\xi^\omega_i`) are given here:

.. list-table:: Crop yields under each scenario (tons/acre)
    :widths: 25 25 25 25
    :header-rows: 1

    * - 
      - Wheat
      - Corn
      - Sugar Beets
    * - Good
      - 3
      - 3.6
      - 24
    * - Average
      - 2.5
      - 3
      - 20
    * - Bad
      - 2
      - 2.4
      - 16

In order to transform the code for the deterministic model above into a
stochastic model which can be manipulated by MPI-SPPy, we need only incorporate
a few extra elements (see :ref:`scenario_creator` for full details). The
``scenario_creator`` function is told the name of the scenario to build, and
builds a Pyomo model for that scenario appropriately:

.. testcode::

    import mpisppy.utils.sputils as sputils

    def scenario_creator(scenario_name):
        if scenario_name == "good":
            yields = [3, 3.6, 24]
        elif scenario_name == "average":
            yields = [2.5, 3, 20]
        elif scenario_name == "bad":
            yields = [2, 2.4, 16]
        else:
            raise ValueError("Unrecognized scenario name")

        model = build_model(yields)
        sputils.attach_root_node(model, model.PLANTING_COST, [model.X])
        model._mpisppy_probability = 1.0 / 3
        return model


The ``scenario_creator`` accomplishes two important tasks

1. It calls the ``attach_root_node`` function. We tell this function which part
   of the objective function (``model.PLANTING_COST``) and which set of variables
   (``model.X``) belong to the first stage. In this case, the problem is only two
   stages, so we need only specify the root node and the first-stage
   information--MPI-SPPy assumes the remainder of the model belongs to the
   second stage.
2. It attaches an attribute called ``_mpisppy_probability`` to the model object. This is the
   probability that the specified scenario occurs. If this probability is not
   specified, MPI-SPPy will assume that all scenarios are equally likely.

Now that we have specified a scenario creator, we can use MPI-SPPy to solve the
farmer's stochastic program. 

Solving the Extensive Form
^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest approach is to solve the extensive form of the model directly.
MPI-SPPy makes this quite simple:

.. testcode::

    from mpisppy.opt.ef import ExtensiveForm

    options = {"solver": "cplex_direct"}
    all_scenario_names = ["good", "average", "bad"]
    ef = ExtensiveForm(options, all_scenario_names, scenario_creator)
    results = ef.solve_extensive_form()

    objval = ef.get_objective_value()
    print(f"{objval:.1f}")


.. testoutput::

    ...
    -108390.0

We can extract the optimal solution itself using the ``get_root_solution``
method of the ``ExtensiveForm`` object:

.. testcode::

    soln = ef.get_root_solution()
    for (var_name, var_val) in soln.items():
        print(var_name, var_val)

.. testoutput::
    
    X[BEETS] 250.0
    X[CORN] 80.0
    X[WHEAT] 170.0


Solving Using Progressive Hedging (PH)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also solve the model using the progressive hedging (PH) algorithm.
First, we must construct a PH object:

.. testcode::

    from mpisppy.opt.ph import PH

    options = {
        "solver_name": "cplex_persistent",
        "PHIterLimit": 5,
        "defaultPHrho": 10,
        "convthresh": 1e-7,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
    }
    all_scenario_names = ["good", "average", "bad"]
    ph = PH(
        options,
        all_scenario_names,
        scenario_creator,
    )


.. testoutput::
    :hide:

    ...

Note that all of the options in the ``options`` dict must be specified in order
to construct the PH object. Once the PH object is constructed, we can execute
the algorithm with a call to the ``ph_main`` method:

.. testcode::

    ph.ph_main()

.. testoutput::
    :hide:

    ...


.. testoutput::
    :options: +SKIP


    [    0.00] Start SPBase.__init__
    [    0.01] Start PHBase.__init__
    [    0.01] Creating solvers
    [    0.01] Entering solve loop in PHBase.Iter0
    [    2.80] Reached user-specified limit=5 on number of PH iterations

Note that precise timing results may differ.  In this toy example, we only
execute 5 iterations of the algorithm. Although the algorithm does not converge
completely, we can see that the first-stage variables already exhibit
relatively good agreement:

.. testcode::

    variables = ph.gather_var_values_to_rank0()
    for (scenario_name, variable_name) in variables:
        variable_value = variables[scenario_name, variable_name]
        print(scenario_name, variable_name, variable_value)

.. testoutput::
    :hide:

    ...
    average X[BEETS]
    ...

.. testoutput::
    :options: +SKIP

    good X[BEETS] 280.6489711937925
    good X[CORN] 85.26131687116064
    good X[WHEAT] 134.0897119350402
    average X[BEETS] 283.2796296293019
    average X[CORN] 80.00000000014425
    average X[WHEAT] 136.72037037055298
    bad X[BEETS] 280.64897119379475
    bad X[CORN] 85.26131687116226
    bad X[WHEAT] 134.08971193504266

The function ``gather_var_values_to_rank0`` can be used in parallel to collect
the values of all non-anticipative variables at the root. In this (serial)
example, it simply returns the values of the first-stage variables.

Solving Using Benders' Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we can solve our example using Benders' decomposition, known as the
L-shaped method in stochastic programming. The setup code is similar to the
previous methods:

.. testcode::

    from mpisppy.opt.lshaped import LShapedMethod

    all_scenario_names = ["good", "average", "bad"]
    bounds = {name: -432000 for name in all_scenario_names}
    options = {
        "root_solver": "cplex_persistent",
        "sp_solver": "cplex_persistent",
        "sp_solver_options" : {"threads" : 1},
        "valid_eta_lb": bounds,
        "max_iter": 10,
    }

    ls = LShapedMethod(options, all_scenario_names, scenario_creator)
    result = ls.lshaped_algorithm()

    variables = ls.gather_var_values_to_rank0()
    for ((scen_name, var_name), var_value) in variables.items():
        print(scen_name, var_name, var_value)

.. testoutput::
    :hide:

    ...

.. testoutput::
    :options: +SKIP

    [    0.00] Start SPBase.__init__
    Current Iteration: 1 Time Elapsed:    0.00 Current Objective: -Inf
    Current Iteration: 2 Time Elapsed:    0.01 Time Spent on Last Master: 0.00 Time Spent Generating Last Cut Set:    0.01 Current Objective: -1296000.00
    Current Iteration: 3 Time Elapsed:    0.02 Time Spent on Last Master: 0.00 Time Spent Generating Last Cut Set:    0.01 Current Objective: -160000.00
    Current Iteration: 4 Time Elapsed:    0.02 Time Spent on Last Master: 0.00 Time Spent Generating Last Cut Set:    0.00 Current Objective: -113750.00
    Converged in 4 iterations.
    Total Time Elapsed:    0.03 Time Spent on Last Master:    0.00 Time spent verifying second stage:    0.00 Final Objective: -108390.00
    good X[BEETS] 250.0
    good X[CORN] 80.0
    good X[WHEAT] 170.0
    average X[BEETS] 250.0
    average X[CORN] 80.0
    average X[WHEAT] 170.0
    bad X[BEETS] 250.0
    bad X[CORN] 80.0
    bad X[WHEAT] 170.0

We see that, for this toy example, the L-shaped method has converged to the
optimal solution within just 10 iterations.


aircond
-------

This is fairly complicated example because it is multi-stage and the
model itself offers a lot of flexibility.  The aircond example is
unusual in that the model file, ``aircond.py``, lives in
``mpisppy.tests.examples`` directory. Scripts and bash files that use
it live in ``examples.aircond``.  A good place to start is the
``aircond_cylinders.py`` file that starts with some functions that
support the main program. The main program makes use of the 
``Config`` object called `cfg` that creates a parser and gets arguments.

The configuration data obtained by the parser are passed directly to the vanilla hub
and spoke creator which knows how to use the arguments from a ``Config`` object.
The arguments unique to aircond are processed by the ``create_kwargs`` function
in the reference model file.

A simple example that uses a few of the options is shown in ``aircond_zhat.bash``, which
also calls the ``xhat4xhat`` program to estimate confidence intervals for the solution
obtained.


hydro
-----

Hydro is a three stage example that was originally coded in PySP and we make extensive use
of the PySP files. Unlike farmer and aircond where the scenario data are created from distributions,
for this problem the scenario data are provided in files.

Using PySPModel
^^^^^^^^^^^^^^^
In the file ``hydro_cylinders_pysp.py`` the lines

::

   from mpisppy.utils.pysp_model import PySPModel
   ...
   hydro = PySPModel("./PySP/models/", "./PySP/nodedata/")

cause an object called ``hydro`` to be created that has the methods needed by vanilla and the hub and
spoke creators as can be seen in the ``main`` function of ``hydro_cylinders_pysp.py``.


Not using PySPModel
^^^^^^^^^^^^^^^^^^^

In the file ``hydro_cylinders.py`` the file ``hydro.py`` is imported because it provides the functions
needed by vanilla hub and spoke creators.


netdes
------

This is a very challenging network design problem, which has many instances each defined by a data file.
For this problem, cross scenario cuts are helpful
so the use of that spoke is illustrated in ``netdes_cylinders.py``.  

sslp
----

This is a classic problem from Ntaimo and Sen with data in PySP format
so the driver code (e.g., ``sslp_cylinders.py`` that makes use of ``sslp.py``) is somewhat similar to the
hydro example except sslp is simpler because it is just two stages.

UC
--

This example uses the ``egret`` package for the underlying unit commitment model
and reads PySP format data using the ``pyomo`` dataportal. Data files for a variety
of numbers of scenarios are provided.

sizes
-----

The sizes example (Jorjani et al, IJPR, 1999) is a two-stage problem with general integers in each stage. The file
``sizes_cylinders.py`` is the usual cylinders driver. There are other examples in the directory, such
as ``sizes_demo.py``, which provides an example of serial execution (no cylinders).
