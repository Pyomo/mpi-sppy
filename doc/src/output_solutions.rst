.. _Output Solutions:

Output Solutions
================

The mechanisms for outputting and accessing solutions depends on how the solutions
were obtained. We will describe some of the possibilities here.

EF (and Mid-level)
------------------

If you are not using a ``WheelSpinner`` object, but rather creating an
``EF`` object directly (or ``PH``, ``APH`` or ``L-shaped`` directly),
you can use the function ``mpisppy.spbase.SPBase.write_tree_solution``
to output the entire solution tree; the function takes a directory
name (string) as positional argument (and optionally a customized
writer function).  You can output just the first stage solution using
``mpisppy.spbase.SPBase.write_tree_solution``; the function takes a
file name (string) as positional argument (and optionally a customized
writer function).

For example, suppose you have an extensive form (EF) object, `ef`, that
is of a type derived from SPBase (which is almost surely is), then you
can print the tree solution to a directory named `efsol` using

::
   
   ef.write_tree_solution("efsol")


See :ref:`SPBase` in the API section for a description of the function signature.

WheelSpinner
------------

The ``WheelSpinner`` class has member functions that can write
solutions that are very similar to the functions on ``SPBase``.
You can use the function ``WheelSpinner.write_tree_solution``
to output the entire solution tree; the function takes a directory
name (string) as positional argument (and optionally a customized
writer function).  You can output just the first stage solution using
``WheelSpinner.write_tree_solution``; the function takes a
file name (string) as positional argument (and optionally a customized
writer function).   See the `farmer_cylinders` example.

xhat for Confidence Intervals
-----------------------------

To get a first-stage (ROOT node) `xhat` for confidence intervalues,
the function ``sputils.first_stage_nonant_npy_serializer`` can be
passed as the ``first_stage_solution_writer`` keyword argument to the
``WheelSpinner.write_tree_solution`` function.  See the
`farmer_cylinders` example.

Customized Writer Function
--------------------------

See ``examples.uc.uc_funcs.py`` for an example called `` scenario_tree_solution_writer``,
which matches the argument name in the ``write_tree_solution`` function. The most
important thing to note is that these functions are passed a scenario model (a Pyomo model) that
is populated with the solution for the given scenario.

Amalgamator
-----------

To get solution output when using `Amalgamator`, you can supply a file name in
``options['write_solution']['first_stage_solution']`` and/or a directory name in
``options['write_solution']['tree_solution']``
