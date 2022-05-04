.. _Access Solutions:

Programmatic Access to Solutions
================================

The mechanisms for outputting and accessing solutions depends on how the solutions
were obtained. We will describe some of the possibilities for programmatic access
to the solutions here.

EF
--

If you are not using a ``WheelSpinner`` object, but rather creating an
``EF`` object directly using ``mpisppy.sputils.create_EF``,
you can use the function ``mpisppy.sputils.ef_scenarios`` to loop over
the scenario models after solution. The function takes one arugment,
which is the EF object.

For example, suppose you have such an extensive form (EF) object, `ef`,
then you
can access the ``Pyomo`` model for every scenario using something like:

::
   
   for sname, smodel in sputils.ef_scenarios(ef):

One can access the variables in the Pyomo model `smodel` inside the loop.




WheelSpinner
------------

While it is fairly straightforward to call a function to write
solutions found by ``WheelSpinner`` (see the :ref:`Output Solutions`
section), accessing the variables programmatically is more complicated
because, in general, the scenarios are not all on the same MPI rank
and not all ranks have access to the final version of the scenarios
that they do have.  If you are not experienced writing programs that
use MPI (and even if you are), you might want to use a solution writer
(see the :ref:`Output Solutions` section). To get behavior from the
solution writers that is customized for your application, you should
supply your own `scenario_tree_solution_writer` function as an
argument to the ``write_tree_solution`` function. See ``examples.uc.uc_funcs.py`` for
one example.  The main thing is the the custom function is passed
scenarios populated with the best upper bound solution regardless of which
cylinder found it.
