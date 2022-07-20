.. _Amalgamator:

Amalgamator
===========

For simple problems that do not need extra specification, ``amalgamator.py``
provides several drivers to solve a problem without writing a program
that creates the cylinders one by one. ``amalgamator.from_module`` enables
a high-level user to create a hub-and-spoke architecture using the command 
line, with only a few lines of code.

The Amalgamator class
-------------------------
The ``Amalgamator`` class basically wraps up the creation and run of a simple
hub-and-spokes architecture.
It creates hub and spokes dictionaries using vanilla,
calls ``sputils.spin_the_wheel`` and finally writes 
the solution to a file or a directory.

It takes as inputs scenario names, a scenario creator, options and
a ``kw_creator`` function. ``kw_creator`` must be a function specific to your
problem, taking the amalgamator options as an input, and giving as an output
additional arguments for the ``scenario_creator``. The amalgamator class
is not flexible with respect to function names in the module.

The ``cfg`` argument is a ``Config`` class object that specifies information 
about the problem, and dictates the way Amalgamator runs. 
It must contains the following attributes for use with cylinders:

* A boolean ``2stage`` or ``mstage`` equal to True, indicating a 2-stage or 
  a multistage problem.

* A list ``cylinders``, containing the names of the desired hub and spokes.

* A list ``extensions``, containing the names of the desired extensions.

* (optional) A dictionary ``write_solution`` with 2 attributes, 
  ``first_stage_solution`` (resp. ``tree_solution``) with a .csv file name to 
  write
  the first stage solution (resp. a directory to write the tree full solution)

* (optional) Various options, e.g. to control the cylinders creation or the
  extensions, or options used to construct ``scenario_creator`` additional 
  arguments
  
.. Note::
   Amalgamator does not work with everything. It only supports the cylinders and
   extensions that have a dedicated method in ``vanilla.py``.


Create Amalgamator from a module and command line
------------------------------------------------- Given an options
``Config`` object (typically called `cfg`) as above,
``amalgamator.Amalgamator_parser`` creates calls the appropriate
functions to add the necessary information for different modules.

The method ``amalgamator.from_module`` uses the two utilities described above.
It takes as input a module name, and calls ``amalgamator.Amalgamator_parser``
to get cylinder options and the number of scenarios from the command line.
Then, it computes the scenario names, and finally creates and
runs an Amalgamator object.

.. Note::
   The module must contains several methods:
   ``scenario_creator``, ``scenario_names_creator``, ``inparser_adder`` and
   ``kw_creator``. The files ``examples.farmer.farmer.py``, ``mpisppy.tests.examples.aircond.py`` contain
   examples of these functions.

The full options dictionary is passed through to ``kw_creator`` so keyword arguments for
scenario creation can be placed in the almalgamator options dictionary.

Notes about ``inparser_adder``
------------------------------

The function adds config arguments unique to the instance. Note that `--branching-factors` can be added
even if something else added it, because the config.add functions bypass duplicates.

   
Amalgamator with EF
-------------------

It is possible to use ``amalgamator.py`` to solve a problem by solving 
directly its extensive form (see the section :ref:`EF Directly`). The options
must then include an attribute ``EF-2stage`` or ``EF-mstage`` set equal to 
True. It uses the ``sputils.create_EF`` method.

Examples
--------

As intended, the examples of use of Amalgamator are quite short. The first
example, ``farmer_ama.py``, solves directly the EF. The model can be specified,
e.g. by taking an integer version of it. This specification can be made via
the command line, thanks to the ``inparser_adder`` method of ``farmer.py``.

Another example uses amalgamator, this time to create a hub-and-spokes 
architecture. ``uc_ama.py`` creates a hub and 2 spokes. A notable feature of
this example is the use of the ``fixer`` extension. This extension needs a 
function "id_fix_list_fct" as a parameter. However, a function cannot be
passed via the command line. "id_fix_list_fct" must thus be an attribute of 
the options of ``amalgamator.from_module``.

Finally, it is possible to use Amalgamator without calling 
``amalgamator.from_module``. The example ``aircond_ama`` starts by
fetching informations from the command line via 
``amalgamator.Aamgomator_parser``, and then modifies the options to get an
appropriate number of scenarios before creating an Amalgamator object. 

