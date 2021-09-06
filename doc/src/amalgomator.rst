.. _Amalgomator:

Amalgomator
===========

For simple problems that do not need extra specification, ``amalgomator.py``
provides several drivers to solve a problem without writing a program
that creates the cylinders one by one. ``amalgomator.from_module`` enables
a high-level user to create a hub-and-spoke architecture using the command 
line, with only a few lines of code.

The Amalgomator class
-------------------------
The ``Amalgomator`` class basically wraps up the creation and run of a simple
hub-and-spokes architecture.
It creates hub and spokes dictionaries using vanilla,
call ``sputils.spin_the_wheel`` and finally writes 
the solution to a file or a directory.

It takes as inputs scenario names, a scenario creator, options and
a ``kw_creator`` function. ``kw_creator`` must be a function specific to your
problem, taking the amalgomator options as an input, and giving as an output
additional arguments for the ``scenario_creator``.

The ``options`` argument is a dictionnary that specifies useful informations 
over our problem, and dictates the way Amalgomator runs. 
It must contains the following attributes:

* An boolean ``2stage`` or ``mstage`` equal to True, indicating a 2-stage or 
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
   Amalgomator does not work with everything. It only supports the cylinders and
   extensions that have a dedicated method in ``vanilla.py``.


Create Amalgomator from a module and command line
-------------------------------------------------
Given an options dictionary as above, ``amalgomator.Amalgomator_parser``
calls the appropriate parsers from ``baseparsers.py`` and completes the options
to add the necessary information for different modules.

The method ``amalgomator.from_module`` uses the two utilities described above.
It takes as input a module name, and calls ``amalgomator.Amalgomator_parser``
to get cylinder options and the number of scenarios from the command line.
Then, it computes the scenario names, and finally creates and
runs an Amalgomator object.

.. Note::
   The module must contains several methods:
   ``scenario_creator``, ``scenario_names_creator``, ``inparser_adder`` and
   ``kw_creator``. ``afarmer.py``, ``aaircond.py`` and ``uc_funcs.py`` contain
   examples of these functions.
   
Amalgomator with EF
-------------------

It is possible to use ``amalgomator.py`` to solve a problem by solving 
directly its extensive form (see the section :ref:`EF Directly`). The options
must then include an attribute ``EF-2stage`` or ``EF-mstage`` set equal to 
True. It uses the ``sputils.create_EF`` method.

Examples
--------

As intended, the examples of use of Amalgomator are quite short. The first
example, ``farmer_ama.py``, solves directly the EF. The model can be specified,
e.g. by taking an integer version of it. This specification can be made via
the command line, thanks to the ``inparser_adder`` method of ``farmer.py``.

Another example uses amalgomator, this time to create a true hub-and-spokes 
architecture. ``uc_ama.py`` creates a hub and 2 spokes. A notable feature of
this example is the use of the ``fixer`` extension. This extension needs a 
function "id_fix_list_fct" as a parameter. However, a function cannot be
passed via the command line. "id_fix_list_fct" must thus be an attribute of 
the options of ``amalgomator.from_module``.

Finally, it is possible to use Amalgomator without calling directly 
``amalgomator.from_module``. The example ``aircond_ama`` is first of all
fetching informations from the command line via 
``amalgomator.Aamgomator_parser``, and then modifying the options to get an
appropriate number of scenarios before creating an Amalgomator object. 

