.. _Drivers:

Drivers
=======

To make use of the hub and spoke system, you must come up with a
driver that instantiates the objects and calls them. Nearly the
last step in most drivers is a call to ``mpisppy.utils.spin_the_wheel``
that calls the hub and spokes. Many of the example drivers take
advanatage of shared code in the ``examples`` directory; see
:ref:`Examples`.

We now proceed to consider driver examples that do not use the shared
services provided with the examples; however, many developers will prefer
to mimic the examples.

Starting with Examples
----------------------

The ``examples`` directory has utilities that set up command line options
and create dictionaries used to create hubs and spokes. The main shared utilities
are

* ``baseparser.py`` that creates command line options.
* ``vanilla.py`` that creates dictionaries used for hub and spoke
  creation. These dictionaries are ultimately fed to
  ``spin_the_wheel``.

The contructors for the vanilla spokes take arguments that vary slightly depending
on the spoke, but all want the args passed in by the args parser,
followed by ``scenario_creator`` function, a ``scenario_denoument`` function
(that can be ``None``), a list of scenario names as ``all_scenario_names``,
and ``cb_data``. Other arguments can be seen in the file ``mpisppy.examples.vanilla.py``
or in the the ``*_examples.py`` files that use it.  Since all require
the first four, in the examples, they are often collected into a tupe called
``beans`` so they can be passed to the constructor for every vanilla spoke.
  
Extending Examples
------------------
  
Many developers
will need to add extensions. Here are few examples:

* In the ``farmer_cylinders.py`` example, there is a block of code to add a ``--crops-mult`` argument that is passed to the scenario create in the ``cb_data`` dictionary.

* In the ``hydro_cylinders.py`` example (which has three stages), ``baseparser.py`` is not used. The branching factors are obtained from the command line and passed to the scenario constructor via ``cb_data`` and also passed to various spokes using ``["opt_kwargs"]["PHoptions"]["branching_factors"]``

* The ``uc_cylinders.py`` example adds arguments that are used to provide data or trigger the inclusion of extensions. The  extension specifications and arguments are added to the dictionaries  (e.g., ``hub_dict``) create by ``vanilla.py``.

Not Using Examples Utilities
----------------------------

Some users who want to use mpi-sppy with hubs and spokes embedded in
other examples and other users may want a deeper understanding because
they want to extend the shared services in ``examples``. For these
users, we provide a few examples that work directly with lower- and
intermediate-level APIs.

No Spokes
^^^^^^^^^

There is a ``sizes`` example that does not use the hub and spoke
system, it just calls PH (or the EF) without any spokes. The code can
be found in ``mpisppy.examples.sizes.sizes_demo.py`` and it also
demonstrates the use of extensions (see :ref:`Extensions`).

Hub and Spokes
^^^^^^^^^^^^^^

An example of a hub with spokes that does not use the shared services
provided in the ``examples`` directory is
``acopf3.ccopf2wood.py``. This shows an example of creating a
dictionaries (e.g. ``hub_dict``) without using ``vanilla.py``. For
more examples of dictionaries for hub and spoke creation,
``vanilla.py`` itself is a good place to look.

A Note about Options
--------------------

As a matter of common practice, there are no checks for unrecognized
keys in options dictionaries in ``mpi-sppy``. This has the advantage
that new spokes and extensions can simply look for any options that
they like. The disadvantage is that developers who use ``mpi-sppy``
cannot count on it to detect spelling errors in options names.
