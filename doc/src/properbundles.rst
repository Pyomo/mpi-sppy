Proper Bundles
==============

Prior to 2024, bundles were constructed for the purpose of solves, but
all other processing (e.g., computing W values) was done on individual
scenarios. We will refer to these as `loose bundles`. This bundling scheme
is very flexible with respect to the numbers of scenarios in each bundle.
There are various if-blocks in the mpisppy code to support this type of bundle.

.. Warning::
   In relase 1.0, loose bundles scheduled to be deprecated.

In 2024, `proper bundles` were supported. After the extensive form
for a proper bundle is created, the original scenarios are more or less
forgotten and all processing takes place for the bundle. At the time
of this writing, these bundles are a little less flexible in that
the number of scenarios per bundle must divide the number of scenarios
and randomizing the assignment of scenarios to bundles is left to the
user (e.g., by using a pseudo-random vector to provide one level
of indirection for the scenario number in the ``scenario_creator`` function).
As of the time of this writing, only two-stage problems are easily supported.
Proper bundles result in faster execution than loose bundles.

See ``mpisppy.generic_cylinders.py`` for an example of their use in
code and see ``examples.generic_cylinders.bash`` for a few proper
bundle command lines.  In addition to being created on the fly and
used, they can be written (but not used in the same run) with
``--pickle-bundles-dir`` (note the the directory specified will be
overwritten), and read before use with ``--unpickle-bundles-dir``.  In
all uses of bundles in ``mpisppy.generic_cylinders.py`` the
``--scenarios-per-bundle`` option must be specified (even when
reading).

.. Note::
   When writing bundles in ``mpisppy.generic_cylinders.py``, all
   ranks are used for forming and writing bundles. Command line
   options related to anything other than proper bundles are ignored.

.. Note::
   Reading and writing bundle pickle files only works with proper bundles, not
   loose bundles.

.. Note::
   If you do pseudo random number generation on-the-fly during scenario creation,
   very careful management of random seeds is required if you want to
   get the same scenarios with proper  bundles that you get without them.

.. Note::
   Unpickled scenarios in proper bundles are not supported in generic_cyliners.
   (The wrappers would need to be more sophisticated.)

.. Note::
   The `scenario_denouement` function might not be called when pickling bundles.

.. Warning::
   Helper functions are *not* pickled, so there is a loose linkage with the
   helper functions in the module.


Modules
-------

In addition to command line options specified in ``mpisppy.utils.config.py``
and supported in ``mpisppy.generic_cylinders.py'',
there are two modules that have most of the support for proper bundles:

  - ``mpisppy.utils.pickle_bundle.py`` has miscellaneous utilities related to picking and other data processing
  - ``mpisppy.utils.proper_bundler.py`` has wrappers for cylinder programs


Multistage
----------

The most flexible way to create proper bundles is to write
your own problem-specific code to do it. The
file ``aircond_cylinders.py`` in the aircond example directory
provides an example.  The latter part of the ``allways.bash`` script
demonstrates how to run it.

There is support for multi-stage bundles in mpi-sppy, but the scenario
probabilities must be uniform and the bundles must span the same number
of entire second stage nodes.

Notes
-----

Pickled bundles are clearly useful for algorithm tuning and algorithm
experimentation. In some, but not all, settings they can also improve
wall-clock performance for a single optimization run. The pickler
(e.g., ``bundle_pickler.py`` in the aircond example) does not use a
solver and can be run once to provide bundles to all cylinders. It can
often be assigned as many ranks as the total number of CPUs
available. Reading the bundles from a pickle file is much faster
than creating them.

The trick is that the bundles must contain entire second stage nodes
so the resulting bundles represent a two-stage problem.

