Pickled Bundles
===============

At the time of this writing, pickled bundles is a little bit beyond
the bleeding edge.  The idea is that bundles are formed and then saved
as dill pickle files for rapid retrieval. The file
``aircond_cylinders.py`` in the aircond example directory provides an
example.  The latter part of the ``allways.bash`` script demonstrates
how to run it.

In the future, we plan to support this concept with higher levels of abstraction.

Pickled bundles are clearly useful for algorithm tuning and algorithm
experimentation. In some, but not all, settings they can also improve
wall-clock performance for a single optimization run. The pickler
(e.g., ``bundle_pickler.py`` in the aircond example) does not use a
solver and can be run once to provide bundles to all cylinders. It can
often be assigned as many ranks as the total number of CPUs
available. Reading the bundles from a pickle file is much faster
than creating them.
