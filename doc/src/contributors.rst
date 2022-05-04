Notes for Code Contributors
===========================

Tests
^^^^^

There are some tests run by github, but before submitting a PR, you should
cd to the examples directory to look at and run ``run_all.py``. Notice that
``run_all.py`` creates some timing benchmark csv files as a side-effect.
As a general rule, you should leave them in your local examples directory
for regression testing and do not push them to the main repository.

import mpi
^^^^^^^^^^

Do not import mpi4py directly. Use

::

   from mpisppy import MPI

You will have access to `MPI.COMM_WORLD` as a result.
