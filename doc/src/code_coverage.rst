.. _code_coverage:

Code Coverage
=============

Running the Full Suite with ``run_coverage.bash``
-------------------------------------------------

The top-level script ``run_coverage.bash`` runs every test phase (serial
pytest tests, MPI tests, and example-based integration tests) under
``coverage``, then combines the results into a single HTML report.

.. code-block:: bash

   bash run_coverage.bash              # defaults to cplex
   bash run_coverage.bash gurobi       # or specify your solver

The script uses ``.coveragerc`` (also at the top level) to configure source
filtering, parallelism, and report output. When it finishes, open
``htmlcov/index.html`` in a browser to explore the results.

The phases mirror the CI jobs in ``.github/workflows/test_pr_and_main.yml``,
so local coverage results should be comparable to CI.

Using ``--python-args`` for Individual Scripts
----------------------------------------------

The test launcher scripts (``run_all.py``, ``afew.py``, ``generic_tester.py``,
etc.) spawn subprocesses via ``mpiexec``, so a simple ``coverage run`` on the
launcher itself does not capture what runs inside those subprocesses. To solve
this, each launcher accepts a ``--python-args`` option that inserts extra
arguments after ``python`` in every subprocess command it builds.

For example, from the ``examples`` directory:

.. code-block:: bash

   python afew.py gurobi_persistent "" \
       --python-args="-m coverage run --parallel-mode --source=mpisppy"
   coverage combine
   coverage report
   coverage html          # optional: browsable HTML report in htmlcov/

The ``--python-args`` value is inserted between ``python`` and the remaining
arguments in each subprocess invocation. A command that would normally be:

.. code-block:: text

   mpiexec -np 3 python -u -m mpi4py farmer_cylinders.py --num-scens 3 ...

becomes:

.. code-block:: text

   mpiexec -np 3 python -u -m coverage run --parallel-mode --source=mpisppy -m mpi4py farmer_cylinders.py --num-scens 3 ...

Because ``--parallel-mode`` is used, each MPI rank writes its own
``.coverage.<hostname>.<pid>`` file. After the run, ``coverage combine``
merges them into a single ``.coverage`` database.

Scripts That Support ``--python-args``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

==============================================  ==========================================
Script                                          Location
==============================================  ==========================================
``run_all.py``                                  ``examples/``
``afew.py``                                     ``examples/``
``tryone.py``                                   ``examples/``
``generic_tester.py``                           ``examples/``
``straight_tests.py``                           ``mpisppy/tests/``
``afew_agnostic.py``                            ``mpisppy/agnostic/examples/``
==============================================  ==========================================

The ``--python-args`` flag can appear anywhere on the command line and is
stripped before the script's own positional argument parsing runs, so it does
not interfere with existing arguments like ``solver_name`` or ``mpiexec_arg``.

Tips
----

* Use ``--source=mpisppy`` (or rely on ``.coveragerc``) to limit coverage to
  the library itself and avoid instrumenting Pyomo, numpy, etc.
* The launcher scripts ``chdir`` into subdirectories before spawning
  subprocesses. When running coverage manually with ``--python-args``, use
  ``--data-file`` with an absolute path so all ``.coverage.*`` files land in
  one place, e.g.
  ``--python-args="-m coverage run --parallel-mode --data-file=/abs/path/.coverage --source=mpisppy"``.
  ``run_coverage.bash`` handles this automatically.
* For a quick smoke test, ``afew.py`` finishes in under a minute and still
  exercises the core hub-and-spoke machinery.
