.. _console_scripts:

Console Scripts
===============

Installing mpi-sppy -- with ``pip install mpi-sppy`` or with
``pip install -e .`` from a clone -- puts three console scripts on your
``PATH``. They are the shortest way to run mpi-sppy: no path to a file
inside the repository, and no need to be in the directory where
mpi-sppy was cloned.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Console script
     - Equivalent module form
   * - ``mpi-sppy-generic-cylinders``
     - ``python -m mpisppy.generic_cylinders``
   * - ``mpi-sppy-mrp-generic``
     - ``python -m mpisppy.mrp_generic``
   * - ``mpi-sppy-one-sided-test``
     - ``python -m mpi_one_sided_test``

Each console script takes exactly the same command-line arguments as the
module form it replaces, so anywhere in this documentation that shows
``python -m mpisppy.generic_cylinders ...`` you may type
``mpi-sppy-generic-cylinders ...`` instead:

.. code-block:: bash

   mpi-sppy-generic-cylinders --module-name farmer --num-scens 3 \
       --EF --EF-solver-name gurobi

What each one does
------------------

``mpi-sppy-generic-cylinders``
    The main driver: command-line access to the hub-and-spoke system,
    the extensive form, confidence intervals, and much more, without
    writing a driver program. See :ref:`generic_cylinders`.

``mpi-sppy-mrp-generic``
    Sequential sampling (the Multiple Replication Procedure). See
    :ref:`Sequential Sampling Confidence Intervals`.

``mpi-sppy-one-sided-test``
    A small MPI one-sided-communication test used to check that your MPI
    installation might be suitable for mpi-sppy. It takes no arguments
    and is meant to be run under ``mpiexec``. See :ref:`Install mpi4py`.

Running in parallel
-------------------

Give the console script to ``mpiexec`` the same way you would give it any
other command:

.. code-block:: bash

   mpiexec -np 3 mpi-sppy-generic-cylinders --module-name farmer \
       --num-scens 3 --solver-name gurobi --max-iterations 10 \
       --default-rho 1 --lagrangian --xhatshuffle

A rank that dies takes the whole job with it, exactly as
``python -m mpi4py`` does: if one rank raises an uncaught exception, the
traceback is printed and ``MPI_Abort`` is called, rather than leaving the
surviving ranks blocked forever in a collective and the ``mpiexec`` job
hung. Importing mpi-sppy is what arranges this, so all of these are covered:

.. code-block:: bash

   mpiexec -np 3 mpi-sppy-generic-cylinders ...
   mpiexec -np 3 python -m mpi4py -m mpisppy.generic_cylinders ...
   mpiexec -np 3 python my_own_driver.py

The protection begins at the ``import mpisppy``, so a per-rank failure in
an earlier import of your own is not covered by the first and third forms.
The ``python -m mpi4py`` runner starts before your script does and so
covers that too; if your driver imports something of its own off a shared
filesystem before it reaches mpi-sppy, prefer it.

Only an exception you do not catch yourself ends the job; one you handle is
your own business. Serial runs re-raise normally, so tracebacks and exit
codes are unchanged. ``APH`` is the exception: it runs its iterations on a
worker thread, and a failure there can still hang the job.

If your ranks are doing genuinely independent work -- a sweep where each
rank runs its own case and never builds an mpi-sppy object across ranks --
you may not want one rank's failure to end the others:

.. code-block:: bash

   MPISPPY_NO_ABORT_HOOK=1 mpiexec -np 40 python sweep.py

A rank that raises then takes only itself down. It must be set before the
process starts, because mpi-sppy installs the hook when it is imported;
setting it from inside a running driver is too late. Do not set it for a
run with cylinders, or for a multi-rank ``--EF`` run: those ranks are in
collectives together and the job will hang instead. It says so on rank 0
when it takes effect.

Troubleshooting
---------------

``command not found`` (or, on Windows, "not recognized")
    The console scripts go into the ``bin`` (``Scripts`` on Windows)
    directory of the Python environment that ran ``pip install``, so
    they are on your ``PATH`` only while that environment is active.
    Activate it (``conda activate ...`` or ``source .../bin/activate``)
    and confirm with ``which mpi-sppy-generic-cylinders`` (``where`` on
    Windows).

Running from a clone without installing
    If you are working from a checkout and have not installed it with

    .. code-block:: bash

       pip install -e ".[mpi]"

    then the console scripts do not exist. Either run that command from
    the top of the clone, or use the file form instead, e.g.
    ``mpiexec -np 3 python -m mpi4py mpisppy/generic_cylinders.py ...``,
    also from the top of the clone.

Stale scripts after moving or renaming the clone
    An editable install records the path to the clone. If you move or
    rename the directory, re-run ``pip install -e ".[mpi]"`` from the new
    location.
