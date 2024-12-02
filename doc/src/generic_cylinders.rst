.. _generic_cylinders:

`generic_cylinders.py`
======================

The program ``mpisppy.generic_cylinders.py`` provides a starting point for
creating a customized program for processing a model or perhaps it will be all
that is needed for a particular model. Not all mpi-sppy features
are implemented in this program, but enough are to provide examples and to get
started. A bash script to illustrate its use is ``examples.generic_cylinders.bash``.

For a new model, you will need to create a python file that will
be refered to as the ``module`` and/or the ``model file``.
The file must contain a :ref:`scenario_creator` and :ref:`helper_functions`.
The file name is given without the the ``.py`` extension as the
``--module-name`` on the command and it should be the first argument. It is
needed even with the ``--help`` argument, e.g.,

.. code-block:: bash
   
    python ../mpisppy/generic_cylinders.py --module-name farmer/farmer --help

.. Note::
   If you want to run cylinders, you need to use ``mpiexec``; however, if you are
   solving the EF directly, you do not need ``mpiexec``.


.. Note::
    This functionality is at the level of alpha-release.

Pickled Scenarios
-----------------

The ``generic_cylinders`` program supports pickling and unpickling
scenarios. When pickling, all ranks are used for pickling, no other
processing is done and command line arguments other than
``pickle-scenarios-dir`` are
ignored.

.. Note::
   When unpickling, `num_scens` might be needed on `cfg` so `num-scens` is
   probably needed on the command line. Consistency with the files in the
   pickle directory might not be checked by the program.

