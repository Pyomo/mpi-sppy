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

A Class in the module
---------------------

If you want to have a class in the module to help create return values
for functions, you will need to have an additional helper file called
``initialize(cfg)``. It is called by ``generic_cylinders.py``
after cfg is populated and can
be used to create a class. Note that the function ``inparser_adder`` cannot
make use of the class because that function is called before ``initialize``.
Here is some sudo-code for this idea:

.. code_block:: python

    my_object = None

    def inparser_adder(cfg):
        cfg.add_to_config('json_file_with_model_params',
                      ...)

    def initialize(cfg):
        global my_object
        my_object = MyClass(cfg)

    def scenario_names_creator(num_scens,...):
        return my_object.scenario_names_creator(num_scens,...)

    def scenario_creator(scen_name, ...):
        return my_object.scenario_creator(scen_name, ...)

    def scenario_denouement(rank, ...):
        return my_object.scenario_denoument(rank, ...)   

        
custom_writer
-------------

Advanced users might want to write their own solution output function. If the
module contains a function called ``custom_writer()``, it will be called
at the end of processing. If you are writing such a function, you should look
in ``generic_cylinders.py`` to see the two places it appears and the arguments
that are passed from each place (your function can test the type
of the first argument to see whence it was called or it can
examine the call stack for more information).


Assuming the first formal parameter in your function is assigned
to variable named wheel in the non-ef case, your function probably should
include something along the lines of this:

.. code_block:: python

   if wheel.spcomm.opt.cylinder_rank == 0:

so that you avoid writing the output from every rank.
