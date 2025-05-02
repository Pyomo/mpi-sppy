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

Advanced manipulation of the hub and spokes dicts
-------------------------------------------------

Advanced users might want to directly manipulate the hub and spoke dicts
immediately before ``spin_the_wheel()`` is called. If the module (or class)
contains a function called ``hub_and_spoke_dict_callback()``, it will be called
immediately before the ``WheelSpinner`` object is created. The ``hub_dict``,
``list_of_spoke_dict``, and ``cfg`` object will be passed to it.
See ``generic_cylinders.py`` for details.

    
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

If you want to have a class in the module to provide helper functions,
your module still needs to have an ``inparse_adder`` function and the module will need
to have a function called ``get_mpisppy_helper_object(cfg)`` that returns
the object.  It is called by ``generic_cylinders.py`` after cfg is
populated and can be used to create a class. Note that the function
``inparser_adder`` cannot make use of the class because that function
is called before ``get_mpisppy_helper_object``.

The class definition needs to include all helper functions other than
``inparser_adder``.  The example ``examples.netdes.netdes_with_class.py``
demonstrates how to implement a class in the module (although in this
particular example, there is no advantage to doing that).

        
custom_writer
-------------

This is an advanced topic. 
Advanced users might want to write their own solution output function. If the
module contains a function called ``custom_writer()``, it will be passed
to the solution writer. Up to four functions can be specified in the module (or the
class if you are using a class):

   - ef_root_nonants_solution_writer(file_name, representative_scenario, bundling_indicator)
   - ef_tree_solution_writer(directory_name, scenario_name, scenario, bundling_indicator)
   - first_stage_solution_writer(file_name, scenario,bundling_indicator)
   - tree_solution_writer(directory_name, scenario_name, scenario, bundling_indicator)

The first two, if present, will be used for the EF if that is select
and the second two for hub and spoke solutions.  For further
information, look at the code in ``mpisppy.generic_cylinders.py`` to
see how these are used and in ``mpisppy.utils.sputils`` for example functions
such as ``first_stage_nonant_npy_serializer``.  There is a very simple
example function in ``examples.netdes.netdes_with_class.py''.

.. Warning::
   These functions will only be used if cfg.solution_base_name has been given a value by the user.

.. Warning::
   Misspelled function names will not result in an error message, nor will they be called, of course.

config-file
-----------

This specifies a text file that may contain any command line options.
Options on the command line take precedence over values set in the file.
There is an example text file in ``examples.sizes.sizes_config.txt``.
This option gets pulled in with with ``cfg.popular_args`` and processed by ``cfg.parse_command_line``.
Note that required arguments such as ``num_scens`` *must* be on the command line.

solver-log-dir
--------------

This specifies a directory where solver log files for *every* subproblem solve.
This directory will be created for the user and must *not* exist in advance.

warmstart-subproblems
---------------------

Loosely speaking, this option causes subproblem solves to be given the
previous iteration solution as a warm-start. This is particularly important
when using an option to lineraize proximal terms.
