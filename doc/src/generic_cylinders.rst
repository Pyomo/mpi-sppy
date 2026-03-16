.. _generic_cylinders:

``generic_cylinders.py``
========================

The program ``mpisppy.generic_cylinders.py`` is the recommended way to
run mpi-sppy. It provides command-line access to the hub-and-spoke
system, the extensive form solver, confidence intervals, and many
other features without requiring you to write a driver program.

Your Model File (Module)
------------------------

Pyomo Models
^^^^^^^^^^^^

Pyomo modellers use ``generic_cylinders.py`` by creating a Python module that provides
certain functions. The module name is given (without the ``.py`` extension)
as the ``--module-name`` argument, and it should be the first argument.
It is needed even with ``--help``:

.. code-block:: bash

    python -m mpisppy.generic_cylinders --module-name farmer/farmer --help

The module must contain:

- ``scenario_creator`` -- see :ref:`scenario_creator`
- ``scenario_names_creator`` -- see :ref:`helper_functions`
- ``kw_creator`` -- see :ref:`helper_functions`
- ``inparser_adder`` -- see :ref:`helper_functions`
- ``scenario_denouement`` (can be ``None``) -- see :ref:`helper_functions`

Optional functions include ``_rho_setter``, ``id_fix_list_fct``,
``hub_and_spoke_dict_callback``, ``custom_writer``, and
``get_mpisppy_helper_object``. See :ref:`helper_functions` for details.

non-Pyomo Models
^^^^^^^^^^^^^^^^

To use mpi-sppy for models not written in Pyomo, scenarios and scenario tree information
can be provided in files. See :ref:`loose_integration` for more information.


Solving the Extensive Form
--------------------------

To solve the EF directly (no MPI required):

.. code-block:: bash

    python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
        --EF --EF-solver-name gurobi

.. note::
   Most command line options relevant to the EF start with ``--EF``.
   Other options are silently ignored when ``--EF`` is specified
   (one exception is ``--solution-base-name``).

Running PH with Spokes
-----------------------

To run Progressive Hedging with bound-computing spokes, use ``mpiexec`` (or ``mpirun``):

.. code-block:: bash

    mpiexec -np 3 python -m mpi4py mpisppy/generic_cylinders.py \
        --module-name farmer --num-scens 3 \
        --solver-name gurobi_persistent --max-iterations 10 \
        --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.01

.. note::
   If you are solving the EF directly, you do not need ``mpiexec``.
   Only decomposition methods (PH, APH, etc.) require MPI.

Choosing a Hub Algorithm
-------------------------

By default, the hub runs synchronous PH. Alternative hub algorithms
are selected with flags:

- (default) PH -- no flag needed
- ``--APH`` -- Asynchronous PH (see :ref:`sec-aph`)
- ``--subgradient-hub`` -- Subgradient method
- ``--fwph-hub`` -- Frank-Wolfe PH
- ``--ph-primal-hub`` -- PH primal

Choosing Spokes
----------------

Spokes provide bounds and heuristic solutions. Enable them with flags:

**Outer bound (lower bound for minimization) spokes:**

- ``--lagrangian`` -- Lagrangian relaxation bound
- ``--lagranger`` -- Lagrangian with reduced-cost fixing
- ``--fwph`` -- Frank-Wolfe PH bound
- ``--subgradient`` -- Subgradient bound
- ``--ph-dual`` -- PH dual bound
- ``--relaxed-ph`` -- Relaxed PH bound
- ``--reduced-costs`` -- Reduced costs bound

**Inner bound (upper bound for minimization) spokes:**

- ``--xhatshuffle`` -- Randomly shuffle scenario solutions
- ``--xhatxbar`` -- Use xbar as a candidate solution

See :ref:`Spokes` for details on each spoke type.

ADMM Decomposition
-------------------

``generic_cylinders`` supports ADMM-based decomposition via the ``--admm``
and ``--stoch-admm`` flags. See :ref:`generic_admm` for full details,
including model module requirements, bundling support, and examples.

Rho Settings
-------------

The penalty parameter rho is critical for PH performance.
See :ref:`rho_setting` for a full description of all rho-related options,
including ``--default-rho``, ``--sep-rho``, ``--coeff-rho``,
``--sensi-rho``, ``--grad-rho``, and adaptive rho updaters.

Extensions via Command Line
----------------------------

Some extensions can be activated directly from the command line:

- ``--fixer`` -- Fix variables that have converged
- ``--mipgaps-json <file>`` -- MIP gap schedule from a JSON file
- ``--user-defined-extensions <module>`` -- Load a custom extension module

See :ref:`Extensions` for details on available extensions.

Solution Output
----------------

To write solutions, use ``--solution-base-name``:

.. code-block:: bash

    python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
        --EF --EF-solver-name gurobi --solution-base-name farmersol

This writes nonanticipativity variable data to files with the given base
name and full scenario solutions to a directory named
``<base-name>_soldir``.

See :ref:`Output Solutions` for more details.

MMW Confidence Intervals
--------------------------

MMW confidence intervals can be computed directly via generic_cylinders
flags such as ``--mmw-num-batches``. See :ref:`MMW Confidence Intervals`
for details.

Pickled Scenarios
-----------------

The ``generic_cylinders`` program supports pickling and unpickling
scenarios. When pickling, all ranks are used for pickling, no other
processing is done and command line arguments other than
``pickle-scenarios-dir`` are ignored.

.. note::
   When unpickling, ``num_scens`` might be needed on ``cfg`` so
   ``--num-scens`` is probably needed on the command line. Consistency
   with the files in the pickle directory might not be checked.

Advanced: ``hub_and_spoke_dict_callback``
-----------------------------------------

Advanced users can directly manipulate the hub and spoke dicts
immediately before ``spin_the_wheel()`` is called. If the module (or class)
contains a function called ``hub_and_spoke_dict_callback()``, it will be
called immediately before the ``WheelSpinner`` object is created. The
``hub_dict``, ``list_of_spoke_dict``, and ``cfg`` object will be passed
to it. See ``generic_cylinders.py`` for details.

Advanced: Using a Class in the Module
--------------------------------------

If you want to have a class in the module to provide helper functions,
your module still needs to have an ``inparser_adder`` function and the
module will need to have a function called
``get_mpisppy_helper_object(cfg)`` that returns the object. It is called
by ``generic_cylinders.py`` after cfg is populated and can be used to
create a class. Note that ``inparser_adder`` cannot make use of the class
because it is called before ``get_mpisppy_helper_object``.

The class definition needs to include all helper functions other than
``inparser_adder``. The example ``examples.netdes.netdes_with_class.py``
demonstrates this (although in that particular example, there is no
advantage to doing so).

``custom_writer``
-----------------

Advanced users can write their own solution output function. If the
module contains a function called ``custom_writer()``, it will be passed
to the solution writer. Up to four functions can be specified in the
module (or the class if you are using a class):

- ``ef_root_nonants_solution_writer(file_name, representative_scenario, bundling_indicator)``
- ``ef_tree_solution_writer(directory_name, scenario_name, scenario, bundling_indicator)``
- ``first_stage_solution_writer(file_name, scenario, bundling_indicator)``
- ``tree_solution_writer(directory_name, scenario_name, scenario, bundling_indicator)``

The first two, if present, will be used for the EF and the second two
for hub-and-spoke solutions. For further information, look at the code
in ``mpisppy.generic_cylinders.py`` and in ``mpisppy.utils.sputils`` for
example functions such as ``first_stage_nonant_npy_serializer``. There is
a simple example in ``examples.netdes.netdes_with_class.py``.

.. warning::
   These functions will only be used if ``cfg.solution_base_name`` has been
   given a value by the user.

.. warning::
   Misspelled function names will not result in an error message, nor will
   they be called.

``config-file``
---------------

This specifies a text file that may contain any command line options.
Options on the command line take precedence over values set in the file.
There is an example text file in ``examples.sizes.sizes_config.txt``.
This option gets pulled in with ``cfg.popular_args`` and processed by
``cfg.parse_command_line``.
Note that required arguments such as ``num_scens`` *must* be on the
command line.

``solver-log-dir``
------------------

This specifies a directory where solver log files for *every* subproblem
solve will be written. This directory will be created for the user and
must *not* exist in advance.

``warmstart-subproblems``
--------------------------

This option causes subproblem solves to be given the previous iteration
solution as a warm-start. This is particularly important when using an
option to linearize proximal terms.

Presolve (FBBT and OBBT)
-------------------------

The ``--presolve`` option enables variable bounds tightening before
solving, which can improve solver performance and numerical stability.

**Feasibility-Based Bounds Tightening (FBBT)**: Uses constraint
propagation to tighten variable bounds. Fast and always applied when
presolve is enabled.

**Optimization-Based Bounds Tightening (OBBT)**: Uses optimization to
find the tightest possible bounds by solving auxiliary min/max problems.
More expensive but can achieve significantly tighter bounds.

.. code-block:: bash

   python -m mpisppy.generic_cylinders farmer.farmer --num-scens 3 --presolve

To enable OBBT in addition to FBBT:

.. code-block:: bash

   python -m mpisppy.generic_cylinders farmer.farmer --num-scens 3 --presolve --obbt

.. warning::
   OBBT may not be compatible with all solver interfaces, particularly
   persistent solvers.

The OBBT implementation uses Pyomo's ``obbt_analysis`` function. For more
details, see ``mpisppy.opt.presolve.py``.
