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

Standard Formats (SMPS, MPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an exception to the ``--module-name`` requirement, the following flags
can be used as the first argument and the appropriate module will be
inferred automatically:

- ``--smps-dir`` — uses ``mpisppy.problem_io.smps_module`` (see :doc:`smps`)
- ``--mps-files-directory`` — uses ``mpisppy.problem_io.mps_module`` (see :ref:`loose_integration`)

For example:

.. code-block:: bash

    python -m mpisppy.generic_cylinders --smps-dir examples/sizes/SMPS \
        --solver-name cplex --EF

Using ``--module-name`` together with ``--smps-dir`` or
``--mps-files-directory`` is an error.


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

Multistage Options
-------------------

For multistage problems (three or more stages), the model's
``inparser_adder`` should register ``branching_factors`` via
``cfg.multistage()`` or ``cfg.add_branching_factors()``.

``--stage2-ef-solver-name``
  Solver to use for forming second-stage EFs during xhat evaluation.
  When set, the ``xhatshuffle`` spoke forms an EF for each second-stage
  node, fixes the first-stage nonants, and solves.  The number of ranks
  allocated to the xhatshuffle spoke must be an integer multiple of the
  number of second-stage nodes.

Example for the hydro model (three stages, branching factors 3 3):

.. code-block:: bash

    mpiexec -np 3 python -m mpi4py mpisppy/generic_cylinders.py \
        --module-name hydro --solver-name cplex --max-iterations 100 \
        --default-rho 1 --lagrangian --xhatshuffle --rel-gap 0.001 \
        --branching-factors "3 3" --stage2-ef-solver-name cplex

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
- ``--wtracker`` -- Track W (Lagrange-multiplier) values per iteration
  and write a convergence report at the end of the run
  (see :ref:`wtracker_extension`)

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

Pickled Scenarios and Bundles
-----------------------------

The ``generic_cylinders`` program supports pickling and unpickling
scenarios and proper bundles, optionally with presolve, a user-supplied
cleanup callback, and an iteration 0 solve baked into the pickle. See
:ref:`pickling` for the full workflow, including the
:ref:`tuning workflow <pickling_tuning_workflow>` and the
:ref:`single-run case <pickling_iter0_parallelism>` where pickling iter0
can be faster than not pickling because all ranks are available during
the pickling phase.

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

``solver-options``
------------------

mpi-sppy passes solver-specific options to the underlying Pyomo
solver plugin via a whitespace-separated string of ``key=value``
pairs. The global flag is ``--solver-options``; every spoke also
takes a per-spoke variant — ``--lagrangian-solver-options``,
``--reduced-costs-solver-options``, ``--subgradient-solver-options``,
and so on — that overlays on top of the global flag for that
spoke's solves.

Example:

.. code-block:: bash

    --solver-options "presolve=2 threads=4"
    --lagrangian-solver-options "mipgap=0.01"

With this invocation, the lagrangian spoke's effective solver
options are ``{presolve=2, threads=4, mipgap=0.01}``: the spoke
flag adds ``mipgap`` and leaves the global ``presolve`` and
``threads`` in place. The hub and the other spokes see the
global dict ``{presolve=2, threads=4}`` unchanged.

.. warning::

   Behavior change in 2026: per-spoke solver-options flags
   **overlay** the global ``--solver-options`` dict; previously
   they **replaced** it for that spoke. In the unlikely event
   you relied on the spoke flag dropping a global key, the
   recipe is to re-spell every key you want in the spoke
   options, or to omit the global ``--solver-options`` flag
   entirely. The new behavior
   is a superset of the old in every case where the spoke flag's
   keys are a subset of the global's, which is the common
   pattern (spoke flag tightens ``mipgap``, leaves the rest
   alone).

Two option names — ``mipgap`` and ``threads`` — are translated
to each solver's native spelling at solve time, so the same CLI
invocation works whether the configured solver is CPLEX, Gurobi,
Xpress, or HiGHS. Other option keys pass through to the solver
unchanged. If you supply a solver-native key directly (e.g.
``--solver-options "mip_rel_gap=0.01"`` for HiGHS), it wins over
any ``mipgap`` set elsewhere.

For iteration-aware mipgap, use ``--iter0-mipgap`` and
``--iterk-mipgap`` (plus their per-spoke variants), or
``--mipgaps-json <path>`` for a mipgap-only schedule.
``--max-solver-threads`` sets a system-level thread cap that wins
over any inline ``threads`` value; use it on shared HPC nodes.

For solver logging, see ``--solver-log-dir`` below — do not try
to enable solver logs through ``--solver-options``.

Solver-options file
^^^^^^^^^^^^^^^^^^^

For richer configurations, ``--solver-options-file <path>`` reads
a JSON file with per-iteration and per-spoke sub-blocks. Schema:

.. code-block:: json

    {
      "default":    {"threads": 4, "presolve": 2},
      "iter0":      {"mipgap": 1e-4},
      "iterk":      {"mipgap": 1e-3},
      "starting_at_iter": {"5": {"mipgap": 1e-5}, "10": {"mipgap": 1e-6}},
      "spokes": {
        "lagrangian": {
          "default":    {"mipgap": 0.01},
          "starting_at_iter": {"5": {"mipgap": 0.001}}
        },
        "reduced_costs": {"iter0": {"mipgap": 0.001}}
      }
    }

Sub-blocks behave per their names:

* ``default``    — applies to every iteration.
* ``iter0``      — only at iteration 0.
* ``iterk``      — at iteration 1 and beyond.
* ``starting_at_iter`` — keyed by iteration number ``N`` (with
  ``N >= 1``); applies from iteration ``N`` onward (so ``"5"``
  first fires at ``k = 5`` and persists until a later
  ``starting_at_iter`` entry overrides it). For options that
  should apply at every iteration, use the ``default`` sub-block
  instead — ``starting_at_iter`` with ``N = 0`` is rejected
  because it would silently outrank ``iter0`` and ``iterk``.
* ``spokes``     — per-spoke overrides keyed by spoke name. Each
  spoke sub-block has the same shape minus ``spokes``.

Per-spoke files (``--lagrangian-solver-options-file <path>``, etc.)
are also accepted; they apply only to that spoke and have the same
shape as a global file (the ``spokes`` sub-block, if present, is
ignored).

Precedence at the same iteration / predicate, lowest to highest:

1. ``--solver-options-file`` entries.
2. Inline ``--solver-options`` (``default`` predicate only).
3. ``--mipgaps-json`` (``starting_at_iter`` only — planned for
   deprecation; superseded by ``--solver-options-file``).
4. ``--iter0-mipgap`` / ``--iterk-mipgap`` / ``--max-solver-threads``
   (CLI sugar).

More specific predicates always win for any iteration that matches
both: at ``k = 7``, an ``starting_at_iter: {"5": …}`` entry overrides
any ``iterk`` entry, even though both apply.

``solver-log-dir``
------------------

This specifies a directory where solver log files for *every* subproblem
solve will be written. This directory will be created for the user and
must *not* exist in advance. File names disambiguate scenario and rank,
so concurrent solves do not clobber one another.

If the per-solve log volume is too high, add
``--hub-only-solver-logs`` to write logs only for hub-side solves
(spoke-side subproblem solves are skipped). ``--hub-only-solver-logs``
requires ``--solver-log-dir`` to also be set.

.. warning::

   Do **not** try to enable solver logging by passing a solver-specific
   log flag through ``--solver-options`` (e.g.
   ``--solver-options "logfile=run.log"`` for CPLEX, or the equivalent
   ``LogFile`` / ``log_file`` for Gurobi / HiGHS). Because every
   subproblem solve across every rank would share that one path, the
   file overwrites itself or interleaves output from concurrent solves;
   the resulting log is rarely useful. Use ``--solver-log-dir`` (with
   ``--hub-only-solver-logs`` if needed) instead.

``warmstart-subproblems``
--------------------------

This option causes subproblem solves to be given the previous iteration
solution as a warm-start. This is particularly important when using an
option to linearize proximal terms.

``turn-off-names-check``
--------------------------

By default, mpi-sppy verifies that all scenarios list the same
non-anticipative variables in the same order.  This catches a common bug
where a ``scenario_creator`` provides variables in an inconsistent order
across scenarios.  Use ``--turn-off-names-check`` to disable this
validation.

.. note::
   This check is automatically disabled when proper bundles are used,
   because bundled scenarios are EFs that have different names across bundles.

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
