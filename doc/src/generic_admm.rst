.. _generic_admm:

ADMM with ``generic_cylinders``
================================

The ``--admm`` and ``--stoch-admm`` flags allow ADMM-based decomposition
to be used with any compatible model module through
:ref:`generic_cylinders <generic_cylinders>`,
eliminating the need for a bespoke driver script per problem.

There are two modes:

- **Deterministic ADMM** (``--admm``): Decomposes a deterministic problem into
  coupled subproblems that share consensus variables. Each subproblem is treated
  as a "scenario" by mpi-sppy.

- **Stochastic ADMM** (``--stoch-admm``): Combines ADMM decomposition with
  stochastic programming. Each ADMM subproblem has its own set of stochastic
  scenarios, yielding composite "ADMM-stochastic" scenario names.

.. Note::
   ADMM uses ``variable_probability`` internally, which is incompatible with
   FWPH. If both ``--admm`` (or ``--stoch-admm``) and ``--fwph`` are specified,
   an error is raised. Proper bundles are not supported with deterministic
   ADMM (``--admm``), but are supported with stochastic ADMM
   (``--stoch-admm``); see :ref:`admm_bundling` below.


Tutorial: Running the ``distr`` Example
-----------------------------------------

The ``examples/distr/`` directory contains a distribution network problem
that is naturally decomposed by region. Each region is an ADMM subproblem
with consensus variables on the inter-region flows.

Prerequisite: mpi-sppy must be installed (``pip install -e .[mpi]``) with
a working MPI installation and a solver (e.g., cplex, gurobi, or xpress).

Running deterministic ADMM
^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the ``examples/distr/`` directory:

.. code-block:: bash

   mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py \
       --module-name distr --admm --num-scens 3 \
       --default-rho 10 --max-iterations 50 --solver-name cplex \
       --lagrangian --xhatxbar --rel-gap 0.01 --ensure-xhat-feas

Here:

- ``--module-name distr`` loads ``distr.py`` as the model module.
- ``--admm`` enables deterministic ADMM decomposition.
- ``--num-scens 3`` specifies three subproblems (regions).
- ``--lagrangian --xhatxbar`` add outer-bound and inner-bound spokes.
- ``-np 3`` is one MPI rank per cylinder (1 hub + 2 spokes).

The output will show PH iterations with bounds converging, just as with
a bespoke ADMM driver.


Running stochastic ADMM
^^^^^^^^^^^^^^^^^^^^^^^^

The ``examples/stoch_distr/`` directory extends the distribution problem
with stochastic scenarios (random production losses). From that directory:

.. code-block:: bash

   mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py \
       --module-name stoch_distr --stoch-admm \
       --num-admm-subproblems 3 --num-stoch-scens 3 \
       --default-rho 10 --max-iterations 50 --solver-name cplex \
       --lagrangian --xhatxbar --rel-gap 0.01

Here:

- ``--stoch-admm`` enables stochastic ADMM.
- ``--num-admm-subproblems 3`` specifies three ADMM subproblems (regions). These are loaded into the ``admm_subproblem_names_creator`` via the config object.
- ``--num-stoch-scens 3`` specifies three stochastic scenarios per region. These are loaded into the ``stoch_scenario_names_creator`` via the config object.
- The total number of "scenarios" seen by mpi-sppy is
  ``num_admm_subproblems * num_stoch_scens = 9``.


Model Module Interface
-----------------------

To use ``--admm`` or ``--stoch-admm`` with ``generic_cylinders``, your model
module must provide the standard functions required by ``generic_cylinders``
plus additional ADMM-specific functions.

Standard functions (always required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the same functions required by any ``generic_cylinders`` model:

- ``scenario_creator(scenario_name, **kwargs)``
- ``scenario_names_creator(num_scens)``
- ``scenario_denouement(rank, scenario_name, scenario)``
- ``kw_creator(cfg)`` — returns a dict of keyword arguments for ``scenario_creator``
- ``inparser_adder(cfg)`` — registers model-specific command-line arguments

See :ref:`scenario_creator` and :ref:`helper_functions` for details.


Additional functions for ``--admm``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: consensus_vars_creator(num_scens, all_scenario_names, **scenario_creator_kwargs)

   Creates the consensus variables dictionary.

   :param int num_scens: number of subproblems
   :param list all_scenario_names: list of all scenario (subproblem) name strings
   :param scenario_creator_kwargs: keyword arguments from ``kw_creator(cfg)``,
       passed via ``**``
   :returns: dict mapping subproblem names to lists of consensus variable
       name strings (e.g., ``{"Region1": ["flow[('DC1', 'DC2')]", ...], ...}``)

The consensus variable names must match the Pyomo variable names on the
scenario models exactly.


Additional functions for ``--stoch-admm``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: consensus_vars_creator(admm_subproblem_names, stoch_scenario_name, **scenario_creator_kwargs)

   Creates the consensus variables dictionary for stochastic ADMM.

   :param list admm_subproblem_names: list of ADMM subproblem name strings
   :param str stoch_scenario_name: name of any one stochastic scenario
       (used to inspect the model for consensus variable names)
   :param scenario_creator_kwargs: keyword arguments from ``kw_creator(cfg)``
   :returns: dict mapping subproblem names to lists of
       ``(variable_name, stage)`` tuples

.. py:function:: admm_subproblem_names_creator(cfg)

   :param cfg: config object
   :returns: list of ADMM subproblem name strings

.. py:function:: stoch_scenario_names_creator(cfg)

   :param cfg: config object
   :returns: list of stochastic scenario name strings

.. py:function:: admm_stoch_subproblem_scenario_names_creator(admm_subproblem_names, stoch_scenario_names)

   Creates the list of composite names for all (subproblem, stochastic scenario)
   pairs.  These composite names are what mpi-sppy treats as "scenario names"
   internally.  The ordering matters: all subproblems for a given stochastic
   scenario should appear consecutively so that scenarios from the same
   stochastic path are grouped together.

   :param list admm_subproblem_names: from ``admm_subproblem_names_creator``
   :param list stoch_scenario_names: from ``stoch_scenario_names_creator``
   :returns: list of composite name strings

   Example implementation (from ``stoch_distr.py``):

   .. code-block:: python

      def admm_stoch_subproblem_scenario_names_creator(
              admm_subproblem_names, stoch_scenario_names):
          return [combining_names(sub, stoch)
                  for stoch in stoch_scenario_names
                  for sub in admm_subproblem_names]

   With 2 subproblems (``Region1``, ``Region2``) and 3 stochastic scenarios,
   this produces::

      ["ADMM_STOCH_Region1_StochasticScenario1",
       "ADMM_STOCH_Region2_StochasticScenario1",
       "ADMM_STOCH_Region1_StochasticScenario2",
       "ADMM_STOCH_Region2_StochasticScenario2",
       "ADMM_STOCH_Region1_StochasticScenario3",
       "ADMM_STOCH_Region2_StochasticScenario3"]

   Note the nesting order: the outer loop is over stochastic scenarios and
   the inner loop is over subproblems.  This groups all subproblems for the
   same stochastic scenario together, which is required for correct scenario
   distribution across MPI ranks.

.. py:function:: split_admm_stoch_subproblem_scenario_name(name)

   The inverse of the combining function: given a composite name, returns
   the original subproblem name and stochastic scenario name.  This function
   must be consistent with ``combining_names`` and
   ``admm_stoch_subproblem_scenario_names_creator``.

   :param str name: a composite ADMM-stochastic scenario name
   :returns: tuple ``(admm_subproblem_name, stoch_scenario_name)``

   Example implementation (from ``stoch_distr.py``):

   .. code-block:: python

      def split_admm_stoch_subproblem_scenario_name(name):
          # name is e.g. "ADMM_STOCH_Region1_StochasticScenario1"
          parts = name.split('_')
          admm_subproblem_name = parts[2]       # "Region1"
          stoch_scenario_name = parts[3]         # "StochasticScenario1"
          return admm_subproblem_name, stoch_scenario_name

   .. Warning::
      This example relies on subproblem and scenario names not containing
      underscores.  If your names contain underscores, use a different
      delimiter or a more robust parsing strategy in both ``combining_names``
      and ``split_admm_stoch_subproblem_scenario_name``.


Creating Your Own ADMM Model
------------------------------

The easiest way to create an ADMM model for use with ``generic_cylinders`` is
to start from one of the ``distr`` examples and adapt it. The steps below
use deterministic ADMM as the starting point; stochastic ADMM follows the
same pattern with additional functions.

Step 1: Copy the template
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy ``examples/distr/distr.py`` (and ``examples/distr/distr_data.py`` if you
want to keep data in a separate file) to a new directory for your model.

Step 2: Define your subproblems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ADMM subproblem corresponds to a "scenario" in mpi-sppy. In
``scenario_names_creator``, return a list of names for your subproblems:

.. code-block:: python

   def scenario_names_creator(num_scens):
       return [f"Subproblem{i+1}" for i in range(num_scens)]

Step 3: Implement ``scenario_creator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your ``scenario_creator`` builds a Pyomo ``ConcreteModel`` for one subproblem.
The model must include all variables that appear in the consensus (coupling)
constraints.

.. code-block:: python

   def scenario_creator(scenario_name, **kwargs):
       cfg = kwargs["cfg"]
       model = build_my_model(scenario_name, cfg)
       # Attach a trivial root node (ADMM wrapper will handle consensus)
       varlist = list()
       sputils.attach_root_node(model, model.Obj, varlist)
       return model

.. Note::
   Pass an empty ``varlist`` to ``attach_root_node``. The ADMM wrapper
   automatically manages which variables are non-anticipative (the consensus
   variables).

Step 4: Implement ``consensus_vars_creator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function tells the ADMM wrapper which variables must agree across
subproblems. Return a dict mapping each subproblem name to a list of
Pyomo variable name strings:

.. code-block:: python

   def consensus_vars_creator(num_scens, all_scenario_names, **kwargs):
       consensus_vars = {}
       # Example: subproblems share a variable "x[link]"
       for name in all_scenario_names:
           consensus_vars[name] = ["x[link_A]", "x[link_B]"]
       return consensus_vars

The variable name strings must exactly match ``var.name`` as it appears on
the Pyomo model (e.g., ``"flow[('DC1', 'DC2')]"``).

Step 5: Implement ``kw_creator`` and ``inparser_adder``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``kw_creator(cfg)`` returns a dictionary that will be unpacked as keyword
arguments to both ``scenario_creator`` and ``consensus_vars_creator``. Put
any data your model needs into this dictionary:

.. code-block:: python

   def kw_creator(cfg):
       my_data = load_data(cfg)
       return {"cfg": cfg, "my_data": my_data}

   def inparser_adder(cfg):
       cfg.num_scens_required()
       cfg.add_to_config("my_param",
                         description="A model-specific parameter",
                         domain=float, default=1.0)

Step 6: Implement ``scenario_denouement``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function is called for each scenario at the end of the solve. It can
be a no-op:

.. code-block:: python

   def scenario_denouement(rank, scenario_name, scenario):
       pass

Step 7: Run
^^^^^^^^^^^^

.. code-block:: bash

   mpiexec -np 3 python -m mpi4py mpisppy/generic_cylinders.py \
       --module-name my_model --admm --num-scens 4 \
       --default-rho 1.0 --max-iterations 100 --solver-name cplex \
       --lagrangian --xhatxbar


Extending to Stochastic ADMM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To support ``--stoch-admm``, additionally implement:

1. ``admm_subproblem_names_creator(cfg)``
2. ``stoch_scenario_names_creator(cfg)``
3. ``admm_stoch_subproblem_scenario_names_creator(admm_subproblem_names, stoch_scenario_names)``
4. ``split_admm_stoch_subproblem_scenario_name(name)``

These functions define the naming convention for composite scenarios. See
``examples/stoch_distr/stoch_distr.py`` for a complete working example.

Your ``scenario_creator`` will receive composite names and must split them
to determine both which ADMM subproblem and which stochastic scenario to build.
Your ``consensus_vars_creator`` returns ``(variable_name, stage)`` tuples
instead of plain strings.


.. _admm_bundling:

Bundling with Stochastic ADMM
-------------------------------

Stochastic ADMM creates one "virtual scenario" per (subproblem, stochastic
scenario) pair.  For problems with many stochastic scenarios, this can mean
a large number of PH scenarios.  **Bundling** groups all stochastic scenarios
within the same subproblem into a single EF bundle, reducing the number of
PH scenarios to one per subproblem.  

To enable bundling, add ``--scenarios-per-bundle`` to a ``--stoch-admm`` run.
Currently, full bundling is required: ``--scenarios-per-bundle`` must equal
``--num-stoch-scens``.

.. code-block:: bash

   mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py \
       --module-name stoch_distr --stoch-admm \
       --num-admm-subproblems 2 --num-stoch-scens 4 \
       --default-rho 10 --max-iterations 50 --solver-name cplex \
       --lagrangian --scenarios-per-bundle 4 --xhatxbar

With ``--num-admm-subproblems 2`` and ``--scenarios-per-bundle 4``, PH sees
only 2 bundles (one per subproblem) instead of 8 virtual scenarios.

How it works
^^^^^^^^^^^^^

The ``AdmmBundler`` (in ``mpisppy/utils/admm_bundler.py``) creates scenarios
on-the-fly inside its ``scenario_creator``, following the same pattern as
``ProperBundler``.  For each bundle it:

1. Creates the constituent stochastic scenarios via the module's
   ``scenario_creator``.
2. Adds dummy consensus variables and computes variable probabilities
   (the same processing that ``Stoch_AdmmWrapper`` performs).
3. Builds an EF from the scenarios using
   ``nonant_for_fixed_vars=True`` so all bundles have identical nonant
   structure.
4. Flattens all consensus variables from all tree levels into a single
   ROOT node.

Because each bundle contains scenarios from only one subproblem, all
scenarios within a bundle share the same real/dummy variable pattern,
ensuring consistent PH coordination.

Model module requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^

Bundled stochastic ADMM requires two additional functions in the model
module beyond the standard ``--stoch-admm`` interface:

.. py:function:: combining_names(admm_subproblem_name, stoch_scenario_name)

   Creates a composite virtual scenario name from a subproblem and
   stochastic scenario.

   :param str admm_subproblem_name: e.g. ``"Region1"``
   :param str stoch_scenario_name: e.g. ``"StochasticScenario1"``
   :returns: str, e.g. ``"ADMM_STOCH_Region1_StochasticScenario1"``

These are the same functions used by ``Stoch_AdmmWrapper`` and are already
present in the ``stoch_distr`` example.

Limitations
^^^^^^^^^^^^

- **Full bundling only**: ``--scenarios-per-bundle`` must equal
  ``--num-stoch-scens``.  Partial bundling (where some but not all stochastic
  scenarios are grouped) is not supported because different stochastic paths
  cannot be correctly coordinated after flattening to ROOT.

- **Deterministic ADMM**: Bundling is not supported with ``--admm``
  (only with ``--stoch-admm``).

- **Inner bounds**: The ``xhatxbar`` and ``xhatshuffle`` spokes may report
  ``inf`` when used with bundles, because the bundle EF models do not have
  the same structure as individual scenarios.  The Lagrangian outer bound
  works correctly.


Reference: CLI Arguments
--------------------------

The following arguments are added by the ADMM support in ``generic_cylinders``:

==========================================  ===========  ============================================
Argument                                    Domain       Description
==========================================  ===========  ============================================
``--admm``                                  bool         Enable deterministic ADMM decomposition
``--stoch-admm``                            bool         Enable stochastic ADMM decomposition
``--scenarios-per-bundle``                  int          Bundle stochastic scenarios (stoch-admm only)
==========================================  ===========  ============================================

.. Note::
   For deterministic ADMM, the number of subproblems is given by ``--num-scens``,
   which should be registered by the model's ``inparser_adder``.