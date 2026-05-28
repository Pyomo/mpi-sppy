.. _generic_admm:

ADMM with ``generic_cylinders``
================================

The ``--admm`` and ``--stoch-admm`` flags allow ADMM-based decomposition
to be used with any compatible model module through
:ref:`generic_cylinders <generic_cylinders>`,
eliminating the need for a custom driver script per problem.

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

.. Note::
   With ``--stoch-admm``, the ``--xhatshuffle`` spoke requires
   ``--stage2-ef-solver-name`` and an error is raised otherwise.  Without
   it, xhatshuffle would fix nonants only along the picked scenario's tree
   path, leaving the ADMM consensus variables in other stochastic outcomes
   unconstrained and producing an invalid (over-optimistic) inner bound.
   Use ``--xhatxbar`` if you want an inner bound without solving a stage-2
   EF; xhatxbar fixes nonants to the PH ``xbar``, which is itself the
   consensus value.


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
a custom ADMM driver.


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


.. _stoch_admm_branching_factors:

Branching factors with ``--stoch-admm``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::

   When using ``--stoch-admm``, the value passed to ``--branching-factors``
   describes the **original** problem's tree (i.e., the branching factors
   **before** the ADMM-stage augmentation).  The stochastic ADMM wrapper
   appends ``num_admm_subproblems`` as the final stage, then republishes
   the augmented branching factors back to the config so that downstream
   consumers (notably xhatshuffle's stage2ef path) see the correct tree
   shape automatically.

   - For a 2-stage-origin problem (e.g., ``stoch_distr``): ``--branching-factors``
     may be omitted entirely.  The wrapper infers ``[num_stoch_scens]`` from
     ``--num-stoch-scens`` and produces the augmented tree
     ``[num_stoch_scens, num_admm_subproblems]``.
   - For an N-stage-origin problem: pass the N-1 original branching factors.
     The wrapper appends ``num_admm_subproblems`` to produce an
     N-level augmented tree.

   **Semantics change (post mpi-sppy 0.13.2):** earlier versions of
   ``setup_stoch_admm`` ignored ``--branching-factors`` entirely and hard-coded
   ``BFs=None`` into ``Stoch_AdmmWrapper``.  As a result, anyone using
   ``--stoch-admm`` together with ``--xhatshuffle --stage2-ef-solver-name``
   had to hand-encode the augmented tree as
   ``--branching-factors "<num_stoch_scens> <num_admm_subproblems>"``.
   **That workaround now produces an incorrect (too deep) tree** and must be
   removed: pass only the original problem's branching factors (or omit the
   flag for 2-stage-origin problems).

A worked example using stage2ef is provided in
``examples/stoch_distr/stoch_admm_stage2ef.bash``.


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
   :no-index:

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

Naming the composite ADMM-stochastic scenarios
""""""""""""""""""""""""""""""""""""""""""""""""

The wrapper treats each (ADMM subproblem, stochastic scenario) pair as
one "scenario" with a composite name.  By default, mpi-sppy combines
the two into a single string using the delimiter ``__ADMM__`` --
``"ADMM_STOCH__ADMM__Region1__ADMM__StochasticScenario1"`` and so on
-- and decodes it back when the wrapper needs to know which ADMM
subproblem a scenario belongs to.  The defaults live in
``mpisppy.utils.stoch_admmWrapper`` as ``default_combining_names``,
``default_split_admm_stoch_subproblem_scenario_name``, and
``default_admm_stoch_subproblem_scenario_names_creator``.

If your subproblem and stochastic-scenario names do not contain the
sentinel ``__ADMM__`` substring, omit all three helpers and the
wrapper uses the defaults automatically.  Otherwise see
"Customizing the naming convention" below.

Customizing the naming convention
"""""""""""""""""""""""""""""""""

To override the defaults, provide ``combining_names`` and
``split_admm_stoch_subproblem_scenario_name`` (both, since they form
an inverse pair) on the module.  Optionally also provide
``admm_stoch_subproblem_scenario_names_creator`` to control the list
ordering.

.. py:function:: combining_names(admm_subproblem_name, stoch_scenario_name)

   Build the composite name from an ADMM subproblem name and a
   stochastic scenario name.  Pairs with
   ``split_admm_stoch_subproblem_scenario_name``.

.. py:function:: split_admm_stoch_subproblem_scenario_name(name)

   The inverse of ``combining_names``: given a composite name, return
   ``(admm_subproblem_name, stoch_scenario_name)``.  Must be defined
   together with ``combining_names`` or both omitted -- defining one
   without the other raises ``RuntimeError`` at ``setup_stoch_admm``
   time.

.. py:function:: admm_stoch_subproblem_scenario_names_creator(admm_subproblem_names, stoch_scenario_names)

   Optional.  Build the list of composite names.  If omitted, the
   wrapper uses the default (which composes your ``combining_names``,
   or the package default if you also omitted that, with the same
   nesting order shown below).

   :param list admm_subproblem_names: from ``admm_subproblem_names_creator``
   :param list stoch_scenario_names: from ``stoch_scenario_names_creator``
   :returns: list of composite name strings

   The ordering matters: all ADMM subproblems for a given stochastic
   scenario should appear consecutively, so that scenarios from the
   same stochastic path are grouped together for correct distribution
   across MPI ranks:

   .. code-block:: python

      def admm_stoch_subproblem_scenario_names_creator(
              admm_subproblem_names, stoch_scenario_names):
          return [combining_names(sub, stoch)
                  for stoch in stoch_scenario_names   # outer
                  for sub in admm_subproblem_names]    # inner

   With 2 subproblems (``Region1``, ``Region2``), 3 stochastic
   scenarios, and the default ``combining_names``, this produces::

      ["ADMM_STOCH__ADMM__Region1__ADMM__StochasticScenario1",
       "ADMM_STOCH__ADMM__Region2__ADMM__StochasticScenario1",
       "ADMM_STOCH__ADMM__Region1__ADMM__StochasticScenario2",
       "ADMM_STOCH__ADMM__Region2__ADMM__StochasticScenario2",
       "ADMM_STOCH__ADMM__Region1__ADMM__StochasticScenario3",
       "ADMM_STOCH__ADMM__Region2__ADMM__StochasticScenario3"]

   .. Note::
      A custom ``combining_names`` /
      ``split_admm_stoch_subproblem_scenario_name`` pair must agree.
      Defining ``admm_stoch_subproblem_scenario_names_creator``
      without the inverse pair is also an error -- the wrapper still
      needs the split function to decode the names you produce.


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
       return model

.. Note::
   For ``--admm``, **do not** call ``sputils.attach_root_node`` in your
   ``scenario_creator``.  ``AdmmWrapper`` builds the scenario tree itself
   (calling ``attach_root_node`` internally with the consensus variables as
   the non-anticipative list); any user-supplied node list would be
   overwritten.  For ``--stoch-admm`` the contract is different — see the
   note in "Extending to Stochastic ADMM" below.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To support ``--stoch-admm``, additionally implement:

1. ``admm_subproblem_names_creator(cfg)``
2. ``stoch_scenario_names_creator(cfg)``

The composite ADMM-stochastic scenario names are built by the wrapper
itself unless you want to customize them; see "Naming the composite
ADMM-stochastic scenarios" above for the defaults and "Customizing
the naming convention" if you need to override.  See
``examples/stoch_distr/stoch_distr.py`` for a complete working
example.

.. Note::

   ``scenario_creator`` for ``--stoch-admm`` differs from the
   deterministic case in one way:

   **It receives a composite name.**  The argument is e.g.
   ``"ADMM_STOCH__ADMM__Region1__ADMM__StochasticScenario3"`` (default
   naming) or whatever your custom ``combining_names`` produces;
   ``scenario_creator`` must decode it -- either by calling
   ``mpisppy.utils.stoch_admmWrapper.default_split_admm_stoch_subproblem_scenario_name``
   or by calling your own ``split_admm_stoch_subproblem_scenario_name``
   -- to recover the ADMM subproblem name and the stochastic scenario
   name, then build the corresponding model.

First-stage attachment via module hooks (recommended)
""""""""""""""""""""""""""""""""""""""""""""""""""""""

Under the hood, ``Stoch_AdmmWrapper`` reads the user-supplied
``_mpisppy_node_list`` and *appends* an ADMM-consensus stage to it
(whereas ``AdmmWrapper`` overwrites the node list).  The wrapper can
attach the root node for you if you provide two module-level hook
functions:

.. code-block:: python

   def first_stage_cost(scenario):
       """Original problem's first-stage cost expression."""
       return scenario.FirstStageCost

   def first_stage_varlist(scenario):
       """Original problem's first-stage variables (NOT ADMM consensus vars)."""
       return scenario._first_stage_vars   # stashed in scenario_creator

When both hooks (``first_stage_cost`` and ``first_stage_varlist``) are
defined on the module, the wrapper calls
``sputils.attach_root_node(scenario, first_stage_cost(scenario),
first_stage_varlist(scenario))`` itself for each scenario before
running its consensus-stage logic.  ``scenario_creator`` no longer
needs to call ``attach_root_node`` (and must not — see error matrix
below).

See ``examples/stoch_distr/stoch_distr.py`` for the canonical
pattern, including how to stash the varlist on the scenario from
inside ``scenario_creator`` so the hook can find it.

.. Note::
   The hooks are **both-or-neither**: defining only one raises
   ``RuntimeError`` at ``setup_stoch_admm`` time.  Mixing the hooks
   with a manual ``attach_root_node`` call also raises.

.. Note::
   ``first_stage_varlist`` may return a mix of scalar ``Var``,
   ``VarData``, and indexed ``Var`` containers.  Indexed containers are
   expanded internally to one consensus entry per ``VarData`` (e.g.
   ``NumBuilt`` becomes ``NumBuilt[2025]``, ``NumBuilt[2026]``, ...),
   so you do not need to unpack indexed Vars before returning them.

Advanced first-stage hooks (optional)
"""""""""""""""""""""""""""""""""""""""

``sputils.attach_root_node`` accepts two further optional parameters,
``surrogate_nonant_list`` and ``nonant_ef_suppl_list`` (see
:ref:`surrogate_nonant_list` and :ref:`ef_supplement_list` for what
each does), for problems that need to mark some first-stage Vars as
surrogates (EF skips their nonant equality) or as EF-supplemental
nonants (extra Vars carried through the EF construction).  If your
problem needs either, define the corresponding optional module-level
hook:

.. code-block:: python

   def first_stage_surrogate_nonant_list(scenario):
       """Optional. Forwarded to attach_root_node's surrogate_nonant_list."""
       return scenario._surrogate_nonants   # stashed in scenario_creator

   def first_stage_nonant_ef_suppl_list(scenario):
       """Optional. Forwarded to attach_root_node's nonant_ef_suppl_list."""
       return scenario._ef_suppl_nonants

Each advanced hook is independent of the other — defining either one
alone is fine — but both depend on the two core hooks
(``first_stage_cost`` and ``first_stage_varlist``) also being defined,
because there is nothing for the wrapper to attach the advanced lists
onto otherwise.  Defining an advanced hook without the core hooks
raises ``RuntimeError`` at ``setup_stoch_admm`` time.

On the legacy path (no core hooks), pass ``surrogate_nonant_list`` and
``nonant_ef_suppl_list`` directly to your own ``sputils.attach_root_node``
call inside ``scenario_creator``; the wrapper inherits whatever you
attached.

First-stage attachment via manual ``attach_root_node`` (legacy)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

If you omit both hooks, ``scenario_creator`` must itself call
``sputils.attach_root_node`` with the original problem's first-stage
cost and varlist (and ``surrogate_nonant_list`` /
``nonant_ef_suppl_list`` if you need them).  Skipping the call (when
no hooks are defined) raises ``RuntimeError`` with a message pointing
at both options.

This path is preserved for backward compatibility with model modules
written before the hooks existed (and for direct uses of
``Stoch_AdmmWrapper`` that bypass ``setup_stoch_admm``).

Consensus vars
""""""""""""""

Your ``consensus_vars_creator`` returns ``(variable_name, stage)``
tuples instead of plain strings.


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

Bundled stochastic ADMM uses the same naming helpers as the
unbundled path -- the defaults work unless the subproblem or
stochastic-scenario names contain the ``__ADMM__`` sentinel.  See
"Customizing the naming convention" above for how to override.

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
``--num-admm-subproblems``                  int          Number of ADMM subproblems (stoch-admm only)
``--num-stoch-scens``                       int          Number of stochastic scenarios (stoch-admm only)
``--scenarios-per-bundle``                  int          Bundle stochastic scenarios (stoch-admm only)
==========================================  ===========  ============================================

.. Note::
   ``--num-admm-subproblems`` and ``--num-stoch-scens`` are registered
   automatically by ``mpisppy.generic.admm.admm_args`` under
   ``generic_cylinders --stoch-admm``; the model module's
   ``inparser_adder`` does not need to re-register them.  Passing
   ``--stoch-admm`` without both flags raises a clear error from
   ``_check_admm_compatibility`` instead of crashing inside the
   model's ``admm_subproblem_names_creator``.

.. Note::
   For deterministic ADMM, the number of subproblems is given by ``--num-scens``,
   which should be registered by the model's ``inparser_adder``.