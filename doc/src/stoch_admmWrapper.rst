.. _stoch_admmWrapper:

StochAdmmWrapper
================

.. automodule:: stoch_distr
   :noindex:
   :show-inheritance:

.. automodule:: distr_data
   :noindex:
   :show-inheritance:

Decomposition using consensus ADMM can be used to allow paral-
lelization efficiencies or for reasons related to information security. In
either case, the input data may be uncertain and we give a decomposi-
tion algorithm based on Progressive Hedging.

A technical report is available at `ool < https://optimization-online.org/2024/08/consensus-admm-under-uncertainty >`

**StochAdmmWrapper** uses progressive hedging implemented in mpi-sppy 
to solve a stochastic problem by breaking them into subproblems.

It is similar in many points to **admmWrapper**, but the key differences are the labelling of scenarios,
the definition of consensus_vars and of scenario_creator.

An example of usage is given below.

Usage
-----

The driver (in the example ``stoch_distr_admm_cylinders.py``) calls ``stoch_admmWrapper.py``,
using the model provided by the model file (in the example ``examples.stoch_distr.stoch_distr.py``).
The file ``stoch_admmWrapper.py`` returns variable probabilities that can be used in the driver to create the PH (or APH) 
object which will solve the subproblems in a parallel way, insuring that merging conditions are respected.

Labelling the scenarios
+++++++++++++++++++++++

In StochAdmmWrapper ``stochastic_scenarios`` precede the decomposition into subproblems.
These scenarios are then decomposed in each ``admm_subproblem``
to create an ``admm_stoch_subproblem_scenario``, also called extended scenario.

For instance, in the stochastic distribution example, in a ``stochastic_scenario``: ``StochasticScenario1``,
in the ``admm_subproblem``: ``Region2``, the ``admm_stoch_subproblem_scenario`` is ``ADMM_STOCH_Region2_StochasticScenario1``.

All these names are required in the driver file ``stoch_distr_admm_cylinders.py`` to create the wrapper file ``stoch_admmWrapper.py``.
The function ``split_admm_stoch_subproblem_scenario_name`` is also needed to obtain the ``admm_subproblem`` and ``stochastic_scenarios``
from the ``admm_stoch_subproblem_scenario``.

Functions needed in the driver
++++++++++++++++++++++++++++++

The driver file requires the `scenario_creator function <scenario_creator>`_ which creates the model for each scenario.

.. py:function:: scenario_creator(admm_stoch_subproblem_scenario_name)
    :no-index:

    Creates the model, which should include the consensus variables.
    However, this function shouldn't attach the consensus variables for the admm subproblems as it is done in stoch_admmWrapper. 
    Therefore, only the stochastic tree as it would be represented without the decomposition needs to be created.

    Args:
        admm_stoch_subproblem_scenario_name (str): the name of the extended scenario that will be created.

    Returns:
        Pyomo ConcreteModel: the instantiated model

The driver file also requires helper arguments that are used in mpi-sppy. They are detailed `in helper_functions <helper_functions>`_
and in the example below.
Here is a summary:

* ``scenario_creator_kwargs`` (dict[str]): key words arguments needed in ``scenario_creator``

* A function that is called at termination in some modules (e.g. PH)
    .. py:function:: scenario_denouement
        :no-index:

        Args:
            rank (int): rank in the cylinder 

            admm_stoch_subproblem_scenario_name (str): name of the extended scenario

            scenario (Pyomo ConcreteModel): the instantiated model

* ``stoch_scenario_names``, ``admm_subproblem_names`` and ``all_admm_stoch_subproblem_scenario_names`` (lists of str)


The driver also needs global information to link the subproblems.
It should be provided in ``consensus_vars``. 

Two types of consensus variables should be added: those that were already appearing
as non-anticipative variables (with their stage), and consensus variables linking the
ADMM subproblems with a stage n if the stochastic problem is orginally a n-stage problem.


``consensus_vars`` (dictionary): 
    * Keys are the subproblems 
    * Values are thelist of pairs (consensus_variable_name (str), stage (int))

.. note::

    Every variable in ``consensus_vars[subproblem]`` should also appear as a variable in the pyomo model of the subproblem.

Using the config system
+++++++++++++++++++++++

In addition to the previously presented data, the driver also requires arguments to create the PH Model and solve it. 
Some arguments may be passed to the user via config, but the cylinders need to be added.


Direct solver of the extensive form
+++++++++++++++++++++++++++++++++++
``stoch_distr_ef.py`` can be used as a verification or debugging tool for small instances.
It directly solves the extensive form using the wrapper ``scenario_creator`` from ``stoch_admmWrapper``.
It has the advantage of requiring the same arguments as ``stoch_distr_admm_cylinders`` because both solve the extensive form.

This method offers a verification for small instances, but doesn't decompose the problem.


.. note::

    ``stoch_admmWrapper`` doesn't collect yet instanciation time.

Stochastic distribution example
-------------------------------
This example consists in solving a stochastic distribution model by decomposing it into regions and
ensuring the flow balance with consensus variables in PH.

The model is stochastic because in a first stage the production at factory
nodes need to be determined, it is the originale non-anticipative variable, 
later the stochastic parameter of the production loss is known. 
Finally everything else can be determined.

To decompose it in ADMM subproblems, an extra stage is added in ``stoch_admmWrapper.py``.

``examples.stoch_distr.stoch_distr.py`` is an example of model creator in stoch_admmWrapper for a (linear) inter-region minimal cost distribution problem.
``stoch_distr_admm_cylinders.py`` is the driver.

Original data dictionaries
+++++++++++++++++++++++++++
The data is created in ``distr_data.py``. Some models are hardwired for 2, 3 and 4 regions. 
Other models are created pseudo-randomly thanks to parameters defined in ``data_params.json``.

In the example the ``inter_region_dict_creator`` (or ``scalable_inter_region_dict_creator``) creates the inter-region information.

.. autofunction:: distr_data.inter_region_dict_creator
    :no-index:

The ``region_dict_creator`` (or ``scalable_region_dict_creator``) creates the information specific to a region,
regardless of the other regions.

.. autofunction:: distr_data.region_dict_creator
    :no-index:

Adapting the data to create the model
+++++++++++++++++++++++++++++++++++++

To solve the regions independantly we must make sure that the constraints (in our example flow balance) would still be respected if the
models were merged. To impose this, consensus variables are introduced. 

In our example a new consensus variable is the flow among regions. Indeed, in each regional model we introduce the inter-region arcs 
for which either the source or target is in the region to impose the flow balance rule inside the region. But at this stage, nothing 
ensures that the flow from DC1 to DC2 represented in Region 1 is the same as the flow from DC1 to DC2 represented in Region 2.
That is why the flow ``flow["DC1DC2"]`` is a consensus variable in both regions: to ensure it is the same.

The purpose of ``examples.stoch_distr.stoch_distr.inter_arcs_adder`` is to do that.

.. autofunction:: stoch_distr.inter_arcs_adder
    :no-index:

.. note::

    In the example the cost of transport is chosen to be split equally in the region source and the region target.
    the only thing needed is that the sum of the costs is the original cost.

    We here represent the flow problem with a directed graph. If, in addition to the flow from DC1 to DC2 represented by ``flow["DC1DC2"]``,
    a flow from DC2 to DC1 were to be authorized we would also have ``flow["DC2DC1"]`` in both regions. 

Once the local_dict is created, the Pyomo model can be created thanks to ``min_cost_distr_problem``.

.. autofunction:: stoch_distr.min_cost_distr_problem
    :no-index:

Transforming data for the driver
++++++++++++++++++++++++++++++++

The driver requires elemnts given by the model.

``all_admm_stoch_subproblem_scenario_names``, ``split_admm_stoch_subproblem_scenario_name``, ``admm_subproblem_names``, ``stoch_scenario_names`` are explained above

``scenario_creator`` is created thanks to the previous functions.

.. autofunction:: stoch_distr.scenario_creator
    :no-index:

The dictionary ``scenario_creator_kwargs`` is created with

.. autofunction:: stoch_distr.kw_creator
    :no-index:

In this example the original model is a two-stage problem. Therefore it doesn't require branching factors ``BFs``.
If the original model had more than two stages, branching factors would need to be added.

The function ``inparser_adder`` requires the user to give ``num_stoch_scens`` (the number of stochastic scenarios), ``num_admm_subproblems`` (the number of regions) during the configuration.
Optional model specific config arguments are added:

    * The stochastic parameters ``spm``, ``cv`` and ``initial_seed`` for the Gaussian generation of the production loss stochastic.
    * If ``scalable`` is used in the command line, a stochastic model with each region of "size" ``mnpr`` is generated
    * ``ensure_xhat_feas`` is a boolean. If true, slacks are added to the distribution centers (with high penalty costs), so that the xhatter always finds a solution, which gives a best incumbent.

.. autofunction:: stoch_distr.inparser_adder
    :no-index:

Contrary to the other helper functions, ``consensus_vars_creator`` is specific to stoch_admmWrapper.
The function ``consensus_vars_creator`` creates the required ``consensus_vars`` dictionary.

.. autofunction:: stoch_distr.consensus_vars_creator
    :no-index:

Understanding the driver
++++++++++++++++++++++++
In the example the driver gets argument from the command line through the function ``_parse_args``

.. py:function:: _parse_args()
    :no-index:

    Gets argument from the command line and add them to a config argument. Some arguments are required.


.. note::

    Some arguments, such as ``cfg.run_async`` and all the methods creating new cylinders
    not only need to be added in the ``_parse_args()`` method, but also need to be called later in the driver.


Non local solvers
+++++++++++++++++

The file ``globalmodel.py`` and ``distr_ef.py`` are used for debugging or learning. They don't rely on ph or admm, they simply solve the
problem without decomposition.

* ``globalmodel.py``

    In ``globalmodel.py``,  ``global_dict_creator`` merges the data into a global dictionary
    thanks to the inter-region dictionary, without creating inter_region_arcs. Then model is created (as if there was 
    only one region without arcs leaving it) and solved.

    However this file depends on the structure of the problem and doesn't decompose the problems.
    Luckily in this example, the model creator is the same as in stoch_distr, because the changes for consensus-vars are neglectible.
    However, in general, this file may be difficultly adapted and inefficient.

* ``stoch_distr_ef.py``

    As presented previously solves the extensive form. 
    The arguments are the same as ``stoch_distr_admm_cylinders.py``, the method doesn't need to be adapted with the model.

