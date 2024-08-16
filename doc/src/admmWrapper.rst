.. _admmWrapper:

AdmmWrapper
===========

.. automodule:: distr
   :noindex:
   :show-inheritance:

.. automodule:: distr_data
   :noindex:
   :show-inheritance:

**AdmmWrapper** uses progressive hedging implemented in mpi-sppy 
to solve a non-stochastic problem by breaking them into subproblems.

An example of usage is given below.

Usage
-----

The driver (in the example ``distr_admm_cylinders.py``) calls ``admmWrapper.py``,
using the model provided by the model file (in the example ``examples.distr.distr.py``).
The file ``admmWrapper.py`` returns variable probabilities that can be used in the driver to create the PH (or APH) 
object which will solve the subproblems in a parallel way, insuring that merging conditions are respected.

Data needed in the driver
+++++++++++++++++++++++++

The driver file requires the `scenario_creator function <scenario_creator>`_ which creates the model for each scenario.

.. py:function:: scenario_creator(scenario_name)

    Creates the model, which should include the consensus variables.
    However, this function should not create any tree.

    Args:
        scenario_name (str): the name of the scenario that will be created.

    Returns:
        Pyomo ConcreteModel: the instantiated model

The driver file also requires helper arguments that are used in mpi-sppy. They are detailed `in helper_functions <helper_functions>`_
and in the example below.
Here is a summary:

* ``scenario_creator_kwargs`` (dict[str]): key words arguments needed in ``scenario_creator``
        
* ``all_scenario_names`` (list of str): the subproblem names

* A function that is called at termination in some modules (e.g. PH)
    .. py:function:: scenario_denouement

        Args:
            rank (int): rank in the cylinder 

            scenario_name (str): name of the scenario

            scenario (Pyomo ConcreteModel): the instantiated model


.. note::

    Subproblems will be represented in mpi-sppy by scenarios. Consensus variables ensure the problem constraints are
    respected, they are represented in mpi-sppy by non-anticipative variables.
    The following terms are associated: subproblems = scenarios (= regions in the example), 
    and nonants = consensus-vars (=flow among regions in the example)


The driver also needs global information to link the subproblems.
It should be provided in ``consensus_vars``. 

``consensus_vars`` (dictionary): 
    * Keys are the subproblems 
    * Values are the list of consensus variables

.. note::

    Every variable in ``consensus_vars[subproblem]`` should also appear as a variable in the pyomo model of the subproblem.

Using the config system
+++++++++++++++++++++++

In addition to the previously presented data, the driver also requires arguments to create the PH Model and solve it. 
Some arguments may be passed to the user via config, but the cylinders need to be added.


Direct solver of the extensive form
+++++++++++++++++++++++++++++++++++
``distr_ef.py`` can be used as a verification or debugging tool for small instances.
It directly solves the extensive form using the wrapper ``scenario_creator`` from ``admmWrapper``.
It has the advantage of requiring the same arguments as ``distr_admm_cylinders`` because both solve the extensive form.

This method offers a verification for small instances, but is slower than ``admmWrapper``
as it doesn't decompose the problem.


.. note::

    ``admmWrapper`` doesn't collect yet instanciation time.

Distribution example
--------------------
``examples.distr.distr.py`` is an example of model creator in admmWrapper for a (linear) inter-region minimal cost distribution problem.
``distr_admm_cylinders.py`` is the driver.

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

In our example a consensus variable is the flow among regions. Indeed, in each regional model we introduce the inter-region arcs 
for which either the source or target is in the region to impose the flow balance rule inside the region. But at this stage, nothing 
ensures that the flow from DC1 to DC2 represented in Region 1 is the same as the flow from DC1 to DC2 represented in Region 2.
That is why the flow ``flow["DC1DC2"]`` is a consensus variable in both regions: to ensure it is the same.

The purpose of ``examples.distr.distr.inter_arcs_adder`` is to do that.

.. autofunction:: distr.inter_arcs_adder

.. note::

    In the example the cost of transport is chosen to be split equally in the region source and the region target.

    We here represent the flow problem with a directed graph. If, in addition to the flow from DC1 to DC2 represented by ``flow["DC1DC2"]``,
    a flow from DC2 to DC1 were to be authorized we would also have ``flow["DC2DC1"]`` in both regions. 

Once the local_dict is created, the Pyomo model can be created thanks to ``min_cost_distr_problem``.

.. autofunction:: distr.min_cost_distr_problem

.. _sectiondatafordriver:

Transforming data for the driver
++++++++++++++++++++++++++++++++

The driver requires five elements given by the model: ``all_scenario_names``, ``scenario_creator``, ``scenario_creator_kwargs``,
``inparser_adder`` and ``consensus_vars``.

``all_scenario_names`` (see above) is given by ``scenario_names_creator``

``scenario_creator`` is created thanks to the previous functions.

.. autofunction:: distr.scenario_creator

The dictionary ``scenario_creator_kwargs`` is created with

.. autofunction:: distr.kw_creator

The function ``inparser_adder`` requires the user to give ``num_scens`` (the number of regions) during the configuration.

.. autofunction:: distr.inparser_adder

Contrary to the other helper functions, ``consensus_vars_creator`` is specific to admmWrapper.
The function ``consensus_vars_creator`` creates the required ``consensus_vars`` dictionary.

.. autofunction:: distr.consensus_vars_creator

Understanding the driver
++++++++++++++++++++++++
In the example the driver gets argument from the command line through the function ``_parse_args``

.. py:function:: _parse_args()

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
    Luckily in this example, the model creator is the same as in distr, because the changes for consensus-vars are neglectible.
    However, in general, this file may be difficultly adapted and inefficient.

* ``distr_ef.py``

    As presented previously solves the extensive form. 
    The arguments are the same as ``distr_admm_cylinders.py``, the method doesn't need to be adapted with the model.

