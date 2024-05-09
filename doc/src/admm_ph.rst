.. _admm_ph:

ADMM PH
=======

**ADMM_PH** uses progressive hedging implemented in mpi-sppy 
to solve a non-stochastic problem by breaking them into subproblems.

An example of usage is given below.

Usage
-----

The driver (in the example ``distr_admm_cylinders.py``) calls ``admm_ph.py``,
thanks to the formated data provided by the model (in the example ``distr.py``).
The file ``admm_ph.py`` returns variable probabilities that can be used in the driver to create the PH (or APH) 
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

The driver file also requires helper arguments that are used in mpi-sppy. They are detailed `in helper_functions <helper_functions>`_.
Here is a summary:

* ``scenario_creator_kwargs``(dict[str]): key words arguments needed in ``scenario_creator``
        
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
    and nonants = consensus-vars (= dummy nodes in the example)


The driver also needs global information to link the subproblems.
It should be provided in ``consensus_vars``. 

``consensus_vars`` (dictionary): 
    * Keys are the subproblems 
    * Values are the list of consensus variables

.. note::

    Every variable in ``consensus_vars[subproblem]`` should also appear as a variable of the subproblem.

Using the config system
+++++++++++++++++++++++

In addition to the previously presented data, the driver also requires arguments to
create the PH Model and solve it. These arguments are asked to the user via config.
.. TBD add external link on precedent line


Direct solver of the extensive form
+++++++++++++++++++++++++++++++++++
``distr_ef.py`` can be used as a verification or debugging tool for small instances.
It directly solves the extensive form using the wrapper ``scenario_creator`` from ``admm_ph``.
It has the advantage of requiring the same arguments as ``distr_admm_cylinders`` because both solve the extensive form.

This method offers a verification for small instances, but is slower than ``admm_ph``
as it doesn't decompose the problem.


.. note::

    ``admm_ph`` doesn't collect instanciation time.

Distribution example
--------------------
``distr.py`` is an example of model creator in admm_ph for a (linear) inter-region minimal cost distribution problem.
``distr_admm_cylinders.py`` is the driver.

Original data dictionaries
+++++++++++++++++++++++++++

In the example the ``inter_region_dict_creator`` creates the inter-region information.

.. autofunction:: distr.inter_region_dict_creator

The ``region_dict_creator`` creates the information specific to a region regardless of the other regions.

.. autofunction:: distr.region_dict_creator

Adapting the data to create the model
+++++++++++++++++++++++++++++++++++++

To solve the regions independantly we must make sure that the constraints (in our example flow balance) would still be respected if the
models were merged. To impose this, dummy nodes are introduced. 

In our example a consensus variable is the slack variable on a dummy node. \\
If ``inter-region-dict`` indicates that an arc exists from the node DC1 (distrubution center in Region 1) to DC2, then we create a dummy 
node DC1DC2 stored both in Region1's local_dict["dummy node source"] and in Region2's local_dict["dummy node target"], with a slack whose
only constraint is to be lower than the flow possible from DC1 to DC2. PH then insures that the slacks are equal.

The purpose of ``distr.dummy_nodes_generator`` is to do that.

.. autofunction:: distr.dummy_nodes_generator

.. note::

    In the example the cost of transport is chosen to be split equally in the region source and the region target.

    We here represent the flow problem with a directed graph. If a flow from DC2 to DC1 were to be authorized we would create
    a dummy node DC2DC1.

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

Contrary to the other helper functions, ``consensus_vars_creator`` is specific to admm_ph.
The function ``consensus_vars_creator`` creates the required ``consensus_vars`` dictionary.

.. autofunction:: distr.consensus_vars_creator

Constructing the driver
+++++++++++++++++++++++


If the config argument ``run_async`` is used, the method used will be asynchronous projective hedging instead of ph.

.. In the following line add link to config page
The other arguments are detailed in config. They include:
* ``config.popular_args()``: the popular arguments, note that ``default_rho`` is needed
* ``xhatxbar``, ``lagrangian``, ``ph_ob``, ``fwph`` each of them create a new cylinder
* ``two_sided_args`` these optionnal arguments include give stopping conditions: ``rel_gap``, ``abs_gap`` for
relative or absolute termination gap and ``max_stalled_iters`` for maximum iterations with no reduction in gap
* ``ph_args`` and ``aph_args`` which are specific arguments required when using PH or APH
* ``tracking_args``: if required, prints additional information from PH on the screen

.. note::

    Some methods, such as ``run_async`` and all the methods creating new cylinders
    not only need to be added in the ``_parse_args()`` method, but also need to be called later in the driver.


Non local solvers
+++++++++++++++++

The file ``globalmodel.py`` and ``distr_ef.py`` are used for debugging. They don't rely on ph or admm, they simply solve the
problem without decomposition.

* ``globalmodel.py``

    In ``globalmodel.py``,  ``global_dict_creator`` merges the data into a global dictionary
    thanks to the inter-region dictionary, without creating dummy nodes. Then model is created (as if there was 
    only one region without dummy nodes) and solved.

    However this file strongly depends on the structure of the problem and doesn't decompose the problems. It may be difficultly adpated
    and inefficient.

* ``distr_ef.py``

    As presented previously solves the extensive form. 
    The arguments are the same as ``distr_admm_cylinders.py``, the method doesn't need to be adapted with the model.

