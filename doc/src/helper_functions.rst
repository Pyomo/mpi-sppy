.. _helper_functions:

Helper Functions in the Model File
====================================

The :ref:`scenario_creator` function is required, but
``generic_cylinders.py`` (see :ref:`generic_cylinders`) also needs
several helper functions in the model file.

All these functions can be found in the example ``examples/farmer/farmer.py``. For many applications, scenario_denouement just has a pass.

Required Functions
------------------

These functions are required by ``generic_cylinders.py``:

.. py:function:: kw_creator(cfg)

    Returns keyword arguments for ``scenario_creator``.

    Args:
        cfg (config object): specifications for the problem

    Returns:
        dict (str): the dictionary of keyword arguments that is used in ``scenario_creator``


.. py:function:: scenario_names_creator(num_scens, start=0)

    Creates the name of the ``num_scens`` scenarios starting from ``start``.

    Args:
        num_scens (int): number of scenarios
        start (int): starting index for the names

    Returns:
        list (str): the list of names

.. py:function:: inparser_adder(cfg)

    Adds arguments to the config object which are specific to the problem.

    Args:
        cfg (config object): specifications for the problem given in the command line

.. py:function:: scenario_denouement(rank, scenario_name, scenario)

    Called at termination by some modules (e.g., PH).

    Args:
        rank (int): rank in the cylinder

        scenario_name (str): name of the scenario

        scenario (Pyomo ConcreteModel): the instantiated model

    .. warning::
        Not all modules call the ``scenario_denouement`` function.

Optional Functions
------------------

These functions are optional and enable additional features:

- ``_rho_setter``: Returns per-variable rho values for PH. See :ref:`rho_setting`.
- ``id_fix_list_fct``: Identifies variables to fix (used with the ``--fixer`` extension).
- ``hub_and_spoke_dict_callback``: Allows direct manipulation of hub/spoke dicts before solving. See :ref:`generic_cylinders`.
- ``custom_writer``: Custom solution output functions. See :ref:`generic_cylinders`.
- ``get_mpisppy_helper_object(cfg)``: Returns an object that provides helper functions via a class. See :ref:`generic_cylinders`.
