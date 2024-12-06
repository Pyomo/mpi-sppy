.. _helper_functions:

Helper functions in the model file 
==================================

The `scenario_creator function <scenario_creator>`_ is required but some modules also need helper functions in the model file.

All these functions can be found in the example ``examples.distr.distr.py`` and 
are documented in `the admm wrapper documentation <admmWrapper.rst#sectiondatafordriver>`_

.. py:function:: kw_creator(cfg)

    Args:
        cfg (config object): specifications for the problem

    Returns:
        dict (str): the dictionary of key word arguments that is used in ``scenario_creator``


.. py:function:: scenario_names_creator(num_scens, start=0)

    Creates the name of the ``num_scens`` scenarios starting from ``start``

    Args:
        num_scens (int): number of scenarios
        start (int): starting index for the names

    Returns:
        list (str): the list of names

.. py:function:: inparser_adder(cfg)
    
    Adds arguments to the config object which are specific to the problem

    Args:
        cfg (config object): specifications for the problem given in the command line

Some modules (e.g. PH) call this function at termination
.. py:function:: scenario_denouement

    Args:
        rank (int): rank in the cylinder 

        scenario_name (str): name of the scenario

        scenario (Pyomo ConcreteModel): the instantiated model

.. Warning::
    Not all modules call the ``scenario_denouement`` function.
