Secret Menu Items
=================

There are many options that are not exposed in ``mpisppy.utils.config.py`` and we list
a few of them here.


display_timing
--------------

This is a PH option that adds a barrier step to collect information about
the time required to solve subproblems. This can be helpful in diagnosis
and tuning of algorithms because for some problems, the variability in
time to solve scenario sub-problems can be quite large.

.. Note::
   This option should be used only when there is exactly one subproblem per rank.

To set the option, use

.. code-block:: python
                
   PHoptions["display_timing"] = True

E.g., if you were adding this to ``examples.farmer.farmer_cylinders`` where the
hub definition dictionary is called ``hub_dict`` you would add

.. code-block:: python

   hub_dict["opt_kwargs"]["PHoptions"]["display_timing"] = True

before passing it to ``spin_the_wheel``.


initial_proximal_cut_count
--------------------------

If the `linearize_proximal_terms` option is specified (see :ref:`linearize_proximal`)
then the option 'initial_proximal_cut_count' controls
the initial number of cuts (default 2).

E.g. if you wanted to specify four cuts
in ``examples.farmer.farmer_cylinders`` where the
hub definition dictionary is called ``hub_dict`` you would add

.. code-block:: python
                
   hub_dict["opt_kwargs"]["PHoptions"]["initial_proximal_cut_count"] = 4

before passing ``hub_dict`` to ``spin_the_wheel``.
