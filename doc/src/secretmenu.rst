Secret Menu Items
=================

There are many options that are not exposed in ``mpisppy.utils.config.py`` and we list
a few of them here.


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


subgradient_while_waiting
-------------------------

The Lagrangian spoke has an additional argument, `subgradient_while_waiting`,
which will compute subgradient steps while it is waiting on new W's from the
hub.
