Secret Menu Items
=================

There are many options that are not exposed in ``mpisppy.utils.baseparsers.py`` and we list
a few of them here.


display_timing
--------------

This is a PH option that adds a barrier step to collect information about
the time required to solve subproblems. This can be helpful in diagnosis
and tuning of algorithms because for some problems, the variability in
time to solve scenario sub-problems can be quite large.

.. Note::
   This option should be used only when there is exactly one scenario per rank.

To set the option, use

::
   PHoptions["display_timing"] = True

E.g., if you were adding this to ``examples.farmer.farmer_cylinders`` where the
hub defintion dictionary is called ``hub_dict`` you would add

::
   hub_dict["opt_kwargs"]["PHoptions"]["display_timing"] = True

before passing it to ``spin_the_wheel``.
