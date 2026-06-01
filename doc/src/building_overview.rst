Overview
========

If your model is written in Pyomo, you create a Python module with the
following functions. The remaining chapters in this section describe
each one in detail.

- ``scenario_creator`` -- builds a Pyomo model for one scenario (see :ref:`scenario_creator`)
- ``scenario_names_creator`` -- returns the list of scenario names (see :ref:`helper_functions`)
- ``kw_creator`` -- returns keyword arguments for the scenario creator (see :ref:`helper_functions`)
- ``inparser_adder`` -- adds problem-specific command-line arguments (see :ref:`helper_functions`)
- ``scenario_denouement`` -- called at termination (can be ``None``; see :ref:`helper_functions`)

Once you have these functions, you can use ``generic_cylinders.py``
(see :ref:`generic_cylinders`) to solve your problem using the EF or
the hub-and-spoke system. See the ``farmer`` directory in ``examples``
for a complete working example (``farmer.py`` and ``farmer_generic.bash``).

For models written in an algebraic modeling language other than Pyomo
(e.g., AMPL or GAMS), see :doc:`agnostic`. For models supplied as
SMPS-format files, see :doc:`smps`.
