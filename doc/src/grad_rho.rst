Gradient-based rho
==================
You can find a detailed example using this code in ``examples.farmer.CI.farmer_rho_demo.py``.

Grad-rho is also supported in ``generic_cylinders.py``.

There are options in ``cfg`` to control dynamic updates:

  * `--dynamic-rho-primal-crit` and `--dynamic-rho-dual-crit` are booleans that trigger dynamic rho
  * `--dynamic-rho-primal-thresh` and `--dynamic-rho-dual-thresh` control how sensitive the trigger is.
    They have default values, so do not need to be set. See
   ``dyn_rho_base.py`` to see how the update is done if you don't like the default values.
  * `--grad-rho-multiplier` is a cummulative multiplier when rho is set or updated. 
