W-based rho
===================

``mpisppy.utils.find_rho.py`` computes rho
using W as cost.

To compute W-based rho, you can call ``get_rhos_from_Ws``
after creating your config object.
You will need to add ``rho_args`` when you set up the config object.
For example with farmer:

.. code-block:: python

   import mpisppy.utils.find_rho as find_rho

   def _parse_args():
      cfg = config.Config()
      ...
      cfg.rho_args()

   def main():
      ... #after the config object is created
      if cfg.rho_setter:
         rho_setter = find_rho.Set_Rhos(cfg).rho_setter
      find_rho.get_rhos_from_Ws('farmer', cfg)

.. Note::
   This will write a rho file
   only if you include ``--rho-file``, 
   and use the rho setter only if you include ``--rho-setter``
   in your bash script.

You can find a detailed example using this code in ``examples.farmer.farmer_rho_demo.py``.


compute_rho
-----------

This function computes rhos for each scenario and variable by using the Ws
The rho values depend on both the scenario and the variable:
``compute_rhos`` returns a dictionnary with the variable names
and a list of corresponding rho values for each scenario.


rhos
----

This function computes a rho for each variable using the dictionnary
returned by ``compute_rhos``.
To do so, it uses an order statistic which you should set with ``--order-stat``.
It needs to be a float between 0 and 1.
It will write the resulting rhos in a csv file
containing each variable name and the corresponding value.

To use it you should include the following in your bash script.

.. code-block:: bash

   --whatpath #file with the Ws you want to use
   --rho-file #file where rhos will be written
   --order-stat #float between 0 and 1


rho_setter
----------

This function provides a rho setter which can be used in PH.
It will set rhos using the values written in your file.


To use it you should include the following in your bash script.

.. code-block:: bash

   --rho-setter 
   --rho-path #file with the rhos you want to use

