Gradient-based rho
==================

``mpisppy.utils.gradient.py`` computes rho 
using the gradient and writes them in a file. 

.. Note::
   This only works for two-stage problems for now.

To compute gradient-based rho, you can call ``grad_cost_and_rho`` 
after creating your config object. 
You will need to add ``gradient_args`` when you set up the config object.
For example with farmer:

.. code-block:: python

   import mpisppy.utils.gradient as grad

   def _parse_args():
      cfg = config.Config()
      ...
      cfg.gradient_args()

   def main():
      ... #after the config object is created
      grad.grad_cost_and_rho('farmer', cfg)

.. Note::
   This will write a rho file (resp. grad cost file) 
   only if you include ``--grad-rho-file`` (resp. ``--grad-cost-file``) 
   in your bash script.

You can find a detailed example using this code in ``examples.farmer.farmer_rho_demo.py``.


find_grad
---------

This function computes the gradient of the objective function for each scenario. 
It will write the resulting gradient costs in a csv file
containing each scenario name, variable name and the corresponding value.

To use it you should include the following in your bash script.

.. code-block:: bash

   --xhatpath #path of your xhat file (.npy)
   --grad-cost-file #file where gradient costs will be written


compute_rhos
------------

This function computes rhos for each scenario and variable 
using the previously computed gradient costs.
The rho values depend on both the scenario and the variable: 
``compute_rhos`` returns a dictionnary with the variable names
and a list of corresponding rho values for each scenario.


rhos
----

This function computes a rho for each variable using the dictionnary 
returned by ``compute_rhos``. 
The method for doing so is the average for now, 
but can be edited in ``_pstat``.
It will write the resulting rhos in a csv file
containing each variable name and the corresponding value.

To use it you should include the following in your bash script.

.. code-block:: bash

   --grad-rho-file #file where gradient rhos will be written
