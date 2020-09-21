.. _PySP conversion:

PySP conversion
===============


Details
^^^^^^^

To use a PySP model in `mpi-sppy`, the critical step is to instantiate a ``PySPModel`` object. The API for the
class is:

.. automodule:: mpisppy.utils.pysp_model
   :members:
   :undoc-members:
   :show-inheritance:

Once a ``PySPModel`` object and an options dictionary have been created, an ``ExtensiveForm`` or ``PH`` object
can be created using the ``PySPModel`` object attributes.
