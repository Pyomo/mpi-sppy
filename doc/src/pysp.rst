.. _PySP conversion:

PySP conversion
===============


Details
^^^^^^^

To use a PySP model in `mpi-sppy`, the critical step is to instantiate
a ``PySPModel`` object. The API for the class is:

.. automodule:: mpisppy.utils.pysp_model.pysp_model
   :members:
   :undoc-members:
   :show-inheritance:

Once a ``PySPModel`` object and an options dictionary have been
created, an ``ExtensiveForm`` or ``PH`` object can be created using
the ``PySPModel`` object attributes.

As reflected in the API for the ``PySPModel`` class there are many ways
to specify the reference model and scenario tree information in PySP.
A few examples are given in directory ``examples/farmer/from_pysp``.
Here are few notes about these examples:

* They are written mainly to provide code examples.

* How to form the EF is shown in, e.g. ``abstract.py``

A more advanced example utilizing the hub-and-spoke system
is available in the file ``examples/hydro/hydro_cylinders_pysp.py``
