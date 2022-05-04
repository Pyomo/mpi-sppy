APIs
====

High-level
----------

For most developers, understanding the API for the high level classes is not
necessary except perhaps when they want to bypass the examples shared services.

EF class
^^^^^^^^

This class forms the extensive form and makes it available to send to a solver. It
also provides some methods to look at the solution.

.. automodule:: mpisppy.opt.ef
   :members:
   :undoc-members:
   :show-inheritance:



PH class
^^^^^^^^

This is the class for PH hub.

.. automodule:: mpisppy.opt.ph
   :members:
   :undoc-members:
   :show-inheritance:

lshaped class
^^^^^^^^^^^^^

This is the class for L-shaped hub.

.. automodule:: mpisppy.opt.lshaped
   :members:
   :undoc-members:
   :show-inheritance:
      


Mid-level
---------

Most developers will not want to interact directly with mid-level classes; however,
they are important to code contributors and to developers who want to create their
own extensions.


phbase.py
^^^^^^^^^

This is the base class for PH and PH-like algorithms.

.. automodule:: mpisppy.phbase
   :members:
   :undoc-members:
   :show-inheritance:

.. _`SPBase`:
      
spbase.py
^^^^^^^^^

SPBase is the base class for many algorithms.

.. automodule:: mpisppy.spbase
   :members:
   :undoc-members:
   :show-inheritance:

scenario_tree.py
^^^^^^^^^^^^^^^^

This provides services for the scenario tree.

.. automodule:: mpisppy.scenario_tree
   :members:
   :undoc-members:
   :show-inheritance:

Low Level
---------

Most developers will not need to understand the low-level classes.

spcommunicator
^^^^^^^^^^^^^^

This class handles communication between ranks.

.. automodule:: mpisppy.cylinders.spcommunicator
   :members:
   :undoc-members:
   :show-inheritance:

spoke
^^^^^

.. automodule:: mpisppy.cylinders.spoke
   :members:
   :undoc-members:
   :show-inheritance:

hub
^^^

.. automodule:: mpisppy.cylinders.hub
   :members:
   :undoc-members:
   :show-inheritance:
   
