.. _EF directly:

Solving the Extensive Form
==========================

When we refer to "solving the EF" we mean passing the extensive form in its
entirety directly to a general-purpose solver (as opposed to decomposition).

The simplest way to solve the EF is via ``generic_cylinders.py`` with the
``--EF`` flag:

.. code-block:: bash

   python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
       --EF --EF-solver-name gurobi

See :ref:`generic_cylinders` for full details on EF-related command-line options.


``mpisppy.opt.ef.ExtensiveForm`` Class
---------------------------------------

For developers who need programmatic access, there is a class for the EF
that roughly matches the "look and feel" of a hub class, but does not
function as a hub.

.. automodule:: mpisppy.opt.ef
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:

      

.. _sputils.create_EF:

Other method: ``mpisppy.utils.sputils.create_EF``
-------------------------------------------------

The use of this function does not require the installation of ``mpi4py``. Its use
is illustrated in ``examples.farmer.farmer_ef.py``. Here are the
arguments to the function:

.. autofunction:: mpisppy.utils.sputils.create_EF
