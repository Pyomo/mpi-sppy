.. _EF directly:

EF Directly
===========

If we are speaking carefully, we would say that any method that solves
a problem of optimization under undertainty using scenarios is solving
the extensive form (EF). Some of the methods solve it by
decomposition. When not speaking carefully, we refer to "solving the
EF" to mean "passing the EF in its entirety directly to a
general-purpose solver." There are two closely related ways
to do this in ``mpi-sppy``.


Preferred method: ``mpisppy.opt.ef.ExtensiveForm``
--------------------------------------------------

There is a class for the EF that roughly matches the "look and feel" of a hub
class, but does not function as a hub.

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
