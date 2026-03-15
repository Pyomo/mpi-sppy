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

      

EF Extensions
-------------

The ``ExtensiveForm`` class supports extensions via the ``extensions`` and
``extension_kwargs`` constructor arguments. EF extensions inherit from
``mpisppy.extensions.extension.EFExtension`` and can override two hooks:

- ``pre_solve()``: called after EF creation, before passing the model to the solver.
- ``post_solve(results)``: called after the solver returns; must return results.

When using ``generic_cylinders.py``, extensions are injected by defining an
``ef_dict_callback(ef_dict, cfg)`` function in the model module. This callback
can modify ``ef_dict`` to add extensions before the EF is solved. Use
``cfg_vanilla.ef_extension_adder(ef_dict, ext_class)`` to add extension classes
(multiple extensions are handled automatically via ``EFMultiExtension``).

See ``examples/farmer/farmer_ef_ext.py`` for a complete example that adds
a minimum expected wheat production constraint to the EF.

.. _sputils.create_EF:

Other method: ``mpisppy.utils.sputils.create_EF``
-------------------------------------------------

The use of this function does not require the installation of ``mpi4py``. Its use
is illustrated in ``examples.farmer.farmer_ef.py``. Here are the
arguments to the function:

.. autofunction:: mpisppy.utils.sputils.create_EF
