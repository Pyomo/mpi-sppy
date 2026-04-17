.. _rho_setting:

Rho Setting
===========

The penalty parameter rho (:math:`\rho`) is central to Progressive Hedging (PH)
and related decomposition methods. mpi-sppy provides several ways to set and
dynamically update rho values. This page consolidates all rho-related options.

Default Rho
-----------

The simplest approach is to set a single rho value for all variables:

.. code-block:: bash

   --default-rho 1.0

This value is used for any variable that does not have a rho set by another
mechanism (e.g., a rho setter function or one of the strategies below).

.. note::
   If no ``_rho_setter`` function is provided in the model module, then
   ``--default-rho`` is required unless one of ``--sep-rho``, ``--coeff-rho``,
   or ``--sensi-rho`` is specified (in which case default-rho is automatically
   set to 1 as a fallback).

Rho Setter Function
--------------------

The model module can define a ``_rho_setter`` function that returns
per-variable rho values. This function is passed to the hub constructor
and is called during setup. See :ref:`helper_functions` for details.

Separation-based Rho (``--sep-rho``)
-------------------------------------

Uses the separation between scenario solutions to set rho. Enabled with:

.. code-block:: bash

   --sep-rho

Coefficient-based Rho (``--coeff-rho``)
----------------------------------------

Sets rho based on objective function coefficients. Enabled with:

.. code-block:: bash

   --coeff-rho

Sensitivity-based Rho (``--sensi-rho``)
----------------------------------------

Sets rho based on sensitivity analysis. Enabled with:

.. code-block:: bash

   --sensi-rho

.. note::
   If existing rho values have been set (e.g., by ``--sep-rho`` or
   ``--coeff-rho``), ``--sensi-rho`` will use them as starting points.
   For this reason, ``--sensi-rho`` should typically appear after
   ``--sep-rho`` or ``--coeff-rho`` in the workflow.

Reduced-costs-based Rho (``--reduced-costs-rho``)
--------------------------------------------------

Sets rho based on reduced costs from LP relaxations. Enabled with:

.. code-block:: bash

   --reduced-costs-rho

Like ``--sensi-rho``, this will use existing rho values if available.

Gradient-based Rho (``--grad-rho``)
------------------------------------

Uses gradient information to set rho values. Enabled with:

.. code-block:: bash

   --grad-rho

A detailed example is in ``examples.farmer.CI.farmer_rho_demo.py``.

The ``--grad-rho-multiplier`` option provides a cumulative multiplier
applied when rho is set or updated.

Dynamic Rho Updates
-------------------

Rho can be updated dynamically during PH iterations using convergence
criteria:

- ``--dynamic-rho-primal-crit``: trigger updates based on primal convergence
- ``--dynamic-rho-dual-crit``: trigger updates based on dual convergence
- ``--dynamic-rho-primal-thresh``: sensitivity threshold for primal trigger
  (has a default value)
- ``--dynamic-rho-dual-thresh``: sensitivity threshold for dual trigger
  (has a default value)

See ``dyn_rho_base.py`` for implementation details on how updates are
computed.

Norm Rho Updater (``--use-norm-rho-updater``)
----------------------------------------------

An adaptive rho strategy based on the norm of primal and dual residuals.
Enabled with:

.. code-block:: bash

   --use-norm-rho-updater

There is also a corresponding converger that can be enabled with
``--use-norm-rho-converger``, which requires ``--use-norm-rho-updater``.

Primal-Dual Rho Updater (``--use-primal-dual-rho-updater``)
-------------------------------------------------------------

An adaptive rho strategy based on primal-dual balancing. Enabled with:

.. code-block:: bash

   --use-primal-dual-rho-updater

Additional options:

- ``--primal-dual-rho-update-threshold``: threshold for triggering updates
- ``--primal-dual-rho-primal-bias``: bias towards primal residual

W-based Rho
------------

A rho based on W values is supported through ``mpisppy.utils.find_rho.py``,
which can compute rho using a cost input file created by the gradient
software. If you want to use W values as the cost, you would need to
modify the code in ``wxbarwriter.py`` to write the final W values in the
right format (i.e., matching the cost output format used by the gradient
software).
