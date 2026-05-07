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

.. note::
   ``--sep-rho`` is scheduled for deprecation; its functionality is
   expected to be subsumed into ``--grad-rho``. Instantiating ``SepRho``
   now emits a ``DeprecationWarning``. See
   `issue #673 <https://github.com/Pyomo/mpi-sppy/issues/673>`_.

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

.. note::
   ``--reduced-costs-rho`` is scheduled for deprecation; reduced-cost
   rho has not been demonstrated to be effective in practice.
   Instantiating ``ReducedCostsRho`` now emits a ``DeprecationWarning``.
   See `issue #673 <https://github.com/Pyomo/mpi-sppy/issues/673>`_.

Gradient-based Rho (``--grad-rho``)
------------------------------------

Uses per-variable gradient information from the scenario subproblems
to set rho. Enabled with:

.. code-block:: bash

   --grad-rho

A detailed example is in ``examples.farmer.CI.farmer_rho_demo.py``.
See :ref:`grad_rho` for the algorithmic description (forthcoming).

Options:

- ``--grad-rho-multiplier`` (default ``1.0``): scalar multiplier
  applied to every computed rho.
- ``--grad-order-stat`` (default ``0.5``): order statistic across
  scenarios used to combine per-scenario rho values into one. ``0``
  selects the min, ``1`` the max, ``0.5`` the mean; values in
  ``(0, 0.5)`` interpolate min-to-mean and values in ``(0.5, 1)``
  interpolate mean-to-max.
- ``--grad-rho-relative-bound`` (default ``100``): floor on the
  primal-residual denominator relative to the consensus value (xbar).
  Prevents tiny ``|x − xbar|`` from inflating rho when a variable is
  near consensus.
- ``--eval-at-xhat`` (default off): when an xhat-producing spoke is
  present (e.g. ``--xhatshuffle``), evaluate the gradient at the best
  xhat seen so far instead of at the latest subproblem iterate values.
  Falls back to the iterate values until the first xhat arrives.
- ``--indep-denom`` (default off): use a single
  probability-weighted, MPI-Allreduced denominator across all
  scenarios instead of a per-scenario denominator.

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
modify the code in ``mpisppy/utils/w_utils/wxbarwriter.py`` to write the final W values in the
right format (i.e., matching the cost output format used by the gradient
software).
