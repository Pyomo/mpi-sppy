.. _risk management:

Risk Management (CVaR)
======================

mpi-sppy supports minimizing (or maximizing) a risk-averse objective that
blends the expected cost with the Conditional Value-at-Risk (CVaR, also called
expected shortfall) of the cost:

.. math::

   \lambda \cdot \mathbb{E}[\text{Cost}] \;+\; \beta \cdot \text{CVaR}_\alpha(\text{Cost})

This is the same "weighted CVaR" formulation used by PySP, where
:math:`\alpha \in (0,1)` is the confidence level, :math:`\beta \ge 0` is the
weight on CVaR, and :math:`\lambda \ge 0` is the weight on the expectation.

How it works
------------

CVaR is added as a per-scenario model transform using the Rockafellar--Uryasev
linearization. For each scenario :math:`s` a non-negative excess variable
:math:`\delta_s` is added together with a single shared Value-at-Risk variable
:math:`\eta`:

.. math::

   \text{minimize} \quad
   \lambda \, \text{Cost}_s + \beta\, \eta
   + \frac{\beta}{1-\alpha}\, \delta_s
   \qquad \text{s.t.} \quad \delta_s \ge \text{Cost}_s - \eta,\;\; \delta_s \ge 0 .

Because :math:`\sum_s p_s = 1` and :math:`\eta` is a single first-stage
(non-anticipative) variable, the risk measure distributes over scenarios and
:math:`\eta` becomes *"just another first-stage variable."* The original
risk-neutral objective is deactivated (but left on the model, so it remains
available for separate :math:`\mathbb{E}[\text{Cost}]` reporting) and replaced
by a new active objective. As a result the extensive form (EF) and **every
cylinder** (PH/APH hub, Lagrangian and subgradient outer-bound spokes, xhat
inner-bound spokes, FWPH, ...) inherit risk aversion with no algorithm changes.

Using it from the command line
------------------------------

With ``generic_cylinders`` (or any driver that calls ``cvar_args``), add
``--cvar`` and, optionally, the weights:

.. code-block:: bash

   # EF solve of the risk-averse farmer
   python ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 \
       --EF --EF-solver-name gurobi --cvar --cvar-weight 2.0 --cvar-alpha 0.8

The flags are:

* ``--cvar`` -- apply the CVaR transform to every scenario (default off).
* ``--cvar-weight`` -- :math:`\beta`, the weight on CVaR (default ``1.0``).
* ``--cvar-alpha`` -- the confidence level :math:`\alpha`, ``0 < alpha < 1``
  (default ``0.95``).
* ``--cvar-mean-weight`` -- :math:`\lambda`, the weight on
  :math:`\mathbb{E}[\text{Cost}]` (default ``1.0``); use ``0`` for pure CVaR.

Using it programmatically
-------------------------

Wrap any ``scenario_creator`` with ``cvar_scenario_creator`` (or call
``add_cvar`` on an already-built scenario):

.. code-block:: python

   import mpisppy.utils.cvar as cvar

   risk_creator = cvar.cvar_scenario_creator(
       my_scenario_creator, cvar_weight=2.0, cvar_alpha=0.8)
   # risk_creator now has the same signature as my_scenario_creator and can be
   # passed to create_EF, the PH constructor, the cfg_vanilla factories, etc.

.. _cvar rho:

Setting rho with CVaR (important)
---------------------------------

.. warning::

   The Value-at-Risk variable :math:`\eta` typically has a **very different cost
   profile** from the model's other first-stage variables: it lives on the scale
   of the *objective* (often orders of magnitude larger than, say, an acreage or
   a production quantity). A single uniform proximal :math:`\rho` that is
   reasonable for the original variables is usually far too small for
   :math:`\eta`, so Progressive Hedging barely moves :math:`\eta` and converges
   extremely slowly (or not at all within an iteration budget).

The robust fix is to let mpi-sppy choose a per-variable :math:`\rho` from the
cost coefficients. **We recommend gradient-based rho** (``--grad-rho``), and
``--sep-rho`` is another good cost-aware option. For example, on the
three-scenario farmer with ``--cvar --cvar-weight 2.0 --cvar-alpha 0.8`` (whose
EF-CVaR optimum is ``-220700``):

.. list-table::
   :header-rows: 1

   * - rho strategy
     - result after up to 100 PH iterations
   * - ``--default-rho 1``
     - 40% gap; the bounds never close because :math:`\eta` is stuck
   * - ``--grad-rho --grad-order-stat 0.5``
     - converges to the optimum (0% gap)
   * - ``--sep-rho``
     - converges to the optimum (0% gap)

So a risk-averse PH run on the farmer looks like:

.. code-block:: bash

   mpiexec -np 3 python -m mpi4py ../../mpisppy/generic_cylinders.py \
       --module-name farmer --num-scens 3 --solver-name gurobi \
       --max-iterations 100 --default-rho 1 --grad-rho --grad-order-stat 0.5 \
       --lagrangian --xhatshuffle --rel-gap 1e-6 \
       --cvar --cvar-weight 2.0 --cvar-alpha 0.8

See :ref:`rho_setting` for the full menu of rho strategies.

Maximization
------------

Maximization objectives are supported as well. Risk aversion then applies to the
*lower* (reward) tail: the excess variable becomes non-positive and the excess
constraint mirrors (:math:`\delta_s \le \text{Cost}_s - \eta`,
:math:`\delta_s \le 0`), so the same ``--cvar`` flags maximize
:math:`\lambda\,\mathbb{E}[\text{Reward}] + \beta\,\text{CVaR}_\alpha`
of the worst-case (lowest) rewards.

Confidence intervals
--------------------

The ``zhat4xhat`` program (see :ref:`zhat introduction`) evaluates whichever
objective is active in each scenario. Because the CVaR transform leaves the
risk-averse objective active, ``zhat4xhat`` automatically evaluates the
risk-averse objective for a candidate ``xhat`` -- no extra flags are needed --
so the resulting interval reflects the risk-averse value rather than the
risk-neutral cost.

Scope and limitations
---------------------

.. admonition:: Scope: single root-stage CVaR, not time-consistent

   mpi-sppy's CVaR support uses a single Value-at-Risk variable :math:`\eta` at
   the root node and applies CVaR to the *total* (end-of-horizon) scenario cost.
   This is the same formulation as PySP. It is a single-period risk measure on
   the total cost and is **not** a nested / time-consistent multistage risk
   measure (there is no per-stage :math:`\eta` and no recursive composition
   across stages). For two-stage problems this is the usual CVaR; for multistage
   problems it measures the risk of the total cost only. **If you need a
   time-consistent (nested) risk measure, please contact the mpi-sppy
   developers.**
