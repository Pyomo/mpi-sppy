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
* ``--cvar-eta-lb`` / ``--cvar-eta-ub`` -- explicit lower / upper bounds for the
  Value-at-Risk variable :math:`\eta` (see :ref:`cvar eta bound`); each overrides
  the automatic bound on that side (default: unset).
* ``--cvar-eta-bound-method`` -- how to bound :math:`\eta` automatically:
  ``fbbt`` (structural, no solves; the default), ``solve`` (relax the easy side;
  for the worst-case side solve the risk-neutral extensive form and take the cost
  at that solution -- a coupled solve, only for small/medium models), or ``none``
  (see :ref:`cvar eta bound`).
* ``--cvar-eta-mipgap`` -- the mip gap for the solves under
  ``--cvar-eta-bound-method solve`` (default ``1e-4``); for the worst-case side it
  is the risk-neutral EF solve gap, so keep it tight.
* ``--cvar-eta-solve-time-limit`` -- seconds for the coupled risk-neutral EF
  solve under ``--cvar-eta-bound-method solve`` (default ``60``); if it does not
  finish in time the worst-case side is left free (``<= 0`` skips it entirely).

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

.. _cvar eta bound:

Bounding the Value-at-Risk variable eta
---------------------------------------

At the optimum the Value-at-Risk variable :math:`\eta` equals
:math:`\text{VaR}_\alpha(\text{Cost})`, which is *provably* within the range of
the cost over all scenarios: :math:`\min_s \text{Cost}_s \le \eta^\star \le
\max_s \text{Cost}_s`. mpi-sppy uses this to give :math:`\eta` -- which is
otherwise a completely free variable living on the (large) objective scale -- a
valid bound. A bounded :math:`\eta` helps some solvers and keeps the
Progressive Hedging subproblems well posed; it can also be *necessary* for the
outer-bound spokes: a Lagrangian or subgradient relaxation dualizes the
non-anticipativity of :math:`\eta`, and with :math:`\eta` free that relaxed
subproblem is easily unbounded (its solves then simply fail).

**This is on by default, with two methods** selected by
``--cvar-eta-bound-method``:

* ``fbbt`` (the default) -- for every scenario, compute the range of the cost
  expression *structurally* with feasibility-based bounds tightening (FBBT); no
  solves. FBBT uses only the cost expression and its variables' bounds, **not the
  constraints**, so it can behave poorly with **big-M** formulations: the big-M
  variable bounds are deliberately large (order :math:`M`), so FBBT returns a
  correspondingly huge (valid but useless) bound. Prefer ``solve`` there, since
  it respects the constraints and integrality.
* ``solve`` -- find the cost range by solving. The two ends are found
  differently, because they need different things:

  * The **easy side** -- "how cheap it could be" for a minimization (or how good
    for a maximization) -- only needs a valid enclosing value, so a
    **relaxation** (LP) over the feasible region is enough; this side is finite
    for any real model, sits far from the VaR, and its looseness is harmless. It
    is solved per scenario, distributed across the ranks.
  * The **worst-case side** -- "how bad it could be" (the maximum cost for a
    minimization, the minimum reward for a maximization) -- *cannot* come from
    maximizing the cost over the feasible region: that is unbounded whenever the
    model can act arbitrarily wastefully (any big-M formulation, or farmer's
    unlimited purchases). Instead we evaluate the cost at a single **feasible
    point** -- the *risk-neutral solution* :math:`x^{\mathrm{RN}}`, which
    minimizes :math:`\mathbb{E}[\text{Cost}]` -- and take
    :math:`\max_s \text{Cost}_s(x^{\mathrm{RN}})` (min for a maximization). This
    is finite and never cuts the optimum, because
    :math:`\eta^\star = \text{VaR}(x^\star) \le \text{CVaR}(x^\star) \le
    \text{CVaR}(x^{\mathrm{RN}}) \le \max_s \text{Cost}_s(x^{\mathrm{RN}})`, where
    the middle step uses :math:`\mathbb{E}(x^{\mathrm{RN}}) \le
    \mathbb{E}(x^\star)`.

  Finding :math:`x^{\mathrm{RN}}` is a **coupled** solve -- a single first-stage
  decision shared by all scenarios -- so ``solve`` builds and solves the
  risk-neutral **extensive form**. That is only tractable when the EF is, which
  is exactly when you would not need decomposition, so ``solve`` is a convenience
  for **small/medium** models where ``fbbt`` leaves the worst-case side unbounded
  but one EF solve is affordable. The solve is **time-boxed** by
  ``--cvar-eta-solve-time-limit``; if it does not reach optimality in time the
  worst-case side is left free (the timed-out solution is discarded, never used,
  so the bound can never be invalid). For a genuinely large model, use ``fbbt``
  or an explicit ``--cvar-eta-ub`` / ``--cvar-eta-lb`` instead.

The easy side reduces the per-scenario relaxation values to a global
enclosing value across *all* scenarios (an MPI reduction); the worst-case side
is already global (one EF over all scenarios). Both ends *enclose* the true
range, so the box always contains the VaR and can never cut off the optimum.

.. note::

   The bound must be **global** (across all scenarios), not per-scenario.
   :math:`\eta` is a single non-anticipative variable, so in the extensive form
   the per-scenario copies are tied together; a bound taken from one scenario's
   cost range could exclude the true VaR (which may sit in a *different*
   scenario's range) and would then cut off the optimum. The easy side is
   therefore reduced to a single global value before bounding, and the
   worst-case side is evaluated on the whole (coupled) extensive form.

**When the automatic bound does nothing.** ``fbbt`` bounds a side only where the
cost is *structurally* bounded; on models with a structurally unbounded cost
(the classic ``farmer``, whose purchases are unbounded above) it leaves that side
free. ``solve`` goes further -- it bounds the worst-case side from the
risk-neutral solution -- but only if that EF solve finishes within
``--cvar-eta-solve-time-limit``; if it times out, or with ``fbbt`` / ``none``,
that side is again free. On a side that is left free you can supply a bound *by
hand* with ``--cvar-eta-lb`` / ``--cvar-eta-ub``. Under ``solve``,
``generic_cylinders`` prints the bounds it computed and, for any side it had to
leave free, names the flag that would bound it, e.g. (here the worst-case EF
solve did not finish in time)::

   CVaR --cvar-eta-bound-method solve: the risk-neutral EF solve did not reach optimality within 60.0s (status: maxTimeLimit); the worst-case side of eta is left free.  Raise --cvar-eta-solve-time-limit or set --cvar-eta-lb/ub.
   CVaR --cvar-eta-bound-method solve computed eta bounds: lower=-167667, upper=free
     eta has no automatic upper bound and is left free; supply --cvar-eta-ub to bound it

*Worked example -- the farmer.* The farmer minimizes total cost = planting cost
:math:`+` purchase cost :math:`-` sales revenue, on 500 acres. Its cost is
unbounded above, so ``fbbt`` cannot bound the upper side; the three-scenario EF,
however, solves instantly, so ``solve`` bounds *both* sides automatically:

* **Lower (profit) side**: the LP relaxations find the most profitable plan,
  giving :math:`\eta \gtrsim -167{,}667` here.
* **Upper (worst-case) side**: the worst-scenario cost at the risk-neutral
  solution, computed from the small EF -- no hand computation needed.

.. code-block:: bash

   # small farmer: solve bounds both sides automatically
   python ../../mpisppy/generic_cylinders.py --module-name farmer --num-scens 3 \
       --EF --EF-solver-name gurobi --cvar --cvar-weight 2.0 --cvar-alpha 0.8 \
       --cvar-eta-bound-method solve

*When the EF is too big to solve in time* (it exceeds
``--cvar-eta-solve-time-limit``, or you would rather not pay for it), supply the
worst-case side by hand from what you know about the model. For the farmer: planting is at most all 500
acres in the most expensive crop, sugar beets at ``$260``/acre
:math:`\Rightarrow 500 \times 260 = 130{,}000`; at an optimum the farmer never
buys more than the cattle-feed requirement (buying more only wastes money), so
purchases cost at most :math:`200 \times 238 + 240 \times 210 = 98{,}000`; sales
revenue is non-negative. Hence :math:`\eta \le 228{,}000`; round up to
``--cvar-eta-ub 250000``. ``--cvar-eta-lb`` and ``--cvar-eta-ub`` *override* the
automatic bound on their side (so you can tighten it or fill in a free side), and
``--cvar-eta-bound-method none`` turns the automatic bound off entirely. An
explicit bound is your responsibility: set it tighter than the true VaR and you
will change the answer, exactly as any incorrect variable bound would.

Programmatically, :func:`add_cvar` and :func:`cvar_scenario_creator` accept
``eta_lb``, ``eta_ub`` and ``auto_eta_bound`` (the FBBT bound); ``SPBase``
performs the FBBT reduction automatically when each extensive-form or cylinder
object is constructed. The ``solve`` bound is available as
:func:`compute_cvar_eta_bounds_by_solve`, whose ``(lb, ub)`` you pass as
``eta_lb`` / ``eta_ub``.

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
