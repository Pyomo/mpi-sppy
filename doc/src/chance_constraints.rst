.. _Chance Constraints:

Chance Constraints
==================

As a placeholder, mpi-sppy supports a PySP-style, sample-average-approximation
(SAA) chance constraint of the form

.. math::

   P(\text{risky constraint holds}) \ge 1 - \alpha .

As in PySP, you define a binary **indicator variable** in each scenario model
together with your own big-M constraints linking it to satisfaction; mpi-sppy
adds the single aggregating constraint that turns the per-scenario indicators
into a probabilistic guarantee.

.. warning::

   **Scope: chance constraints are supported only for the extensive-form (EF)
   solve.** A chance constraint :math:`\sum_s p_s z_s \ge 1 - \alpha` is a single
   constraint that couples a (binary) indicator variable from every scenario, so
   -- unlike CVaR -- it does not separate across scenarios and is **not**
   inherited by the PH / APH / Lagrangian / xhat decomposition cylinders. This
   matches PySP, which only solves the EF. Decomposing a chance constraint (e.g.
   Lagrangian dualization of the coupling constraint with a scalar price) is
   possible but is a substantially harder effort with an integrality duality
   gap. **If you need chance constraints under decomposition, please contact the
   mpi-sppy developers.**

The indicator convention
------------------------

For each scenario ``s`` define a binary variable ``z_s`` with

.. math::

   z_s = 1 \iff \text{the risky constraint is satisfied in scenario } s ,

and add your own big-M constraint(s) enforcing the link (so that ``z_s = 1``
forces satisfaction). mpi-sppy then adds

.. math::

   \sum_s p_s\, z_s \ \ge\ 1 - \alpha ,

i.e. ``E[z] >= 1 - alpha``: the probability mass of *satisfying* scenarios is at
least :math:`1 - \alpha`, so the violation probability is at most
:math:`\alpha`. Setting ``alpha = 0`` forces satisfaction in every scenario (a
robust constraint); a larger ``alpha`` buys a cheaper objective by letting the
worst (most expensive to satisfy) scenarios fail. The indicator must be
**binary** for the constraint to be exact; a continuous "indicator" yields a
relaxation and triggers a warning.

If the indicator variable is indexed, mpi-sppy adds one chance constraint per
index of the variable.

Command line (generic_cylinders)
--------------------------------

Use the ``--EF`` flag together with:

- ``--cc-indicator-var NAME`` -- the name of your per-scenario binary indicator
  (its presence enables the chance constraint);
- ``--cc-alpha ALPHA`` -- the allowed violation probability, ``0 <= alpha < 1``
  (default ``0.0``).

For example, the bundled capacity example builds enough capacity to meet demand
with probability at least :math:`1 - \alpha`:

.. code-block:: bash

   python -m mpisppy.generic_cylinders \
       --module-name examples/chance_constraint/cc_capacity \
       --num-scens 10 --EF --EF-solver-name gurobi \
       --cc-indicator-var served --cc-alpha 0.2

With the deterministic ramp demands in that example, the cost-minimizing
capacity is the :math:`(1-\alpha)`-quantile of demand (here, ``80``). Requesting
a chance constraint without ``--EF`` is rejected at parse time.

Programmatic use
----------------

When you build the EF yourself (via ``ExtensiveForm`` or
``sputils.create_EF``), call :func:`mpisppy.utils.chance_constraint.add_chance_constraint`
on the assembled model after construction and before solving:

.. code-block:: python

   import mpisppy.utils.sputils as sputils
   from mpisppy.utils import chance_constraint

   ef = sputils.create_EF(scenario_names, scenario_creator)
   chance_constraint.add_chance_constraint(
       ef, cc_indicator_var_name="served", cc_alpha=0.2)
   # ... now solve ef ...

.. autofunction:: mpisppy.utils.chance_constraint.add_chance_constraint
