.. _vss:

Value of the Stochastic Solution (VSS)
======================================

The **Value of the Stochastic Solution** measures how much is gained by
modeling the uncertainty instead of collapsing it to a single "average"
future and solving that deterministic problem. It compares two
*here-and-now* (first-stage) decisions: the one from the full stochastic
program, and the one from the **mean-value problem** (replace every random
parameter by its expected value and solve the resulting deterministic
model). VSS is the expected extra cost of using the mean-value decision
once the real uncertainty shows up. A large VSS justifies building a
stochastic model; a near-zero VSS says the deterministic shortcut was
almost as good.

.. warning::

   **Computing VSS can be expensive.** The mean-value solve is cheap, but
   the ``EEV`` term (below) re-solves the recourse problem for *every*
   scenario with the first stage fixed — a full second pass over all
   scenarios. On large or integer models, ``--vss`` can roughly **double**
   the run time. It is off by default.

The three numbers (minimization)
--------------------------------

Write the two-stage program with first-stage vector :math:`x`, random data
:math:`\xi`, and second-stage value function :math:`Q(x,\xi)`:

.. math::

   RP = \min_x \; \big\{\, c\cdot x + \mathbb{E}_\xi[\,Q(x,\xi)\,] \,\big\}.

- **RP** — the *Recourse Problem* optimum: the here-and-now solution the run
  already computes. It is **exact** from an ``--EF`` run and the
  **incumbent** (with a bracket) from a decomposition run.
- **EV** — the *mean-value problem*
  :math:`\min_x \{ c\cdot x + Q(x,\bar\xi)\}` with :math:`\bar\xi` the
  average of the data over the scenario set. Its first-stage solution is
  :math:`\bar x`.
- **EEV** — the *Expected result of the EV solution*: pin the first stage at
  :math:`\bar x` and evaluate honestly across every scenario,
  :math:`EEV = c\cdot\bar x + \mathbb{E}_\xi[\,Q(\bar x,\xi)\,]`.

Then

.. math::

   VSS = EEV - RP \;\; (\ge 0).

For a **maximization** model the sign flips: :math:`VSS = RP - EEV`
(still :math:`\ge 0`). ``mpi-sppy`` reads the model sense and reports VSS as
a non-negative "cost of using the average."

VSS is not EVPI (Expected Value of Perfect Information,
:math:`RP - WS`), which measures the value of *knowing* the future rather
than *modeling* that you do not. ``--vss`` reports VSS only.

Usage
-----

Add ``--vss`` to a ``generic_cylinders`` run. The scenario module must
define ``average_scenario_creator`` (the same contract used by
:ref:`jensens`); it builds the mean-value scenario whose first-stage
solution VSS evaluates. The report is printed at the end and does not change
the solution.

.. code-block:: bash

   cd examples/farmer
   python ../../mpisppy/generic_cylinders.py --module-name farmer \
       --num-scens 3 --EF --EF-solver-name gurobi --solver-name gurobi --vss

produces::

   ================= VSS report =================
     RP  (stochastic solution, here-and-now):        -108390   [EF, exact]
     EV  (mean-value problem objective)     :        -118600
     EEV (EV first stage over scenarios)    :        -107240
     VSS = EEV - RP (sense=min)             :           1150   (1.06% of |RP|)
   =============================================

Exact vs. bracketed RP
~~~~~~~~~~~~~~~~~~~~~~~~

From an ``--EF`` run, ``RP`` is exact. From a decomposition (cylinders) run
that did not close the optimality gap, ``RP`` is only known to lie between
the outer and inner bounds, so VSS is reported both as a conservative point
value (against the incumbent) and as a bracket::

     RP  (stochastic solution, here-and-now):        -108389   [decomposition incumbent]
         optimality bracket [outer, inner]  : [-108508, -108389]
     ...
     VSS = EEV - RP (sense=min)             :        1149.33   (1.06% of |RP|)
         VSS bracket from RP bracket        : [1149.33, 1268.39]

The width of the VSS bracket is exactly the unclosed optimality gap; tighten
the run (or use ``--EF``) for a sharper VSS.

Infeasible mean-value solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the mean-value first stage :math:`\bar x` is infeasible when fixed in
some scenario (the model lacks relatively complete recourse), then that
scenario's recourse cost is :math:`+\infty`, so ``EEV`` and ``VSS`` are
``+inf``. This is a meaningful result — the deterministic shortcut is
unusable — and the report names the offending scenarios. See
:ref:`feasible_xhat` for the related "fix a candidate, repair if needed"
concern; VSS deliberately does **not** repair :math:`\bar x`, since a
repaired point would no longer be the mean-value decision.

Limitations
-----------

This version is **two-stage only** and cannot be combined with proper
bundles, ADMM, or ``--cvar`` (each rewrites the objective or the
scenario/first-stage structure, so ``RP`` and ``EEV`` would no longer be
comparable). ``--vss`` fails fast at setup if the model or configuration is
unsupported, so a long run is never wasted.
