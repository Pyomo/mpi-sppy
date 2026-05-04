.. _feasible_xhat:

Per-scenario-feasible candidate xhat
====================================

.. warning::

   This is an advanced topic. Most ``mpi-sppy`` users do **not** need
   ``feasible_xhat_creator``. Reach for it only when downstream code
   pins a candidate first-stage point and solves a per-scenario
   subproblem with **no fallback** — for example, the pin-dual
   Lagrangian inner step in a rho-setting algorithm. Ordinary xhat
   evaluation (including the Jensen's xhat path; see :ref:`jensens`)
   tolerates per-scenario infeasibility by silently dropping the
   candidate and continuing, and that is usually what you want.

The contract
------------

A scenario module that participates exposes:

.. code-block:: python

   def feasible_xhat_creator(*, solver_name,
                             solver_options=None,
                             **scenario_creator_kwargs):
       """Return {nodename: np.ndarray} -- a candidate first-stage
       point that is feasible to fix in every real scenario's
       per-scenario subproblem. Two-stage: only "ROOT" is populated."""

Discovery is via ``getattr(module, "feasible_xhat_creator", None)``,
parallel to ``average_scenario_creator`` (see :ref:`jensens`). The
returned dict is in the cache form consumed by
``Xhat_Eval._fix_nonants``; the ``np.ndarray`` is in
``_mpisppy_node_list[0].nonant_vardata_list`` order.

Two-stage only, for now. Multi-stage extensions would need a candidate
per non-leaf node, plus a story for inter-stage feasibility coupling
that does not exist in the two-stage case.

Why this convention exists
--------------------------

The Jensen's xhat path (see :ref:`jensens`) already exposes the
average-scenario solution as a candidate first-stage. It tolerates
infeasibility: if the candidate cannot be pinned in some real
scenario, ``_jensens_evaluate_xhat`` returns ``None`` and the spoke
silently moves on. That is the right behavior for an inner-bound
spoke whose only job is to opportunistically improve a bound.

Pin-dual algorithms are different. They pin the candidate, solve a
per-scenario MIP (or LP) at the pin, and use the resulting dual
information to drive the outer algorithm. If the pin is infeasible in
some scenario, the per-scenario subproblem has no solution and there
is no graceful fallback at that step. The candidate **must** be
feasible to pin in every real scenario.

That is a strictly stronger contract than "Jensen's xhat plus luck,"
and it is the contract that ``feasible_xhat_creator`` provides.

The two helpers in ``mpisppy.utils.xhat_helpers``
-------------------------------------------------

``feasible_xhat_creator`` implementations are short. ``mpi-sppy``
ships two reusable engines that do the heavy lifting; the
implementations call one of them and apply a model-specific repair.

``average_xhat_nonants(average_scenario_creator, *, solver_name, ...)``
   Builds the model returned by ``average_scenario_creator``,
   optionally LP-relaxes it, solves it, and returns the ROOT first-stage
   values as ``np.ndarray``. One deterministic solve over the average
   data.

``lp_xbar_nonants(scenario_creator, scenario_names, *, solver_name, ...)``
   For each real scenario, builds the model, applies
   ``core.relax_integer_vars``, solves, and returns the
   probability-weighted average of ROOT first-stage values across
   scenarios. ``K`` LP solves, where ``K`` is the number of scenarios.

These two are **not interchangeable** for models with binary first-stage:
averaging data and averaging solutions do not commute when the optimal
first-stage is not a continuous function of the data. The averaged-data
problem can omit first-stage activity that some real scenario individually
needs; the per-scenario LP-xbar instead carries any activity that any
scenario's LP wanted positive into the average, where a feasibility-
preserving rounding rule can promote it. For models with continuous
first-stage, the distinction collapses.

Choosing between the two engines is the caller's responsibility. The
caller knows whether averaging data preserves enough information to
cover per-scenario feasibility; the framework cannot detect that from
the model.

The rounding rule is also yours
-------------------------------

The output of either engine is a real-valued vector that has to be
turned into a feasible candidate. Whether the right rule is
``np.ceil``, ``np.floor``, ``np.round``, identity, or a per-component
try-and-check is a model-specific decision that depends on
**monotonicity of recourse feasibility in each first-stage variable**:

* If raising :math:`x_e` from 0 to 1 only loosens recourse
  constraints (as for "open the arc" binaries in netdes), ``np.ceil``
  is feasibility-preserving.
* If the variable indexes a covering decision (open the facility) and
  more open never tightens recourse, ``np.round`` typically suffices.
* If recourse feasibility is non-monotone in the variable, neither
  rule is safe and the implementation must do something model-specific
  (a proof-of-feasibility per-component repair, an aggregation across
  scenarios, etc.).

``mpi-sppy`` does **not** ship an automatic rounder. Even within a
single model, different first-stage variables can need different
rules; per-component try-and-check degenerates into solving an
MIP-feasibility problem in itself. The repair belongs in the
``feasible_xhat_creator``, where the model author has the domain
knowledge.

File layout
-----------

By convention these helpers live in
``examples/<model>/<model>_auxiliary.py``, **not** in the main
example file. The introductory example (``farmer.py``, ``netdes.py``,
...) stays focused on what a first-time reader needs:
``scenario_creator``, ``scenario_names_creator``, ``kw_creator``,
``inparser_adder``, ``sample_tree_scen_creator``,
``scenario_denouement``. Auxiliary functions used only by advanced
machinery go in the ``_auxiliary`` sibling.

.. admonition:: Discovery
   :class: note

   When one of the xhat-spoke flags below is set,
   ``cfg_vanilla._find_feasible_xhat_creator`` first tries
   ``getattr(scenario_module, "feasible_xhat_creator", None)`` on the
   user's main scenario module; if that is ``None`` it imports
   ``<module_name>_auxiliary`` and looks there. Discovery is gated on
   the flag, so the auxiliary import does not happen unless requested.
   Downstream consumers (e.g. findW) that bypass the cylinder system
   import the auxiliary module directly and call
   ``feasible_xhat_creator`` themselves.

In-cylinder use: ``--<xhatter>-try-feasible-xhat-first`` flags
--------------------------------------------------------------

The four xhat spokes that ship with ``mpi-sppy`` accept a
``feasible_xhat_creator`` candidate via a per-spoke flag, parallel to
``--*-try-jensens-first``:

* ``--xhatshuffle-try-feasible-xhat-first``
* ``--xhatxbar-try-feasible-xhat-first``
* ``--xhatlooper-try-feasible-xhat-first``
* ``--xhatspecific-try-feasible-xhat-first``

When set, the spoke calls the module's ``feasible_xhat_creator`` once
before entering its main loop, fixes the candidate as the first-stage
nonants, evaluates the expected objective across all real scenarios,
and -- if the evaluation is feasible -- sends that as its first inner
bound. Implementation lives in
``_JensensMixin._try_feasible_xhat`` in
``mpisppy/cylinders/_jensens_mixin.py``; the spoke ``main()`` methods
call it once after ``_try_average_scenario_xhat``.

.. admonition:: Mutually exclusive with ``--*-try-jensens-first``
   :class: warning

   ``--<xhatter>-try-jensens-first`` and
   ``--<xhatter>-try-feasible-xhat-first`` are mutually exclusive on
   the same spoke. ``cfg_vanilla._maybe_attach_feasible_xhat`` raises
   at spoke-setup time if both are enabled, with a message naming the
   conflicting CLI options.

   The two pre-loop candidates serve overlapping purposes: Jensen's
   often gives a tighter incumbent bound when its candidate happens to
   be feasible everywhere, while ``feasible_xhat_creator`` is
   guaranteed feasible by contract but can be a looser incumbent. Per
   spoke, pick whichever fits the model's structure -- not both.

   Across spokes, mixing is fine: one xhat spoke can be configured
   with ``--xhatshuffle-try-jensens-first`` while another runs with
   ``--xhatxbar-try-feasible-xhat-first``.

Worked example: farmer (continuous first-stage)
-----------------------------------------------

Farmer's first-stage variable ``DEVOTED_ACRES`` is bounded
``NonNegativeReals``, and farmer has relatively complete recourse via
the buy/sell variables (``QuantityPurchased``,
``QuantitySubQuotaSold``, ``QuantitySuperQuotaSold``), so any feasible
acreage allocation -- including the average-scenario optimum -- is
feasible to pin in every real scenario. No rounding is needed.

``examples/farmer/farmer_auxiliary.py``:

.. code-block:: python

   from mpisppy.utils.xhat_helpers import average_xhat_nonants
   from farmer import average_scenario_creator


   def feasible_xhat_creator(*, solver_name, solver_options=None,
                             **scenario_creator_kwargs):
       arr = average_xhat_nonants(
           average_scenario_creator,
           solver_name=solver_name,
           scenario_creator_kwargs=scenario_creator_kwargs,
           solver_options=solver_options,
       )
       return {"ROOT": arr}

This is the simplest case the convention has to handle, and it
illustrates an important point about the convention: callers always
go through ``feasible_xhat_creator`` rather than calling
``average_xhat_nonants`` directly. If a downstream model swap replaces
farmer with a binary-first-stage model, only the auxiliary file has
to change; the call site at the consumer (e.g., findW) is unchanged.

Worked example: netdes (binary, arc-open monotonicity)
------------------------------------------------------

Netdes ``model.x[e]`` is ``Binary`` for each candidate arc. The
recourse constraint is :math:`y_e \le u_e \, x_e`; raising :math:`x_e`
from 0 to 1 only loosens this bound, and the flow-balance constraints
do not involve ``x``. So opening more arcs cannot make any per-
scenario subproblem less feasible -- ``np.ceil`` is feasibility-
preserving for the arc-open variables.

The right *engine* for netdes is **not** ``average_xhat_nonants``.
The averaged-data problem can leave some :math:`x_e` at 0 because the
average demand pattern does not need that arc; a real scenario with
peakier demand may need it. The averaged-solution path is
``lp_xbar_nonants``: any arc that any scenario's LP wanted positive
contributes positively to the average, and ``np.ceil`` then promotes
it to 1.

``examples/netdes/netdes_auxiliary.py``:

.. code-block:: python

   import numpy as np
   from mpisppy.utils.xhat_helpers import lp_xbar_nonants
   from netdes import scenario_creator, scenario_names_creator


   def feasible_xhat_creator(*, solver_name, solver_options=None,
                             num_scens=None, **scenario_creator_kwargs):
       if num_scens is None:
           from parse import parse
           num_scens = parse(scenario_creator_kwargs["path"],
                             scenario_ix=None)["K"]
       snames = scenario_names_creator(num_scens)
       arr = lp_xbar_nonants(
           scenario_creator, snames,
           solver_name=solver_name,
           scenario_creator_kwargs=scenario_creator_kwargs,
           solver_options=solver_options,
       )
       return {"ROOT": np.ceil(arr - 1e-9)}

The :math:`-10^{-9}` margin keeps integer-valued LP solutions from
being inadvertently bumped up by floating-point dust.

Worked example: sslp (binary, set-covering monotonicity)
--------------------------------------------------------

Sslp ``model.FacilityOpen[j]`` is ``Binary``. Opening more facilities
never tightens ``DemandConstraint`` (more capacity available) or
``ClientConstraint`` (the LHS does not involve ``FacilityOpen``). The
shipped model also carries a high-``Penalty`` ``Dummy`` slack, so any
pin is technically feasible; the rounded LP-xbar is still a
meaningful low-slack candidate for the pin-dual machinery.

Sslp does not currently ship an ``average_scenario_creator``, so the
auxiliary skips the ``average_xhat_nonants`` engine entirely and goes
straight to ``lp_xbar_nonants``. The feasibility-preserving rule
chosen here is ``np.round``.

``examples/sslp/sslp_auxiliary.py``:

.. code-block:: python

   import numpy as np
   from mpisppy.utils.xhat_helpers import lp_xbar_nonants
   from sslp import scenario_creator, scenario_names_creator


   def feasible_xhat_creator(*, solver_name, solver_options=None,
                             num_scens, **scenario_creator_kwargs):
       snames = scenario_names_creator(num_scens)
       arr = lp_xbar_nonants(
           scenario_creator, snames,
           solver_name=solver_name,
           scenario_creator_kwargs=scenario_creator_kwargs,
           solver_options=solver_options,
       )
       return {"ROOT": np.round(arr)}

See also
--------

* :ref:`jensens` -- Jensen's bound and the
  ``--*-try-jensens-first`` flags. Shares the
  ``average_scenario_creator`` convention but uses it for a different
  contract (silently-skip-on-infeasibility candidate xhat, plus an
  outer-bound path that ``feasible_xhat_creator`` does not address).
* :ref:`scenario_creator` -- the core scenario-module conventions
  (``scenario_creator``, ``scenario_names_creator``, ...) that are
  prerequisites for everything in this document.
