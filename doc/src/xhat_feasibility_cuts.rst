.. _xhat_feasibility_cuts:

Xhat Feasibility Cuts
=====================

For two-stage and multi-stage problems **without complete recourse**, a
candidate first-stage solution ``xhat`` proposed by an xhatter spoke
(``xhatlooper``, ``xhatshufflelooper``, ``xhatspecific``,
``xhatxbar``) can turn out to be infeasible in one or more scenarios.
Today those candidates are simply discarded and nothing prevents the
same ``xhat`` from being proposed again a few iterations later.

When this feature is enabled, an xhatter that detects such an
infeasibility emits a **no-good feasibility cut** on the first-stage
variables, and the hub installs that cut into every scenario's
subproblem. The result is that the same ``xhat`` (and any other
assignment with the same pattern on binaries) is excluded from future
consideration.

This is a first-milestone implementation: it is **valid only when every
first-stage (nonant) variable is binary**. See :ref:`xhat_feas_cuts_boundaries`
below.

Enabling the Feature
--------------------

``generic_cylinders`` exposes a single integer flag:

.. code-block:: bash

   --xhat-feasibility-cuts-count N

where ``N`` is the maximum number of feasibility cuts the xhatter may
emit per iteration. The default is ``0``, which disables the feature
entirely. Any positive integer both turns the feature on and sizes the
shared-memory buffer the xhatter uses to send cuts.

Nothing else needs to be specified. The hub-side installer
(``mpisppy.extensions.xhat_feasibility_cut_extension.XhatFeasibilityCutExtension``)
is attached automatically through :ref:`cfg_vanilla <drivers>` when the
flag is positive.

Example
-------

.. code-block:: bash

   python -m mpisppy.generic_cylinders \
       --module-name my_binary_first_stage_model \
       --num-scens 10 \
       --solver-name gurobi \
       --max-iterations 50 \
       --default-rho 1.0 \
       --lagrangian \
       --xhatshuffle \
       --xhat-feasibility-cuts-count 3

The cap of 3 says "emit at most 3 cuts per xhatter iteration". Cuts
accumulate across iterations inside each scenario's
``_mpisppy_model.xhat_feasibility_cuts`` constraint container.

.. _xhat_feas_cuts_boundaries:

Scope and Limitations
---------------------

The first-milestone cut is the textbook no-good inequality

.. math::

   \sum_{i:\, \hat{x}_i = 1} (1 - x_i) + \sum_{i:\, \hat{x}_i = 0} x_i \;\geq\; 1

which is valid only when every :math:`x_i` is binary. If any
first-stage nonant is integer (not bounded to :math:`\{0, 1\}`) or
continuous, the cut cannot exclude the infeasible ``xhat`` correctly,
so the feature **refuses to run** rather than silently generate
invalid relaxations:

.. code-block:: text

   RuntimeError: --xhat-feasibility-cuts-count > 0 requires every
   first-stage (nonant) variable to be binary; found non-binary nonant
   '<var name>' (key (<node>, <i>)) on scenario '<sname>' with
   domain <domain>. The first-milestone feasibility-cut generator is
   no-good-only. Support for integer and continuous first-stage
   variables is planned as a follow-up milestone (pyomo Benders /
   Farkas extension).

The error is raised at hub setup time (before any PH work begins),
so a misconfiguration is caught immediately.

**Integer first-stage variables with bounds** :math:`[0, 1]` **are
accepted** — semantically those are binary. Declaring a var as
``pyo.Integers`` with ``bounds=(0, 1)`` works just as well as
``pyo.Binary``.

Multi-stage
-----------

The no-good cut is linear in whichever nonants the xhatter fixed, and
does not depend on the recourse-cost formulation. It is therefore
valid for multi-stage problems as well as two-stage, as long as every
nonant at every node is binary. The same startup check is applied
across all nodes (``ROOT``, ``ROOT_0``, etc.).

Interaction with Proper Bundles
-------------------------------

The hub installer appends cuts to each scenario's
``xhat_feasibility_cuts`` constraint container. When a scenario
object is actually a proper bundle, the cut is installed against the
bundle's canonical nonant set (``s._mpisppy_data.nonant_indices``)
exactly as the cross-scenario cut machinery does. Nonanticipativity
inside the bundle ensures the cut takes effect on every per-scenario
block.

Follow-up Milestones
--------------------

- Lifting the binary-only restriction requires generating **Farkas
  feasibility cuts** from the dual ray of an infeasible second-stage
  LP. The upstream
  ``pyomo.contrib.benders.benders_cuts.BendersCutGeneratorData``
  currently only produces optimality cuts; supporting the
  infeasibility case is a Pyomo PR. Once that lands, the xhatter will
  be able to emit valid cuts for integer and continuous first-stage
  variables as well (LP recourse only).
- **Cut-pool management** is deferred to
  `issue #670 <https://github.com/Pyomo/mpi-sppy/issues/670>`_. Today
  cuts accumulate indefinitely in the per-scenario constraint
  container, the same way cross-scenario cuts do in
  ``CrossScenarioExtension``. For long runs, use the
  ``--xhat-feasibility-cuts-count`` cap to bound the per-iteration
  growth.

See Also
--------

- :ref:`Spokes` — overview of the xhat spokes.
- :ref:`Extensions` — the broader extension mechanism.
- ``doc/xhat_feasibility_cuts_design.md`` — the design document with
  the full milestone plan and the ``V1/V2/V3`` scope table.
