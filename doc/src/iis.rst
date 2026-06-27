.. _iis:

Diagnosing Infeasible Xhats with an IIS
=======================================

An xhatter (incumbent-finder) proposes a candidate first-stage
solution ``xhat``, fixes the non-anticipative variables at those
values, and solves every scenario subproblem. If any scenario is
infeasible at that ``xhat``, the candidate is rejected and the
xhatter quietly moves on to the next one. That silence is the right
behavior almost always: candidate rejection is routine and the search
recovers on its own.

It is the *wrong* behavior when your model is supposed to have
(relatively) complete recourse and — because of a modeling subtlety or
a data issue — actually does not. Then the run can spin for a long
time finding *no* incumbent, with nothing to explain why.

The ``--xhatter-write-iis`` option turns that dead end into a
diagnosis. The first time an xhatter rejects a candidate because a
scenario subproblem is infeasible, mpi-sppy computes an **IIS**
(irreducible infeasible set) for the offending subproblem using
`pyomo.contrib.iis
<https://pyomo.readthedocs.io/en/stable/explanation/analysis/iis.html>`_
and writes it to a file. The IIS is the minimal set of constraints and
variable bounds that are mutually in conflict — i.e. *"with these
first-stage decisions, scenario X is infeasible because of this handful
of constraints."*

.. important::

   This runs **at most once per cylinder (per MPI rank)**. The IIS
   computation is expensive, so a guard ensures it cannot fire on every
   rejected candidate, every iteration. See `Run-once semantics`_
   below — the per-rank behavior in particular is easy to be surprised
   by.

Enabling the Feature
--------------------

Three flags control it (all are off / defaulted unless you set them):

``--xhatter-write-iis``
   Boolean switch that turns the feature on. Off by default.

``--xhatter-iis-method`` (``auto`` | ``ilp`` | ``explanation``)
   Which ``pyomo.contrib.iis`` facility to use. Default ``auto``.

   - ``ilp`` — ``write_iis``: writes a standard ``.ilp`` IIS file using
     a **commercial** solver's native IIS engine. Requires a
     *persistent* interface to cplex, gurobi, or xpress.
   - ``explanation`` — ``compute_infeasibility_explanation``: writes a
     textual minimal-infeasible-system report. Works with **any**
     Pyomo solver (it relaxes constraints rather than calling a native
     IIS engine), so it is the path to use with open solvers.
   - ``auto`` — picks ``ilp`` when the configured ``--solver-name`` is
     cplex/gurobi/xpress, otherwise ``explanation``.

``--xhatter-iis-dir <path>``
   Directory for the IIS files. Default: the current working directory.
   The directory is created if it does not exist.

Example
-------

.. code-block:: bash

   python -m mpisppy.generic_cylinders \
       --module-name my_model \
       --num-scens 10 \
       --solver-name gurobi \
       --max-iterations 50 \
       --default-rho 1.0 \
       --lagrangian --xhatshuffle \
       --xhatter-write-iis

The first time the ``xhatshuffle`` spoke evaluates a candidate that is
infeasible for one of its local scenarios, it writes (with gurobi)
``XhatShuffleInnerBound_<scenario>.ilp`` to the current directory and
prints a line naming the file. No further IIS is written by that spoke.

Output File Naming
------------------

File names follow the ``--solver-log-dir`` convention,
``<cylinder>_<scenario>``, with an extension by method:

- ``ilp`` method: ``<cylinder>_<scenario>.ilp``
- ``explanation`` method: ``<cylinder>_<scenario>.iis.log``

where ``<cylinder>`` is the cylinder class name (e.g.
``XhatShuffleInnerBound``) and ``<scenario>`` is the offending
scenario's name (for a multi-stage stage2-EF xhatter it is the
second-stage tree-node name). Because each rank owns different
scenarios, naming by scenario means concurrent writers on different
ranks never collide.

.. _Run-once semantics:

Run-once Semantics
------------------

"Just once" is **per cylinder, i.e. per MPI rank**:

- The first time *this* xhatter detects an infeasible xhat subproblem,
  it writes an IIS for the **first infeasible local scenario** on this
  rank, then sets a guard. No further IIS is emitted by this cylinder
  for the rest of the run.
- In a parallel run, ranks have independent state, so you may get **up
  to one IIS file per rank** that encountered an infeasible local
  scenario — each named by its offending scenario.

Per-rank (rather than globally-once) is deliberate: the infeasible
scenario may not live on rank 0, so electing a single writer would
suppress the diagnostic on exactly the rank that has the problem. One
file per rank, each named by its scenario, gives you the full picture
while still bounding the cost to one IIS computation per worker.

If the IIS computation itself fails (for example ``ilp`` was requested
but no persistent commercial solver is installed), the failure is
reported and the run continues unperturbed; the guard is set anyway, so
a failed attempt is never retried.

Limitations
-----------

- **One scenario per rank.** Only the first infeasible local scenario
  is explained, matching the "just once" intent and bounding cost.
- **``ilp`` needs a persistent commercial solver.** ``write_iis`` uses
  the native IIS engine of cplex/gurobi/xpress through their persistent
  interfaces. With an open solver, use (or let ``auto`` select) the
  ``explanation`` method.
- The feature only fires from xhatters. Routine subproblem
  infeasibility during the main PH/APH algorithm is out of scope.

See Also
--------

- :ref:`xhat_from_file` — handy for reproducing an infeasibility on
  demand: write a known-infeasible first-stage vector to a file, hand
  it in with ``--xhat-from-file``, and the IIS path fires
  deterministically.
- :ref:`Spokes` — overview of the xhat spokes.
- ``doc/designs/iis_on_xhat_infeasible_design.md`` — the design document.
