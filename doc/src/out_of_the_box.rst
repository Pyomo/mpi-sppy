.. _out_of_the_box:

Out-of-the-box auto-configuration
=================================

The ``--out-of-the-box`` option lets a relatively new user obtain a *sensible*
mpi-sppy run with almost no knowledge of the library's internals. You supply a
model module (and its scenario data); mpi-sppy introspects the environment and
the model and assembles a defensible configuration -- algorithm, solver, spokes,
flexible rank split, and proper bundling -- instead of requiring a hand-crafted
hub/spoke command line.

The spirit is *"here is my model, go,"* followed by a clear explanation of what
was chosen and how to do better.

.. note::
   Out-of-the-box (OOTB) only *fills gaps*. **Any option you set explicitly
   always wins** -- OOTB never overrides it. So you can start from
   ``--out-of-the-box`` and override individual choices as you learn.

Basic usage
-----------

Add ``--out-of-the-box`` to an otherwise minimal ``generic_cylinders`` command
line (your model still needs its scenario count -- ``--num-scens`` for two-stage
problems, ``--branching-factors`` for multistage):

.. code-block:: bash

    # serial -> too few ranks for cylinders, so OOTB solves the EF
    python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
        --out-of-the-box

    # 3+ ranks available; OOTB decomposes when the problem is big/hard enough
    mpiexec -np 3 python -m mpi4py -m mpisppy.generic_cylinders \
        --module-name farmer --num-scens 6 --out-of-the-box

OOTB prints the configuration it chose, the **equivalent explicit command line**
(so you can reproduce, learn from, and tweak it), runs the model, and then prints
a prioritized **Suggestions** list.

.. note::
   OOTB decomposes only when it expects the decomposition to pay off. The base
   tier estimates how long the monolithic EF would take and, if that is within
   budget, solves the EF even when several ranks are available -- because for a
   small or fast-solving model the EF *is* the right call. On a fast machine with
   a commercial solver, the bundled examples (farmer, sizes, aircond) are cheap
   enough that ``--out-of-the-box`` chooses the EF for all of them. To exercise
   the cylinder path on a small problem, either request a spoke (any
   decomposition flag, e.g. ``--lagrangian``, forces a decomposition) or use the
   ``--out-of-the-box-minus`` tier, which has no size estimate and decomposes
   whenever there are more than a couple of scenarios:

   .. code-block:: bash

       mpiexec -np 3 python -m mpi4py -m mpisppy.generic_cylinders \
           --module-name farmer --num-scens 6 --out-of-the-box-minus

What OOTB decides
-----------------

In order, OOTB chooses:

#. **Solver.** The first installed solver in a preference order (persistent
   commercial, then commercial, then a free QP-capable solver, then LP/MIP-only).
   An LP/MIP-only solver (cbc, glpk) automatically adds
   ``--linearize-proximal-terms`` because it cannot take the quadratic PH prox.
   If you pass ``--solver-name`` it is used as-is (and carried over to
   ``--EF-solver-name`` if OOTB ends up solving the EF).
#. **Extensive form vs. decomposition.** With fewer than three ranks there is no
   useful cylinder configuration (hub + at least two spokes), so OOTB solves the
   **EF**. Above the rank floor, OOTB still solves the EF when the whole problem
   is small enough to expect a quick monolithic solve (see *Effort and the EF
   gate* below). Otherwise it decomposes. If you explicitly request a
   decomposition (any spoke or a non-default hub) and have enough ranks, OOTB
   never substitutes the EF.
#. **Spokes (a small, widened core).** Starting from a minimal core of one outer
   bound (``--lagrangian``) and one inner/incumbent spoke (``--xhatshuffle``),
   OOTB adds further spokes only while every cylinder would still keep at least a
   couple of ranks. The preference is to give a few cylinders width rather than
   pile on many weak single-rank spokes. So six ranks become three cylinders,
   widened -- not six single-rank cylinders.
#. **Flexible rank split.** Ranks are split across cylinders unevenly: cheaper
   cylinders (the xhat family) get a smaller share via per-spoke
   ``--*-rank-ratio`` flags. (This is a crude cold-start split; the right split
   depends on relative subproblem solve cost.)
#. **Proper bundling.** When there are many scenarios, OOTB forms proper bundles,
   choosing the largest ``--scenarios-per-bundle`` that divides the scenario
   count, leaves at least as many bundles as ranks, and keeps a bundle's modeled
   solve effort within budget.
#. **A few extra defaults**, each backed off if you addressed the same concern:
   ``--default-rho 1`` and the ``--grad-rho`` rho setter, ``--rel-gap 0.01``,
   ``--max-iterations 100``, and ``--dynamic-rho-primal-crit``.

Transparency
------------

OOTB prints the choices and the equivalent command line up front, and a
**Suggestions** list after the run (so the suggestions can reflect how the run
went). For example, a serial farmer run reports::

    [out-of-the-box] tier 'base', policy 2026-06-28
      - solver: gurobi_persistent (first available in preference order)
      - --EF: only 1 ranks; decomposition needs >= 3
      - --EF-solver-name gurobi_persistent: EF solver (gurobi_persistent)
    [out-of-the-box] equivalent command line:
      mpiexec -np 1 python -m mpi4py -m mpisppy.generic_cylinders \
          --module-name farmer --num-scens 3 --EF --EF-solver-name gurobi_persistent
    ...
    [out-of-the-box] Suggestions:
      * Ran the monolithic EF because only 1 MPI rank(s) were available; with
        >= 3 ranks OOTB would decompose (hub + bound spokes).

The equivalent command line is anchored with the module and scenario
specification and lists every flag OOTB added, so you can paste it (dropping
``--out-of-the-box``) to reproduce or modify the run.

Effort tiers (how deeply OOTB inspects the model)
-------------------------------------------------

Three mutually-exclusive flags select how deeply OOTB looks at the model; they
share one interpreter and one policy file and differ only in how much they
inspect. Every decision uses the best fact available and degrades gracefully to
a suggestion when a fact is missing.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Flag
     - Instantiates
     - What it can decide
   * - ``--out-of-the-box-minus``
     - nothing
     - EF gate by scenario *count*; solver by availability. Cannot size proper
       bundles (no model size information).
   * - ``--out-of-the-box`` *(default)*
     - one probe scenario
     - Size-aware EF gate; integrality- and size-aware bundling. The recommended
       tier.
   * - ``--out-of-the-box-plus``
     - *(reserved)*
     - Planned: instantiate all scenarios and do a brief timed solve for
       solve-time / gap information. Currently behaves like the base tier.

``--out-of-the-box-plus`` is **not** an autotuner: it is "out-of-the-box with
more information," making the same one-shot decisions as the base tier. (It is
reserved for a future release; today it is equivalent to ``--out-of-the-box``.)

Effort and the EF gate
^^^^^^^^^^^^^^^^^^^^^^^

The base tier instantiates one scenario to read its size profile (continuous /
integer variable counts and nonant counts) and models solve *effort* from it.
The numbers in the policy file are **calibrated to roughly seconds**, so the EF
budget reads as a wall-clock target: OOTB solves the EF when the whole problem's
modeled effort is within ``ef_effort_budget`` (about that many seconds), and
sizes bundles so a bundle is at most a small multiple as hard as a single
scenario. Integer content scales superlinearly, so an integer-heavy model
decomposes at a far smaller scenario count than a continuous one.

``--inspect-only`` (dry run)
----------------------------

``--inspect-only`` does the inspection, prints the configuration, the equivalent
command line, and config-time suggestions, then **stops before the production
run**. It is independent of OOTB (on its own it just verifies that one scenario
instantiates), but pairs naturally with it:

.. code-block:: bash

    # plan the run, print the equivalent command line, do not solve
    python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
        --out-of-the-box --inspect-only

``--inspect-only`` takes an optional **assumed rank count** for HPC planning:
``--inspect-only 512`` plans as if 512 ranks were available -- so you can get the
recommended command line for a large job *from a login node, without launching
it*. Everything else (installed solvers, model size) still comes from the real
session; only the rank count is hypothetical.

Policy files
------------

OOTB's choices are driven by a dated, declarative **policy file** under
``mpisppy/generic/ootb_policies/`` -- data, not code, interpreted by a thin
Python routine so every decision is explainable. A bare ``--out-of-the-box``
uses the newest shipped default policy; ``--out-of-the-box PATH`` uses the policy
file at ``PATH`` (the optional value of the flag *is* the path). Policy files
with different *foci* may ship side by side, distinguished by filename; you
select one by passing its path. The run logs which policy and ``policy_version``
it used.

The policy holds the solver preference order, the EF budget, the spoke ladder and
rank ratios, the bundle-effort model, and the extra-option defaults. Its numbers
are produced by the calibration tool (below), not hand-guessed.

.. _ootb_validator:

Validating a policy file
------------------------

A policy file is checked by a validator that confirms it is well-formed and that
its recommendations make sense -- and, on demand, actually run:

.. code-block:: bash

    # static schema + decision checks on the default policy
    python -m mpisppy.generic.ootb_validate

    # also exercise the real example models (needs a solver to instantiate)
    python -m mpisppy.generic.ootb_validate --examples

    # also actually run the recommended configs and flag problem cases
    python -m mpisppy.generic.ootb_validate --run --json report.json

The validator has three layers: static schema checks (every referenced flag is
real, keys/types are right); decision checks (the EF gate fires when it should,
forced decomposition wins, bundling is valid, no conflicting rho setters); and
run checks that execute the recommended configurations and **flag** two cases for
a human to review -- an EF that misses a 1% gap in ten minutes, and cylinders
that max out on iterations. It produces a human-readable and a machine-readable
(``--json``) report. The fast, solver-free layers gate continuous integration;
the run tier is for nightly / local use.

.. _ootb_calibrator:

Calibrating the effort numbers
------------------------------

The effort coefficients and budget are produced by a calibration tool from timed
solves on the example models, so they track measured wall-clock time on a
reference machine rather than being guesses:

.. code-block:: bash

    python -m mpisppy.generic.ootb_calibrate --solver-name gurobi \
        --output mpisppy/generic/ootb_policies/ootb_policy_<date>.json

The tool times extensive-form solves over a spread of bundle sizes, fits the
continuous / integer / nonant coefficients (choosing the integer exponent by best
fit), keeps the coefficients in seconds units so the budgets read as seconds, and
writes a new dated policy with provenance. Solve time is machine- and
solver-dependent (and noisy for MIPs), so a calibrated policy is per reference
machine and approximate; re-run the tool to recalibrate for your environment.

Limitations
-----------

OOTB aims for a *defensible* configuration, not an *optimal* one. It does not
tune convergence parameters to optimality and does not search a parameter space.
For full control, write an explicit hub/spoke command line (the rest of
:ref:`generic_cylinders` documents every option) -- and remember that OOTB emits
exactly such a command line for you to start from.
