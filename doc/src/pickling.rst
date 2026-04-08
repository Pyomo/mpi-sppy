.. _pickling:

Pickling Scenarios and Bundles
==============================

``mpisppy`` can pickle scenarios and proper bundles to disk so that
later PH / EF / MMW runs unpickle them instead of rebuilding from
scratch every time. This page covers the full pickling workflow:
the basic write / read flags, the pre-pickle preprocessing pipeline
(presolve, user callback, iter0 solve), and how to use pickling as
part of an algorithm-tuning workflow.

For background on proper bundles themselves see :ref:`Pickled-Bundles`.
The cylinder driver that exposes most of these flags is described in
:ref:`generic_cylinders`.

Why Pickle?
-----------

There are two distinct reasons to pickle:

1. **Reuse across runs.** Building scenarios (and especially proper
   bundles) can be expensive. Once they are pickled, every downstream
   tuning / experimentation run reads them from disk in a fraction of
   the time it would take to rebuild them.
2. **Front-load deterministic work.** Presolve, model-specific cleanup,
   and the iteration 0 solve are all deterministic given the model.
   They can be paid once at pickle time and then skipped (or warm-
   started) on every later run.

Basic Pickling and Unpickling
-----------------------------

The ``generic_cylinders`` driver writes and reads pickles via two
pairs of flags:

- ``--pickle-scenarios-dir <DIR>`` / ``--unpickle-scenarios-dir <DIR>``
  for individual scenarios.
- ``--pickle-bundles-dir <DIR>`` / ``--unpickle-bundles-dir <DIR>``
  for proper bundles. ``--scenarios-per-bundle`` must also be given on
  both the writing and reading runs.

When the driver is asked to write pickles, **all ranks are used for
pickling** and most other command line options are ignored on that run.
Pickling is a separate phase from solving.

.. warning::
   The directory passed to ``--pickle-bundles-dir`` /
   ``--pickle-scenarios-dir`` is overwritten. Do not point it at a
   directory you care about.

.. note::
   When unpickling, options such as ``--num-scens`` are still required
   because ``cfg`` needs them. Consistency between the command line and
   the files in the pickle directory is not always checked.

.. note::
   Unpickled scenarios inside proper bundles are not supported by
   ``generic_cylinders`` directly — the wrappers would need to be more
   sophisticated. Pickle the bundles, not the scenarios, when bundling.

.. note::
   The ``scenario_denouement`` function might not be called when
   pickling bundles.

.. warning::
   Helper functions are **not** pickled, so there is a loose linkage
   with the helper functions in the module. The module that built the
   pickle and the module that consumes it must be source-compatible.

Pre-Pickle Preprocessing
------------------------

By default a pickle captures exactly what ``scenario_creator`` returns:
a freshly built Pyomo model with no preprocessing applied and no solver
state. Several deterministic operations can optionally run between
``scenario_creator`` and ``dill_pickle``, so the cost is paid once and
shared by every downstream run.

The pipeline, when all stages are enabled, is:

.. code-block:: text

   scenario_creator(sname)
     → SPPresolve (FBBT / optional OBBT)         # --presolve-before-pickle
     → <pre_pickle_function>(model, cfg)         # --pre-pickle-function NAME
     → solve iter0 (store values + duals)        # --iter0-before-pickle
     → dill_pickle(model)

Each stage is independently controlled by a command line flag and can
be enabled or skipped. When more than one is enabled, the order above
is preserved: presolve runs before the user callback, the callback runs
before the iter0 solve, and pickling is the last step.

For proper bundles, every stage operates on the **bundled extensive
form** (i.e., on whatever is in ``local_subproblems``). Intra-bundle
cross-scenario propagation in presolve happens automatically.

Stage 1: ``--presolve-before-pickle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Runs ``SPPresolve`` over the rank-local scenarios (or bundles) before
pickling. This is exactly the same machinery used by ``--presolve``
when solving directly (see the presolve discussion in
:ref:`generic_cylinders`), so the tightened bounds and any
cross-rank ``Allreduce`` synchronization match what you would get at
solve time. The difference is that the cost is paid **once** and baked
into the pickle.

OBBT can be turned on in addition to FBBT via the existing
``--obbt`` flag. Be aware that turning on OBBT at pickle time
introduces a solver dependency on the pickling job — if you pickle on
a machine without a solver installed (for example, an agnostic
AMPL / GAMS workflow or a CI environment), do not enable
``--obbt`` for pickling.

.. code-block:: bash

   python -m mpisppy.generic_cylinders --module-name farmer --num-scens 12 \
       --pickle-bundles-dir farmer_pickles --scenarios-per-bundle 3 \
       --presolve-before-pickle

Stage 2: ``--pre-pickle-function``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Names a user callback that is invoked between presolve and the iter0
solve. The argument is a dotted Python path to a function with the
signature:

.. code-block:: python

   def my_pre_pickle_fn(model, cfg):
       """Called once per scenario or bundle just before it is pickled.
       Free to mutate `model` in place. Return value is ignored.
       """

The function does not have to live in your model module — any importable
function works, so you can keep generic cleanup utilities in a shared
module.

Typical uses:

- Apply selected ``pyomo.contrib.preprocessing`` transformations
  (coefficient tightening, redundant constraint removal, variable
  aggregation, zero-term elimination) that you trust for your model.
- Fix variables to known-tight values you can compute outside the
  solver.
- Delete dominated constraints identified by domain knowledge.
- Rename or reorganize components for faster downstream access.

.. code-block:: bash

   python -m mpisppy.generic_cylinders --module-name farmer --num-scens 12 \
       --pickle-bundles-dir farmer_pickles --scenarios-per-bundle 3 \
       --presolve-before-pickle \
       --pre-pickle-function farmer_cleanup.fix_known_vars

.. note::
   The callback is opt-in. If ``--pre-pickle-function`` is not given,
   no user code runs in this stage. The flag takes a function name
   rather than relying on a magic module-level attribute precisely so
   that nothing happens silently.

Stage 3: ``--iter0-before-pickle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Solves each scenario (or bundle EF) once with its original objective —
no PH ``W``, no PH proximal term, i.e. a PH iteration 0 solve — and
stores the result inside the pickle. Variable ``.value`` attributes
and any suffixes attached to the model survive pickling, so the
solution and its dual information are available to downstream
consumers.

By default the same solver as the rest of the run is used. To override
just for the pickling phase:

- ``--pickle-solver-name <NAME>`` selects a different solver for the
  pickle-time iter0 solve (e.g. a fast LP solver even though downstream
  uses a MIP solver).
- ``--pickle-solver-options <STRING>`` overrides solver options for the
  pickle-time solve.

.. code-block:: bash

   python -m mpisppy.generic_cylinders --module-name farmer --num-scens 12 \
       --pickle-bundles-dir farmer_pickles --scenarios-per-bundle 3 \
       --presolve-before-pickle --iter0-before-pickle \
       --solver-name gurobi_persistent --pickle-solver-name gurobi

.. warning::
   Pickling with an LP-only solver and then running with a MIP solver
   downstream is allowed. The LP-relaxed variable values still serve as
   a starting point, but for MIPs this is **not always** a good warm
   start — some integer-feasibility-sensitive solvers can be slower
   when started from a non-integer point. If in doubt, use the same
   solver at pickle time and run time.

.. warning::
   If the iter0 solve fails (infeasible, time limit, interrupt, or
   solver error), the pickling job **shuts down**. There is no
   "pickle anyway with a warning" fallback. Producing pickles with
   silently bad state would be worse than the job stopping. Fix the
   underlying problem and rerun.

Duals and Reduced Costs in the Pickle
"""""""""""""""""""""""""""""""""""""

When ``--iter0-before-pickle`` is set, ``mpisppy`` attaches a Pyomo
``IMPORT`` suffix for duals (and reduced costs) to each model **before**
the iter0 solve. After the solve, those suffix values become part of
the pickled model.

.. important::
   Pickles produced with ``--iter0-before-pickle`` carry dual and
   reduced cost values on a Pyomo ``IMPORT`` suffix. Downstream
   consumers (Lagrangian / Lagranger spokes, fixer extensions, custom
   user code) can read those duals from the unpickled model without
   having to re-solve. If you do not want this behavior, do not enable
   ``--iter0-before-pickle``.

How Downstream Runs Use the Iter0 Solution
"""""""""""""""""""""""""""""""""""""""""""

There are two tiers of consumption:

1. **Warm start (default).** PH still runs iter0 on the unpickled
   models, but each subproblem solve now starts from pre-populated
   variable values. For MIPs this becomes a MIP start and is usually a
   significant speedup; for LPs it is less useful without a basis.
2. **Skip iter0 entirely.** A follow-up flag ``--iter0-from-pickle``
   tells PH to read the variable values from the unpickled scenarios,
   compute ``xbar`` directly, perform the first ``W`` update, and go
   straight to iter1. The PH side detects this case via a flag stored
   inside ``_mpisppy_data`` on the model when the pickle was written.

Pickle Metadata
^^^^^^^^^^^^^^^

Each pre-processed pickle records which stages ran, which presolve
options were used, and which solver was invoked, on
``model._mpisppy_data.pickle_metadata``. The metadata travels inside
the pickle automatically, so it survives file moves and is easy to
inspect after the fact.

.. _pickling_tuning_workflow:

Tuning Workflow with Pickling
-----------------------------

A common reason to pickle is **algorithm tuning**: you want to try many
combinations of rho settings, spoke combinations, fixer thresholds,
extension parameters, and so on, against the **same** scenarios. The
scenario build is identical for every tuning run, so doing it again
each time is pure waste.

A typical tuning workflow:

1. Pick a representative problem size (and bundle structure, if you
   are bundling).
2. **Pickle once.** Run ``generic_cylinders`` with ``--pickle-bundles-dir``
   (or ``--pickle-scenarios-dir``) and any of the pre-pickle stages
   that are useful for your model. Common combinations:

   - ``--presolve-before-pickle`` alone: pay FBBT once.
   - ``--presolve-before-pickle --iter0-before-pickle``: also bake in
     the iteration 0 solution (and duals) so every tuning run starts
     warm.
   - Add ``--pre-pickle-function ...`` if you have model-specific
     cleanup you trust.

3. **Tune from the pickle.** Every tuning run uses
   ``--unpickle-bundles-dir <DIR>`` (or ``--unpickle-scenarios-dir``)
   instead of rebuilding scenarios. The build / presolve / iter0 cost
   is gone, so the run-to-run wallclock difference becomes a clean
   measurement of the tuning change.
4. When you find good settings, lock them in and rebuild pickles only
   when the underlying model changes.

This workflow especially benefits from ``--iter0-before-pickle``: many
tuning experiments differ only in iter1+ behavior, so eliminating
iter0 from every iteration of the experiment loop is a real time win.

.. _pickling_iter0_parallelism:

Single-Run Case: Pickling Iter0 to Use All CPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a less obvious case for pickling: **even if you only intend
to solve once**, pickling first can be faster than not pickling. The
reason is parallelism allocation.

When you run ``generic_cylinders`` with cylinders, the available MPI
ranks are split across the hub and the spokes. Iteration 0 of PH (the
deterministic warm-start solve) only runs on the hub side; the spoke
ranks are doing other work or are idle for the iter0 phase.

When you run a pickling job with ``--iter0-before-pickle``, **all** the
available MPI ranks are used for forming bundles and running iter0
solves — there are no cylinders during pickling. So if your machine
has many cores and you have many bundles, the iter0 solves can be
spread across every available rank rather than only the ranks
allocated to the hub during a normal cylinder run.

The result: in some settings, the wall-clock cost of (pickle with
``--iter0-before-pickle``) + (cylinder run that consumes the warm
start) can be **lower** than the cost of a single direct cylinder run
that has to do iter0 on a smaller hub allocation.

This is not universal — for small models or small node counts the
overhead of the extra serialization round trip dominates. But on
larger HPC allocations with many bundles, it is a real option worth
measuring.
