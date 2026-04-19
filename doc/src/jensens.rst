.. _jensens:

Jensen's Bound as potential starting bound
==========================================

.. warning::

   **Jensen's bound is a valid lower bound for a minimization problem when the recourse value function
   is convex in the random parameters**

   Necessary conditions for a valid *outer* bound:

   1. The second-stage problem is an LP — no integer recourse.
   2. Random parameters appear only in objective coefficients and in
      the right-hand side, not in ways that break convexity of the
      recourse value in those parameters.
   3. Two-stage structure.

   ``mpi-sppy`` checks (1) automatically on the outer-bound path and
   refuses to compute the outer bound when non-nonant integer/binary Vars
   exist in the recourse variables while (2) and (3) are the user's
   responsibility.

   **Inner bounds (xhatters) are valid regardless of the above.** The
   xhat path uses the expected-value scenario only as a *source of a
   candidate first-stage solution*; that candidate is then honestly
   evaluated across the real scenarios, so Jensen's convexity
   assumption never enters the validity argument.

What the options do
-------------------

Two-stage only (for now). Each supported spoke gains one boolean flag:

Outer-bound (lower-bounder) spokes

* ``--lagrangian-try-jensens-first``
* ``--lagranger-try-jensens-first``
* ``--subgradient-try-jensens-first``
* ``--reduced-costs-try-jensens-first``

Inner-bound (xhat) spokes

* ``--xhatshuffle-try-jensens-first``
* ``--xhatxbar-try-jensens-first``
* ``--xhatlooper-try-jensens-first``
* ``--xhatspecific-try-jensens-first``

When the flag is set, before the spoke enters its usual iterations it
builds a single *expected-value (EV) scenario* (via the module's
``expected_value_creator``; see below), solves it with the spoke's
configured solver, and then:

* on the outer-bound path, sends the solver's best dual bound on the EV
  problem as its first outer bound;
* on the xhat path, takes the EV first-stage solution, evaluates it
  across all scenarios, and reports the expected cost as its first
  inner bound if it is feasible for all scenarios.

.. admonition:: Solver options and what is extracted as the bound
   :class: note

   The EV solve uses ``iterk_solver_options`` (the spoke's
   "production" tolerances), not ``iter0_solver_options``. The EV
   solve is a one-shot deterministic solve, not a first-iteration
   subproblem solve.

   On the outer-bound path the value sent to the hub is the **solver's
   best dual bound** on the EV problem (Pyomo's
   ``results.problem.lower_bound`` for minimize, ``upper_bound`` for
   maximize), not the incumbent objective value. With the dual bound,
   a non-zero MIP gap on the EV solve does **not** invalidate
   Jensen's outer bound — the dual bound is a valid lower bound on
   the EV optimum regardless of gap, and Jensen's inequality
   transitively makes it a valid lower bound on the expected
   recourse cost. (For an LP the dual bound and the incumbent
   coincide, so this distinction collapses.)

   On the xhat path the dual bound is ignored; only the first-stage
   nonant values are used, and they are honestly re-evaluated across
   the real scenarios via ``Xhat_Eval.evaluate``. A loose MIP gap on
   the EV solve there just means a possibly-worse candidate xhat,
   never an invalid bound.

The spoke then continues its normal loop. There is **no** "iteration
-1" in PH, APH, L-shaped, or any hub — the Jensen's work happens
entirely inside a spoke.

**What the expectation is taken over.** The EV data must be the sample mean
of the data for every scenario in the run (*not* the expectation of
any underlying continuous distribution). This matches how ``mpi-sppy``
treats the outer problem: the sample-average approximation (SAA) is
what is actually being solved, and Jensen's bound applies directly to
that problem.

Authoring ``expected_value_creator``
------------------------------------

A scenario module that wants to participate must define:

.. code-block:: python

   def expected_value_creator(scenario_name, **kwargs):
       """Return a Pyomo model with the same shape as scenario_creator(...),
       but built from expectation-valued random data.

       Two-stage: _mpisppy_probability must be 1.0 (a single
       deterministic scenario, not a member of a probabilistic
       ensemble).
       """

.. admonition:: Under the Hood
   :class: note

   Discovery is via ``getattr(module, "expected_value_creator", None)``; 
   if a flag is set but the module does not define the function, 
   ``cfg_vanilla`` raises a clear error at spoke-setup time.
   

The recommended authoring pattern — which ``examples/farmer/farmer.py``
now demonstrates — is to split the work into two underscore helpers
that are shared between ``scenario_creator`` and
``expected_value_creator``:

.. code-block:: python

   def _scenario_data(scenario_name, **kwargs):
       """Pure-Python random data as a plain dict. No Pyomo."""
       ...

   def _build_model(scenario_name, data, *, probability, **kwargs):
       """Build Pyomo model from the data dict. Shared build path."""
       ...

   def scenario_creator(scenario_name, **kwargs):
       data = _scenario_data(scenario_name, **kwargs)
       prob = 1.0 / kwargs["num_scens"] if kwargs.get("num_scens") else "uniform"
       return _build_model(scenario_name, data, probability=prob, **kwargs)

   def expected_value_creator(scenario_name, **kwargs):
       # NOTE: could be multi-threaded for large num_scens.
       snames = scenario_names_creator(kwargs["num_scens"])
       datas  = [_scenario_data(s, **kwargs) for s in snames]
       avg    = _average_scenario_data(datas)
       return _build_model(scenario_name, avg, probability=1.0, **kwargs)

Benefits of this split:

* One source of truth for "what does the Pyomo model look like." If
  ``scenario_creator`` and ``expected_value_creator`` each build the
  model inline, they will eventually drift.
* Separating data from model makes averaging trivial: average the
  dict, not the Pyomo components.
* Every rank independently calls ``expected_value_creator`` and gets
  an identical model. No collective communication needed.

Seed management
---------------

This is the single most common source of bugs in user-authored
``expected_value_creator`` functions. Please read this section before
writing your own.

**Use a local RNG per call, not a module-level global.**
The ``farmer`` example deliberately avoids the
``farmerstream = np.random.RandomState()`` module-level pattern and
instead does:

.. code-block:: python

   def _scenario_data(scenario_name, ..., seedoffset=0):
       scennum = sputils.extract_num(scenario_name)
       rng = np.random.RandomState(scennum + seedoffset)   # local, fresh
       ...

The module-level pattern is thread-unsafe: two threads calling the
same function race on ``.seed()`` and ``.rand()`` and produce garbage.
With a local RNG, ``expected_value_creator`` can safely be
parallelized later (e.g. via ``concurrent.futures.ThreadPoolExecutor``).
The on-disk results are byte-for-byte identical to the shared-stream
pattern for a given ``(scennum, seedoffset)``.

**Draw order must be deterministic and explicit.**
Do not rely on Pyomo's component-build order to sequence your random
draws. Build the data dict in a plain Python loop first; then hand
that dict to ``_build_model``. The underscore-helper pattern does
this automatically.

**``seedoffset`` must be threaded through both creators.**
If your module already accepts ``seedoffset`` in
``scenario_creator`` (as the confidence-interval code expects), make
sure ``expected_value_creator`` accepts and forwards it too.
Otherwise, a confidence-interval run with a non-zero ``seedoffset``
would get a Jensen bound computed from a scenario set that does not
match the one the run is actually solving.

**Thread-safety.**
The ``expected_value_creator`` you ship should be safe to call from
multiple threads concurrently. With local RNGs (above), it is — and
the farmer example's docstring sketches how to wrap the per-scenario
loop in a ``ThreadPoolExecutor``. With a shared module-level RNG, it
is not.

Worked example: farmer
----------------------

See ``examples/farmer/farmer.py`` for the pedagogical reference
implementation. The refactor split the original monolithic
``scenario_creator`` into ``_scenario_data`` (pure-Python yield
generation using a local ``RandomState``) and ``_build_model`` (Pyomo
construction). ``expected_value_creator`` averages the yield dicts
from every would-be scenario and hands the average to ``_build_model``
with ``probability=1.0``.

To see it in action:

.. code-block:: bash

   mpiexec -np 3 python -m mpi4py mpisppy/generic_cylinders.py \
       --module-name farmer --num-scens 3 --default-rho 1 \
       --solver-name cplex --lagrangian --xhatshuffle \
       --lagrangian-try-jensens-first --xhatshuffle-try-jensens-first \
       --max-iterations 2

The lagrangian spoke will send its Jensen's outer bound before running
iteration 0, and the xhatshuffle spoke will evaluate the EV first-
stage solution as its first xhat.

(A variant that loads scenario data from ``.dat`` files rather than
generating it with an RNG can be seen in ``examples/sizes/sizes.py``,
which uses a Pyomo AbstractModel.)

Interaction with each spoke's own iteration 0
---------------------------------------------

* **lagrangian**: its normal iteration 0 trivial bound (Lagrange
  multipliers :math:`W = 0`) is itself wait-and-see and therefore
  already a valid outer bound. The value of this flag is
  that the EV bound arrives *before* iteration 0 finishes, which can
  matter a lot when there are many scenarios and iteration 0 is slow.
  The flag does not replace or disable iteration 0. For some
  problems Jensen's bound is tighter than the trivial iteration 0
  bound.
* **subgradient**: iteration 0 solves all scenarios. The EV bound
  arrives earlier.
* **reduced_costs**: inherits the lagrangian behavior.
* **lagranger**: like lagrangian, but takes nonants from the hub and
  computes its own W rather than receiving W. Iteration 0 is still
  the trivial W=0 wait-and-see bound, so the same reasoning applies
  as for lagrangian. (note that this bounder seldom works well)

Convexity limitations, in more detail
-------------------------------------

The integer-recourse check is *necessary but not sufficient*. A
continuous-recourse two-stage LP can still violate convexity in
:math:`\xi` if, for example, random parameters enter the recourse
problem's technology matrix in a way that makes the recourse value
non-convex in those parameters. ``mpi-sppy`` does not detect this
automatically — if you know your problem has matrix-randomness,
verify convexity before enabling the outer-bound flag.

The xhat path has no such restriction: the candidate solution is
honestly evaluated, so any two-stage recourse structure (including
integer recourse and non-convex recourse) is acceptable there.
