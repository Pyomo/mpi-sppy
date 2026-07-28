.. _w_oscillation:

W-vector oscillation: detection and interruption
=================================================

The ``w_oscillation`` extension (``mpisppy.extensions.w_oscillation``,
class ``WOscillationMonitor``) watches the Progressive Hedging dual weight
(``W``) vector while a synchronous PH hub runs. It can **detect** oscillation /
cycling in ``W`` and report it, and it can optionally **interrupt** the
oscillation -- slamming (fixing) the offending variables -- to break the cycle.

Oscillating weights -- a ``W`` trajectory that flips sign repeatedly or whose
swings fail to damp out -- are a common and convergence-killing symptom for
mixed-integer problems. Watson and Woodruff (*"Progressive Hedging Innovations
for a Class of Stochastic Mixed-Integer Resource Allocation Problems"*,
Computational Management Science, 2011) describe the mechanism in their §2.1:
the weight update is ``w += rho * (x - xbar)``, so a too-large rho lets ``w``
"shoot past" its optimal value and thrash, especially in MIPs where a change in
one integer variable induces changes in others that are then reversed.
(As an aside, we note that for MIPs a rho that is too small can also result in
oscillation.)

The extension is activated entirely by command-line flags on
``generic_cylinders.py`` (and any ``Config``-based driver):

============================== =================================================
Flag                           Effect
============================== =================================================
``--detect-W-oscillations``    Detect and **report** oscillation (pure
                               observation; no change to the optimization).
``--interrupt-W-oscillations`` **Act** on detected oscillation (slamming).
                               Runs the detection engine, but the report is
                               opt-in (see below).
============================== =================================================

With **neither** flag the extension is never constructed. Both flags take the path to a JSON control file.

.. note::

   This is a **hub** extension for **synchronous PH**. The hooks run under any
   ``PHBase`` hub, but the oscillation and cadence notions assume synchronous
   iterations; APH is not specially wired. Outer/inner-bound spokes are
   untouched.

Relationship to wtracker
------------------------

For a broad view of how ``W`` evolves -- moving means, standard deviations, and
coefficient of variation across every nonant/scenario trace -- use the
:ref:`wtracker_extension`. ``wtracker`` keeps the full history and is a general
diagnostic; ``w_oscillation`` is the focused layer that flags the specific
traces that are *cycling* and (optionally) acts on them, and it keeps only a
small bounded ring buffer rather than the whole history.


Detection
---------

How it works
^^^^^^^^^^^^

Each PH iteration, in the ``miditer`` hook (so the freshest *post-update*
``W`` is in place), the extension captures the ``W`` vector for every local
scenario into a bounded ring buffer. After ``warmup_iters`` samples exist, and
then every ``check_every`` iterations, it runs the selected detectors.

Detection is **per (scenario, nonant)**, but reporting and acting are **per
nonant**, so the per-scenario verdicts are reduced across scenarios with the
same per-node communicators that x-bar uses: a ``SUM`` of the number of
scenarios that flagged each nonant and a ``MAX`` of the per-trace statistics.
Cylinder **rank 0** then writes the report. Because the inputs are reduced to
be identical on every rank, **the report does not depend on how scenarios are
distributed across MPI ranks**.

Detection methods
^^^^^^^^^^^^^^^^^^

The ``methods`` block of the control file selects one or both detectors and
overrides their per-method defaults. New detectors can be added without any CLI
change.

``zero_crossings``
""""""""""""""""""

A port of PySP's ``sorgw`` plugin. For each (scenario, nonant) ``W`` trajectory
(optionally only the last ``window`` samples) it computes:

- ``WZeroCrossings`` -- the number of sign changes of ``W`` (ignoring entries
  with ``|W| < tol``);
- ``DiffZeroCrossings`` -- the number of sign changes of the consecutive
  differences ``ΔW``;
- ``diffs_ratio`` -- a damping ratio: the mean of ``|ΔW|`` over the back
  (newer) half of the trace divided by the mean over the front (older) half. A
  ratio well below 1 means the swings are shrinking (converging); a ratio near
  or above 1 means they are not damping.

The trace is **flagged** if *any* threshold is met. Keys (defaults in
parentheses): ``tol`` (``1e-6``), ``window`` (``null`` = whole retained
history), ``thresh_w_crossings`` (``2``), ``thresh_diff_crossings`` (``3``),
``thresh_diffs_ratio`` (``0.2``).

``w_hash_recurrence``
"""""""""""""""""""""

The Watson-Woodruff §2.4 ("Detecting Cyclic Behavior") cycle detector. For each
nonant it hashes the **per-scenario** ``W`` vector and flags a *recurrence* of
that vector -- the same hash seen again within a look-back window -- which
signals a genuine cycle. ``min_period`` excludes period-1 "recurrence" so a
*constant* ``W`` (i.e. convergence) is never mistaken for a cycle.

In the distributed setting the hashed vector spans scenarios on different
ranks, so the extension forms a **distribution-independent signature**: each
rank sums identity-mixed 64-bit hashes of its local scenarios' values, and the
partial sums are combined with an ``MPI.SUM`` reduction. Because addition is
commutative, the signature is independent of the scenario-to-rank mapping. (This
is an additive / multiset hash; see Bellare & Micciancio, EUROCRYPT 1997, and
Clarke et al., ASIACRYPT 2003.) Keys: ``window`` (``20``), ``quantum``
(``1e-6``; ``W`` is quantized to this before hashing), ``min_period`` (``2``).

Detection control file
^^^^^^^^^^^^^^^^^^^^^^^

Keys (besides ``methods``):

- ``output_csv`` (**required**) -- path for the per-nonant aggregate report;
  written by cylinder rank 0.
- ``per_scenario_csv`` (optional, default ``null``) -- path for a
  per-(scenario, nonant) detail file; off by default.
- ``warmup_iters`` (``5``) -- do not evaluate until this many ``W`` samples
  exist.
- ``check_every`` (``1``) -- evaluate the detectors every this many iterations
  after warm-up.
- ``report_mode`` -- ``on_detect`` (a row the first time a nonant becomes
  flagged; the default), ``every_check`` (a row at every check), or ``final``
  (one report at the end of the run).
- ``min_scenarios_to_report`` (``1``) / ``min_frac_to_report`` (``null``) -- a
  nonant is reported once at least this many scenarios (or this fraction of the
  scenarios at its node) flag it.

An example is shipped at ``examples/sizes/config/w_oscillation.json``. It
enables both detectors, keeps a 20-iteration window, and (via
``min_frac_to_report`` of ``0.5``) reports a nonant once at least half of the
scenarios at its node flag it:

.. literalinclude:: ../../examples/sizes/config/w_oscillation.json
   :language: json

Detection output
^^^^^^^^^^^^^^^^

The **aggregate** CSV has a header row and one row per flagged nonant per
detection event, with columns::

  iteration, node, variable, method, num_scenarios_total,
  num_scenarios_flagged, max_w_crossings, max_diff_crossings,
  max_diffs_ratio, cycle_period

Method-specific columns are blank for the other method (e.g. ``cycle_period``
is populated only for ``w_hash_recurrence``).

The example above leaves ``per_scenario_csv`` at ``null``, so only the
aggregate report is written. To also emit the per-(scenario, nonant) detail
file -- one row per flagged trace per check, gathered to rank 0 -- set
``per_scenario_csv`` to a path (other keys fall back to their defaults):

.. code-block:: json

    {
        "output_csv": "w_oscillations.csv",
        "per_scenario_csv": "w_oscillations_per_scenario.csv",
        "report_mode": "every_check",
        "methods": {
            "zero_crossings": {}
        }
    }

The detail file has columns ``iteration, node, scenario, variable, method,
w_crossings, diff_crossings, diffs_ratio, w_value``. Only flagged rows are
gathered, so the volume is bounded by what is actually oscillating; on a badly
thrashing problem it can still be large, which is why it is off by default.


Interrupting oscillation
------------------------

Passing ``--interrupt-W-oscillations <file>`` makes the extension *act* on the
nonants it flags, in ``miditer`` before that iteration's solve, so the change
takes effect immediately.

Actions are strictly detection-gated: on an iteration where the detectors flag
no nonant (or none reaches ``min_scenarios_flagged`` scenarios), nothing is
slammed -- the run is untouched. A slam is one-way: the variable it fixes
*stays* fixed for the remainder of the run, even after its oscillation flag
clears (there is no unfix path).

Reporting is opt-in
^^^^^^^^^^^^^^^^^^^

Interruption needs the detection **engine** to know which nonants are cycling,
but it does **not** automatically write the cycling **report**. A pure
``--interrupt-W-oscillations`` run drives the engine to act and announces each
interruption with a log line (see `What you will see`_), and writes **no** CSV.

To also get the report, ask for detection explicitly, in either of two ways:
add ``--detect-W-oscillations <file>`` alongside the interrupt flag, or embed a
``detect`` block inside the interrupt JSON. The ``detect`` block takes the same
keys as a standalone `Detection control file`_ (so ``output_csv`` is still
required, and ``methods`` selects the detectors). Either way the detection
settings you supply serve double duty -- they produce the report *and* become
the engine the interrupter acts on. With neither, a built-in default detector
drives the actions silently.

For example, an interrupt file that also writes the report:

.. code-block:: json

    {
        "action": "slam",
        "trigger": { "start_iter": 100 },
        "slam": {
            "directives_file": "examples/sizes/config/slamming_directives.csv"
        },
        "detect": {
            "output_csv": "w_oscillations.csv",
            "methods": { "zero_crossings": {} }
        }
    }

Actions
^^^^^^^

``action`` (**required**) must be ``slam``:

``slam``
   Fix **one** flagged nonant per slam event -- the highest-priority one that
   can actually be slammed -- via the existing :ref:`slammer <slammer>` action
   layer, with successive slams separated by a cooldown of at least
   ``iters_between_slams`` iterations (default ``3``). Fixing is drastic and
   permanent -- a slammed variable stays fixed for the rest of the run, even
   after its oscillation flag clears -- and fixing just one cycling variable
   often re-settles the others, so even when many nonants are flagged only one
   is slammed per event (which one is decided by the directives file's
   ``priority`` column, not by any measure of oscillation severity). The cooldown matters because the detectors judge a trailing history
   window: a nonant that is re-settling after a fix keeps its flag until the
   old oscillation ages out of the window, so "still flagged" is *not* yet
   evidence of "still cycling" -- the cooldown gives each fix time to work
   before the next variable is fixed (set it to ``1`` to slam on every flagged
   iteration). The cooldown starts only when a slam actually lands; an event
   where nothing was slammable retries on the next flagged iteration. The
   ``slam`` block also names a ``directives_file`` -- a slammer-style
   directives CSV (by-name patterns, a direction such as ``lb`` / ``ub`` /
   ``nearest`` / ``max``, and a ``priority``). Among the flagged nonants the
   slammer picks by that ``priority`` column (largest first, ties by name), so
   the priority ranking decides which one is fixed. Watson-Woodruff §2.4's
   native remedy -- fixing a cycling variable to its per-scenario maximum -- is
   exactly a directives file of ``...,max,...``.

Trigger
^^^^^^^

The ``trigger`` block controls *when* and *which* nonants are acted on:

- ``start_iter`` (``5``) -- the first iteration at which interruption may occur.
  Once past it, slamming is paced by its own ``iters_between_slams`` cooldown
  (see the ``slam`` action above).
- ``min_scenarios_flagged`` (``1``) -- a nonant is acted on once at least this
  many scenarios flag it.

The trigger is independent of the detector's ``warmup_iters`` / ``check_every``
(which govern *reporting*). If you want to avoid acting on early noise, set
``start_iter`` no smaller than ``warmup_iters``.

Interrupt control file
^^^^^^^^^^^^^^^^^^^^^^^

An example is shipped at
``examples/sizes/config/w_oscillation_interrupt.json``. Pair it with the
detection example to also get the report, e.g.::

  --detect-W-oscillations examples/sizes/config/w_oscillation.json
  --interrupt-W-oscillations examples/sizes/config/w_oscillation_interrupt.json

.. literalinclude:: ../../examples/sizes/config/w_oscillation_interrupt.json
   :language: json

What you will see
^^^^^^^^^^^^^^^^^

Every time the extension acts, it prints one rank-0 progress line, for
example::

  [   12.34] W-oscillation interruption [iter 7]: 3 nonant(s) flagged; slammed 1 nonant(s)

This line is always emitted (it does not require ``--verbose``); it is the only
output of a report-less interrupt run. On iterations where the slam cooldown
suppresses a slam, nothing is printed. Detailed per-slam reporting comes from
the slammer itself under ``--verbose``.


Scope, MPI, and limitations
---------------------------

- **Synchronous PH only.** See the note at the top of this page.
- **MPI / distribution independence.** The aggregate report and the
  per-nonant flagged set are computed with per-node ``SUM`` / ``MAX``
  reductions (and, for ``w_hash_recurrence``, a distribution-independent
  additive signature), so they are identical on every rank regardless of the
  scenario-to-rank mapping. The interrupter acts on that rank-identical flagged
  set in a fixed order, so the slammer's per-node ``min`` / ``max`` reduction is
  reached symmetrically on every rank.
- **Multistage.** Detection and reporting iterate the scenario tree node by
  node and support multistage problems. The action selection is rank-coherent for
  two-stage problems and single-rank-per-node multistage; a node split across
  ranks would need an extra reduction to agree on the action, which is not done
  (the same limitation the slammer documents).

See ``doc/designs/w_oscillation_design.md`` for the full design and rationale.
