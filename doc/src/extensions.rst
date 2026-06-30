.. _Extensions:

Extensions
==========

In order to allow for extension or modification of code behavior, many of
the components (hubs and spokes) support callout points. The objects
that provide the methods that are called are referred to as `extensions`.
Instead of using an extension, you could just hack in a function call,
but then every time ``mpi-sppy`` is updated, you would have to remember
to hack it back in. By using an extension object, the addition
or modification will remain available. Perhaps more important, the
extension can be included in some applications, but not others.

There are a number of extensions, particularly for PH, that are provided
with ``mpi-sppy`` and they provide examples that can be used for the
creation of more. Extensions can be found in ``mpisppy.extensions``.
Note that some things (e.g. some xhatters) can be used as a cylinder
or as an extension. A few other things (e.g., cross scenario cuts) need
both an extension and a cylinder.

Many extensions are supported in :ref:`generic_cylinders` via
command-line flags:

- ``--fixer`` -- activates the fixer extension
- ``--slamming-directives-file <file>`` -- activates the slammer extension
- ``--detect-W-oscillations <file>`` -- activates W-oscillation detection
  (see :ref:`w_oscillation`)
- ``--interrupt-W-oscillations <file>`` -- activates W-oscillation
  interruption (rho reduction and/or slamming; implies detection;
  see :ref:`w_oscillation`)
- ``--mipgaps-json <file>`` -- activates the mipgapper extension
- ``--user-defined-extensions <module>`` -- loads a custom extension
- ``--grad-rho`` -- activates gradient-based rho (see :ref:`rho_setting`)
- ``--use-norm-rho-updater`` -- activates the norm rho updater
- ``--use-primal-dual-rho-updater`` -- activates the primal-dual rho updater

The rest of this help file describes extensions released with mpisppy along
with some hints for including them in your own cylinders driver program.

Multiple Extensions
-------------------

To employ multiple PH extensions, use ``mpisppy.extensions.extension import MultiExtension``
that allows you to give a list of extensions that will fire in order
at each callout point. See, e.g. ``examples.sizes.sizes_demo.py`` or
``examples.farmer.CI.farmer_rho_demo.py`` for an
example of use.

.. note::
   The ``MultiExtension`` constructor in ``mpisppy.extensions.extensions.py``
   takes a list of extensions classes in addition to the optimization object
   (e.g. inherited from ``PHBase``). However, ``cfg_vanilla.py`` wants
   to see the class ``MultiExtension`` in the hub or cylinder dict entry
   for ``["opt_kwargs"]["extensions"]`` and then wants to see a list of
   extension classes in ``["opt_kwargs"]["extension_kwargs"]["ext_classes"]``.
   Some examples do both, which can be little confusing.


PH extensions
-------------

Some of these can be used with other hubs. An extension object can be
passed to the PH constructor and it is assumed to have methods defined
for all the callout points in PH (so all of the examples do). To see 
the callout points look at ``phbase.py``. Extensions can also specify
callout points in the `Hub` `SPCommunicator` object: these callout points
are especially useful for writing custom `Spoke` objects which can then
interact with the hub PH object. To see the callout points look at
``cylinders/hub.py``; an example of such an extension is the
"cross-scenario cut" extension defined in ``extensions/cross_scen_extension.py``
and associated spoke object defined in ``cylinders/cross_scen_spoke.py``.

If you want to use more than one extension, define a main extension that has
a reference to the other extensions and can call their methods in the
appropriate order. Extensions typically access low level elements of
``mpi-sppy`` so writing your extensions is an advanced topic. We will
now describe a few of the extensions in the release.

mipgapper.py
^^^^^^^^^^^^

This is a good extension to look at as a first example. It takes a
dictionary with iteration numbers and mipgaps as input and changes the
mipgap at the corresponding iterations. The dictionary is provided in
the options dictionary in ``["gapperoptions"]["mipgapdict"]``.  There
is an example of its use in ``examples.sizes.sizes_demo.py``.

Instead of an options dictionary, when run with cylinders the options
``["gapperoptions"]["starting_mipgap"]`` and ``["gapperoptions"]["mipgap_ratio"]``
can be set. The ``starting_mipgap`` will be the initial value used,
and as the cylinders close the relative optimality gap the extension will set the subproblem
mipgaps as the ``min(starting_mipgap, mipgap_ratio * problem_ratio)``, where
the ``problem_ratio`` is the relative optimality gap on the overall problem
as computed by the cylinders.

This extension can also be used with the Lagrangian and subgradient spokes.

fixer.py
^^^^^^^^

This extension provides methods for fixing nonanticipative variables (usually integers) for
which all scenarios have agreed for some number of iterations. There
is an example of its use in ``examples.sizes.sizes_demo.py`` also
in ``examples.uc.uc_ama.py``. The ``uc_ama`` example illustrates
that when ``amalgamator`` is used ``"id_fix_list_fct"`` needs
to be on the ``Config`` object so the amalgamator can find it.

.. note::

   For the iteration zero fixer tuples, the iteration counts are just
   compared with None. If you provide a count for iteration zero, the
   variable will be fixed if it is within the tolerance of being converged.
   So if you don't want to fix a variable at iteration zero, provide a
   tolerance, but set all count values to ``None``.

reduced_cost_fixer
^^^^^^^^^^^^^^^^^^

This extension provides methods for fixing nonanticipative variables based on their expected
reduced cost as calculated by the ReducedCostSpoke. The aggressiveness of the
fixing can be controled through the ``zero_rc_tol`` parameter (reduced costs
with magnitude below this value will be considered 0 and not eligible for fixing)
and the ``fix_fraction_target`` paramemters, which set a maximum fraction of
nonanticipative variables to be fixed based on expected reduced costs. These two
parameters iteract with each other -- the expected reduced costs are sorted by
magnitude, and if the `fix_fraction_target`` percental is below ``zero_rc_tol``,
then fewer than ``fix_fraction_target`` variables will be fixed. Further, to
have a defined expected reduced cost, all nonant variable values *must be* at
the same bound in the ReducedCostSpoke.

Variables will be unfixed if they no longer meet the expected reduced cost
criterion for fixing, e.g., the variable's expected reduced cost became too
low or the variable was not at its bound in every subproblem in the ReducedCostSpoke.

relaxed_ph_fixer
^^^^^^^^^^^^^^^^

This extension will fix nonanticipative variables at their bound if they are at
their bound in the RelaxedPHSpoke for that subproblem. It will similarily unfix
nonanticipative variables which are not at their bounds in the RelaxedPHSpoke.
Because different nonanticipative variables are fixed in different suproblems,
it will also unfix nonanticipative variables if their value is *not* at the the current
consensus solution xbar (because the variable was not fixed in a different subproblem
and therefore came off its bound).

xhat
^^^^

Most of the xhat methods can be used as an extension instead of being used
as a spoke, when that is desired (e.g. for serial applications).

integer_relax_then_enforce
^^^^^^^^^^^^^^^^^^^^^^^^^^

This extension is for problems with integer variables. The scenario subproblems
have the integrality restrictions initially relaxed, and then at a later point
the subproblem integrality restrictions are re-enabled. The parameter ``ratio``
(default = 0.5) controls how much of the progressive hedging algorithm, either
in the iteration or time limit, is used for relaxed progressive hedging iterations.
The extension will also re-enforce the integrality restrictions if the convergence
threshold is within 10\%  of the convergence tolerance.

This extension can be especially effective if (1) solving the relaxation
is much easier than solving the problem with integrality constraints or (2) the
relaxation is reasonably "tight".

.. _slammer:

slammer
^^^^^^^

This extension does preference-driven *slamming*: it forces (fixes) a
nonanticipative variable according to pre-specified user preferences while the
hub is running, to break a stall or cycle. Unlike the other fixers above --
which fix on *agreement* and so can infer a direction automatically -- slamming
is meant for variables that are *not* settling, where there is no agreement to
read a direction from, so the directions are supplied by the user in a
directives file. This is the intended use rather than an enforced precondition:
slamming does not test convergence per variable; any variable matched by a
``can_slam`` rule is eligible (see the eligibility rules below). (Slamming is
distinct from the ``SlamMin`` / ``SlamMax`` *spokes*, which are non-destructive
incumbent finders that never perturb the hub.)

From ``generic_cylinders.py`` the extension is activated **only** when a
directives file is supplied, so a run with no slamming options behaves exactly
as it does today:

- ``--slamming-directives-file <file>`` -- the directives file (its presence
  activates the extension)
- ``--slam-start-iter <K>`` -- first hub iteration at which slamming may occur
  (default 1)
- ``--iters-between-slams <M>`` -- once started, slam at most once every ``M``
  iterations (default 1)

Supplying ``--slam-start-iter`` or ``--iters-between-slams`` without the file is
an error.

The directives file is a CSV keyed by nonant name with shell-style wildcards
(``*``/``?``; the index brackets ``[`` ``]`` are matched literally). Each row
gives a name pattern, an optional ``can_slam`` flag (``1``/``0``; ``0`` carves
out an exception), an ordered ``|``-separated list of ``directions`` from
``{lb, ub, nearest, anywhere, min, max}`` (the first applicable one is used),
and a ``priority`` (the eligible nonant with the largest priority is slammed
first). Matching is last-match-wins, so write broad defaults first and
exceptions after; only variables matched by a ``can_slam`` rule are ever
slammed. A multi-index name contains a comma and so must be quoted. A pattern
that matches no variable in the model is a hard error (it is almost always a
typo) and the message names the file. A worked example translated from PySP
ships at ``examples/sizes/config/slamming_directives.csv``.

The available ``directions``, and the value each fixes the variable to, are:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Direction
     - Value the variable is fixed to
   * - ``lb``
     - the variable's lower bound
   * - ``ub``
     - the variable's upper bound
   * - ``nearest``
     - whichever bound (``lb`` or ``ub``) the current consensus value ``xbar``
       is closer to
   * - ``anywhere``
     - ``xbar`` itself, rounded to the nearest integer for integer and binary
       variables (using ``--rounding-bias``)
   * - ``min``
     - the minimum of the variable's value across all scenarios that share its
       scenario-tree node
   * - ``max``
     - the maximum of the variable's value across all scenarios that share its
       scenario-tree node

A direction is *applicable* only when it produces a finite value (for example,
``lb`` is skipped for a variable that has no lower bound). The directions in a
row are tried in order and the first applicable one is used; if none applies,
that variable is not slammed at this event.

Slamming is triggered by iteration count: no variable is slammed before
``--slam-start-iter``, and from then on at most one variable is slammed every
``--iters-between-slams`` iterations. At each such event the single eligible
variable with the largest ``priority`` is slammed (ties are broken by name, so
the choice is the same on every rank). A variable is eligible only if it is
matched by a ``can_slam`` rule, is not already fixed (by the modeler or another
fixer), and is not a surrogate variable. Once a variable is slammed it stays
fixed for the remainder of the run.

The directions ``lb``, ``ub``, ``nearest``, and ``anywhere`` use only data that
is already identical on every rank, so they need no communication; ``min`` and
``max`` perform a single small reduction across scenarios, and only when such a
slam actually occurs. See ``doc/designs/slamming_design.md`` for design details
and rationale.

WXBarWriter and WXBarReader
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is an extension to write xbar and W values and another to read them.
An example of their use is shown in ``examples.sizes.sizes_demo.py``

norm_rho_updater
^^^^^^^^^^^^^^^^

This extension adjust rho dynamically. The code is in ``mpisppy.extensions.norm_rho_updater.py``
and there is an accompanying converger in ``mpisppy.convergers.norm_rho_converger``. From
``generic_cylinders.py``, enable it with ``--use-norm-rho-updater``; a
hand-wired example using the underlying classes directly is preserved in
``examples.farmer.archive.farmer_cylinders.py``. This is the original
Gabe H. dynamic rho.


rho_setter
^^^^^^^^^^

Per variable rho values (mainly for PH) can be set using a function
that takes a scenario (a Pyomo ``ConcreteModel``) as its only
argument. The function returns a list of (id(vardata), rho)
tuples. The function name can be given the the ``vanilla.ph_hub``
constructor or in the hub dictionary under ``opt_kwargs`` as the
``rho_setter`` entry. (The function name is ultimately passed to the
``phabase`` constructor.)

There is an example of the function in the sizes example (``_rho_setter``).

SepRho
^^^^^^

Set per variable rho values using the "SEP" algorithm from

Progressive hedging innovations for a class of stochastic mixed-integer resource allocation problems
Jean-Paul Watson, David L. Woodruff, Compu Management Science, 2011
DOI 10.1007/s10287-010-0125-4

One can additional specify a multiplier on the computed value (default = 1.0).
If the cost coefficient on a non-anticipative variable is 0, the default rho value is used instead.

CoeffRho
^^^^^^^^

Set per variable rho values proportional to the cost coefficient on each non-anticipative variable,
with an optional multiplier (default = 1.0) that is applied to the computed value. If the coefficient is 0, the default rho value is used instead.

primal_dual_rho
^^^^^^^^^^^^^^^

Increase or decrease rho for every variable to keep primal and dual convergence balance. If
the primal residual is greater than ``update_threshold`` times the dual residual, then all
rhos are increased by the ``update_threshold``, and conversely all rhos are decreased if
the dual residual is greater than ``update_threshold`` time the primal residual. The user
can also specify a ``primal_bias`` (default 1.0) which will emphasize primal convergence
when greater than 1 and emphasize dual convergence if less than 1.

This extension is especially useful if the rhos provided by the user (or some other extension)
are believed to be "in balance", such that per-variable updates are not needed (and can sometimes
hinder algorithmic progress when different nonanticipative variables play similar roles in
the subproblem optimization problems).

.. _wtracker_extension:

wtracker_extension
^^^^^^^^^^^^^^^^^^

The wtracker_extension outputs a report about the convergence (or really, lack thereof) of
W values.
An example of programmatic use is shown in ``examples.sizes.sizes_demo.py``.

From ``generic_cylinders.py``, enable it with ``--wtracker``. Related options:

- ``--wtracker`` -- enable the extension (default off)
- ``--wtracker-file-prefix <str>`` -- prefix for the rank-by-rank output
  files (default: empty)
- ``--wtracker-wlen <int>`` -- moving-window length, in iterations,
  used by the convergence statistics (default 20)
- ``--wtracker-reportlen <int>`` -- maximum number of rows in each
  ranked report (default 100)
- ``--wtracker-stdevthresh <float>`` -- ignore moving standard deviations
  below this value when counting "converged" traces (default: use
  ``E1_tolerance``)

At the end of the run, each rank writes three files using its prefix:

- ``<prefix>_summary_iter<N>_rank<r>.txt`` -- text summary with counts
  and totals
- ``<prefix>_stdev_iter<N>_rank<r>.csv`` -- top entries sorted by
  moving standard deviation
- ``<prefix>_cv_iter<N>_rank<r>.csv`` -- top entries sorted by moving
  absolute coefficient of variation

Each CSV row is indexed by ``(varname, scenario_name)`` and gives the
windowed mean and stdev of W for that nonant/scenario pair. This is a
diagnostic tool intended for tuning rho and convergence behavior; it
adds time and memory and is not recommended for production runs.


.. _w_oscillation:

w_oscillation
^^^^^^^^^^^^^

The ``w_oscillation`` extension (``mpisppy.extensions.w_oscillation``)
*detects* oscillation / cycling in the PH dual weight (W) vector, reports
it to a CSV, and can optionally *interrupt* it. Oscillating (sign-flipping,
non-damping) weights are a common, convergence-killing symptom for
mixed-integer problems. With only ``--detect-W-oscillations`` the extension
is **pure observation**: it changes no rho values and fixes no variables, so
a run with detection enabled follows exactly the same optimization trajectory
as one without. With ``--interrupt-W-oscillations`` it additionally acts on
the detected oscillation (see `Interrupting oscillation`_ below).

For a broader view of how W evolves -- moving means, standard deviations,
and coefficient of variation across all nonant/scenario traces -- use the
:ref:`wtracker_extension`; ``w_oscillation`` is the focused layer that
flags the specific traces that are *cycling*.

From ``generic_cylinders.py``, enable it with::

  --detect-W-oscillations <control.json>

The JSON control file selects and parameterizes the detection methods and
controls the output. Keys:

- ``output_csv`` (required) -- the per-nonant aggregate report; written by
  cylinder rank 0.
- ``per_scenario_csv`` (optional) -- a per-(scenario, nonant) detail file
  (gathered to rank 0); off by default.
- ``warmup_iters`` -- do not evaluate until this many W samples exist
  (default 5).
- ``check_every`` -- evaluate the detectors every this many iterations
  after warm-up (default 1).
- ``report_mode`` -- ``on_detect`` (a row the first time a nonant is
  flagged; the default), ``every_check``, or ``final`` (one report at the
  end).
- ``min_scenarios_to_report`` / ``min_frac_to_report`` -- a nonant is
  reported once at least this many scenarios (or this fraction of them)
  flag it.
- ``methods`` -- a (non-empty) object selecting one or both detectors;
  per-method keys override documented defaults.

Two detection methods are available:

- ``zero_crossings`` -- a port of PySP's ``sorgw`` plugin: per
  (scenario, nonant) it counts sign changes of W and of the consecutive
  differences, plus a back-half/front-half damping ratio of ``|ΔW|``, and
  flags a trace when any of ``thresh_w_crossings`` (default 2),
  ``thresh_diff_crossings`` (default 3), or ``thresh_diffs_ratio``
  (default 0.2) is met.
- ``w_hash_recurrence`` -- the Watson-Woodruff "Progressive Hedging
  Innovations" (Computational Management Science, 2011) §2.4 cycle
  detector: it hashes the per-scenario W vector for each nonant and flags
  a *recurrence* of that vector (a genuine cycle of period
  ``min_period`` or more, so convergence is not mistaken for a cycle).
  Keys: ``window`` (default 20), ``quantum`` (default 1e-6),
  ``min_period`` (default 2).

An example control file is shipped at
``examples/sizes/config/w_oscillation.json``. It enables both detectors,
keeps a 20-iteration window, and reports a nonant once at least half of the
scenarios (``min_frac_to_report``) flag it:

.. literalinclude:: ../../examples/sizes/config/w_oscillation.json
   :language: json

That example leaves ``per_scenario_csv`` at ``null``, so only the aggregate
report is written. To also emit the per-(scenario, nonant) detail file --
one row per flagged trace per check, gathered to rank 0 -- set
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
w_crossings, diff_crossings, diffs_ratio, w_value``.

The aggregate CSV has a header row and one row per flagged nonant per
detection event, with columns ``iteration, node, variable, method,
num_scenarios_total, num_scenarios_flagged, max_w_crossings,
max_diff_crossings, max_diffs_ratio, cycle_period``. The report is
independent of how scenarios are distributed across MPI ranks.

Interrupting oscillation
""""""""""""""""""""""""

Passing ``--interrupt-W-oscillations <file>`` makes the extension *act* on
the nonants it flags, to break the cycle. Interruption implies detection, so
the detection report is still written; if ``--detect-W-oscillations`` is not
also given, the detection configuration is taken from an optional ``detect``
block inside the interrupt file, otherwise a default detector is used. The
interrupt JSON keys are:

- ``action`` (required) -- ``rho_reduction``, ``slam``, or ``both``. The
  recommendation is to choose one; ``both`` simply applies each in turn (a
  nonant slamming has fixed is unaffected by a subsequent rho change).
- ``trigger`` -- when to act: ``start_iter`` (default 5), then every
  ``iters_between_actions`` iterations (default 3); a nonant is acted on once
  at least ``min_scenarios_flagged`` (default 1) scenarios flag it.
- ``rho_reduction`` (for ``rho_reduction`` / ``both``) -- multiply each
  flagged nonant's rho by ``factor`` (default 0.5, must be in ``(0, 1)``),
  floored at ``min_rho`` (default 1e-3, must be ``> 0`` since PH requires a
  positive rho). Reducing rho relaxes the proximal pull that drives the
  overshoot.
- ``slam`` (for ``slam`` / ``both``) -- ``directives_file``, a
  :ref:`slammer <slammer>`-style directives CSV. The extension drives the
  slammer's action layer directly on the flagged nonants (rather than the
  slammer's own iteration-count trigger). Watson-Woodruff §2.4's native
  remedy -- fixing a cycling variable to its per-scenario maximum -- is just a
  directives file of ``...,max,...``.

An example is shipped at
``examples/sizes/config/w_oscillation_interrupt.json``; pair it with the
detection example, e.g.::

  --detect-W-oscillations examples/sizes/config/w_oscillation.json
  --interrupt-W-oscillations examples/sizes/config/w_oscillation_interrupt.json

.. literalinclude:: ../../examples/sizes/config/w_oscillation_interrupt.json
   :language: json

Interruption is for synchronous PH; like detection it is a hub extension and
leaves the spokes untouched.

See ``doc/designs/w_oscillation_design.md`` for the full design.


gradient_extension
^^^^^^^^^^^^^^^^^^
The gradient_extension sets gradient-based rho for PH. From
``generic_cylinders.py``, enable it with ``--grad-rho``; a standalone
demo is in ``examples.farmer.CI.farmer_rho_demo.py``. There are options
in ``cfg`` to control dynamic updates.

mult_rho_updater
^^^^^^^^^^^^^^^^

This extension does a simple multiplicative update of rho; consequently, the update is cumulative.

cross-scenario cuts
^^^^^^^^^^^^^^^^^^^
Two-stage models only. This extension adds cross scenario cuts as calculated
by the cross-scenario cut spoke. See the implementation paper for details.
A hand-wired example using the underlying classes is preserved in
``examples/farmer/archive/cs_farmer.py``.


Distributed Subproblem Presolve
===============================
This functionality is available for all Hub and Spoke algorithms which inherit from
``SPBase``. It can be enabled by passing ``presolve=True`` into the constructor.

Leveraging the existing feasibility-based bounds tightening (FBBT) available in Pyomo, this
presolver will tighten the bounds on all variables, including the non-anticipative variables.
If the non-anticipative variables have different bounds, the bounds among the non-anticipative
variables will be synchronized to utilize the tightest available bound.

In its current state, the user might opt-in to presolve for two reasons:

1. For problems without relatively complete recourse, utilizing the tighter bounds on the
   non-anticipative variables and speed convergence and improve primal and dual bounds. In
   rare cases it might also detect infeasibility.

2. For problems where a "fixer" extension or spoke is used, determining tight bounds on the
   non-anticipative variables may improve the fixer's performance.

.. Note::
   Like many solvers, the presolver will convert infinite bounds to 1e+100.

.. Note::
   This capability requires the auto-persistent pyomo solver interface (APPSI) extensions
   for Pyomo to be built on your system. This can be achieved by running ``pyomo build-extensions``
   at the command line.

.. Note::
   The APPSI capability in Pyomo is under active development. As a result, the presolver
   may not work for all Pyomo models.


variable_probability
====================

This is experimental as of February 2021; use with caution.  The main use-case is
to allow zero-probability variables.

A function similar to ``rho_setter`` can be passed to the ``SPBase``
constructor via the ``PHBase`` construtor as the
``variable_probability`` argument to allow for per variable
probability specification. So it can be passed through by ``vanilla``
via ``ph_hub``. The function should return (vid, probability) pairs.
If the function needs arguments, pass them via
the ``SPBase`` option ``variable_probability_kwargs``

The variable probabilities impact the computation of
``xbars`` and ``W``.

.. Note::
   The only xhatter that is likely to work with variable probabilities is xhatxbar. The others
   are likely to execute without error messages but will not find good solutions.


Objective function considerations
---------------------------------

If variables with by-variable probability are in the objective function, it is
up to the scenario creator code to deal with it. This is not so difficult for
zero-probability variables.

zero-probability variables
--------------------------

When you
create the scenario, you probably want to fix zero probability variables and perhaps give
them a zero coefficient if they appear in the objective. Fixed
variables will not get a nonanticipativity constraint in bundles. If you
create the EF directly, you probably want to set
``nonant_for_fixed_vars`` to `False` in the call to ``create_EF``. If
you are not calling ``create_EF`` directly, but rather using the
``mpisppy.opt.ef.ExtensiveForm`` object, add ``nonant_for_fixed_vars``
to the dict passed as its ``options`` argument with the value
``False``.

.. Note::
   The ``W`` value for a zero-probability variable will be stay at zero.


Fixed variables may cause trouble if you are relying on the internal
PH convergence metric.

.. Note::
   You must declare variables to be in the nonant list even for those scenarios where they have
   zero probability if they are in other scenarios that share a scenario tree node at the variable's stage.


If some variables have zero probability in all scenarios, then you will need to set the option
``do_not_check_variable_probabilities`` to True in the options for ``spbase``. This will result in skipping the checks for
all variable probabilities! So you might want to set this to False to verify that the probabilities sum to one
only for the Vars you expect before setting it to True.

Scenario_lp_mps_writer_dir
--------------------------

This extension writes an lp file and an mps file with the model, a
json file with (a) list(s) of scenario tree node names and
nonanticaptive variables, and a ``{scenario}_rho.csv`` file with the
per-nonant rho values, for each scenario before the iteration zero
solve of PH or APH. Note that for two-stage problems, all json files
(and all rho files) will be the same. See ``mpisppy.generic_cylinders.py``
for an example of use. In that program it is activated with the
``--scenario-lp-mps-writerdir`` option that specifies a directory that
does not exist. The extension
writes the files to this directory and for each scenario
the base name of the four files written is the scenario name.

Unless you know exactly why you need this, you probably don't.
