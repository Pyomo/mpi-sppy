.. _checkpointing:

Checkpointing and Resuming a Run
================================

A long Progressive Hedging run can be stopped and picked up later. The intended
use is a planned stop: a multi-day study that ends each day and resumes the next
morning on the same cluster, losing at most the work done since the last
checkpoint.

Two words are used throughout this page. A **run** is one execution of
the software. A **study** is the whole piece of work: one run, or several linked
by checkpoints. Iteration *numbers* belong to the study and carry across a
resume, while the two ways to bound the work are named for what each one
counts: ``--max-iterations`` bounds the run being started, and
``--stop-at-iteration-number`` bounds the study. A run ends at whichever of
them arrives first.

A resumed run picks up from the last iteration that has a **published** checkpoint, which is
not always the iteration at which the previous run stopped. How far back that is
set by ``--checkpoint-every-iterations``; read `Choosing K`_ before relying on
this for anything expensive.

Checkpointing is entirely opt-in. With neither ``--checkpoint-dir`` nor
``--resume-from`` the machinery is not attached at all, and a run that does not
ask for it pays nothing.

.. note::
   The current implementation covers a **synchronous PH hub**, run on its own
   or with spokes, on any number of ranks per cylinder, with plain scenarios,
   proper bundles, or stoch-ADMM. Other hub types (notably ``--APH``) are
   refused at startup. See ``doc/designs/checkpointing_design.md`` for the full
   design and the phased rollout.

Writing a checkpoint
--------------------

Give a directory and the run keeps a checkpoint up to date as it goes::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --time-limit 28800 \
      --checkpoint-dir ./ckpt --checkpoint-every-iterations 10

A checkpoint always describes a **completed** PH iteration -- never a partial
one -- and only one is kept at a time, so the directory holds the most recent
iteration that was both completed and due a write. Leaving
``--checkpoint-every-iterations`` off makes every completed iteration due one,
which is rarely what you want on a real model -- hence the ``10`` above, and
`Choosing K`_.

``--time-limit`` is one way to end a run without counting iterations, and the
easiest to schedule a day around; ``--rel-gap``, ``--abs-gap``, the convergence
threshold and a user converger all end one just as cleanly. It is how you get a
run to stop on its own before a scheduler kills it, but be clear about what it
does: the elapsed time is compared against
the limit **once per iteration, at the top**, and nowhere else. It is not
checked during a solve. So a run whose iterations take an hour can overshoot the
limit by nearly an hour, and you have to set the limit below your wall-clock
allocation by at least one iteration's worth of solve time for it to help at
all.

It also does not force a checkpoint to be written. That is what
``--checkpoint-before-seconds`` is for (`Guarding against a scheduler
timeout`_).

The options
~~~~~~~~~~~

There are six, and they are the whole interface:

``--checkpoint-dir DIR``
   Where checkpoints are written. Giving it is what turns writing on. Without
   it or ``--resume-from`` none of this machinery is attached; with
   ``--resume-from`` alone it is attached with writing switched off, so the
   run reads a checkpoint and writes none.

``--checkpoint-every-iterations K``
   Write at every K-th completed iteration instead of every one. The default is
   ``1``, but on a real model you will want it higher -- see `Choosing K`_,
   which is the one decision here that repays some thought. Leaving it off does
   not turn iteration-driven writes off: it writes at every iteration, which is
   the safest cadence and the most expensive one.

``--checkpoint-before-seconds S``
   Write once, at the first iteration boundary where one more iteration as
   long as the last would carry the run past S seconds from its start. For a
   run that will be stopped by a clock rather than by its iteration limit --
   see `Guarding against a scheduler timeout`_.

``--stop-at-iteration-number T``
   End the study at iteration ``T``, counted across every run linked by
   checkpoints. Unset by default, which leaves ``--max-iterations`` -- the
   bound on this run alone -- as the only thing that ends a run. Setting it is
   how you say "finish at 500 however many mornings that takes"; see
   `Resuming`_.

``--resume-from DIR``
   Start from the checkpoint in ``DIR`` instead of from scratch. Iteration
   numbering carries on from there rather than restarting, because the numbers
   count the study rather than this run.

``--checkpoint-backend``
   How model state is stored. ``dill-reload`` is the default and currently the
   only implemented value, so there is no reason to set it.

.. warning::
   **Nothing is published until the first iteration completes.** A run killed
   during startup, or during the iteration-0 solve, leaves no checkpoint at all
   -- not an empty one, nothing to resume from. For the case this feature exists
   for, large MIP subproblems, that iteration-0 solve is often the longest in
   the whole run, so the window is not small.

Choosing K
~~~~~~~~~~

**What K means.** A checkpoint is written at the end of a completed iteration
whose number is a multiple of K. The numbers are the PH iteration numbers you
see in the log, counted from the start of the study, so with ``K = 10`` the
checkpoints are iterations 10, 20, 30, and so on. K is not a countdown from
whenever the current run happened to begin: it does not restart at a resume,
and there is no drift. The default is ``1``, which writes at every iteration.

For example, with ``K = 10`` a run stopped by ``--time-limit`` after 34
completed iterations leaves a checkpoint of iteration 30, and iterations 31
through 34 are lost. Resuming picks up at 31 and the next checkpoint is
iteration 40 -- not 41, because the count follows the study, not the run that
resumes it. Had that run instead stopped at 34 by exhausting ``--max-iterations``,
iteration 34 would have been written as well, for the reason in the next
paragraph.

**The one exception.** The final iteration of an exhausted iteration budget is
always written, whatever K is -- either budget, whichever ended the run. With
``--max-iterations 100`` and ``K = 30`` the checkpoints are iterations 30, 60,
90 and 100. Resuming to carry a study further is ordinary, and that last
iterate is known-good and already in memory, so it is not worth discarding to
save one write. No other kind of stop can be caught this way: a time limit,
``--rel-gap``, ``--abs-gap``, the convergence threshold and a user converger
are all tested partway through the *next* iteration, by which point there is
nothing coherent left to write.

Changing K between a stop and a resume is allowed; like the iteration limit,
it describes how the run is managed rather than what problem is being solved,
so it is not part of the check that decides whether a checkpoint may be
resumed.

Two cases worth calling out
"""""""""""""""""""""""""""

**Setting K equal to the iteration limit.** With ``--max-iterations 40
--checkpoint-every-iterations 40`` you get exactly one checkpoint, at iteration
40, and pay for exactly one write. If all you want is the option of extending
the study later by resuming for more iterations, that is the cheapest way to
get it.

The catch is that it only works when the iteration limit is what stops the run.
Anything else -- ``--rel-gap``, ``--abs-gap``, the convergence threshold, a user
converger, ``--time-limit``, a crash -- stops it before iteration 40, no write
has happened, and **there is nothing to resume from**. That failure does not
announce itself. The checkpoint directory exists, and in a cylinders run it
already holds a ``spokes/`` subdirectory, because the xhat spokes write their
incumbents on a different trigger. So it looks like checkpointing worked, and
you find out the next day::

  CheckpointMismatch: No checkpoint manifest at './ckpt/manifest.json'.

If you want the cheap single checkpoint, either turn the other stopping
criteria off (``--rel-gap 0.0 --abs-gap 0.0``) or accept that an early
convergence may leave you nothing. 

.. _Guarding against a scheduler timeout:

**Guarding against a scheduler timeout.** On a batch system the thing to avoid
is the scheduler killing the job partway through a solve. 

``--time-limit`` does not force a write. On its own, what you resume from is
the last multiple of K, exactly as for any other stop. 

``--checkpoint-before-seconds S`` closes that gap::

  ... --max-iterations 500 --time-limit 28800 \
      --checkpoint-dir ./ckpt --checkpoint-every-iterations 10 \
      --checkpoint-before-seconds 28800

At the end of each completed iteration it asks whether *another* iteration that takes
as long as the most recent iteration
would take the run past S seconds since the run started, and writes now if it would.

It writes at most once. 

**S is yours to size, and nothing is added to it.** In particular the checkpointing
itself is not: it takes as long as any other checkpoint of your models, and it
starts when S has all but arrived. Read a write's cost off the ``toc`` lines in
your own log (`What it costs`_) and leave room for it, along with anything else
that must happen before the scheduler's axe falls. Setting S equal to
``--time-limit``, as above, is the usual choice when ``--time-limit`` is in use: the write then lands at the
first iteration boundary where one more iteration as long as the last would
cross the time limit. Because it writes only once, iterations the run still
completes after it -- when later ones turn out faster -- are covered only by
the every-K writes.

Writing only at iteration boundaries is deliberate. PH computes xbar, updates
the dual weights, gives extensions their mid-iteration hook, and only then
solves; a run that stops on ``--time-limit`` or on convergence stops *before*
that solve, leaving the model describing half an iteration. Rather than try to
unwind that -- an open-ended problem, since any extension may have changed rho,
fixed variables or added cuts -- the checkpoint is simply taken at the last
point where everything agrees.

It is also why a run that ends before finishing iteration 1 publishes nothing,
as `The options`_ warns: no iteration completed, so there is no iterate to
write.

Each write is bracketed by a pair of timestamped ``toc`` lines, so the log shows
how long it took::

  [ 1234.56] Writing checkpoint at iteration 42 to ./ckpt
  [ 1261.03] Checkpoint written at iteration 42

A write that fails mid-run -- the disk filling up, a network filesystem
hiccup -- does not stop the optimization. The failure is reported loudly,
the previously published checkpoint stays intact and resumable, and
the next checkpoint point tries again: the next multiple of
``--checkpoint-every-iterations``, the ``--checkpoint-before-seconds`` write
if it has not happened yet, or the last iteration of the budget, whichever
comes first, if the run gets that far. A run that stops on convergence, on
the gap between the cylinders' bounds, on ``--max-stalled-iters`` or at
``--time-limit`` writes nothing more. A failed ``--checkpoint-before-seconds``
write is not retried: that trigger fires once. A write that fails at the last
iteration of the budget is not retried either; a resume starts from the
checkpoint before it. Conditions detectable at setup (an unwritable directory,
a model that cannot be serialized) still stop the run at startup, before any
solving is done.

Resuming
--------

Point a new run at the directory::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --resume-from ./ckpt

The resumed run continues from the checkpointed iterate rather than starting
over. It does **not** re-solve the subproblems at iteration 0.
Iteration numbering continues where it left off, but the budget
does not: ``--max-iterations`` bounds the run you are starting. To resume a run
that stopped at iteration 4 and do two more, pass ``--max-iterations 2``, and
the study ends at iteration 6.

You can use ``--stop-at-iteration-number`` to say where the *study* should end
and mpi-sppy will work the rest out::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --stop-at-iteration-number 500 --resume-from ./ckpt --checkpoint-dir ./ckpt

That run does at most 100 iterations, and stops earlier if it reaches
iteration 500 of the study. Submitting the same command each morning walks a
500-iteration study forward in 100-iteration days without anyone having to
subtract. The ``--checkpoint-dir`` is what makes that work: ``--resume-from``
only reads, so without it each morning would resume from the same checkpoint
and redo the same 100 iterations. A run that starts from a checkpoint at or
past 500 reports that the study is already finished and does nothing, rather
than quietly running on.

Trying it out
-------------

The quickest way to convince yourself is to run the same problem twice -- once
straight through, once stopped and resumed -- and compare. On ``farmer``, a
deterministic LP, the two should agree exactly.

All three runs below share the same options; only the last line of each
differs::

  GC="mpiexec -np 3 python -m mpi4py -m mpisppy.generic_cylinders \
      --module-name farmer --num-scens 3 --default-rho 1 \
      --solver-name cplex --lagrangian --xhatshuffle \
      --rel-gap 0.0 --abs-gap 0.0"

Run them from ``examples/farmer``, or use ``--module-name
mpisppy.tests.examples.farmer`` from anywhere.

The two gap options are there because this example is small enough to converge
on the inter-cylinder gap at iteration **1**, which would leave nothing to stop
and resume. Turning them off makes the iteration limit the thing that ends the
run. That is a property of the example, not of checkpointing.

**1. A reference run** of six iterations, with no checkpointing at all -- this
is the answer to compare against::

  $GC --max-iterations 6 --solution-base-name ref_soln

**2. The same run, stopped after four iterations**, leaving a checkpoint::

  $GC --max-iterations 4 --checkpoint-dir ./ckpt

**3. Resume it** and let it finish the remaining two iterations -- two, not
six, because the limit is this run's::

  $GC --max-iterations 2 --resume-from ./ckpt --solution-base-name resumed_soln

``--max-iterations 6 --stop-at-iteration-number 6`` would do the same thing
without the subtraction: six iterations offered, the study ending at 6.

What you should see
~~~~~~~~~~~~~~~~~~~

The resumed run reports what it restored -- the hub's iterate, the spoke's best
solution, and the spoke's place in its own scenario walk::

  Restored the checkpointed incumbent for XhatShuffleInnerBound (objective -108382.2222...)
  Resuming from checkpoint in ./ckpt (iteration 4)
  Restored the checkpointed xhatshuffle cursor (pass 6668; the next epoch starts from scen2)

It then starts at iteration 5 -- there is no iteration-0 solve and no repeat of
iterations 1 through 4 -- and ends on the same numbers the reference run did:

=====================  ==============  ================  ==========  ==========
run                        Best Bound    Best Incumbent    Rel. Gap    Abs. Gap
=====================  ==============  ================  ==========  ==========
reference (6)            -108931.8045      -108382.2222      0.505%    549.5823
resumed (4 then 2)       -108931.8045      -108382.2222      0.505%    549.5823
=====================  ==============  ================  ==========  ==========

The written solutions match too::

  diff ref_soln.csv resumed_soln.csv
  diff -r ref_soln_soldir resumed_soln_soldir

Both are clean. Farmer is a deterministic LP, so here "very close" is in fact
"identical"; on a MIP under default solver settings expect agreement to a
tolerance instead, since a resumed run can land on a different optimal solution
(see `What resume guarantees`_).

To exercise the multi-rank path as well, give each cylinder more than one rank
-- ``-np 6`` with three cylinders gives each of them two::

  GC="mpiexec -np 6 python -m mpi4py -m mpisppy.generic_cylinders \
      --module-name farmer --num-scens 6 --default-rho 1 \
      --solver-name cplex --lagrangian --xhatshuffle \
      --rel-gap 0.0 --abs-gap 0.0"

``./ckpt/hub/gen_0004/`` then holds a ``hub_rank_NNNN.pkl`` and one ``.dill``
per scenario **for every rank**, and the manifest that publishes the generation
is written only once all of them have finished.

What must match, and what may change
------------------------------------

A checkpoint records the configuration it was written with and refuses to load
into a run that differs, rather than producing a subtly wrong answer. Resuming
requires the same number of MPI ranks with the same scenarios on each, and a
configuration that matches everywhere except a short list of settings that a resume
may legitimately change:

* **the budget, and every termination criterion** -- ``--max-iterations``,
  ``--stop-at-iteration-number``, ``--time-limit``, ``--rel-gap``,
  ``--abs-gap``, ``--intra-hub-conv-thresh`` and ``--max-stalled-iters``.
  All of them say when to stop rather than what is being solved, so all of
  them may differ.
* **solver choice and how it is driven** -- ``--solver-name``, solver options
  and thread counts, mipgaps, and every per-cylinder solver setting.
* **display, tracking and output destinations**, and the checkpoint options
  themselves;
* **which cylinders run**, except ``--reduced-costs`` and
  ``--cross-scenario-cuts``, which also change the hub's own models. With
  ``--ph-primal-hub`` a different set of dual spokes changes where the hub
  goes, but not what problem it is solving.

Everything else must match -- including options your own model module
registers. That is deliberate: checking by default is what stops a farmer
checkpoint from being resumed with ``--farmer-with-integers`` and quietly
answering the linear program.

There are three practical consequences, and they come from different places.

**Adding an option to your model module invalidates the checkpoints you
already have.** Registering one more option in ``inparser_adder`` changes the
configuration being compared, so the next resume stops at startup with
``CheckpointMismatch`` naming the directory. That is the check doing its job
rather than a defect -- it cannot tell a harmless new option from one that
changes the problem.

**Editing the model's code is not detected, and a resume ignores the edit.**
The check compares configuration, not source code. Change an equation or the
data ``scenario_creator`` builds without changing any option, and the resume
goes ahead -- on the models saved in the checkpoint, which replace the ones
the edited code just built. Keep the model code unchanged until a study
reaches its end.

**A checkpoint is only as portable as the pickles inside it.** Under
``dill-reload`` it holds serialized Pyomo models, so upgrading mpi-sppy, Pyomo
or Python between the write and the resume can leave them unreadable even when
every option still matches. The on-disk layout carries a format version and is
checked, but that only catches layout changes mpi-sppy made deliberately; it
promises nothing about yesterday's pickles loading into today's libraries.
Finish a study on the software stack it started on.

What resume guarantees
----------------------

For the target case -- large MIP subproblems -- a resumed run **continues
correctly**, and never loses or regresses the bounds or the incumbent the run
tracks.

The reloaded models carry the recourse values from the last solve before the
stop, so the first resumed solve *can* warm-start from them -- but only with
``--warmstart-subproblems``, which is off by default and which mpi-sppy does
not turn on for you. Without it that first solve is cold, exactly as every
other solve in the run is. It is **not** bit-for-bit reproducible against a hypothetical
uninterrupted run: multi-threaded MIP solves are not deterministic and admit
multiple optima, so the resumed iterates may differ. That is expected, not a
bug.

Bounds and the incumbent are carried forward as valid best-so-far values. A
resumed run never reports a worse best-so-far than its checkpoint, provided
the spoke that held the incumbent is still in the run and its file holds that
incumbent: a spoke writes an improvement at the end of the loop pass that
reported it, or when the spoke finishes if its loop never got that far; it
writes a restored incumbent, with the xhatshuffle cursor read beside it, to
this run's directory as soon as it restores it; and a failed write is warned
about and not retried. An outer bound that reaches the hub after the last
checkpoint is written is not in it, so the stopped run's final line can show a
better outer bound than the resumed run starts from. An incumbent a spoke
wrote after that checkpoint is restored, since the resumed hub takes its
incumbent from the spokes' files. Each spoke keeps its own file, so dropping
one leaves its incumbent behind: resuming ``--xhatshuffle`` as ``--xhatxbar``
starts without the answer the first one found, and says so.

The xhat extensions that run inside the hub (``XhatLooper``, ``XhatXbar``,
``XhatClosest`` and ``XhatSpecific``) are not covered by that promise. They
evaluate the run's final iterate once, after the last checkpoint has been
written, and keep the result to themselves. A resumed run evaluates its own
final iterate, which can come out worse than the one before the stop.

In a cylinders run the best solution does not live on the hub: the spoke that
found it holds it. So each spoke that looks for one -- the xhat spokes, the
L-shaped xhatter and the two slammers -- keeps its own small file under
``spokes/`` in the checkpoint directory, holding the best solution it has
found, written by variable name whenever that solution improves. A resumed
spoke reads it back and reports it to the hub, which is why a resumed
cylinders run starts from the answer it already had rather than from nothing.

Those files are deliberately not synchronised with the hub's: a spoke writes
when it has something new to record, the hub writes at iteration boundaries,
and neither waits for the other. A spoke whose file is missing -- because the
earlier run stopped before it found anything, or because that spoke was not in
the earlier run at all -- simply starts without an incumbent and says so in
the log. A spoke on several ranks keeps one file per rank, and if they do not
all hold the same incumbent -- a write failed on one rank, say -- it starts
without one too, and says why.

Each file is named for the spoke's class and for which spoke of that class it
is, rather than for the cylinder's position in the wheel, because which
cylinders run is on the list above that a resume may change. Resuming without
``--lagrangian`` therefore still finds the xhat spoke's incumbent, and two
spokes of one class still read their own. The one change that cannot be
absorbed is dropping one of two spokes *of the same class*: the survivor then
looks like the one that was removed, and the resume warns that the incumbent
it adopts may have belonged to the other one. That incumbent is still a
feasible solution for the same model.

An ``xhatshuffle`` spoke also records **where it had got to** in its walk
through the scenarios -- the scenario its next epoch starts from, and its
position and direction in the order -- so a resumed spoke's walk continues the
way the uninterrupted one's would rather than starting over. The pass count
in its log line counts passes of a loop that spins while it waits on the hub,
so it varies from run to run. The scenario order itself is not stored -- the shuffle is
seeded to a fixed value, so a resumed spoke reproduces it exactly -- only the
position in it, which is why a run whose scenario list has changed discards the
position (with a warning) and explores from the start again. The other xhat
spokes re-evaluate from scratch whenever the hub sends new values, so they have
no such position and store none.

The two dual cylinders, ``--relaxed-ph`` and ``--ph-dual``, keep a file of
their own under ``spokes/`` too: their dual weights W, written at the end of
every iteration. A resumed dual cylinder puts them back and carries on from
them, which matters most under ``--ph-primal-hub``, where the hub's W is built
from them. The file also records the probability-weighted sum of W for each
variable, and the resume refuses the file if the weights it put back do not
reproduce it -- an edited file, or changed scenario probabilities. Only W crosses the
checkpoint: a dual cylinder's own extensions, such as ``--grad-rho`` on
``--ph-dual``, start fresh, and the resume says so.

On a deterministic LP or QP solve the primal trajectory can come back
bit-identical, but that is a bonus rather than the guarantee.

Disk usage
----------

Exactly one checkpoint is kept. Retaining older generations is not supported.
The new checkpoint is written in full before the old one is deleted, so the
**peak** on disk is two generations -- size disk quotas for that, not for one.
Under the default ``dill-reload`` backend a checkpoint holds a serialized copy
of every local scenario model, which for large MIPs is not small.

Publication is atomic: the new generation is staged, moved into place, and only
then does a manifest rewrite commit it. A run killed at any point leaves a
complete, resumable checkpoint referenced -- never a half-written one -- and
the next successful write reclaims anything the interrupted one left behind.

Use one checkpoint directory per run. Two runs sharing one share a manifest and
will overwrite each other. A run that is not resuming from its own
``--checkpoint-dir`` removes the ``spokes/`` files an earlier run left there,
so a later resume cannot pick up another study's incumbent or dual weights.

Cylinders that span several ranks
---------------------------------

Each rank holds a different slice of the scenarios, so one checkpoint is the
whole set of per-rank files, and it is committed only once every rank has
written its own. If any rank cannot write, none of them publishes: the previous
checkpoint stays on disk as the resumable one and the run carries on; the
next checkpoint point tries again if the run gets that far, as `Writing a
checkpoint`_ describes. There is no state in which some ranks have advanced
their checkpoint and others have not.

Each rank that failed logs its own cause. Rank 0 always logs the failure too,
naming the failing rank when it is not one itself.

The cost is that the ranks wait for each other at each write, which the
bracketing ``toc`` lines include -- the slowest rank sets the pace.

Nothing in this coordination reaches beyond one cylinder: a hub and its spokes
never wait on each other, and neither do two spokes.

What it costs
-------------

The cost is one model serialization per checkpoint, so what it comes to per
iteration is that divided by K. The figures below are the default ``K = 1``,
the worst case, measured over ten iterations against the same run without
``--checkpoint-dir``:

===========================  ==========  ============  ==========
instance                     no ckpt     with ckpt     overhead
===========================  ==========  ============  ==========
farmer, 3 scenarios (LP)       0.62 s        0.88 s        43%
farmer, 50 scenarios (LP)      1.34 s        4.85 s       262%
sizes, 3 scenarios (MIP)       2.28 s        3.15 s        38%
sizes, 10 scenarios (MIP)      5.05 s        7.46 s        48%
===========================  ==========  ============  ==========

.. warning::
   **These are toy models, and the percentages do not carry over.** Farmer has
   three first-stage variables; ``sizes`` is a small MIP. Serializing one of
   their scenarios takes milliseconds, and a real model's can take seconds. Do
   not scale these figures by your scenario count to predict your own overhead
   -- the per-scenario cost is the part that changes, and it changes by orders
   of magnitude.

   What does carry over is the *shape* of the result: the overhead is
   serialization time against solve time. The 50-scenario farmer run is the
   pathological end -- solves so cheap that writing dominates and the run takes
   three times as long. A large MIP whose solves take minutes is the other end,
   where even a slow serialization is a small share.

The bracketing ``toc`` lines are what you should actually go by: their
difference is essentially the whole overhead, so one calibration run on your own
model tells you the cost directly. If it is too high,
``--checkpoint-every-iterations`` buys most of it back in exchange for repeating
some iterations after a stop; see `Choosing K`_.

Requirements and limitations
----------------------------

**dill is required.** ``--checkpoint-backend dill-reload`` is the default and
currently the only implemented backend, and it needs the optional ``dill``
package::

  pip install mpi-sppy[extras]

**Your scenario models must be serializable.** A model can be made
unserializable by what its ``scenario_creator`` closes over -- most commonly a
Pyomo rule written as a nested function that reads ``cfg`` directly, which pulls
the whole configuration object into the model. See :ref:`scenario_creator` for
the pattern and the fix. Checkpointing serializes every scenario on every rank
at setup rather than discovering the problem at the first write, and the error
names the offending rule and the scenario it was found in.

**The synchronous PH hub only.** ``--APH`` and the other hub types are refused
at startup when either ``--checkpoint-dir`` or ``--resume-from`` is given, as
are an unwritable directory (checked from every rank -- on a cluster a path can
be writable from some nodes and not others), an unimplemented backend, scenario
names that would collide once made filename-safe, and any configuration where
the checkpointing extension would not actually be attached. ``--EF`` and the
write-only modes (``--pickle-bundles-dir``, ``--pickle-scenarios-dir``,
``--write-scenario-lp-mps-files-dir``) are refused for the same reason: none of
them runs the iterative algorithm a checkpoint describes. The intent is that
checkpointing either works or says so at startup, rather than running for hours
and writing nothing.

**Extension and converger state is part of a checkpoint.** Extensions that
accumulate their own state across iterations carry it across a stop: the rho
updaters (``--use-norm-rho-updater``, ``--sep-rho``, ``--sensi-rho``,
``--grad-rho``, and ``MultRhoUpdater`` when a driver attaches it), the fixer
(``--fixer``), the slammer (``--slamming-directives-file``) and the W tracker
(``--wtracker``). So a resumed run using one of them follows the same
trajectory an uninterrupted run would, rather than merely continuing correctly
from the right models.

The shipped convergers need nothing carried: each either recomputes what it
compares from the current iterate or, like the primal-dual converger, rebuilds
it correctly from the resumed models.

The two fixers driven by a spoke, ``--rc-fixer`` (with ``--reduced-costs``) and
``--relaxed-ph-fixer``, carry their own state too, and a resumed run using
either continues correctly from the right models, but neither is promised to
follow the uninterrupted run's trajectory. The reduced-costs spoke is not
checkpointed and starts over on a resume. The relaxed-PH spoke resumes from
its checkpointed W (see the dual cylinders above), but its buffers reach the
hub asynchronously, and an exact retrace is not tested. The PH tracker
(``--ph-track-progress``) continues the hub's files across a resume; on a
spoke it starts them over.

The rho-setting extensions do not *recompute* rho at the resume itself: the
checkpointed rho -- including whatever adaptation had happened by the write --
carries over, and the extensions resume their per-iteration updates from there.

``integer_relax_then_enforce`` decides when to enforce integrality from a
fraction of a budget. Give every run of the study
``--stop-at-iteration-number`` and the iteration fraction is of the study, so
a resumed run enforces at the same iteration an uninterrupted one would.
Without it, each run applies the fraction to its own ``--max-iterations``, and
a resumed run enforces on its own schedule. The time fraction is always of the
run's own ``--time-limit``. A run that stops before the study's fraction stops
with integrality still relaxed, unless the time or convergence condition
enforced it first; the run that resumes it enforces it.

Two things this does not cover:

* **Your own extension carries nothing unless you say so.** If it keeps state
  on itself that decides what it does next -- a history, a counter, a record of
  what it has already changed -- implement ``checkpoint_state()`` and
  ``restore_state(state)`` on it. Return plain data keyed by variable *name* or
  by ``(node name, index)``, never Pyomo objects: a resume replaces every
  model, so a saved variable reference addresses something that no longer
  exists. If it keeps no such state, implement both anyway: return None from
  ``checkpoint_state()`` and do nothing in ``restore_state(state)``.

  This is required, not optional. A run with ``--checkpoint-dir`` or
  ``--resume-from`` refuses to start -- before anything is solved -- if an
  attached extension implements neither method, or only one of them, and the
  message names each one. From the outside "keeps nothing" and "nobody
  decided" look the same, and the second one is a run that quietly stops
  retracing an uninterrupted one.

  Two of the shipped extensions keep state and do not carry it yet, so a
  checkpointed run refuses them: ``WOscillationMonitor`` and
  ``CrossScenarioExtension``.
* **A converger is held to the same rule.** It implements
  ``checkpoint_state()``/``restore_state(state)``, as a no-op pair if it keeps
  no state, or a checkpointed run refuses it at startup. That matters more than
  it sounds: a converger decides when the run stops, so one that accumulates
  history and starts fresh could terminate a resumed run at a different
  iteration than an uninterrupted one. See :ref:`checkpointing_your_converger`.

If you resume with a *different* set of extensions than the checkpoint was
written with, that is allowed -- the hub's iterate is still valid -- and the run
reports each piece of state it could not hand to anybody.

**W and xbar input files are not read on a resumed run.** ``--init-W-fname``
and ``--init-Xbar-fname`` initialize a study; a resumed run takes both from the
checkpoint. Leaving the flags on the command you resubmit each morning is
harmless -- they are skipped, with a line in the log saying so.

**The order you attach extensions in does not affect what is checkpointed.**
The write happens at a dedicated point in the iteration loop, after every
extension's end-of-iteration hook has run. So if your own extension uses that
hook to change rho, fix a variable, relax a domain or add a cut, the change is
part of that iteration's checkpoint and is there when you resume.
