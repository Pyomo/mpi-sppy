.. _checkpointing:

Checkpointing and Resuming a Run
================================

A long Progressive Hedging run can be stopped and picked up later. The intended
use is a planned stop: a multi-day study that ends each day and resumes the next
morning on the same cluster, without losing the work done so far.

Checkpointing is entirely opt-in. With no ``--checkpoint-dir`` the machinery is
not attached at all, and a run that does not ask for it pays nothing.

.. note::
   The current implementation covers a **serial PH hub**. Multi-rank runs,
   bundles, and cylinder (hub-and-spoke) runs are planned but not yet
   supported. See ``doc/designs/checkpointing_design.md`` for the full design
   and the phased rollout.

Writing a checkpoint
--------------------

Give a directory and the run keeps a checkpoint up to date as it goes::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --time-limit 28800 \
      --checkpoint-dir ./ckpt

A checkpoint is written at the end of every completed PH iteration, and only
one is kept, so the file on disk always describes the most recent *completed*
iteration. Pairing ``--checkpoint-dir`` with ``--time-limit`` is the
planned-stop recipe -- set the day's budget and the run stops itself with a
resumable checkpoint in place.

Checkpointing less often
~~~~~~~~~~~~~~~~~~~~~~~~

Serializing every scenario every iteration costs roughly 7--25 ms per scenario
per iteration. Against a MIP whose solves take minutes that is noise, but on
many cheap scenarios it can dominate the run.
``--checkpoint-every-iterations K`` writes less often::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 50 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --checkpoint-dir ./ckpt --checkpoint-every-iterations 10

**What K means.** A checkpoint is written at the end of a completed iteration
whose number is a multiple of K. The numbers are the PH iteration numbers you
see in the log, counted from the start of the study, so with ``K = 10`` the
checkpoints are iterations 10, 20, 30, and so on. K is not a countdown from
whenever the current run happened to begin: it does not restart at a resume,
and there is no drift. The default is ``1``, which writes at every iteration.

**What it costs you.** Only one checkpoint is kept, so at any moment the
directory holds the most recent multiple of K that completed. If the run stops
anywhere else -- a time limit, convergence, a crash -- the iterations after
that point are gone and the resumed run redoes them. That is at most K-1
iterations, and it is the whole trade: you are buying back write time with
work you are willing to repeat.

For example, with ``K = 10`` a run that stops at iteration 34 leaves a
checkpoint of iteration 30, and iterations 31 through 34 are lost. Resuming
picks up at 31 and the next checkpoint is iteration 40 -- not 41, because the
count follows the study, not the resumed leg.

**The one exception.** The final iteration of an exhausted ``--max-iterations``
budget is always written, whatever K is. With ``--max-iterations 100`` and
``K = 30`` the checkpoints are iterations 30, 60, 90 and 100. Raising the limit
and resuming is a normal way to extend a study, and that last iterate is
known-good and already in memory, so it is not worth discarding to save one
write. No other kind of stop can be caught this way: a time limit, the
convergence threshold and a user converger are all tested partway through the
*next* iteration, by which point there is nothing coherent left to write.

Changing K between a stop and a resume is allowed; like the iteration limit,
it describes how the run is managed rather than what problem is being solved,
so it is not part of the check that decides whether a checkpoint may be
resumed.

Writing only at iteration boundaries is deliberate. PH computes xbar, updates
the dual weights, gives extensions their mid-iteration hook, and only then
solves; a run that stops on ``--time-limit`` or on convergence stops *before*
that solve, leaving the model describing half an iteration. Rather than try to
unwind that -- an open-ended problem, since any extension may have changed rho,
fixed variables or added cuts -- the checkpoint is simply taken at the last
point where everything agrees.

One consequence: **a run that ends before finishing iteration 1 publishes no
checkpoint at all.** No iteration completed, so there is no iterate to resume
from.

Each write is bracketed by a pair of timestamped ``toc`` lines, so the log shows
how long it took::

  [ 1234.56] Writing checkpoint at iteration 42 to ./ckpt
  [ 1261.03] Checkpoint written at iteration 42

A write that fails mid-run -- the disk filling up, a network filesystem
hiccup -- does not stop the optimization. The failure is reported loudly in
the log, the previously published checkpoint stays intact and resumable, and
the next iteration boundary tries again. Conditions detectable at setup (an
unwritable directory, a model that cannot be serialized) still stop the run
at startup, before any solving is done.

Resuming
--------

Point a new run at the directory::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --resume-from ./ckpt

The resumed run continues from the checkpointed iterate rather than starting
over. It does **not** re-solve the subproblems at iteration 0 -- for large MIPs
that solve is often the most expensive in the run, and its answer would be
thrown away. Iteration numbering continues where it left off, so
``--max-iterations`` bounds the run as a whole rather than each leg of it.

What must match, and what may change
------------------------------------

A checkpoint records the configuration it was written with and refuses to load
into a run that differs, rather than producing a subtly wrong answer. Resuming
requires the same number of MPI ranks with the same scenarios on each, and a
configuration that matches everywhere except a short list of settings a resume
may legitimately change:

* **the budget** -- ``--max-iterations``, ``--time-limit``, the gap and
  stalling thresholds;
* **solver choice and how it is driven** -- ``--solver-name``, solver options
  and thread counts, mipgaps, and every per-cylinder solver setting. Tightening
  a mipgap on day two continues the same problem rather than redefining it;
* **display, tracking and output destinations**, and the checkpoint options
  themselves;
* **which cylinders run.** The hub's primal trajectory does not depend on the
  spokes.

Everything else must match -- including options your own model module
registers. That is deliberate: checking by default is what stops a farmer
checkpoint from being resumed with ``--farmer-with-integers`` and quietly
answering the linear program.

The practical consequence is that a checkpoint is tied to the mpi-sppy and
model version that wrote it. Adding a new option to your model module will
cause existing checkpoints to be refused.

What resume guarantees
----------------------

For the target case -- large MIP subproblems -- a resumed run **continues
correctly**, and never loses or regresses the best solution found so far.

The reloaded models carry the recourse values from the last solve before the
stop, so the first resumed solve *can* warm-start from them -- but only with
``--warmstart-subproblems``, which is off by default and which mpi-sppy does
not turn on for you. Without it that first solve is cold, exactly as every
other solve in the run is. It is **not** bit-for-bit reproducible against a hypothetical
uninterrupted run: multi-threaded MIP solves are not deterministic and admit
multiple optima, so the resumed iterates may differ. That is expected, not a
bug.

Bounds and the incumbent are carried forward as valid best-so-far values. A
resumed run never reports a worse best-so-far than its checkpoint.

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
will overwrite each other.

What it costs
-------------

By default a checkpoint is written at every completed iteration, so the cost is
one model serialization per iteration. Measured over ten iterations, against
the same run without ``--checkpoint-dir``:

===========================  ==========  ============  ==========
instance                     no ckpt     with ckpt     overhead
===========================  ==========  ============  ==========
farmer, 3 scenarios (LP)       0.62 s        0.88 s        43%
farmer, 50 scenarios (LP)      1.34 s        4.85 s       262%
sizes, 3 scenarios (MIP)       2.28 s        3.15 s        38%
sizes, 10 scenarios (MIP)      5.05 s        7.46 s        48%
===========================  ==========  ============  ==========

Roughly 7-25 ms per scenario per iteration. For the case this is designed for
-- large MIP subproblems where a single solve takes minutes -- that is
negligible. For many cheap scenarios it dominates, and the 50-scenario farmer
run above takes over three times as long.

The bracketing ``toc`` lines are an honest report of it: their difference is
essentially the whole overhead, so a calibration run tells you the cost on your
own models. If that cost is too high, ``--checkpoint-every-iterations`` buys
most of it back in exchange for repeating some iterations after a stop; see
`Checkpointing less often`_ above.

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
the pattern and the fix. Checkpointing checks one scenario at setup rather than
discovering the problem at the first write, and the error
names the offending rule.

**The synchronous PH hub only.** ``--APH`` and the other hub types are refused
at startup when either ``--checkpoint-dir`` or ``--resume-from`` is given, as
is a hub with more than one rank, an unwritable directory, an unimplemented
backend, scenario names that would collide once made filename-safe, and any
configuration where the checkpointing extension would not actually be
attached. The intent is that checkpointing either works or says so at startup,
rather than running for hours and writing nothing.

**Extension and converger state is not yet part of a checkpoint.** Extensions
that accumulate their own state across iterations -- the rho updaters,
``fixer``, ``slammer`` -- start fresh on a resumed run, and so does a
converger given with ``ph_converger`` (the resume warns about the latter in
the log). The restored *model* state is correct, and the run continues
correctly from it, but a resumed run using one of these will not follow the
same trajectory as an uninterrupted one. The rho-setting extensions
(``--sep-rho``, ``--coeff-rho``, ``--sensi-rho``, ``--grad-rho``) do not
recompute rho at the resume itself: the checkpointed rho -- including
whatever adaptation had happened by the write -- carries over, and the
extensions resume their per-iteration updates from there.

**A custom extension that changes models at the end of an iteration must be
attached first.** The checkpoint is written from the checkpointing extension's
end-of-iteration hook, and extensions run that hook in the order they were
attached, with the checkpointing one attached before anything you add. So if
your own extension uses that hook to change rho, fix a variable, relax a
domain or add a cut, it acts *after* the checkpoint for that iteration has
been written -- the change is missing from the checkpoint and is not redone
when you resume. Attach such an extension ahead of the checkpointing one. No
extension shipped with mpi-sppy is affected; this applies only to extensions
supplied with ``--user-defined-extensions``. A future release will write from
a dedicated point in the iteration loop so that ordering stops mattering.
