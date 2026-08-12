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

Give a directory and the run writes one checkpoint when it terminates::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --time-limit 28800 \
      --checkpoint-dir ./ckpt

A checkpoint is written at the end of every completed PH iteration, and only
one is kept, so the file on disk always describes the most recent *completed*
iteration. Pairing ``--checkpoint-dir`` with ``--time-limit`` is the
planned-stop recipe -- set the day's budget and the run stops itself with a
resumable checkpoint in place.

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

  [ 1234.56] Writing checkpoint (termination) at iteration 42 to ./ckpt
  [ 1261.03] Checkpoint written (termination) at iteration 42

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
correctly and warm-started**, and never loses or regresses the best solution
found so far. It is **not** bit-for-bit reproducible against a hypothetical
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

Publication is atomic: the files are written, then a manifest is rewritten to
point at them. A run killed during a write leaves the previous complete
checkpoint intact and referenced, never a half-written one.

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
discovering the problem hours later at the terminal checkpoint, and the error
names the offending rule.

**Extension state is not yet part of a checkpoint.** Extensions that
accumulate their own state across iterations -- the rho updaters, ``fixer``,
``slammer`` -- start fresh on a resumed run. The restored *model* state is
correct, and the run continues correctly from it, but a resumed run using one
of these will not follow the same trajectory as an uninterrupted one.
