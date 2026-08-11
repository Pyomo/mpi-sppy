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

``--checkpoint-at-termination`` is on by default, so the checkpoint is written
when the run ends for *any* internal reason: convergence, ``--max-iterations``,
or hitting ``--time-limit``. Pairing it with ``--time-limit`` is the planned-stop
recipe above -- set the day's budget and the run stops itself and checkpoints.
Turn it off with ``--disable-checkpoint-at-termination``.

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

A checkpoint records the layout it was written with and refuses to load into a
run that does not match, rather than producing a subtly wrong answer. Resuming
requires:

* the same number of MPI ranks, and the same scenarios on each rank;
* the same **structural** options -- the ones that change the shape of the
  scenario models or the meaning of the state stored in them:
  ``--default-rho``, ``--linearize-proximal-terms``,
  ``--linearize-binary-proximal-terms``,
  ``--proximal-linearization-tolerance``, the ``--smoothing`` settings, the
  ``--cvar`` settings, ``--module-name``, ``--num-scens``,
  ``--branching-factors``, and ``--scenarios-per-bundle``.

Everything else is free to change. In particular **the iteration limit, the
time limit, and the display/verbosity options may all differ** on a resume --
picking a run back up with a different budget is the whole point, so those are
deliberately outside the check.

A mismatch is reported with an explicit message naming what differs.

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

**The terminal checkpoint runs after ``scenario_denouement``.** Standard
denouement functions only report, but one that re-solves or mutates a model
would have those changes captured in the checkpoint.
