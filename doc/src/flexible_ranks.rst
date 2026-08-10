.. _flexible_ranks:

Flexible (Unequal) Rank Assignments
===================================

By default every cylinder in a run gets the same number of MPI ranks:
``mpiexec -np 12`` with a hub and two spokes gives four ranks to each.
That is rarely how the work is actually distributed. A PH hub solves
every subproblem on every iteration and scales well with ranks; a
Lagrangian spoke also solves subproblems but may need fewer iterations;
an xhat shuffle spoke is comparatively lightweight and may be fine with
one or two ranks. Splitting the pool evenly leaves the hub short while
the xhat spoke idles.

Flexible rank assignment lets you give each cylinder a share of the
rank pool.

Specifying ratios
-----------------

Each spoke has a ``--<spoke>-rank-ratio`` option giving its target share
*relative to the hub*, which is always the reference at 1.0. There is no
hub option. All ratios default to 1.0, which is the equal-rank behavior.

.. code-block:: bash

   mpiexec -np 14 python -m mpi4py ../../mpisppy/generic_cylinders.py \
       --module-name farmer --num-scens 100 \
       --solver-name gurobi --default-rho 1 --max-iterations 50 \
       --lagrangian --lagrangian-rank-ratio 0.5 \
       --xhatshuffle --xhatshuffle-rank-ratio 0.25

With ratios hub 1.0, lagrangian 0.5, xhatshuffle 0.25 over 14 ranks,
that gives 8 ranks to the hub, 4 to the Lagrangian spoke, and 2 to the
xhat spoke. The allocation is printed at startup, so you can confirm
what you got without working it out by hand.

Only relative magnitudes matter: ``1.0 / 0.5 / 0.25`` and
``4 / 2 / 1`` request the same split. Ratios are used rather than
explicit counts so the same command line behaves sensibly at any
``-np``.

The available options are:

================================= ========================
option                            spoke
================================= ========================
``--lagrangian-rank-ratio``       Lagrangian outer bound
``--xhatshuffle-rank-ratio``      xhat shuffle inner bound
``--xhatxbar-rank-ratio``         xhat xbar inner bound
``--subgradient-rank-ratio``      subgradient outer bound
``--fwph-rank-ratio``             Frank-Wolfe PH
``--relaxed-ph-rank-ratio``       relaxed PH
``--ph-dual-rank-ratio``          PH dual
``--ph-xfeas-spoke-rank-ratio``   PH xfeas
================================= ========================

The ``reduced_costs`` spoke deliberately has no such option; see
`Limitations`_ below.

How ranks are apportioned
-------------------------

Ranks are apportioned by the largest-remainder (Hare quota) method, then
every cylinder is given a floor of one rank. The counts always sum to
exactly ``-np``, and every cylinder always runs, so an awkward ratio
cannot silently starve a cylinder of all its ranks. Requesting more
cylinders than ranks is an error, since the floor of one each is then
impossible.

Uneven division is not warned about; the apportionment simply rounds.

What it does not change
-----------------------

When every ratio is 1.0 — the default, and the only possibility before
this feature existed — the run takes exactly the code path it always
did. The ratios themselves are the switch: nothing about the
unequal-rank machinery is reachable until you ask for a non-default
ratio.

That also means the fallback is free. If the unequal-rank path
misbehaves on some MPI build, set the ratios back to 1.0 and you are
back on the long-proven path, with no other changes to your run.

Internally, an unequal-rank run puts its MPI window on ``COMM_WORLD``
rather than per-stratum communicators, and a cylinder reading a
per-scenario field assembles it from however many of the sending
cylinder's ranks hold the scenarios it needs. That machinery is
described in ``doc/designs/flexible_rank_assignments.md``; you do not
need to know it to use the feature.

Limitations
-----------

The spokes listed above are supported at any ratio, in both two-stage
and multistage problems. The ``reduced_costs`` spoke is not: it consumes
a per-scenario reduced-cost field whose assembly across unequal rank
counts was never implemented, because its only consumer was a
since-removed rho setter. It therefore has no rank-ratio option and runs
at the hub's rank count.

If some other cylinder reads a per-scenario field that has no
multi-source assembly, the run fails at startup — during window
creation, not part-way through a solve — with a message naming the
cylinder and the field, and suggesting you run the cylinders that
exchange it at equal rank counts. You will not get a silently
mis-assembled buffer.

Diagnosing an unequal-rank run
------------------------------

A cylinder assembling a field from several sending ranks can catch that
sender part-way through publishing. Such a read is either rejected and
retried, or — for fields whose consumers re-evaluate anyway — accepted
with mixed contents. This is expected and self-correcting, but it means
a bounds cylinder can appear to report less often than you expect.

Every unequal-rank run therefore counts these reads and prints a
per-field summary as each cylinder finalizes, letting you distinguish a
coherence problem from a merely slow upstream sender. The buckets, how
to read the reported miss rate, and the option for printing the counts
periodically during a run rather than only at the end are described
under ``coherence_diagnostics_period`` in :ref:`secretmenu`.

Equal-rank runs do no multi-source reads, and print nothing.
