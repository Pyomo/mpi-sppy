.. _secretmenu:

Secret Menu Items
=================

There are many options that are not exposed in ``mpisppy.utils.config.py`` and we list
a few of them here.


coherence_diagnostics_period
----------------------------

On an unequal-rank run (any cylinder given a ``rank_ratio`` other than
1.0) a per-scenario field is assembled from several of the sending
cylinder's ranks. Such a read can straddle a publish, with some sources
answering from before the write and some from after. The reader then
either rejects the read, or -- for a field whose consumers re-evaluate
anyway -- accepts a blended one.

Every such read is counted, always, and each cylinder prints a per-field
summary as it finalizes:

.. code-block:: text

   coherence diagnostic [LagrangianOuterBound] DUALS: total=1812, new_accepted=38, not_new=1664, rejected_incoherent=50, rejected_cross_reader=60, accepted_mixed=0, miss rate=6.07%

The buckets partition ``total``:

``new_accepted``
   The sources agreed on an advanced write id; the data was used.

``not_new``
   The write id did not advance, so there was nothing to take -- the
   sender has not published since the last accepted read.

``rejected_incoherent``
   This rank's own sources disagreed, so the read was rejected and will
   be retried. This is the fundamental coherence miss.

``rejected_cross_reader``
   This rank's sources agreed, but another rank of the same cylinder saw
   a different write id -- usually because *it* straddled the publish,
   and it records the miss itself.

``accepted_mixed``
   A relaxed field's sources disagreed and the blended assembly was used
   anyway.

The point of the split is to separate two conditions that look alike
from the outside. If a bounds cylinder seems to report rarely and
``not_new`` dominates, the upstream sender is simply slow and the reader
is fine. If the rejection and mixed buckets dominate, reads are being
lost or blended to publish straddling. The reported ``miss rate`` is
everything a straddle cost -- ``rejected_incoherent`` plus
``rejected_cross_reader`` plus ``accepted_mixed``, over ``total``.

That summary needs no option. Setting ``coherence_diagnostics_period``
additionally prints a cylinder's own running counts every N multi-source
reads, which is useful when watching a run live rather than reading it
afterwards. It is set per cylinder, on the dict for the cylinder you
want to watch (hub or spoke):

.. code-block:: python

   hub_dict["opt_kwargs"]["options"]["coherence_diagnostics_period"] = 500

before passing the dicts to ``spin_the_wheel``. It is not exposed as a
CLI flag, so under ``generic_cylinders.py`` it must be set by modifying
the configured dicts in code.

Three things to know when reading the output. Counters are kept per
``Field`` and aggregated over the cylinder's ranks, so a cylinder that
reads one field from more than one peer cylinder cannot tell you which
peer the misses came from. The periodic line is printed before the
current read has been bucketed, so its buckets sum to ``total - 1``;
only the finalization summary is exactly partitioned. And an equal-rank
run does no multi-source reads at all, so it prints nothing.

The counters are also available programmatically, as
``SPCommunicator.coherence_counters``, keyed by ``Field``.


initial_proximal_cut_count
--------------------------

If the `linearize_proximal_terms` option is specified (see :ref:`linearize_proximal`)
then the option 'initial_proximal_cut_count' controls
the initial number of cuts (default 2).

E.g. if you wanted to specify four cuts in a hand-wired driver such as
``examples.farmer.archive.farmer_cylinders`` (where the hub definition
dictionary is called ``hub_dict``) you would add

.. code-block:: python

   hub_dict["opt_kwargs"]["PHoptions"]["initial_proximal_cut_count"] = 4

before passing ``hub_dict`` to ``spin_the_wheel``. When running through
``generic_cylinders.py`` instead, this option is not currently exposed
as a CLI flag and must be set by modifying the configured ``hub_dict``
in code.


subgradient_while_waiting
-------------------------

The Lagrangian spoke has an additional argument, `subgradient_while_waiting`,
which will compute subgradient steps while it is waiting on new W's from the
hub.
