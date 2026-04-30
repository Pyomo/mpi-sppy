.. _xhat_from_file:

Supplying an Initial Xhat from a File
=====================================

Every xhat spoke (``xhatlooper``, ``xhatshufflelooper``,
``xhatspecific``, ``xhatxbar``) will optionally read a first-stage
solution ``xhat`` from a ``.npy`` file, evaluate it across all
scenarios once, and report the resulting inner bound — **before** its
normal exploration loop starts.

When This is Useful
-------------------

- **Warm start from a prior run on a similar instance.** If you have
  already solved a related instance (same first-stage structure, same
  nonant order), the previous run's xhat is often a good starting
  candidate for the current run. Feed it in and the xhatter reports
  that inner bound immediately instead of waiting for normal
  exploration to stumble onto something comparable. This is the
  most common use case.
- **User-supplied heuristic candidate.** If your domain knowledge or
  a hand-computed heuristic gives you a promising ``xhat``, supplying
  it as the first thing the xhatter evaluates often shortens the time
  to a useful inner bound.
- **Testing infeasibility-driven features.** Combined with
  the xhat feasibility-cuts feature (PR #671), you can write a known-infeasible
  ``xhat`` to a ``.npy`` file and hand it in; the feasibility-cut
  path then fires end-to-end, letting you verify that the same xhat
  is not revisited on the next iteration.

Enabling the Feature
--------------------

``generic_cylinders`` exposes a single string flag:

.. code-block:: bash

   --xhat-from-file <path>

where ``<path>`` points at a ``.npy`` file whose contents is a
one-dimensional numpy array holding the first-stage values **in the
same order as the problem's root-node nonant list**. Order-sensitive
— the ordinary pyomo iteration order over
``scenario._mpisppy_node_list[0].nonant_vardata_list`` for any local
scenario. (If you generated the file from a previous mpi-sppy run,
the order matches automatically.)

The flag is off by default; the feature is only active when the flag
is supplied.

File Format
-----------

``.npy`` only, via the existing ``mpisppy.confidence_intervals.ciutils.read_xhat``
helper. That is the canonical mpi-sppy xhat on-disk format: the
MMW confidence-interval code already uses it
(``--mmw-xhat-input-file-name``), and several examples write xhats
this way (``ciutils.write_xhat``).

To produce a compatible file from a script:

.. code-block:: python

   import numpy as np
   xhat_values = [1.0, 0.0, 1.0]   # in nonant order
   np.save("my_xhat.npy", np.array(xhat_values, dtype=float))

Example
-------

.. code-block:: bash

   python -m mpisppy.generic_cylinders \
       --module-name my_model \
       --num-scens 10 \
       --solver-name gurobi \
       --max-iterations 50 \
       --default-rho 1.0 \
       --lagrangian --xhatshuffle \
       --xhat-from-file prior_run_xhat.npy

Each rank reads the file once, the xhat spoke evaluates it, and the
resulting inner bound (if finite) is sent to the hub before the
spoke starts its normal shuffle loop.

Scope and Limitations
---------------------

**Two-stage only.** V1 supports two-stage problems only, matching
``ciutils.read_xhat``. Multi-stage is planned as a follow-up; for
now, enabling the flag on a multi-stage run raises

.. code-block:: text

   RuntimeError: --xhat-from-file is two-stage only; multi-stage
   support is planned as a follow-up.

**Length must match.** The file's vector length must equal the
problem's root-node nonant count. A mismatch raises at spoke startup
with an error naming both counts — no silent truncation or padding.

**Missing file is a hard error.** The path must exist when the spoke
starts; a missing file is not silently treated as "feature off".

Interaction with ``--*-try-jensens-first``
------------------------------------------

Both ``--xhat-from-file`` and Jensen's ``--*-try-jensens-first`` (see
the Jensen's-bound docs) contribute a single candidate xhat before
the xhatter's normal loop. They can be used together. The explicit
file-supplied candidate is evaluated **first**, then Jensen's, then
the normal exploration. ``update_if_improving`` keeps whichever is
best, so correctness does not depend on order; the ordering is a
predictability and log-readability choice.

Interaction with ``--xhat-feasibility-cuts-count``
--------------------------------------------------

If both flags are on and the file-supplied xhat turns out to be
infeasible in some scenario, the xhatter's infeasibility path fires
and a feasibility cut is emitted (see the xhat feasibility-cuts feature (PR #671)
for the mechanics). The hub installs the cut into every scenario,
and the same xhat is not revisited. This combination is the
recommended way to exercise feasibility cuts in an end-to-end test:
hand in a known-infeasible binary vector via ``--xhat-from-file``
with ``--xhat-feasibility-cuts-count=1`` and watch the cut land.

Follow-up Milestones
--------------------

- Multi-stage support (per-node xhat file or a multi-node format).
- Additional file formats (CSV, JSON) if a concrete use case appears.

See Also
--------

- :ref:`Spokes` — overview of the xhat spokes.
- the xhat feasibility-cuts feature (PR #671) — the companion feature for
  non-complete-recourse problems.
- ``doc/xhat_from_file_design.md`` — the design document.
