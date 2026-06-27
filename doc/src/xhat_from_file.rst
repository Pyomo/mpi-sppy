.. _xhat_from_file:

Supplying an Initial Xhat from a File
=====================================

Every xhat spoke (``xhatlooper``, ``xhatshufflelooper``,
``xhatspecific``, ``xhatxbar``) will optionally read an ``xhat`` from a
file, evaluate it across all scenarios once, and report the resulting
inner bound — **before** its normal exploration loop starts. Two file
formats are accepted:

- a ``.csv`` nonant tree (``node_name, variable_name, value``), matched
  to the model **by variable name**, for **any number of stages**; and
- a ``.npy`` holding a bare root-node vector, matched **by position**,
  for **two-stage** problems only.

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

The format is chosen by the file extension: ``.csv`` for the by-name
nonant tree, anything else (``.npy``) for the positional root vector.
The flag is off by default; the feature is only active when supplied.

File Formats
------------

**CSV nonant tree (any number of stages).** Lines are
``node_name, variable_name, value`` with **node-local** variable names;
lines beginning with ``#`` are comments. This is matched to the model
by name, so it is **not** order-sensitive, and a file may carry more
nodes than a given run needs (extras are ignored). Produce one directly
from a run with ``--write-xhat-file`` (below), or by hand:

.. code-block:: text

   # node_name, variable_name, value
   ROOT, DevotedAcreage[CORN0], 80.0
   ROOT, DevotedAcreage[SUGAR_BEETS0], 250.0
   ROOT, DevotedAcreage[WHEAT0], 170.0

**Positional ``.npy`` (two-stage only).** A one-dimensional numpy array
of the root-node values **in nonant order** — the pyomo iteration order
over ``scenario._mpisppy_node_list[0].nonant_vardata_list`` for any
local scenario. This is the format the MMW confidence-interval code uses
(``--mmw-xhat-input-file-name``, via ``ciutils.read_xhat``):

.. code-block:: python

   import numpy as np
   xhat_values = [1.0, 0.0, 1.0]   # in nonant order
   np.save("my_xhat.npy", np.array(xhat_values, dtype=float))

Producing a file with ``--write-xhat-file``
-------------------------------------------

A run writes its incumbent xhat as the canonical CSV nonant tree with:

.. code-block:: bash

   --write-xhat-file <path>

It works for any number of stages and identically for EF and cylinders
runs (both route through ``sputils.write_nonant_tree_csv``), so the file
from one run is directly readable by ``--xhat-from-file`` in another.
Programmatically, ``WheelSpinner.write_tree_nonants(path)`` and
``sputils.ef_nonants_csv(ef, path)`` write the same format.

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

**Multi-stage needs the CSV.** The ``.csv`` nonant tree supports any
number of stages. The positional ``.npy`` is two-stage only; pointing
``--xhat-from-file`` at a ``.npy`` on a multi-stage run raises, naming
the ``.csv`` format as the multi-stage path.

**Names / lengths must match.** For a ``.csv``, every node and variable
the run needs must be present, by node-local name, or startup raises
naming the missing item. For a ``.npy``, the vector length must equal
the root-node nonant count — no silent truncation or padding.

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

See Also
--------

- :ref:`Spokes` — overview of the xhat spokes.
- the xhat feasibility-cuts feature (PR #671) — the companion feature for
  non-complete-recourse problems.
- :ref:`iis` — write an IIS when an xhatter hits an infeasible
  scenario; pairs naturally with this feature for reproducing an
  infeasibility on demand.
- ``doc/designs/multistage_xhat_write_design.md`` — the design document
  for the multi-stage CSV nonant tree (read and write).
- ``doc/designs/xhat_from_file_design.md`` — the original two-stage
  design document.
