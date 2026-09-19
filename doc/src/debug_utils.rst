.. _debug_utils:

Buffer Sanity Inspector
=======================

The ``mpisppy.debug_utils`` package provides a passive content-check
utility for the MPI-RMA send and receive buffers used by the hub-and-
spoke system. It is intended as a debugging aid when you suspect that
a buffer is being written to from somewhere unexpected (for example,
when a spoke sees a shutdown signal that the hub did not send).

The inspector does *not* modify producer code, does *not* introduce
new MPI traffic, and is no-cost when not invoked.

When to Use This
----------------

- **A spoke is acting on data it should not have received.** A spurious
  ``SHUTDOWN`` is the canonical example, but the same idea applies to
  any field — nonants outside their bounds, a write_id that went
  backwards, NaN data on a buffer the hub claims to have published, etc.
- **A new field/cylinder is being introduced** and you want a cheap
  invariant check during development.
- **Reproducing intermittent buffer-content bugs** where adding a print
  in the hot path is too noisy unless gated.

What the Inspector Checks
-------------------------

Generic checks (run for every field):

- Trailing ``write_id`` slot is a finite, non-negative, integer-valued
  double.
- Send buffers: the trailing slot equals ``buf.id()``.
- Receive buffers: the trailing slot is not less than ``buf.id()``
  (the last id that ``get_receive_buffer`` accepted). An optional
  ``ctx.last_write_id`` provides an additional, stricter baseline.
- Data region: no ``inf`` values; no ``NaN`` values once
  ``write_id >= 1``.
- Padding region (between ``logical_len`` and ``padded_len``) remains
  ``NaN`` — its canonical state from ``communicator_array``. A finite
  value anywhere in padding is a write that ran past the field's
  logical length.

Per-``Field`` checks:

- ``SHUTDOWN``: only two legitimate states — ``NaN`` data with
  ``write_id == 0`` (initial, no publish yet) or ``data[0] == 1.0``
  with ``write_id >= 1`` (``Hub.send_terminate`` has fired). Anything
  else, including ``data[0] == 0.0``, is treated as corruption.
- ``NONANT``: data length is a positive multiple of
  ``ctx.get_nonant_count()`` (the publisher may hold several local
  scenarios, so the buffer can be wider than one scenario's worth);
  componentwise bounds against ``[ctx.nonant_lower, ctx.nonant_upper]``
  are only checked when the bound arrays match the buffer length.
- ``NONANT_LOWER_BOUNDS`` / ``NONANT_UPPER_BOUNDS``: length check;
  consistency with the counterpart bound when supplied via ``ctx``.
- ``OBJECTIVE_INNER_BOUND`` / ``OBJECTIVE_OUTER_BOUND``: length 1.
- ``BEST_XHAT``: length at least ``ctx.get_nonant_count()``; nonant
  prefix within bounds when supplied.

Manual Use
----------

.. code-block:: python

   from mpisppy.debug_utils import inspect_buffer, InspectContext
   from mpisppy.cylinders.spwindow import Field

   ctx = InspectContext(nonant_count=spbase.nonant_length)
   rep = inspect_buffer(some_recv_buf, Field.NONANT, ctx, verbose=True)
   if not rep.ok:
       print(rep)

``Report`` is a small dataclass with ``ok``, ``findings`` (list of
strings), ``severity`` (``"warn"`` or ``"error"``), and an optional
``dump`` populated when ``verbose=True``. The inspector never raises;
the caller decides whether to log, raise, or treat the read as stale.

Command-Line Trigger at Cylinder Shutdown
-----------------------------------------

The ``--inspect-buffers-on-shutdown`` flag, exposed through the
standard ``Config`` system (``popular_args``), causes each spoke to
run the inspector on its ``SHUTDOWN`` receive buffer **at the moment a
shutdown is detected** (inside ``got_kill_signal``, only when the
signal fires — not on every poll). Findings, with rank info, are
printed when the report is not ok:

.. code-block:: bash

   mpiexec -np N python -m mpi4py my_driver.py --inspect-buffers-on-shutdown

When the flag is unset (the default), the inspector is never called
and the shutdown-poll cost is unchanged.

Choice of trigger point: a spurious ``SHUTDOWN`` is most diagnostic at
the moment of detection — the relevant buffer state has just arrived
and has not yet been overwritten by later activity. The check fires
once per spoke per cylinder shutdown, regardless of whether the
signal was legitimate; legitimate shutdowns produce an empty
findings list and print nothing.

Extending: Adding a Field Checker
---------------------------------

Each per-field check is a function with the signature
``(buf, report, ctx) -> None`` that appends findings to ``report``
when invariants are violated. Register it in the
``CHECKERS`` dict in ``mpisppy/debug_utils/buffer_inspect.py``:

.. code-block:: python

   def _check_my_field(buf, report, ctx):
       data = buf.value_array()
       if len(data) != some_expected_length:
           report.add(f"MY_FIELD wrong length: {len(data)}", severity="error")

   CHECKERS[Field.MY_FIELD] = _check_my_field

Producers are intentionally left untouched; any context the checker
needs (lengths, bounds, scenario tree info) is passed in via
``InspectContext``.

See Also
--------

The internal design document is at
``doc/designs/async_buffer_sanity_design.md``, including the
invariants the inspector relies on and explicit non-goals
(cross-cylinder consensus, module-level history state).
