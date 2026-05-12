###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Sanity-check utilities for send and receive buffers.

Producers of the buffers (hub and spoke code) are intentionally not modified.
Callers that want to validate a buffer instantiate an InspectContext (any
field optional) and call inspect_buffer(buf, field, ctx).

See doc/designs/async_buffer_sanity_design.md (in progress) for the rationale
behind each check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from mpisppy.cylinders.spwindow import Field


# ---- public dataclasses -----------------------------------------------------


@dataclass
class InspectContext:
    """Caller-supplied context; every field optional.

    Inspector skips any check whose required context is None.

    spbase, when supplied, is queried via getattr for fallback values
    (e.g. nonant_count -> spbase.nonant_length). Explicit fields on the
    context take precedence over spbase-derived ones.
    """
    expected_write_id: Optional[int] = None
    last_write_id: Optional[int] = None
    nonant_count: Optional[int] = None
    nonant_lower: Optional[np.ndarray] = None
    nonant_upper: Optional[np.ndarray] = None
    spbase: Optional[Any] = None

    def get_nonant_count(self) -> Optional[int]:
        if self.nonant_count is not None:
            return self.nonant_count
        return getattr(self.spbase, "nonant_length", None)


@dataclass
class Report:
    ok: bool = True
    findings: list[str] = field(default_factory=list)
    severity: str = "warn"   # "warn" or "error"
    dump: Optional[str] = None

    def add(self, msg: str, severity: str = "warn") -> None:
        self.ok = False
        self.findings.append(msg)
        if severity == "error":
            self.severity = "error"

    def __str__(self) -> str:
        head = "OK" if self.ok else f"FAIL ({self.severity})"
        body = "\n".join(f"  - {f}" for f in self.findings) or "  (no findings)"
        out = f"[buffer_inspect] {head}\n{body}"
        if self.dump:
            out += f"\n  dump:\n{self.dump}"
        return out


# ---- main entry point -------------------------------------------------------


_INT_TOL = 1e-9


def inspect_buffer(buf,
                   fld: Field,
                   ctx: Optional[InspectContext] = None,
                   *,
                   send: bool = False,
                   verbose: bool = False) -> Report:
    """Inspect a SendArray or RecvArray for sanity.

    Args:
        buf: a FieldArray (SendArray or RecvArray) from spcommunicator.
        fld: the Field this buffer carries.
        ctx: optional caller context (expected_write_id, bounds, ...).
        send: True for SendArray, False for RecvArray.
        verbose: include a raw dump in the returned Report.

    Returns:
        Report with ok / findings / severity / optional dump.
    """
    if ctx is None:
        ctx = InspectContext()
    report = Report()

    _check_generic(buf, report, ctx, send=send)
    checker = CHECKERS.get(fld)
    if checker is not None:
        checker(buf, report, ctx)

    if verbose:
        report.dump = _format_dump(buf, fld)

    return report


# ---- generic checks ---------------------------------------------------------


def _check_generic(buf, report: Report, ctx: InspectContext, *, send: bool) -> None:
    logical = buf.array()
    raw_id = logical[-1]
    write_id = _check_write_id_slot(raw_id, report)

    _check_padding_is_nan(buf, report)

    if write_id is None:
        # write_id slot itself was malformed; data-NaN check can't reason
        # about whether NaNs are "expected initial state" or corruption.
        return

    _check_data_nan_consistency(buf, write_id, report)

    if send:
        if write_id != buf.id():
            report.add(
                f"send buffer write_id slot ({write_id}) != buf.id() ({buf.id()})",
                severity="error",
            )
    else:
        # buf.id() == last write_id that get_receive_buffer accepted via
        # _pull_id; the trailing slot must never go below it.
        if write_id < buf.id():
            report.add(
                f"recv write_id {write_id} < buf.id() {buf.id()} "
                "(trailing slot went backwards since last accepted read)",
                severity="error",
            )
        if ctx.last_write_id is not None and write_id < ctx.last_write_id:
            report.add(
                f"recv buffer write_id ({write_id}) regressed below "
                f"ctx.last_write_id ({ctx.last_write_id})",
                severity="error",
            )

    if ctx.expected_write_id is not None and write_id != ctx.expected_write_id:
        report.add(
            f"write_id {write_id} != ctx.expected_write_id {ctx.expected_write_id}"
        )


def _check_write_id_slot(raw_id: float, report: Report) -> Optional[int]:
    if not np.isfinite(raw_id):
        report.add(f"write_id slot is non-finite: {raw_id!r}", severity="error")
        return None
    rounded = round(raw_id)
    if abs(raw_id - rounded) > _INT_TOL:
        report.add(
            f"write_id slot {raw_id!r} is not integer-valued",
            severity="error",
        )
        return None
    if rounded < 0:
        report.add(f"write_id slot {rounded} is negative", severity="error")
        return None
    return int(rounded)


def _check_padding_is_nan(buf, report: Report) -> None:
    # communicator_array initializes the entire window to NaN and only
    # rewrites the logical view. Padding that is no longer NaN means
    # something has written past the field's logical length.
    full = buf.window_array()
    logical_len = buf.logical_len()
    padded_len = buf.padded_len()
    if padded_len == logical_len:
        return
    pad = full[logical_len:padded_len]
    if not np.all(np.isnan(pad)):
        bad = np.where(~np.isnan(pad))[0]
        report.add(
            f"padding region modified: {len(bad)} non-NaN slot(s) "
            f"at offsets {bad.tolist()[:8]}{'...' if len(bad) > 8 else ''}",
            severity="error",
        )


def _check_data_nan_consistency(buf, write_id: int, report: Report) -> None:
    data = buf.value_array()
    has_nan = bool(np.any(np.isnan(data)))
    has_inf = bool(np.any(np.isinf(data)))
    if write_id >= 1 and has_nan:
        report.add(
            "data contains NaN but write_id >= 1 (publish should have "
            "overwritten initial NaN)",
            severity="error",
        )
    if has_inf:
        report.add("data contains inf", severity="error")


# ---- field-specific checkers -----------------------------------------------


def _check_shutdown(buf, report: Report, ctx: InspectContext) -> None:
    # Only two legitimate states exist:
    #   - NaN with write_id == 0: initial state from communicator_array,
    #     no publish has happened yet.
    #   - 1.0 with write_id >= 1: Hub.send_terminate has fired.
    # No producer ever writes 0.0 or any other value, so anything else
    # is a stomp, an RMA race, or a producer bug.
    data = buf.value_array()
    val = data[0]
    raw_id = buf.array()[-1]
    write_id = int(round(raw_id)) if np.isfinite(raw_id) else None

    if write_id == 0 and np.isnan(val):
        return

    if val != 1.0:
        report.add(
            f"SHUTDOWN data[0]={val!r}; only 1.0 is ever published "
            "(or NaN with write_id==0 for the initial state)",
            severity="error",
        )
    if val == 1.0 and write_id is not None and write_id < 1:
        report.add(
            f"SHUTDOWN data[0]==1.0 but write_id=={write_id}; "
            "only send_terminate writes 1.0 and it bumps the id",
            severity="error",
        )


def _check_nonant(buf, report: Report, ctx: InspectContext) -> None:
    # NONANT buffers are sized to the *publisher's* sum of per-scenario
    # nonant counts across that publisher's local scenarios:
    #     len(data) == nonant_count * len(publisher.local_scenarios).
    # When the publisher holds many scenarios (e.g. all 24 leaves of a
    # multi-stage Aircond on one hub rank), len(data) >> nonant_count.
    # The buffer's logical length is already enforced at registration
    # by _validate_recv_field, so here we only insist that the size
    # be a positive multiple of nonant_count, and that the componentwise
    # bounds compare runs only when the buffer happens to be a single
    # scenario wide.
    data = buf.value_array()
    n = ctx.get_nonant_count()
    if n is not None and n > 0 and (len(data) == 0 or len(data) % n != 0):
        report.add(
            f"NONANT data length {len(data)} is not a positive multiple "
            f"of nonant_count {n}",
            severity="error",
        )
    lo, hi = ctx.nonant_lower, ctx.nonant_upper
    raw_id = buf.array()[-1]
    write_id = int(round(raw_id)) if np.isfinite(raw_id) else 0
    # Bounds compare only makes sense once data has been published.
    if write_id < 1:
        return
    if lo is not None and len(lo) == len(data):
        bad = np.where(data < lo)[0]
        if bad.size:
            report.add(
                f"NONANT below lower bound at {bad.size} index(es); "
                f"first: idx {int(bad[0])} value {float(data[bad[0]])!r} "
                f"< lo {float(lo[bad[0]])!r}"
            )
    if hi is not None and len(hi) == len(data):
        bad = np.where(data > hi)[0]
        if bad.size:
            report.add(
                f"NONANT above upper bound at {bad.size} index(es); "
                f"first: idx {int(bad[0])} value {float(data[bad[0]])!r} "
                f"> hi {float(hi[bad[0]])!r}"
            )


def _check_lower_bounds(buf, report: Report, ctx: InspectContext) -> None:
    _check_bound_pair(buf, report, ctx, is_lower=True)


def _check_upper_bounds(buf, report: Report, ctx: InspectContext) -> None:
    _check_bound_pair(buf, report, ctx, is_lower=False)


def _check_bound_pair(buf, report: Report, ctx: InspectContext,
                      *, is_lower: bool) -> None:
    data = buf.value_array()
    # Length: if caller knows nonant_count, the bound buffer should match.
    # (Conservative: total_number_nonants is the real length, but typical
    # two-stage problems set both equal, and a strict total-nonant ctx
    # field would have to be threaded; lean on nonant_count for now.)
    n = ctx.get_nonant_count()
    if n is not None and len(data) != n:
        report.add(
            f"{'LOWER' if is_lower else 'UPPER'}_BOUNDS data length "
            f"{len(data)} != expected {n}",
            severity="error",
        )
    raw_id = buf.array()[-1]
    write_id = int(round(raw_id)) if np.isfinite(raw_id) else 0
    if write_id < 1:
        return
    # If the caller passed the *other* bound, check componentwise consistency.
    other = ctx.nonant_upper if is_lower else ctx.nonant_lower
    if other is None or len(other) != len(data):
        return
    if is_lower:
        bad = np.where(data > other)[0]
        msg_dir = "lower > upper"
    else:
        bad = np.where(data < other)[0]
        msg_dir = "upper < lower"
    if bad.size:
        report.add(
            f"{msg_dir} at {bad.size} index(es); first: idx {int(bad[0])} "
            f"this={float(data[bad[0]])!r} other={float(other[bad[0]])!r}",
            severity="error",
        )


def _check_objective_scalar(buf, report: Report, ctx: InspectContext) -> None:
    data = buf.value_array()
    if len(data) != 1:
        report.add(
            f"objective bound data length {len(data)} != 1",
            severity="error",
        )
    # Finiteness already handled by _check_data_nan_consistency once
    # write_id >= 1; nothing else generic to assert about a scalar bound.


def _check_best_xhat(buf, report: Report, ctx: InspectContext) -> None:
    # BEST_XHAT = [nonant values..., per-scenario costs...]; we don't
    # have scenario count in ctx, so length is only loosely constrained.
    data = buf.value_array()
    n = ctx.get_nonant_count()
    if n is not None and len(data) < n:
        report.add(
            f"BEST_XHAT data length {len(data)} < nonant_count {n}",
            severity="error",
        )
        return
    raw_id = buf.array()[-1]
    write_id = int(round(raw_id)) if np.isfinite(raw_id) else 0
    if write_id < 1 or n is None:
        return
    xhat = data[:n]
    lo, hi = ctx.nonant_lower, ctx.nonant_upper
    if lo is not None and len(lo) == n:
        bad = np.where(xhat < lo)[0]
        if bad.size:
            report.add(
                f"BEST_XHAT nonant portion below lower bound at "
                f"{bad.size} index(es); first: idx {int(bad[0])} "
                f"value {float(xhat[bad[0]])!r} < lo {float(lo[bad[0]])!r}"
            )
    if hi is not None and len(hi) == n:
        bad = np.where(xhat > hi)[0]
        if bad.size:
            report.add(
                f"BEST_XHAT nonant portion above upper bound at "
                f"{bad.size} index(es); first: idx {int(bad[0])} "
                f"value {float(xhat[bad[0]])!r} > hi {float(hi[bad[0]])!r}"
            )


CHECKERS: dict[Field, Callable[[Any, Report, InspectContext], None]] = {
    Field.SHUTDOWN: _check_shutdown,
    Field.NONANT: _check_nonant,
    Field.NONANT_LOWER_BOUNDS: _check_lower_bounds,
    Field.NONANT_UPPER_BOUNDS: _check_upper_bounds,
    Field.OBJECTIVE_INNER_BOUND: _check_objective_scalar,
    Field.OBJECTIVE_OUTER_BOUND: _check_objective_scalar,
    Field.BEST_XHAT: _check_best_xhat,
}


# ---- dump helper ------------------------------------------------------------


def _format_dump(buf, fld: Field) -> str:
    logical = buf.array()
    full = buf.window_array()
    return (
        f"    field      = {fld.name} ({int(fld)})\n"
        f"    buf.id()   = {buf.id()}\n"
        f"    data_len   = {buf.data_len()}\n"
        f"    logical_len= {buf.logical_len()}\n"
        f"    padded_len = {buf.padded_len()}\n"
        f"    logical    = {logical.tolist()}\n"
        f"    padded     = {full.tolist()}"
    )
