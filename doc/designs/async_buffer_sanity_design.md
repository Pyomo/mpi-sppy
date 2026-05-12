# Async buffer sanity inspector

Status: sketch implemented on branch `async-buffer-sanity-design`; design open
for review.

Motivating incident: an xhat spoke received a `SHUTDOWN` signal that was not
sent by `Hub.send_terminate`, suggesting one of the MPI-RMA buffer regions
was being corrupted. The library had no runtime sanity check on buffer
contents; `_validate_recv_field` at `mpisppy/cylinders/spcommunicator.py:307`
only checks layout at registration time. This design adds a passive set of
content checks that can be called manually during debugging or wired in via
an environment variable on hot paths.

---

## 0. Goals

1. Inspect any `SendArray` or `RecvArray` after-the-fact for content
   sanity, given an optional caller-supplied context.
2. Make producers (`hub.py`, `spwindow.py`, send paths) untouchable;
   inspection is purely a consumer concern.
3. Provide a single entry point that works for both send and receive
   buffers and for every `Field`.
4. Catch the SHUTDOWN-stomp signature first, with a framework that
   extends to other fields.

Non-goals:

- Cross-cylinder agreement checks (the existing `synchronize=True` path
  inside `get_receive_buffer` already does this, and we are not paying
  another barrier+allreduce).
- Module-level state that remembers per-buffer history. Tempting for
  detecting trailing-slot oscillation, but it hides state in the
  inspector and grows unbounded as buffers come and go.
- Modifying any producer to publish extra metadata.

---

## 1. Invariants we lean on

These come from reading `communicator_array` at
`mpisppy/cylinders/spcommunicator.py:33-58` and the `FieldArray` hierarchy
at lines 61-148.

1. `communicator_array` allocates a padded MPI memory region and
   initializes the *entire* region (data + id + padding) to NaN, then
   writes 0.0 into the trailing logical slot (the write_id).
2. The hub's send path only ever writes into the logical `value_array()`
   region and bumps the trailing write_id via
   `SendArray._next_write_id()` (line 122).
3. RMA `get` copies the full padded buffer over.
4. `RecvArray._pull_id()` (line 142) records the last id whose
   corresponding payload was accepted by `get_receive_buffer`.

Consequences the inspector relies on:

- The padding region (`window_array()[logical_len:padded_len]`) must
  remain NaN. A finite value anywhere in padding is a write that ran
  past the field's logical length.
- A data slot may be NaN *only* when write_id == 0 (no publish has
  happened). Once write_id >= 1, NaN in data is a corrupted publish.
- `buf.array()[-1]` (the just-arrived trailing slot) must be >=
  `buf.id()` (the last accepted id). A regression means the trailing
  slot was rewritten to a smaller value.
- For a `SendArray`, the trailing slot must equal `buf.id()` between
  publishes.

---

## 2. API

Module: `mpisppy/debug_utils/buffer_inspect.py`.

```python
@dataclass
class InspectContext:
    expected_write_id: Optional[int] = None
    last_write_id:    Optional[int] = None   # caller-tracked baseline
    nonant_count:     Optional[int] = None
    nonant_lower:     Optional[np.ndarray] = None
    nonant_upper:     Optional[np.ndarray] = None
    spbase:           Optional[Any] = None   # duck-typed; fallback source

    def get_nonant_count(self) -> Optional[int]: ...

@dataclass
class Report:
    ok: bool = True
    findings: list[str]
    severity: str = "warn"   # "warn" | "error"
    dump:     Optional[str]  # populated when verbose=True

def inspect_buffer(buf, fld: Field, ctx: Optional[InspectContext] = None,
                   *, send: bool = False, verbose: bool = False) -> Report
```

`InspectContext` fields are all optional; the inspector silently skips
any check whose context is missing. `spbase` is looked up via
`getattr(spbase, "nonant_length", None)`, so a partial mock that exposes
just `nonant_length` works. Explicit fields on the context take
precedence over `spbase`-derived ones.

`Report.severity` ladders from warn to error. Caller decides what to do
with a non-OK report — log, raise, or set `_is_new = False`. The
inspector never raises.

`verbose=True` populates `Report.dump` with a small text block
containing `buf.array()`, `buf.window_array()`, `buf.id()`,
`logical_len`, `padded_len`, and the field name. Cheap for small
fields; for large fields the caller decides.

---

## 3. Generic checks (run for every field)

`_check_generic` (called first):

| Check | Severity | Condition |
|---|---|---|
| trailing slot finite | error | `not np.isfinite(buf.array()[-1])` |
| trailing slot is integer-valued | error | `abs(raw - round(raw)) > 1e-9` |
| trailing slot non-negative | error | `write_id < 0` |
| send: trailing slot == `buf.id()` | error | when `send=True` |
| recv: trailing slot >= `buf.id()` | error | when `send=False` |
| recv: trailing slot >= `ctx.last_write_id` | error | when supplied |
| trailing slot == `ctx.expected_write_id` | warn | when supplied |
| no inf in data | error | always |
| no NaN in data when write_id >= 1 | error | publish should have overwritten initial NaN |
| padding region all-NaN | error | `not np.all(np.isnan(window_array()[logical_len:padded_len]))` |

---

## 4. Per-field registry

`CHECKERS: dict[Field, Callable]`. Entries currently implemented:

- `SHUTDOWN`: data[0] in {0.0, 1.0}; if 1.0 then write_id >= 1. The
  initial state (data NaN, write_id 0) is allowed.
- `NONANT`: length == `ctx.get_nonant_count()`; data in
  `[ctx.nonant_lower, ctx.nonant_upper]` componentwise (when supplied).
- `NONANT_LOWER_BOUNDS` / `NONANT_UPPER_BOUNDS`: length check; if the
  caller passes the *other* bound (via `ctx.nonant_upper` /
  `ctx.nonant_lower`), check componentwise consistency.
- `OBJECTIVE_INNER_BOUND` / `OBJECTIVE_OUTER_BOUND`: length == 1.
- `BEST_XHAT`: length >= `ctx.get_nonant_count()` (the buffer also
  carries per-scenario costs); the nonant prefix is bounds-checked when
  bounds are supplied.

Fields without an entry (`DUALS`, `RELAXED_NONANT`,
`CROSS_SCENARIO_CUT`, `CROSS_SCENARIO_COST`, `EXPECTED_REDUCED_COST`,
`SCENARIO_REDUCED_COST`, `RECENT_XHATS`, `BEST_OBJECTIVE_BOUNDS`,
`WHOLE`) get generic checks only. They can be filled in as needs arise.

Adding a checker: write a function with signature
`(buf, report, ctx) -> None`, register it in `CHECKERS`. No producer
changes required.

---

## 5. Wiring

Two modes, both opt-in.

**Manual.** A developer or test imports `inspect_buffer` and pokes it at
a buffer of interest. Example uses already in
`mpisppy/debug_utils/buffer_inspect.py` smoke tests:

```python
from mpisppy.debug_utils.buffer_inspect import inspect_buffer, InspectContext
rep = inspect_buffer(shutdown_buf, Field.SHUTDOWN, send=False, verbose=True)
if not rep.ok:
    print(rep)
```

**CLI-gated check at the shutdown moment.** `_BoundSpoke.got_kill_signal`
in `mpisppy/cylinders/spoke.py:24-30` is the most likely place to catch
the motivating bug. A new flag `inspect_buffers_on_shutdown` is added
in `Config.popular_args` and propagated through `cfg_vanilla.shared_options`
into `opt.options`. The hook runs the inspector only when the kill
fires *and* the flag is set:

```python
fired = bool(shutdown_buf[0] == 1.0)
if fired and self.opt.options.get("inspect_buffers_on_shutdown"):
    self._inspect_buffers_on_shutdown(shutdown_buf)
```

`_inspect_buffers_on_shutdown` sweeps every registered receive and send
buffer through `inspect_buffer`, not just SHUTDOWN. SHUTDOWN goes first
and verbose (the diagnostic dump lands in the warning); the rest run
non-verbose. `InspectContext(spbase=self.opt)` is threaded through so the
per-field checkers that need nonant length pick it up via the spbase
fallback. The sweep is what gives us real-buffer false-positive coverage
for every checker (not just SHUTDOWN) once the smoke runs.

A failed inspection emits a `RuntimeWarning` (not a `print`) so the signal
is filterable, capturable in tests via `warnings.catch_warnings(record=True)`,
and escalatable to a hard error via `python -W error::RuntimeWarning:mpisppy.cylinders.spoke`.
We do not `raise` here: `got_kill_signal` runs during the collective shutdown
path, and aborting on one rank would leave peers blocked on the next barrier;
the inspector's job at this site is to observe the suspect shutdown, not
abort it. Hot-path call sites added later (e.g. `update_nonants`) may
choose to raise on `rep.severity == "error"`.

When the flag is unset (default), the inspector is not called. We fire
at the moment of detection rather than every poll because a spurious
shutdown is most diagnostic when the buffer state has just arrived and
not yet been overwritten by later activity.

**Smoke coverage.** `mpisppy/tests/straight_tests.py` runs the existing
multi-stage Aircond cylinder invocation (PH hub + lagranger + fwph +
xhatshuffle on 4 ranks) with `--inspect-buffers-on-shutdown` and
`python -W error::RuntimeWarning:mpisppy.cylinders.spoke`. The shutdown
sweep visits every buffer used by that run -- SHUTDOWN, NONANT,
OBJECTIVE_INNER_BOUND, OBJECTIVE_OUTER_BOUND, BEST_XHAT, plus any
others these cylinders register -- so a healthy run is also a
no-false-positives guard for the corresponding checkers. A regression
that produces a warning trips the escalation and the subprocess exits
non-zero. `NONANT_LOWER_BOUNDS` / `NONANT_UPPER_BOUNDS` remain uncovered
by smoke because the cylinders in this run don't produce them; a separate
smoke with a reduced-costs or nonant-bounds spoke would close that gap.

The unit-level test `TestSpokeGotKillSignalWarning` in
`mpisppy/tests/test_buffer_inspect.py` drives `Spoke.got_kill_signal` on
a stub: with a hand-stomped SHUTDOWN buffer (warning fires); with a legit
SHUTDOWN (silent); with the flag off (sweep not invoked); and with a
multi-buffer sweep that mixes healthy NONANT recv/send buffers with a
stomped OBJECTIVE_INNER_BOUND recv (exactly one warning, naming the bad
field).

Other hot paths (`update_nonants`, `sync_bounds`, etc.) can be wired
the same way later. They are intentionally not wired in this round so
that the env-gated overhead surface stays small while we shake the
inspector out.

---

## 6. Detecting the motivating SHUTDOWN bug

Suspected signature: an xhat spoke sees `shutdown_buf[0] == 1.0` while
the hub has not called `send_terminate`. At least one of the following
inspector findings should fire under that scenario:

1. `SHUTDOWN data[0]==1.0 but write_id==0` — the hub never published.
2. `recv write_id N < buf.id() M` — the trailing slot regressed after
   a previously-accepted read.
3. `padding region modified: K non-NaN slot(s) at offsets ...` — an
   adjacent field's write overran into the SHUTDOWN region's padding.

If none of the three fire and the spurious shutdown still occurs, the
write_id slot and data slot are both consistent with a legitimate
shutdown; the next hypothesis would shift away from "stomp" toward a
producer that publishes shutdown out of band.

---

## 7. Future work

- Fill `DUALS`, `RECENT_XHATS`, `CROSS_SCENARIO_*` checkers as they
  become useful.
- Add a `synchronize=True` variant of the SHUTDOWN read in a debug mode
  to cross-check write_id agreement (already a one-line change on
  `spoke.py:29`).
- An EF-side inspector for the EF solve path is a separate effort; the
  buffer layout differs.

---

## 8. Files

- `mpisppy/debug_utils/__init__.py` — package marker.
- `mpisppy/debug_utils/buffer_inspect.py` — the inspector.
- `mpisppy/cylinders/spoke.py` — env-gated hook in `got_kill_signal`.
- `doc/designs/async_buffer_sanity_design.md` — this document.
