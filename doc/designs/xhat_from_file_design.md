# Design: Supply an Initial Xhat from a File

**Status:** Draft — up for discussion. Nothing is implemented yet.
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-04-23

## Motivation

Two independent uses call for the same mechanism:

1. **Warm start / user-supplied candidate.** A user often has a first-stage
   solution they want to try first — from a prior run, a hand-computed
   heuristic, a rolling-horizon neighbor, or a solution transferred from
   a different but related instance. Today, the only way to inject an
   `xhat` into a running cylinder system is to modify code; there is no
   CLI surface for it.
2. **Testing infeasibility-driven features.** The new xhat feasibility
   cuts (PR #671, issue #601) install a cut when an xhatter finds an
   infeasible `xhat`. End-to-end testing that path currently requires
   engineering a scenario that happens to be infeasible for *some*
   `xhat` the xhatter will naturally propose. A supply-your-own-`xhat`
   flag makes it trivial: write the known-infeasible binary vector to
   a file and hand it in.

The same file-read path serves both use cases — so ship it once.

## Non-goals

- Persistent xhat swapping mid-run. The file is read once at spoke
  startup; the xhatter evaluates it, reports the inner bound (or
  infeasibility), and then continues with its normal exploration. No
  re-read mechanism.
- A cross-language data exchange format. We use what mpi-sppy already
  uses.
- Multi-stage. V1 is two-stage-only (matches the current `.npy`
  reader's scope); multi-stage is a named follow-up.

## Related work in the tree

- **`mpisppy/confidence_intervals/ciutils.py`** has
  `read_xhat(path, num_stages=2)` and `write_xhat(xhat, path,
  num_stages=2)`. Both only handle `num_stages=2` today; both use
  `numpy.save`/`numpy.load` on the `xhat['ROOT']` vector. The MMW
  path uses these via `--mmw-xhat-input-file-name`.
- **Jensen's bound design** (PR #657,
  `doc/jensens_bound_design.md`) adds an opt-in `--*-try-jensens-first`
  flag to each xhatter. On startup, the xhatter builds an average
  scenario, solves it, uses the first-stage solution as an xhat,
  evaluates it across all scenarios, and reports the resulting inner
  bound — then continues with normal iteration. Our new feature
  wants to do exactly the same thing with a file-supplied xhat
  instead of an average-scenario-derived one. **Precedence with
  Jensen's is the main new design decision; see §4.**
- **`Xhat_Eval.evaluate(nonant_cache)`**
  (`mpisppy/utils/xhat_eval.py::Xhat_Eval.evaluate`) already does the
  right thing: fix nonants to the supplied values, `solve_loop` across
  local scenarios, compute the expected objective. Whatever mechanism
  loads the file just needs to pack the values into a `nonant_cache`
  and call `evaluate`.

## Proposed architecture

Single flag on the xhat spokes, read once at startup:

```
--xhat-from-file <path>
```

### Format

`.npy` only, via the existing `ciutils.read_xhat` helper.

- It is already the canonical mpi-sppy xhat on-disk format (MMW uses
  it; examples write it).
- It is already restricted to two-stage, which aligns with our V1
  non-goal.
- Extending to more formats (CSV, JSON) is a follow-up, only if a real
  use case appears. Each format adds surface area and edge cases
  (column order, header presence, numeric precision); none are worth
  paying for speculatively.

### Where the read happens

At the start of each xhat spoke's `main()`, immediately **after**
`self.xhat_prep()` (which sets up `Xhat_Eval`) and **before** the
spoke's normal iteration loop. Mirrors the Jensen's hook point
verbatim so the two features share the same pre-loop slot.

Concretely, add a small helper to `XhatBase` (the extension; not the
cylinder base) that:

1. Reads the `.npy` file via `ciutils.read_xhat`.
2. Packs the values into a `nonant_cache` shaped like
   `s._mpisppy_data.nonant_indices` order. Same packing logic as
   Jensen's mixin §5.2 (`_pack_nonant_cache`).
3. Calls `self.opt.evaluate(nonant_cache)` to get an `Eobj` (or
   `None` on infeasibility).
4. If finite: `self.update_if_improving(Eobj)` → send inner bound
   and best-xhat to the hub as the first inner-bound report.
5. If `None`: xhat was infeasible. The xhatter's normal infeasibility
   handling takes over (including the `--xhat-feasibility-cuts-count`
   emission if that feature is enabled), and the spoke continues into
   its regular loop.

The helper lives on `XhatBase` so every xhatter (`xhatlooper`,
`xhatshufflelooper`, `xhatspecific`, `xhatxbar`) picks it up for
free. The spokes themselves stay untouched beyond the one-liner
invocation at the top of `main()`.

### CLI + cfg plumbing

- New `cfg.xhat_from_file_args()` method registering the flag with
  default `None` (feature off).
- Added to `mpisppy/generic/parsing.py::parse_args` alongside the
  other `xhatXxx_args()` calls.
- `shared_options` in `cfg_vanilla` carries the path string through
  to every xhat spoke's options dict, parallel to how we're
  propagating `xhat_feasibility_cuts_count`.
- Startup-time validation: if the flag is set, the file must exist
  and be readable. Hard-fail with a clear error message at `xhat_prep`
  time (not later when we try to load).

## Precedence with Jensen's `--*-try-jensens-first`

Both features contribute a candidate xhat to evaluate once, before the
spoke's main loop. Both use `Xhat_Eval.evaluate(nonant_cache)` and
`update_if_improving(Eobj)`. They do not conflict mechanically — but
we need a rule for ordering if both are set.

**Rule:** *file-supplied xhat first, then Jensen's, then the spoke's
normal loop.* Rationale:

- The file is an *explicit* user hint — the user went to the trouble
  of writing an `.npy`. If both are on, they probably want to see
  what the explicit candidate does before Jensen's.
- Jensen's is a cheap-to-compute heuristic first pick; a natural
  fallback when the file-supplied one is infeasible.
- `update_if_improving` keeps whichever is better, so correctness
  is invariant to order; only the ordering of the first `send_bound`
  report differs. Predictable ordering matters for test stability
  and log readability.

Implementation: the two features contribute candidates to the front
of an (internal, trivial) list:

```
candidates = []
if xhat_from_file_path is not None:
    candidates.append(("from-file", load_nonant_cache_from_file()))
if jensens_enabled:
    candidates.append(("jensens", solve_ev_and_pack()))
for label, nc in candidates:
    Eobj = self.opt.evaluate(nc)
    if Eobj is not None:
        self.update_if_improving(Eobj)
    # infeasibility falls through; feas-cut path (if on) fires
```

No mixin surgery needed; both features just agree to share the
"candidates" list if we want to refactor later. For V1, each feature
writes its own snippet at the top of `main()` and the doc commits to
the above order.

## Multi-stage

Deferred, same as for Jensen's and for `ciutils.read_xhat` itself.
For V1, the startup check raises if `num_stages != 2`, with a pointer
to this design doc's follow-up section. Mechanically, multi-stage
needs either (a) a per-node `.npy` or (b) a format that carries node
keys — both are future work. The feature is still useful in its
two-stage form, and shipping that now does not foreclose the
multi-stage extension.

## Interaction with `--xhat-feasibility-cuts-count`

This is the test-motivated case. When both flags are on:

1. Spoke reads the file, evaluates.
2. `Xhat_Eval.evaluate` detects infeasibility (or not). On
   infeasibility, `_try_one` → `_maybe_emit_feasibility_cut` fires
   (the code landed in PR #671), sending a no-good row through
   `Field.XHAT_FEASIBILITY_CUT`.
3. The hub's `XhatFeasibilityCutExtension` installs the cut into
   every scenario.
4. On the next xhatter iteration, the same `xhat` is now excluded
   by the installed cut.

Step 4 gives us the end-to-end assertion the current
`test_xhat_feasibility_cuts.py` unit tests cannot make: *the
infeasible xhat is not revisited*. A new integration test:

- Writes a known-infeasible binary vector to an `.npy`.
- Runs `generic_cylinders` with `--xhat-from-file` +
  `--xhat-feasibility-cuts-count 1` + a minimal binary-first-stage
  model that is infeasible at that vector but feasible elsewhere.
- Asserts (via run output or a post-run probe) that a cut got
  installed.

This is the testing payoff that motivated the feature. It does not
drive the design — but it shapes what "first-milestone scope" means
below.

## Open questions

- **Should the file-read be rank-zero-only plus bcast, or should every
  rank read independently?** `ciutils.read_xhat` is called per-rank
  today; with a small shared file system this is fine, but on
  clusters with flaky shared storage, rank-0-read + `bcast` is safer.
  V1: mirror the existing `ciutils` behavior (per-rank read). Revisit
  if a user reports trouble.
- **What happens when the file's vector length doesn't match the
  problem's root-node nonant count?** Hard-fail at load time with a
  clear message. No truncation, no padding.
- **Should the inner bound reported from the file xhat update the
  `best_solution_cache`?** Yes — `update_if_improving` does this by
  default (`spoke.py:173-190`). The file-supplied xhat is a real
  candidate and should be reported like any other.

## First-milestone scope

A discrete, reviewable PR delivering:

1. `cfg.xhat_from_file_args()` registering `--xhat-from-file`.
2. `XhatBase` helper that loads the file, packs the nonant cache,
   evaluates, and updates if improving.
3. Each of the four xhat spokes calls that helper at the top of
   `main()` (right after `xhat_prep`, before Jensen's if present).
4. Hard-fail paths: missing file, nonant-length mismatch, multi-stage.
5. `cfg_vanilla.shared_options` propagates the path.
6. Wired into `mpisppy/generic/parsing.py` so `generic_cylinders`
   accepts the flag.
7. Tests:
   - Unit: loader returns the right vector; wrong-length raises.
   - Integration (no MPI): mock `Xhat_Eval.evaluate` and assert the
     helper calls it with the file-loaded vector.
   - Integration (the testing-PR-671 payoff): `run_all.py` entry
     that runs the USAR `wheel_spinner` (binary first-stage) with a
     pre-computed `.npy` file and `--xhat-feasibility-cuts-count=1`,
     then asserts the run completes — and if we can arrange a
     known-infeasible xhat, that a feasibility cut fired.
8. User-facing `doc/src/xhat_from_file.rst` wired into
   `doc/src/index.rst`. Includes the "use this to test feasibility
   cuts" recipe.

Follow-ups (not in V1):

- Multi-stage support.
- CSV/JSON support.
- Re-read mid-run (unlikely to ever be worth it).
