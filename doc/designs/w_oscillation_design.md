# W-oscillation detection and interruption — design

**Status:** PR1 *detection + reporting* (pure observation, no behavior
change) is implemented in ``mpisppy/extensions/w_oscillation.py``; PR2
*interruption* (acts on the detector to break the oscillation) is not yet
started. Shipped as **two review-sized PRs**.
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-06-30

Related work:

- **wtracker** (`mpisppy/utils/w_utils/`) already tracks the W vector and
  computes moving statistics; `wtracker.py` even lists "track oscillations,
  which are important for MIPs" as an explicit TODO, and ships a crude
  `check_cross_zero`. This design is the planned successor to that TODO. The
  user documentation for the detection PR **cross-references the wtracker
  docs** as the way to get the full W trajectory and moving-stat reports;
  this extension is the focused *oscillation* layer on top of the same data.
- **sorgw** (PySP's `pysp/plugins/sorgw.py`) — the zero-crossing detector
  this design adopts as method A (§5.1).
- **Watson–Woodruff "Progressive Hedging Innovations for a Class of Stochastic
  Mixed-Integer Resource Allocation Problems"** (Computational Management
  Science, 2011; optimization-online 2089), §2.4 "Detecting Cyclic Behavior" —
  the W-vector-hashing cycle detector this design adopts as method B (§5.2).
  Its §2.1 (SEP / cost-proportional rho) motivates the rho-reduction action
  (§9).
- **slammer** (`mpisppy/extensions/slammer.py`, `doc/designs/slamming_design.md`)
  — the slamming *mechanism* whose action layer (its §7) PR2's "slam" action
  invokes. The slamming design explicitly anticipated "a separate future
  project that detects stalls/cycles" as the consumer of that mechanism;
  **this is that project.**

---

## 0. Vocabulary

We say a nonanticipative variable's dual weight **W oscillates** when, for a
given (scenario, nonant) pair, the trajectory of `W` over PH iterations
changes sign repeatedly or fails to damp — the hallmark of PH cycling, which
is common and convergence-killing for MIPs. Watson–Woodruff §2.1 describes the
mechanism: "Oscillation can occur when the `w` values are updated too
aggressively or converge from both sides, particularly in MIPs, because the
changes in the value of one integer variable can induce changes in others,
which are then reversed if the `w` multiplier 'shoots past' its optimal
value." Detection is **per (scenario, nonant)**; a nonant is *reported* /
*acted on* using an aggregate across scenarios (§6).

---

## 1. Goals and non-goals

### Goals

1. A **hub extension** that, while a PH hub runs, observes the W vector and
   **detects oscillation** per the methods in §5, configured entirely from a
   **JSON control file**.
2. **Report** detected oscillations to a **CSV file** (header + one row per
   detection event), filename from the JSON, **written by cylinder rank 0**.
   Each row names the **full variable name** and the oscillation evidence
   relative to the JSON's control parameters (e.g. how many crossings, in how
   many scenarios).
3. (PR2) Optionally **interrupt** detected oscillation, via **rho reduction**
   and/or **slamming**, configured from a second JSON control file.
4. **Pluggable detection methods.** The JSON selects which method(s) run and
   parameterizes each, so methods can be added without touching the CLI.
5. **Reuse, don't duplicate, the W-tracking code.** Capture the W trajectory
   with the same idiom wtracker uses; cross-reference wtracker in the docs.
6. **Total backward compatibility.** A run with neither new flag behaves
   exactly as today; PR1 never changes the algorithm (pure observation).

### Non-goals

- **APH.** The hooks run under any `PHBase` hub, but oscillation/cadence
  notions assume synchronous iterations. Scope is **synchronous PH**; APH is
  not specially wired (consistent with the project's "PH-only, C++ APH
  incoming" posture and the slamming design's APH caveat).
- **Spokes.** This is a hub extension. Outer/inner-bound spokes are untouched.
- **Replacing wtracker.** wtracker remains the general W-logging / moving-stat
  tool; this extension is the narrower oscillation detector/actor that reuses
  its capture idiom.
- **A new slamming mechanism.** PR2 *calls* the existing Slammer action layer;
  it does not reimplement fixing.

---

## 2. User interface

Two flags on the generic driver (and any Config-based driver), each taking a
**JSON filename**:

| Flag | Effect |
|---|---|
| `--detect-W-oscillations PATH` | Activates the extension in **detect+report** mode; controls in the JSON at `PATH`. |
| `--interrupt-W-oscillations PATH` | (PR2) Activates **interruption**; controls in the JSON at `PATH`. Implies detection — if `--detect-W-oscillations` is absent, a default detector config is used (or the interrupt JSON may carry a `detect` block, see §9). |

Presence of a flag activates the extension (mirroring how
`--slamming-directives-file` activates the Slammer). Neither flag ⇒ extension
never constructed ⇒ identical behavior to today.

### 2.1 Detection JSON schema (PR1)

```jsonc
{
  "output_csv": "w_oscillations.csv",  // per-nonant aggregate; rank-0 writes (required)
  "per_scenario_csv": null, // optional per-(scenario,nonant) detail file (§7.1); null = off
  "warmup_iters": 5,        // don't evaluate until this many W samples exist
  "check_every": 1,         // evaluate detectors every N iterations after warmup
  "report_mode": "on_detect", // "on_detect" | "final" | "every_check"
  "min_scenarios_to_report": 1, // a nonant is reported once >= this many
                                // scenarios flag it (int; or use "frac" below)
  "min_frac_to_report": null,   // alternative: fraction of scenarios (0..1)
  "methods": {              // which detectors run; keys are method names
    "zero_crossings": {     // method A (§5.1); params override defaults
      "tol": 1e-6,
      "window": null,                 // null = whole trajectory so far
      "thresh_w_crossings": 2,
      "thresh_diff_crossings": 3,
      "thresh_diffs_ratio": 0.2
    },
    "w_hash_recurrence": {  // method B (§5.2)
      "window": 20,         // look back this many iterations for a repeat
      "quantum": 1e-6,      // W values quantized to this before hashing
      "min_period": 2       // ignore period-1 (constant W = convergence)
    }
  }
}
```

Only methods **present** in `"methods"` are run; a method's omitted params
fall back to documented defaults. `report_mode` defaults to `on_detect`
(write a row the first time, and each subsequent `check_every`, a nonant
crosses into the flagged state). `final` writes one report at
`post_everything` (sorgw's behavior). The exact set of knobs per method lives
with that method (§5).

### 2.2 Interruption JSON schema (PR2)

```jsonc
{
  "action": "rho_reduction",      // "rho_reduction" | "slam" | "both"
                                  // docs encourage one or the other; "both"
                                  // simply applies both, no coordination (§9)
  "trigger": {
    "min_scenarios_flagged": 1,   // act when >= this many scenarios flag a nonant
    "start_iter": 5,
    "iters_between_actions": 3
  },
  "rho_reduction": { "factor": 0.5, "min_rho": 1e-3 },
  "slam": { "directives_file": "slam_directives.csv" }  // feeds the Slammer
}
```

---

## 3. Where it lives

- New module `mpisppy/extensions/w_oscillation.py`, class
  `WOscillationMonitor(Extension)`. One extension does both jobs: a
  **detection engine** (always on when either flag is set) plus two consumers
  of its results — a **reporter** (PR1) and an **interrupter** (PR2). This
  keeps a single W-capture path and a single source of truth for "what is
  oscillating."
- Detectors live as small, independently testable callables/classes in the
  same module (or a `w_oscillation/` subpackage if it grows): each takes one
  W trajectory and returns a per-trajectory verdict + stats (§5).
- Config plumbing in `config.py` / `cfg_vanilla.py` / `mpisppy/generic/`
  (§10), following the Slammer wiring exactly.

---

## 4. Hook placement and the per-iteration flow

The PH per-iteration order around the extension hooks is: *xbar computed →
W updated →* `miditer()` *→* `solve_loop()` *→* `enditer()`. So at
`miditer(k)` the freshest **post-update** `W^k` is in place, and any change we
make (rho, slam) takes effect for that iteration's `solve_loop`.

| Hook | Action |
|---|---|
| `pre_iter0` | Parse JSON; build the detector set; build the nonant→name map; allocate the bounded W history buffer. |
| `miditer` | **Capture** `W^k` into the buffer (reusing wtracker's grab idiom). After `warmup_iters`, every `check_every`: **evaluate** detectors locally, **reduce** across scenarios (§6), and — PR1 — **report** flagged nonants; PR2 — **interrupt** per the trigger. |
| `post_everything` | If `report_mode == "final"`, run one last evaluate+reduce+report. Close the CSV. Rank-0 summary line. |

Capturing in `miditer` (post-update W) is the natural sampling point for the
*updated* dual trajectory; it differs from `Wtracker_extension`, which grabs
in `enditer` (same numeric W that iteration, but framed as pre-update). We
reuse the one-line grab (`[w._value for w in s._mpisppy_model.W.values()]`)
and keep our **own bounded ring buffer** of length
`max(window over active methods)` rather than wtracker's keep-everything
dict — bounding memory for long runs (wtracker explicitly warns it is
"intended for diagnostics, not everyday use"). The full-history path remains
available via wtracker itself (cross-referenced in the docs).

---

## 5. Detection methods

A **detector** consumes one W trajectory `w[0..m]` for a single (scenario,
nonant) and returns `(flagged: bool, stats: dict)`. The engine runs every
selected detector on every local (scenario, nonant) trajectory. Per-method
defaults are the sorgw values where applicable.

### 5.1 Method A — `zero_crossings` (from sorgw)

Counts sign changes of `W` and of the consecutive differences `ΔW`, plus a
damping ratio, over the trajectory (optionally only the last `window`
samples):

- `WZeroCrossings` — number of sign changes of `w[i]` (ignoring `|w| < tol`).
- `DiffZeroCrossings` — number of sign changes of `Δw[i] = w[i+1]-w[i]`.
- `diffs_ratio = mean(|Δw|, back half) / mean(|Δw|, front half)` — ratio ≥ 1
  means the swings are **not** damping.

Flagged iff any of:
`WZeroCrossings >= thresh_w_crossings (2)` **or**
`DiffZeroCrossings >= thresh_diff_crossings (3)` **or**
`diffs_ratio >= thresh_diffs_ratio (0.2)`.

This is a faithful port of sorgw's `Of_Interest` test; the per-trajectory
counting logic is pure Python and unit-testable with no MPI (§11).

### 5.2 Method B — `w_hash_recurrence` (Watson–Woodruff §2.4)

Watson–Woodruff §2.4 ("Detecting Cyclic Behavior") detects a cycle by
**repeated occurrence of the per-scenario weight vector** for a variable:

> "To detect cycles, we chose to focus on repeated occurrences of `w_s(i)`
> vectors, implemented using a simple hashing scheme to minimize impact on
> run-time. Once a cycle is detected for any decision variable `x(i)`, the
> value of `x(i)` is immediately fixed to `max_{s∈S} x_s(i)`…"

So for each nonant `i`, the object hashed is the **vector across scenarios**
`w(i) = (w_s(i))_{s∈S}` at the current iteration; a **recurrence** of that
vector (same hash seen before, within a look-back window) signals a cycle.
This keys directly on **W** (not x/x-bar), making it the natural complement to
method A (vector recurrence vs. per-trajectory sign counting).

Note the paper's *native remediation* is "fix `x(i)` to `max_s x_s(i)`" — i.e.
**slam to max**, which is exactly the Slammer `max` direction (so PR2's
method-B-driven action is a one-line wiring, §9).

**Distributed implementation (our addition — the paper was serial).** The
hashed vector spans scenarios that live on different ranks, so we compute a
**distribution-independent signature** with a sum reduction (fitting the §6
framework). Each rank forms, per nonant `i`, a partial signature over its local
scenarios at node `ndn`:

```
term_s(i) = hash64( scenario_index_s, round(w_s(i) / quantum) )   # 64-bit
partial_sig(i) = Σ_{local s} term_s(i)   (mod 2^64)
```

then `comms[ndn].Allreduce(partial_sig, global_sig, op=MPI.SUM)`. Mixing
`scenario_index` into each term preserves per-scenario identity, and the sum is
commutative, so `global_sig(i)` is **independent of how scenarios are mapped to
ranks** — two iterations whose quantized `w(i)` vectors match (per scenario)
get the same signature. (64-bit collisions are vanishingly rare; the paper
likewise accepted a "simple hashing scheme" and "few variables are fixed…
minimal impact.")

Each rank keeps a per-nonant ring buffer of the last `window` global
signatures. **Flagged** iff the current signature equals one from `min_period`
to `window` iterations back — a genuine cycle of period ≥ `min_period`.
`min_period ≥ 2` is required so a **constant** W (convergence, period-1
"recurrence") is *not* mistaken for a cycle. Reported stats: detected period
and the signature.

#### 5.2.1 Background and keywords (to read more)

Two well-studied ideas underpin method B; these are the search terms / references
for anyone extending it:

- **Order-/distribution-independent set hashing** — the trick that makes the
  signature `Σ hash(scenario_index, value)` independent of how scenarios are
  spread across ranks. Keywords: *incremental hashing*, *multiset hashing*,
  *homomorphic / set hashing*, **AdHash / XHash / MuHash**. References:
  Bellare & Micciancio, "A New Paradigm for Collision-Free Hashing:
  Incrementality at Reduced Cost" (EUROCRYPT 1997); Clarke, Devadas, van Dijk,
  Gassend & Suh, "Incremental Multiset Hash Functions and Their Application to
  Memory Integrity Checking" (ASIACRYPT 2003). The additive (sum-mod-2^64)
  variant is what we use; XOR is the alternative if a different collision
  profile is wanted.
- **Recurrence / cycle detection by state hashing** — storing a digest of each
  visited state and flagging a repeat. This is exactly Watson–Woodruff §2.4's
  "simple hashing scheme," and the same idea used in explicit-state model
  checking: keywords *hash compaction*, *bitstate hashing*; references Wolper &
  Leroy, "Reliable Hashing without Collision Detection" (CAV 1993); Holzmann,
  "An Analysis of Bitstate Hashing" (1998). If O(1) memory is ever preferred
  over the ring buffer, the classic sequence-cycle algorithms apply: **Floyd's**
  (tortoise-and-hare) and **Brent's** cycle detection.

### 5.3 Adding methods

A new detector is a new entry in the method registry keyed by its JSON name,
with a `(trajectory, params) -> (flagged, stats)` signature and a defaults
dict. No CLI or wiring change needed.

---

## 6. Data flow and MPI reductions

Detection is per (scenario, nonant); reporting/acting is per nonant, so we
**reduce across scenarios** (the "primarily sum reductions" the spec calls
for). The nonant index `i` aligns across scenarios at a node, so for each
node `ndn`:

1. **Local pass.** For each active method, over local scenarios at `ndn`,
   accumulate per-nonant:
   - `n_flagged[i]` += 1 per local scenario whose trajectory is flagged
     (**SUM**),
   - `max_w_crossings[i]`, `max_diff_crossings[i]`, `max_diffs_ratio[i]`
     (**MAX**), and any other reported stat.
2. **Reduce.** `opt.comms[ndn].Allreduce(local, global, op=MPI.SUM)` for the
   counts and `op=MPI.MAX` for the maxes — the exact `norm_rho_updater` /
   `dyn_rho_base` idiom, on the per-node communicators x-bar already uses.
   Results are identical on every rank.
3. **Decide.** A nonant is *reported* (PR1) / *eligible to act on* (PR2) iff
   `n_flagged[i] >= min_scenarios_to_report` (or the fraction variant). Total
   scenarios per node is known, so the fraction is exact.
4. **Write.** Cylinder **rank 0** appends rows to the CSV; all other ranks
   skip the write. Because step 2 makes the inputs rank-identical, rank 0's
   rows do not depend on how scenarios were distributed across ranks (a
   property the MPI test asserts, §11).

Method B folds into the **same SUM pass**: its per-nonant signature
(§5.2) is itself an `op=MPI.SUM` Allreduce over `comms[ndn]`, computed
alongside the method-A counts, after which recurrence is decided locally from
each rank's (now rank-identical) ring buffer. So one reduction round per check
covers every active method — "primarily sum reductions," as specified.

For the common **two-stage** case there is a single node `ROOT`; multistage
falls out of iterating `s._mpisppy_node_list` and using `comms[ndn]` per node.

---

## 7. CSV output

Header + one row per (reported nonant, method, detection event):

| Column | Meaning |
|---|---|
| `iteration` | PH iteration at which the check fired |
| `node` | node name (`ROOT` for two-stage) |
| `variable` | **full nonant name** (e.g. `DevotedAcreage[SUGAR_BEETS]`) |
| `method` | detector name (`zero_crossings`, `w_hash_recurrence`, …) |
| `num_scenarios_total` | scenarios at this node |
| `num_scenarios_flagged` | how many flagged this nonant (the SUM reduction; for method B, scenarios participating in the recurring vector) |
| `max_w_crossings` | method A: max over scenarios |
| `max_diff_crossings` | method A: " |
| `max_diffs_ratio` | method A: " |
| `cycle_period` | method B: detected period |

Method-specific stat columns are the union over active methods (missing ⇒
blank). Filename and (optionally) which columns to emit come from the JSON.

### 7.1 Per-scenario detail file (optional)

When `per_scenario_csv` is set, the extension also emits a **per-(scenario,
nonant) breakdown** — useful for seeing *which* scenarios are driving an
oscillation, not just the aggregate. The aggregate report (§6/§7) uses a SUM/MAX
**reduction**, which discards per-scenario identity; the detail file instead
**gathers** the (small) set of flagged per-scenario rows to rank 0
(`comms[ndn].gather` of only the rows a rank flagged), and rank 0 writes them as
one file. Columns: `iteration, node, scenario, variable, method`, plus the
per-trajectory stats (`w_crossings, diff_crossings, diffs_ratio` for method A;
the scenario's quantized `w` value and whether its node's vector recurred for
method B).

Two notes: (1) the detail is most meaningful for **per-trajectory** methods
(A), since method B's verdict is a property of the whole cross-scenario vector,
not one scenario; for B we report each participating scenario's `w` value at the
recurrence. (2) Only flagged rows are gathered, so the volume is bounded by what
is actually oscillating — but on a badly thrashing problem this can still be
large, so the file is **off by default**.

---

## 8. Reuse of wtracker

- **Capture:** identical one-liner
  (`[w._value for w in s._mpisppy_model.W.values()]`) over
  `opt.local_scenarios`; the nonant→name alignment is wtracker's
  `varnames` construction. We factor this into a tiny shared helper if it
  reads cleanly, else replicate the one line (it is trivial and already
  duplicated in `wxbarutils`).
- **Docs:** the detection PR's user doc cross-references the wtracker docs for
  the full W trajectory + moving-stat (`stdev` / `CV`) reports, positioning
  this extension as the focused oscillation layer.
- We do **not** reuse wtracker's keep-everything buffer (memory); we keep a
  bounded ring buffer (§4).

---

## 9. Interruption (PR2)

When a nonant is flagged (and the trigger in §2.2 fires), act in `miditer`
before the solve:

- **`rho_reduction`** — multiply that nonant's `rho` by `factor (<1)` in every
  local scenario, floored at `min_rho > 0` (consistent with the merged rho>0
  enforcement, #767). Reducing rho relaxes the proximal pull that is driving
  the overshoot/cycle. The motivation is in Watson–Woodruff §2.1: the update is
  `w_s += ρ(x_s − x̄)`, so a too-large ρ is exactly what lets `w` "shoot past"
  its optimum and thrash; their SEP rho is designed to approach `w*` "from
  below" to avoid this. Reducing ρ on a *detected-cycling* nonant is the
  dynamic analogue. Which nonants to touch comes from the rank-identical
  reduced set (§6), so the change is coherent with no extra communication; the
  per-scenario rho write is local (push to persistent solver via
  `update_var`).
- **`slam`** — invoke the **Slammer action layer** (`slammer.py` §7) on the
  flagged nonant: pick by the directives file's priority/direction and `fix()`
  it across all scenarios. This is exactly the consumer the slamming design
  anticipated; PR2 wires the detector's "this nonant is cycling" signal into
  Slammer's `slam(ndn_i, direction)` rather than Slammer's built-in
  iteration-count trigger. **Method B's paper-native remediation is precisely
  this:** Watson–Woodruff §2.4 fixes the cycling `x(i)` to `max_s x_s(i)`,
  which is the Slammer `max` direction — so a directives file of `*,1,max,…`
  driven by the detector reproduces their §2.4 behavior exactly.
- **`both`** — the user docs **encourage choosing one or the other**, but if
  `action == "both"` we **simply apply both** with no coordination/escalation
  logic: for each flagged nonant this event, run the `rho_reduction` action
  *and* the `slam` action. (No special-casing of the interaction: a nonant the
  slam directives `fix()` becomes `is_fixed()`, so the subsequent rho change on
  it is inert — harmless; and a nonant slamming declines, e.g. not in the
  directives file, still gets its rho reduced. The two mechanisms act on
  whatever each is configured to act on.) Keep-it-simple was the explicit
  decision (§13).

PR2 must demonstrate interruption **helps** (§11): on a model tuned to cycle,
the interrupted run reaches a better gap / converges in fewer iterations than
the plain run in the same budget.

---

## 10. Config / CLI plumbing and backward compatibility

Following the Slammer wiring precisely:

- **`config.py`:** a `w_oscillation_args()` method adding
  `--detect-W-oscillations` (PR1) and `--interrupt-W-oscillations` (PR2), both
  `domain=str, default=None` (filename-valued, like
  `--slamming-directives-file`). Registered in `mpisppy/generic/parsing.py`.
- **`cfg_vanilla.py`:** `add_w_oscillation(hub_dict, cfg)` appends
  `WOscillationMonitor` via `extension_adder` (the `MultiExtension` path) and
  attaches `hub_dict["opt_kwargs"]["options"]["w_oscillation_options"]` =
  `{detect_json, interrupt_json, verbose}`.
- **`mpisppy/generic/extensions.py`:** activate iff
  `cfg.detect_W_oscillations is not None or cfg.interrupt_W_oscillations is not None`.
- **Validation:** JSON parsed and validated at `pre_iter0`; unknown method
  names, bad params, a missing `output_csv`, and (PR2) interrupt-without-a-
  valid-action are hard errors. Parse/validate failures raise on all ranks
  (inputs are rank-identical).
- **Compatibility contract:** no flag ⇒ extension never built ⇒ byte-identical
  behavior. PR1's extension is pure observation (no rho/fix writes) ⇒ even
  *with* `--detect-W-oscillations`, the optimization trajectory is unchanged.

---

## 11. Testing

- **Unit (no MPI):** each detector on synthetic trajectories — a known
  oscillating series (e.g. `[+a,-a,+a,-a,…]`) flags with the expected
  crossing counts; a monotone/damping series does not; threshold boundaries
  (exactly `thresh` crossings) behave as specified; `window` truncation works.
- **Reduction/aggregation (no MPI):** drive the per-node SUM/MAX aggregation
  with a fake multi-scenario fixture; assert `num_scenarios_flagged` and the
  maxes.
- **End-to-end (serial):** **farmer** with a deliberately **high rho on one
  variable** (e.g. `DevotedAcreage[SUGAR_BEETS]` via a `_rho_setter`) to
  induce W oscillation; run `generic_cylinders --detect-W-oscillations`;
  assert the CSV is written, has the header, and flags that variable. Mirrors
  the existing `TestGenericCylindersWtracker` end-to-end test. A **sizes**
  (MIP) variant is a natural second case, since integer cycling is the prime
  motivation.
- **MPI (`mpiexec -np 2`/`3`):** the same induced-oscillation run; assert the
  rank-0 CSV is **independent of scenario→rank distribution** (the reduction
  is correct).
- **PR2 — improvement:** on the cycle-tuned model, assert the interrupted run
  improves on the plain run (fewer iters to a target gap, or better gap at the
  iteration cap).
- **Coverage harness:** add `mpisppy/tests/test_w_oscillation.py` to
  `run_coverage.bash` **and** `.github/workflows/test_pr_and_main.yml` in the
  **same commit** (else codecov/patch reports 0%).

---

## 12. Phased rollout

**PR1 — detection + reporting.**
`w_oscillation.py` (engine + reporter + detector A + detector B), the
`--detect-W-oscillations` flag and JSON, config/cfg_vanilla/generic wiring,
CSV writer (rank 0), unit + reduction + end-to-end + MPI tests, and a user-doc
page cross-referencing wtracker. Pure observation ⇒ green on its own,
backward compatible.

**PR2 — interruption.**
`--interrupt-W-oscillations` flag and JSON, the rho-reduction action, the
slam action (calling the Slammer action layer), the trigger layer, the
`both` = apply-both rule (§9), and an improvement test. Builds on PR1's engine.

Each PR is review-sized and green independently (per the project's
phased-PR-for-redesigns practice).

---

## 13. Decisions

Settled in review (DLW, 2026-06-27); recorded for provenance.

1. **Method B = Watson–Woodruff §2.4** — hash the per-scenario `w(i)` vector,
   flag on recurrence (§5.2). Distributed via a **distribution-independent
   sum-reduced signature** (identity-mixed per-scenario 64-bit hashes).
   *Confirmed* ("I like the idea"). Background/keywords for the technique are in
   §5.2.1 (incremental/multiset hashing; state-hash cycle detection).
2. **`both` interruption (Q2)** — docs **encourage one or the other**; if both
   are configured, **just apply both**, no coordination/escalation logic (§9).
3. **Per-scenario reporting (Q3)** — **yes**, an optional `per_scenario_csv`
   detail file (gather-based, off by default) is in scope for PR1 (§7.1).
4. **Capture hook (Q4)** — **`miditer`** (post-update W), §4. (Differs from
   `Wtracker_extension`, which grabs in `enditer`.)

### Still open

- **`quantum` / `min_period` / `window` defaults** for method B — proposed
  `1e-6 / 2 / 20`; tune against the induced-oscillation test once it exists.
