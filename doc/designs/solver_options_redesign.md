# Solver-options redesign

Status: design complete; phased implementation in progress. This document
covers the current state (§1), goals and non-goals (§2–3), resolved open
questions (§4), the proposed design (§5), and the migration / compatibility
plan (§6). Phase 1 (dormant layered representation) lands with this doc;
phases 2–8 are scheduled per §6.4.

Backward-compatibility constraint: every CLI flag and CLI value-syntax that
works today must continue to work after the redesign. Programmatic-API
changes are allowed (with deprecation), but command-line invocations in
existing scripts and `examples/` must keep producing the same solver-side
behavior.

---

## 0. Scope

"Solver options" here means the key/value settings that mpi-sppy passes
through to the underlying Pyomo solver plugin (e.g. `mipgap`, `threads`,
`time_limit`, `mip_rel_gap`) — *not* hub/spoke convergence parameters,
*not* PH algorithmic options (rho, smoothing, etc.), and *not* the choice
of solver (`solver_name`) except where it directly couples to options.

In scope:

- How the user supplies options on the command line.
- How those options are parsed, merged, and routed to per-cylinder /
  per-iteration solve calls.
- How EF, PH iter0, PH iterk, and per-spoke overrides interact.

Also in scope, with caveat:

- Auto-translating option names across solvers (e.g. `mipgap` for
  CPLEX/Gurobi vs `mip_rel_gap` for HiGHS) — but **only for `mipgap`
  and `threads`**. The current code does not translate at all. See
  §2 goal #4 and §3 non-goal on other keys.

Out of scope:

- Auto-translation for any option key other than `mipgap` and
  `threads`.
- Solver-pool / solver-per-rank dispatch.

### 0.1 Solver-log options are not a `--solver-options` use case

Passing solver-log flags (e.g. CPLEX `logfile`, Gurobi `LogFile`,
HiGHS `log_file`) through `--solver-options` is almost always wrong in
mpi-sppy. Each subproblem solves many times per iteration across many
ranks; a single shared log path either overwrites itself, interleaves
output from concurrent solves, or pins every rank to the same file —
none of which produces a useful log.

The mpi-sppy-blessed mechanism is `--solver-log-dir <directory>`
(`popular_args`, config.py:213), which writes one log file per
subproblem solve into the named directory, with names that
disambiguate scenario and rank. To limit log volume to just hub-side
solves (skipping spoke-side subproblem solves entirely), combine it
with `--hub-only-solver-logs` (config.py:219). Both flags are exposed
in `generic_cylinders.py` via `popular_args()`.

Nothing in this redesign changes that. The layered representation,
options-file, and translation table cover `mipgap`, `threads`, and
similar solver-tuning knobs — not solver logging. Examples below use
`presolve=2` (or similar non-log knobs) for illustrating parser and
merge mechanics, deliberately avoiding `logfile`.

---

## 1. Current state ("as-is")

### 1.1 CLI flags

All solver-option-related CLI flags are registered in
`mpisppy/utils/config.py`. The two reusable registration helpers are:

- `Config.add_solver_specs(prefix="")` — config.py:168. Adds
  `{prefix_}solver_name` and `{prefix_}solver_options`.
- `Config.add_mipgap_specs(prefix="")` — config.py:182. Adds
  `{prefix_}iter0_mipgap` and `{prefix_}iterk_mipgap`.

The flags actually exposed today, by group:

**Global (added by `popular_args()` at config.py:233):**

| Flag                    | Type   | Default | Notes                                        |
|-------------------------|--------|---------|----------------------------------------------|
| `--solver-name`         | str    | None    | config.py:172 (via `add_solver_specs("")`)   |
| `--solver-options`      | str    | None    | space-delimited `k=v` string, config.py:177  |
| `--max-solver-threads`  | int    | None    | config.py:265                                 |
| `--solver-log-dir`      | str    | None    | config.py:210; one log file per solve        |

**PH iteration mipgaps** (added per-call by code that uses `add_mipgap_specs("")`):

| Flag             | Type  | Default | Notes           |
|------------------|-------|---------|-----------------|
| `--iter0-mipgap` | float | None    | config.py:186   |
| `--iterk-mipgap` | float | None    | config.py:191   |

**EF (`EF_base()` at config.py:422):**

| Flag                 | Type  | Default | Notes                                  |
|----------------------|-------|---------|----------------------------------------|
| `--EF-solver-name`   | str   | None    | via `add_solver_specs("EF")`           |
| `--EF-solver-options`| str   | None    | via `add_solver_specs("EF")`           |
| `--EF-mipgap`        | float | None    | config.py:425                           |

**Per-spoke / per-cylinder overrides** (each call site invokes
`add_solver_specs(prefix)` and sometimes `add_mipgap_specs(prefix)`):

- `--lagrangian-solver-name`, `--lagrangian-solver-options`,
  `--lagrangian-iter0-mipgap`, `--lagrangian-iterk-mipgap`
- `--reduced-costs-solver-name`, `--reduced-costs-solver-options`
- `--subgradient-solver-name`, `--subgradient-solver-options`,
  `--subgradient-iter0-mipgap`, `--subgradient-iterk-mipgap`
- `--relaxed-ph-solver-name`, `--relaxed-ph-solver-options`
- `--ph-dual-solver-name`, `--ph-dual-solver-options`
- `--lagranger-solver-name`, `--lagranger-solver-options` (lagranger
  has its own ad-hoc iter0/iterk wiring; see §1.5)
- `--obbt-solver-options` — config.py:325, OBBT presolve only
- `--pickle-solver-name`, `--pickle-solver-options` —
  config.py:1256/1263, used for the iter0 solve done at pickle time

**Mipgap schedule and automatic gapper (`Config.gapper_args()` at config.py:610):**

These flags are registered separately from `add_solver_specs` /
`add_mipgap_specs`, and the mipgap value they produce is written to
the live options dict by the `Gapper` extension at runtime, not
through `shared_options`/`apply_solver_specs`. See §1.5 item 9.

| Flag                        | Type   | Default | Notes                                      |
|-----------------------------|--------|---------|--------------------------------------------|
| `--mipgaps-json`            | str    | None    | config.py:617. JSON file `{"<iter>": <gap>}` (global only — no per-spoke variant). |
| `--starting-mipgap`         | float  | None    | config.py:622. Enables automatic-gapper mode. |
| `--mipgap-ratio`            | float  | 0.1     | config.py:627. Ratio of overall opt gap to subproblem mipgap. |

Per-spoke automatic-gapper variants exist (e.g. `--lagrangian-starting-mipgap`,
`--lagrangian-mipgap-ratio`) via `gapper_args(name="lagrangian")`. Per-spoke
*schedule* variants do not — the JSON schedule flag is registered only
when the `name` argument is `None` (config.py:616).

**Misc adjacent:**

- `--stage2-ef-solver-name` — config.py:407, multistage xhat stage-2 EF.

There is no global `--solver-options-file` flag and no JSON/YAML loader
for arbitrary solver options. The JSON support that does exist
(`--mipgaps-json`) is mipgap-only.

### 1.2 String → dict parsing

`mpisppy/utils/sputils.py:648` defines `option_string_to_dict(ostr)`:

- Splits on whitespace, then each piece on `=`.
- Auto-coerces values: `int` first, then `float`, then string.
- Bare key (no `=`) becomes `key: None`.
- Already-a-dict input is returned unchanged (Jan 27, 2026 fast path —
  sputils.py:651,670).
- Empty / `None` returns `{}`.
- Malformed (more than one `=` in a piece) raises `RuntimeError`.

The inverse is `option_dict_to_string` (sputils.py:690).

Implication: the CLI value `"mipgap=0.01 threads=4 presolve=2"` works,
but **values cannot contain spaces or `=`**. Quoting in the shell is on
the user.

### 1.3 Plumbing: CLI flag → solver call

The path from a flag value to the underlying `SolverFactory(...).solve()`
call has six steps:

```
CLI                                         (user-typed string)
  │
  ▼
cfg.solver_options                          (str on the Config object)
  │
  ▼  cfg_vanilla.shared_options(cfg)        (cfg_vanilla.py:53)
shoptions["iter0_solver_options"]: dict
shoptions["iterk_solver_options"]: dict     (deepcopy of the same dict)
  │
  ▼  cfg_vanilla.{ph_hub, lagrangian_spoke, ...}
hub_dict / spoke_dict ["opt_kwargs"]["options"][...]
  │
  ▼  cfg_vanilla.apply_solver_specs(name, spoke, cfg)   (cfg_vanilla.py:113)
spoke options overridden if --{name}-solver-options was given
  │
  ▼  PHBase.__init__(options=...)            (phbase.py:284-286)
self.iter0_solver_options
self.iterk_solver_options
self.current_solver_options := iter0  (then reassigned to iterk at phbase.py:1052)
  │
  ▼  PHBase.solve_loop(solver_options=self.current_solver_options, ...)
                                              (phbase.py:979 iter0;
                                               phbase.py:1140 iterk)
  │
  ▼  SPOpt.solve_one(solver_options=...)      (spopt.py:119)
for k, v in solver_options.items():
    s._solver_plugin.options[k] = v          (spopt.py:183-187)
results = s._solver_plugin.solve(s, ...)     (spopt.py:229)
```

Key dict keys, by where they live:

- `cfg.solver_options` — raw string from CLI.
- `shoptions["iter0_solver_options"]`, `shoptions["iterk_solver_options"]`
  — parsed dicts; live on the hub/spoke `options` dict for the rest of
  the run.
- `self.current_solver_options` on `PHBase` — the dict actually handed to
  `solve_loop` for the next batch of subproblem solves.

### 1.4 Merge / override rules in `shared_options`

`cfg_vanilla.shared_options()` (cfg_vanilla.py:53–111) builds the dicts
in this order. Every step *mutates* the iter0/iterk dicts produced by the
previous one, so later steps win:

1. Initialize `iter0_solver_options = {}` and `iterk_solver_options = {}`
   (cfg_vanilla.py:63-64).
2. If `cfg.solver_options` is set, parse it once and copy it into
   *both* iter0 and iterk (cfg_vanilla.py:78-81). iterk gets a deepcopy.
3. If `--max-solver-threads`, write
   `[iter0|iterk]_solver_options["threads"]` (cfg_vanilla.py:83-85).
4. If `--iter0-mipgap`, write `iter0_solver_options["mipgap"]`
   (cfg_vanilla.py:86-87).
5. If `--iterk-mipgap`, write `iterk_solver_options["mipgap"]`
   (cfg_vanilla.py:88-89).

`apply_solver_specs(name, spoke, cfg)` (cfg_vanilla.py:113–129) then runs
*per spoke that opted in*, with the same shape but reading
`{name}_solver_options`, `{name}_iter0_mipgap`, etc. Important quirk:
after potentially overwriting iter0/iterk dicts wholesale at line 119-120,
it **re-applies** `--max-solver-threads` at lines 127-129 to keep the
global thread cap honored.

### 1.5 Asymmetries and pitfalls already in the as-is

Things a user can hit today that a redesign should at minimum not regress
on, and ideally fix:

1. **`--solver-options` is one string, applied to both iter0 and iterk.**
   You cannot pass option *X* in iter0 only via the CLI without also
   setting a separate `--iter0-mipgap`-style flag. The only per-iter
   knobs at CLI level are `iter0_mipgap` / `iterk_mipgap`.
2. **`mipgap` is privileged.** It has dedicated CLI flags and is written
   into the dict under the literal key `"mipgap"`. `threads` is similarly
   privileged via `--max-solver-threads`. Other Pyomo-solver options have
   no first-class CLI representation and must go through the
   space-delimited `--solver-options` string.
3. **No solver-name awareness.** `mipgap` works for CPLEX/Gurobi/Xpress
   but is not the right key for HiGHS (`mip_rel_gap`). The framework
   forwards keys verbatim. Examples in the repo (e.g.
   `examples/run_uc.py:96` uses `mip_rel_gap=0.5` with `appsi_highs`)
   work because the user spelled the key correctly.
4. **Per-spoke override is all-or-nothing per dict.** When
   `--lagrangian-solver-options` is set, `apply_solver_specs` *replaces*
   the spoke's iter0/iterk dict with the parsed lagrangian string
   (cfg_vanilla.py:119-120) — there is no merge with the global
   `--solver-options`. Only `max_solver_threads` is restored afterward.
5. **`option_string_to_dict` numeric coercion is unconditional.** A
   solver option whose value is a string that looks like an int or float
   will be coerced. Values containing `=` or spaces are unrepresentable.
6. **Pickle-time iter0 has its own pair** (`--pickle-solver-name`,
   `--pickle-solver-options`) that bypass the iter0/iterk machinery
   above. Same for OBBT (`--obbt-solver-options`) and stage-2 EF
   (`--stage2-ef-solver-name`, no options companion).
7. **Lagranger** sets its iter0/iterk dicts directly in
   `lagranger_spoke()` rather than going through `apply_solver_specs`,
   so its merge semantics differ subtly from sibling spokes.
8. **Persistent solvers** (`gurobi_persistent`, `cplex_persistent`) take
   the same options dict and the same `s._solver_plugin.options[k]=v`
   loop (spopt.py:183-187). The only persistent-specific branches are
   for `set_objective` (spopt.py:171-177), `save_results=False`
   (spopt.py:193-194), and a Gurobi `LogFile` workaround when
   `--solver-log-dir` is used (spopt.py:215-217). No persistent-specific
   option translation.
9. **The mipgap schedule lives in a parallel mechanism.** The
   `--mipgaps-json` schedule and the automatic-gapper mode
   (`--starting-mipgap` / `--mipgap-ratio`) are not handled by
   `shared_options` / `apply_solver_specs` at all. Instead, the
   `Gapper` extension (`mpisppy/extensions/mipgapper.py`) reads
   `cfg.mipgaps_json` at config-build time (cfg_vanilla.py:393-411)
   and at `pre_iter0` / `miditer` mutates the live options dict
   directly: `self.ph.current_solver_options["mipgap"] = mipgap`
   (mipgapper.py:46-52). Consequences: (a) the schedule wins against
   anything `shared_options` set, because Gapper runs later in the
   iteration; (b) per-spoke schedule files don't exist (only the
   global flag is registered); (c) automatic-gapper mode is genuinely
   adaptive — it reads the hub/spoke bound gap each iteration — so it
   cannot be expressed as a static layer.

### 1.6 Representative current usage

CLI (from `examples/run_uc.py:96`):

```bash
mpiexec -np 3 python uc/cs_uc.py \
    --max-iterations=1 --default-rho=1 --num-scens=3 \
    --solver-options="mip_rel_gap=0.5 threads=1" \
    --linearize-proximal-terms --solver-name=appsi_highs
```

Programmatic, bypassing `cfg_vanilla` entirely (older style; e.g.
`examples/sslp/sslp.py:221`):

```python
options["iter0_solver_options"] = {"mipgap": 0.01}
options["iterk_solver_options"] = {"mipgap": 0.02, "threads": 4}
```

EF, plumbed through `solver_spec` rather than the iter0/iterk path
(`examples/farmer/CI/farmer_ef.py:74`):

```python
solver_options = solver_spec.solver_specification(cfg, "EF")
if solver_options is not None:
    for option_key, option_value in solver_options.items():
        s._solver_plugin.options[option_key] = option_value
```

### 1.7 Summary of files that participate

| File                             | Role                                              |
|----------------------------------|---------------------------------------------------|
| `mpisppy/utils/config.py`        | CLI flag registration (`add_solver_specs`, `add_mipgap_specs`, `EF_base`, individual spoke arg-adders) |
| `mpisppy/utils/sputils.py`       | `option_string_to_dict` / `option_dict_to_string` |
| `mpisppy/utils/cfg_vanilla.py`   | `shared_options`, `apply_solver_specs`, hub/spoke factories that copy options dicts into hub/spoke dicts |
| `mpisppy/phbase.py`              | Stores `iter0_solver_options`, `iterk_solver_options`, `current_solver_options`; flips between them around iter0 |
| `mpisppy/spopt.py`               | `solve_one` — applies the dict to `s._solver_plugin.options` and calls `solve` |
| `mpisppy/utils/solver_spec.py`   | Alternative entry used by EF and confidence-interval code |
| `mpisppy/opt/ef.py`              | EF-specific solve options handling                |
| `mpisppy/cylinders/*spoke*.py`   | Spoke-specific consumption (lagranger, reduced-costs persistent check) |

---

## 2. Goals for the redesign

Decisions captured from DLW review of the as-is, 2026-05-07:

1. **Cleaner per-iteration overrides.** Today the only per-iteration
   knobs at CLI level are `--iter0-mipgap` and `--iterk-mipgap`.
   Generalize to an "after-iteration-N" predicate (a set of options
   that applies starting at a user-specified iteration N), so users
   can express e.g. "tighten the gap after iteration 5" without
   recompiling.
2. **Cleaner per-spoke overrides via merge-with-global.** Today
   `apply_solver_specs` replaces a spoke's iter0/iterk dict wholesale
   when `--{name}-solver-options` is given. Change to merge: spoke
   options layer on top of the global dict. A user who wants the
   current "replace" behavior can still get it by re-spelling every
   key, so merge is strictly more general as the default. Exact merge
   depth is open — see §4.
3. **Options-from-file companion to `--solver-options`.** Add a flag
   that loads solver options from a file. Motivation: the existing
   space-delimited `key=value` CLI string cannot represent values
   containing spaces or `=`. Format and merge order with the inline
   string are open — see §4.
4. **Solver-name-aware translation for `mipgap` and `threads` only.**
   These two options are by far the most-used and have different keys
   across solvers (CPLEX/Gurobi `mipgap` vs HiGHS `mip_rel_gap`;
   `threads` vs `Threads`). Translate these two automatically based
   on `solver_name`. All other keys remain pass-through, the user's
   responsibility.

CLI-compat constraint (repeat): every flag in §1.1 keeps working
unchanged for existing scripts.

## 3. Non-goals

- Changing `solver_name` semantics.
- Changing the Pyomo solver-plugin interface.
- Removing any existing CLI flag (compat constraint).
- Promoting additional Pyomo solver options (beyond `mipgap` and
  `threads`) to first-class CLI flags. Other options stay accessible
  via `--solver-options` and the new options-file only.
- Solver-name-aware key translation for any option other than `mipgap`
  and `threads`.

## 4. Open questions

The bigger questions in the previous draft are resolved by §2; what
remains:

1. **Options-file format and merge order.** JSON only, YAML, or both?
   When `--solver-options-file <path>` and `--solver-options "k=v ..."`
   are both supplied, who wins? Proposal to discuss: file is the base,
   inline string overlays. (CLI overlays file feels right because the
   inline string is the more "immediate" surface.)
DLW: CLI overlays


2. **Spoke-override merge depth.** Flat dict union, or anything more
   structured? Today's surface is flat (`{key: value}`), so a flat
   union is the minimum-change implementation. Anything richer would
   only matter if we add nested per-iteration sub-dicts (see #3).
DLW: flat union makes sense

3. **"After-iteration-N" surface.** How does the user specify N? Two
   sketches:
   - Generalize the current pattern: a flag `--after-iter-N-mipgap`
     where `N` is literal (probably awkward).
   - Express it only in the options-file: a top-level section like
     `{"starting_at_iter": {"5": {"mipgap": 1e-3}}}`.
   File-only keeps the CLI surface flat and avoids inventing many new
   flags. Probably the right call if the file format lands first.
DLW: File only. But the file will have to override iterk values or it won't make sense, right?

4. **Lagranger deprecation specifics.** Direction agreed: lagranger's
   custom iter0/iterk handling is deprecated; it routes through the
   same path as siblings, emits a `DeprecationWarning`, and raises if
   a user passes a combination the unified path cannot honor.
   Backward compatibility on the internal wiring is not required.
   Open: warning message text and removal timeline.
DLW: Open timeline. Just say that Lagranger will be deprecated in the future because it does not seem to
work as well as other outer bound options.

5. **Per-spoke `--mipgaps-json` variants.** Today only the global
   `--mipgaps-json` flag is registered (config.py:616, gated on
   `name is None`). The per-spoke automatic-gapper flags
   (`--{name}-starting-mipgap`, `--{name}-mipgap-ratio`) exist, but
   per-spoke schedule flags do not. Should the redesign add
   `--{name}-mipgaps-json` for symmetry with the rest of the per-
   spoke options surface? Adds one flag per Gapper-using spoke; trivial
   to implement (drop the `name is None` gate at config.py:616 and
   plumb in `add_gapper`).
DLW: yes, let's do it. Resolved — incorporated into §5.7 and the
phase-5 work in §6.4. `Config.gapper_args(name)` registers
`--{name}-mipgaps-json` for every per-spoke prefix it is called with.

## 5. Proposed design

### 5.1 One sentence

Replace the `(iter0_solver_options, iterk_solver_options)` pair with an
ordered list of *predicate-scoped overlays*, fed by a new JSON
options-file plus all of today's CLI flags, merged by flat dict union,
and translated for `mipgap` / `threads` only at the last moment before
the values reach the Pyomo solver plugin.

### 5.2 Internal data model: layered overlays

A solve's effective options are produced by walking a list of layers
in order and folding each into a running dict (last write wins per
key, flat dict union). Each layer has a *predicate* (when does this
layer apply) and an *options dict* (what to write).

```python
SolverOptionsLayer = TypedDict("SolverOptionsLayer", {
    "when": Predicate,   # see below
    "options": dict[str, Any],
})

# Predicates:
#   "default"                — always
#   "iter0"                  — iteration 0 only
#   "iterk"                  — iterations k >= 1
#   ("starting_at_iter", N: int)   — iterations k >= N
```

`iterk` is sugar for `("starting_at_iter", 1)` and is kept as a separate
predicate solely for compatibility with `--iterk-mipgap`. EF and other
non-iterating solves treat every layer with `when in {"default",
"iter0"}` as applying, and ignore `iterk` / `starting_at_iter` layers.

A method `PHBase._effective_solver_options(k: int) -> dict` walks
`self.solver_options_layers` in order, picking layers whose predicate
matches `k`, and returns the merged dict. This replaces the
`current_solver_options` flip at `phbase.py:1052`.

The existing `--mipgaps-json` schedule (§1.1, §5.7) folds naturally
into this model: each `{"<N>": gap}` entry becomes a layer with the
same predicates the layered system already has — `iter0` for `N=0`,
`iterk` for `N=1`, `("starting_at_iter", N)` for `N >= 2`. So the static
schedule needs no new predicate; only the integration is new (§5.7).

### 5.3 New options-file

**Shipped.** `--solver-options-file <path>` is registered in
`Config.add_solver_specs()`, so each per-spoke prefix
automatically gets `--{name}-solver-options-file` too. Format is
JSON (stdlib only). YAML can be added later as an alias if there's
demand.

Schema (by example):

```json
{
  "default":   {"threads": 4, "presolve": 2},
  "iter0":     {"mipgap": 1e-4},
  "iterk":     {"mipgap": 1e-3},
  "starting_at_iter": {
    "5":  {"mipgap": 1e-5},
    "10": {"mipgap": 1e-6}
  },
  "spokes": {
    "lagrangian": {
      "default":    {"mipgap": 0.01},
      "starting_at_iter": {"5": {"mipgap": 0.001}}
    },
    "reduced_costs": {
      "iter0": {"mipgap": 0.001}
    }
  }
}
```

`starting_at_iter` keys are JSON strings (since JSON object keys must be
strings); they are coerced to ints at load time. Per-spoke sub-blocks
mirror the top-level shape.

A new helper `mpisppy.utils.sputils.load_solver_options_file(path) ->
list[SolverOptionsLayer]` reads the file and returns the layer list
(global section); a sibling helper extracts the per-spoke sublayers.

### 5.4 Merge precedence (two-axis rule)

Folding all the layers into one effective dict for iteration `k` is a
*flat* dict union (per §4 q2 — last write wins per key), but the
*order* in which layers fold is governed by two axes, applied in
order:

**Axis 1 — Predicate specificity (most-general first; most-specific
last, so most-specific wins):**

```
   default
      ≺  iter0       (only at k = 0)
      ≺  iterk       (k ≥ 1)
              ≺  starting_at_iter:N₁    (k ≥ N₁, N₁ ≥ 1)
              ≺  starting_at_iter:N₂    (k ≥ N₂, N₁ < N₂)
              ≺  ...              (sorted by ascending N)
```

`iter0` and `iterk` are disjoint, so the comparison only matters for
predicates that all match the current `k`. `starting_at_iter:N` is strictly
more specific than `iterk` whenever it matches, because the user
named a precise N. `N = 0` is not allowed: it would silently outrank
`iter0` / `iterk` for every iteration; a layer that should apply
universally must use the `default` predicate instead.

**Axis 2 — Source order, within a single predicate:**

```
   options-file section                       (general --solver-options-file)
      ≺  inline --solver-options              (default predicate only)
      ≺  --mipgaps-json schedule entry        (mipgap-only, predicate
                                               per entry; see §5.7)
      ≺  --iter0-mipgap / --iterk-mipgap /
         --max-solver-threads                 (CLI sugar; canonical
                                               keys mipgap, threads)
```

This is the "CLI overlays file" rule (§4 q1) — but it only breaks
ties *within the same predicate*. Across predicates, axis 1 wins.

`--mipgaps-json` sits above the general options-file because it is a
mipgap-specific surface: a user who reaches for the schedule flag is
expressing more intent about mipgap than someone who set a `mipgap`
key inside the general options-file, and should override it. It sits
below the dedicated `--iter0-mipgap` / `--iterk-mipgap` flags so a
single CLI sugar still wins for the iteration it scopes — though in
practice users rarely combine schedule + sugar for the same N.

**Per-spoke layers** apply on top of the merged global dict. Within a
spoke, axes 1 and 2 apply the same way, with `--{name}-solver-options`
and `--{name}-iter0-mipgap`/`--{name}-iterk-mipgap` taking the CLI-
sugar role:

```
  global merged dict
      ≺  spoke options-file section (axis-1 ordered)
      ≺  spoke --{name}-solver-options        (default predicate)
      ≺  spoke --{name}-iter0-mipgap /
         --{name}-iterk-mipgap                (predicate-scoped)
```

Worked example — the case that motivated this rule:

```
CLI:  --iterk-mipgap=0.001
file: { "starting_at_iter": { "5": { "mipgap": 1e-5 } } }
```

| k | predicates that match    | folded order (axis 1, then 2)                  | result mipgap |
|---|--------------------------|------------------------------------------------|---------------|
| 0 | `iter0`                  | (no `iter0` writer here) → unset               | unset         |
| 3 | `iterk`                  | CLI `--iterk-mipgap`                           | `0.001`       |
| 7 | `iterk`, `starting_at_iter:5`  | CLI `--iterk-mipgap`, then file `starting_at_iter:5` | `1e-5`        |

The CLI's `--iterk-mipgap` writes first (axis 1 puts `iterk` before
`starting_at_iter:N`), then the file's `starting_at_iter:5` overwrites it because
it is strictly more specific. CLI does not "win" against a more
specific predicate — only against a same-predicate file entry.

A natural consequence: if both file and CLI set the *same* key with
the *same* predicate, CLI wins (axis 2). If file sets a key with a
*more specific* predicate that matches, file wins (axis 1). This
matches the §4 q3 follow-up — yes, file's `starting_at_iter:N` overrides
both file `iterk` and CLI `--iterk-mipgap` for `k ≥ N`.

Solver-name-aware translation (§5.6) is applied to the final folded
dict, not at any intermediate layer.

### 5.5 Per-spoke override semantics — behavior change

Today, `apply_solver_specs(name, spoke, cfg)` *replaces* a spoke's
iter0/iterk dict wholesale when `--{name}-solver-options` is given
(cfg_vanilla.py:119–120). Under the new model, per-spoke options are
overlays on top of the global merged dict. A user who depended on the
old replace-behavior recovers it by re-spelling every key they want;
since merge is strictly more general, this is a backward-compatible
behavior *broadening*, not a contraction.

This is the only user-visible behavior change for existing CLI
invocations. It must be called out in §6 (migration plan) and in the
release notes.

### 5.6 Solver-name-aware translation

A new helper in `mpisppy/utils/sputils.py`:

```python
def translate_solver_options(opts: dict, solver_name: str) -> dict:
    """Return a copy of opts with mipgap/threads renamed to the
    solver's actual key, where it differs from mpi-sppy's canonical
    key. Other keys are passed through unchanged."""
```

Translation table (initial proposal — verify before implementing):

| Canonical | cplex / cplex_persistent | gurobi / gurobi_persistent | xpress / xpress_persistent | highs / appsi_highs |
|-----------|--------------------------|----------------------------|----------------------------|---------------------|
| `mipgap`  | `mipgap`                 | `mipgap`                   | `mipgap`                   | `mip_rel_gap`       |
| `threads` | `threads`                | `Threads`                  | `threads`                  | `threads`           |

Collision rule: if both the canonical key (`mipgap`) and the
solver-specific key (`mip_rel_gap`) are present in the merged dict —
e.g. because the user wrote `mip_rel_gap` directly in
`--solver-options` *and* also passed `--iter0-mipgap` — the
solver-specific key wins (translation does not overwrite an existing
value at the destination key). The user explicitly named the
solver-specific key, so respect that intent.

Translation runs at the latest possible moment: inside `solve_one`,
just before the `for option_key,option_value in solver_options.items()`
loop at `spopt.py:183-187`. Stored options on `PHBase` always use the
canonical names, so a single layered structure is solver-agnostic
until the solver is actually known.

Solver name comes from `self.options["solver_name"]`, which is what's
already used to instantiate the solver plugin. If `solver_name` is
`None` (shouldn't happen at solve time, but guard anyway), translation
is a no-op.

### 5.7 Mipgap schedule and Gapper integration

The redesign reuses the existing `--mipgaps-json` flag (registered in
`Config.gapper_args()` at config.py:610) rather than introducing a new
mipgap-schedule flag. The current Gapper machinery splits cleanly into
two cases that get different treatment.

**Static schedule (`--mipgaps-json` only):**

At config-build time (`shared_options` / `apply_solver_specs`), parse
the JSON file and append one layer per entry to
`solver_options_layers`:

| JSON key | Layer predicate         | Layer dict       |
|----------|-------------------------|------------------|
| `"0"`    | `iter0`                 | `{"mipgap": v}`  |
| `"1"`    | `iterk`                 | `{"mipgap": v}`  |
| `"N"` (N≥2) | `("starting_at_iter", N)`  | `{"mipgap": v}`  |

These layers enter axis 2 at the `--mipgaps-json` source level (§5.4),
so they overlay any `mipgap` set in the general options-file, while
single CLI sugar flags (`--iter0-mipgap`, `--iterk-mipgap`) still win
for the iteration they scope. Axis 1 (specificity) handles N stacking.

This *eliminates* the runtime mutation of `current_solver_options` at
`mipgapper.py:46-52` for the static-schedule case: the schedule is a
static configuration of layers, materialized once at startup, and
`_effective_solver_options(k)` produces the right dict at every k
without the Gapper extension being involved.

**Automatic gapper (`--starting-mipgap` / `--mipgap-ratio`):**

This mode is genuinely adaptive: at each `miditer`, the Gapper reads
the hub/spoke bound gap (`compute_gaps()` at mipgapper.py:69) and sets
mipgap to `problem_rel_gap * mipgap_ratio` if that's tighter than
`starting_mipgap`. There is no static layer that can express this.

Treatment: keep `Gapper` as a runtime extension, but rework
`set_mipgap` to write into a designated *dynamic layer* on
`PHBase.solver_options_layers` rather than mutating
`current_solver_options`. Sketch:

- At setup, PHBase reserves a layer slot at the very top of axis 2
  (above all CLI sugar) tagged `dynamic_gapper`.
- Gapper's `set_mipgap(g)` writes `{"mipgap": g}` into that layer's
  options dict, scoped to predicate `("starting_at_iter", k_now)` (or
  simpler: `default`, since the dynamic value is meant to apply
  immediately and persist until the next dynamic write).
- `_effective_solver_options(k)` picks up the dynamic layer last, so
  the adaptive gap wins over everything the user configured —
  matching today's runtime-overwrite semantics.

The `Config.gapper_args()` registration is unchanged; the `--mipgaps-
json` flag and `--starting-mipgap` flag keep their current names and
defaults (CLI compat).

**Per-spoke schedule flags (resolved per §4 q5).** Add
`--{name}-mipgaps-json` for each Gapper-using spoke prefix, by removing
the `name is None` gate at config.py:616 inside `Config.gapper_args()`.
`add_gapper` (cfg_vanilla.py:393) reads the per-spoke flag the same way
it reads the global one and contributes per-spoke layers to the
spoke's section of `solver_options_layers`.

### 5.8 Lagranger deprecation

Per §4 q4: emit a `DeprecationWarning` at `lagranger_spoke()` setup
saying lagranger is slated for removal in a future release because it
does not seem to perform as well as the other outer-bound options.
No removal timeline committed in this redesign. Internal wiring is
*not* refactored to fold into `apply_solver_specs` in this round —
that's tracked separately and is conditional on whether lagranger
survives at all.

### 5.9 Component-level changes

| File                                 | Change                                                                                                                                |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `mpisppy/utils/sputils.py`           | + `load_solver_options_file(path)`, `translate_solver_options(opts, solver_name)`, `_canonical_solver_options_table` constant. `option_string_to_dict` unchanged. |
| `mpisppy/utils/config.py`            | `add_solver_specs(prefix)` also registers `{prefix_}solver_options_file` (str, default None). All existing flags keep the same names, types, and defaults.        |
| `mpisppy/utils/cfg_vanilla.py`       | `shared_options` outputs `solver_options_layers` (a list) instead of (or in addition to) `iter0_solver_options` / `iterk_solver_options`. `apply_solver_specs` overlays per-spoke layers rather than replacing dicts. Reapplication of `max_solver_threads` becomes a `default` layer at the top of the global stack. |
| `mpisppy/phbase.py`                  | Replace `iter0_solver_options` / `iterk_solver_options` / `current_solver_options` with `solver_options_layers` plus an `_effective_solver_options(k)` accessor. iter0/iterk attribute access supported via deprecated property shims that warn once and return the merged dict for that predicate.           |
| `mpisppy/spopt.py`                   | In `solve_one`, ask the caller for an effective options dict (or compute it via `phbase` reference), then run it through `translate_solver_options(opts, self.options["solver_name"])` before the loop at `spopt.py:183-187`.                                                                                  |
| `mpisppy/cylinders/lagranger.py`     | Emit `DeprecationWarning` at setup. No other change.                                                                                                                                                                                                                                                          |
| `mpisppy/extensions/mipgapper.py`    | `set_mipgap` no longer reads or writes `self.ph.current_solver_options` (mipgapper.py:46-52 deleted). Instead, write to the reserved `dynamic_gapper` layer on `PHBase.solver_options_layers`. Static-schedule branch (mipgapper.py:78-84) is no longer needed once `--mipgaps-json` becomes static layers in `cfg_vanilla` — Gapper's `__init__` skips `mipgapdict` handling and only runs in automatic-gapper mode. See §5.7.   |
| `mpisppy/utils/cfg_vanilla.py`       | (continued) `add_gapper` (cfg_vanilla.py:393) parses `--mipgaps-json` and appends one layer per JSON entry to `solver_options_layers`, instead of stuffing `mipgapdict` into `gapperoptions`. Automatic-mode flags continue to flow through `gapperoptions` for the runtime extension.                                                                                                                                          |
| `mpisppy/utils/solver_spec.py`       | EF path: read the same `default` / `iter0` layers from the new file (EF treats `iter0` as applying — see §5.2). Translation runs the same way at solve time.                                                                                                                                                  |

### 5.10 Programmatic API compatibility

Examples like `examples/sslp/sslp.py:221` set
`options["iter0_solver_options"]` and `options["iterk_solver_options"]`
directly, bypassing `cfg_vanilla`. PHBase will still accept these keys
on the input options dict, fold them into the layers list, and emit a
single `DeprecationWarning` per run pointing at the new
`solver_options_layers` shape. No removal in this round.

### 5.11 Things deferred to §6

§6 should enumerate every flag in §1.1 and confirm that under §5 it
produces the same dict the solver plugin sees today, with the single
documented exception in §5.5 (per-spoke overlay vs replace).

## 6. Migration / compatibility plan

### 6.1 Per-flag compatibility audit

Every CLI flag in §1.1, with the dict the solver plugin will see under
the new design vs. today. "Compat" means the produced dict at solve
time is identical to today's for any reasonable user invocation.

| Flag                                          | Today                                                 | Under new design                                                                | Compat?            |
|-----------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------|--------------------|
| `--solver-name`                               | passed to `SolverFactory`                             | unchanged                                                                       | ✓                  |
| `--solver-options "k=v ..."`                  | parsed once; copied to iter0 and iterk dicts          | parsed once; becomes a `default` layer                                          | ✓ (see §6.2 note 1)|
| `--max-solver-threads`                        | written into iter0/iterk under `threads`              | `default` layer with `{threads: N}`; translated for the solver at solve time    | ✓                  |
| `--solver-log-dir`                            | per-solve logfile path                                | unchanged plumbing                                                              | ✓                  |
| `--iter0-mipgap`                              | iter0 dict `mipgap`                                   | `iter0` predicate layer with canonical `mipgap` key                             | ✓                  |
| `--iterk-mipgap`                              | iterk dict `mipgap`                                   | `iterk` predicate layer with canonical `mipgap` key                             | ✓                  |
| `--EF-solver-name`                            | EF path                                               | unchanged (translation runs same way)                                           | ✓                  |
| `--EF-solver-options`                         | EF path; verbatim dict                                | parsed; applied to EF solve; canonical `mipgap`/`threads` translated            | ✓ (note 1)         |
| `--EF-mipgap`                                 | EF path; `mipgap` key                                 | EF `default` layer (EF is a single solve)                                       | ✓                  |
| `--{lagrangian,reduced-costs,subgradient,relaxed-ph,ph-dual,lagranger}-solver-name`           | per-spoke solver-plugin override                      | unchanged                                                                       | ✓                  |
| `--{name}-solver-options`                     | replaces spoke iter0/iterk dict (cfg_vanilla.py:119-120) | overlays global merged dict                                                  | **CHANGED — §6.2** |
| `--{name}-iter0-mipgap`, `--{name}-iterk-mipgap`                                                | per-spoke iter0/iterk `mipgap` key                    | per-spoke predicate layer                                                       | ✓                  |
| `--obbt-solver-options`                       | OBBT presolve dict; single solve                      | parsed; applied to OBBT solve as a single dict                                  | ✓                  |
| `--pickle-solver-name`, `--pickle-solver-options`                                              | pickle-time iter0 solve                               | unchanged plumbing (pickle path is its own thing)                               | ✓                  |
| `--stage2-ef-solver-name`                     | multistage stage-2 EF                                 | unchanged                                                                       | ✓                  |
| `--mipgaps-json`                              | runtime mutation by Gapper                            | static layers from JSON, materialized at config-build time (§5.7); Gapper no longer mutates for this case | ✓ (semantically identical for static schedules) |
| `--starting-mipgap`, `--mipgap-ratio`         | runtime adaptive mutation by Gapper                   | unchanged behavior; Gapper writes to a reserved `dynamic_gapper` layer instead of `current_solver_options` | ✓ |
| `--{name}-starting-mipgap`, `--{name}-mipgap-ratio`                                            | per-spoke adaptive                                    | unchanged                                                                       | ✓                  |

**Note 1 (translation effects).** For `--solver-options` /
`--EF-solver-options`, keys other than `mipgap` and `threads` pass
through verbatim, exactly as today. For `mipgap` and `threads`, the
key written into `s._solver_plugin.options` after translation may
differ from today *only when the user wrote the canonical key
(`mipgap`) for a solver whose actual key differs (e.g. HiGHS
`mip_rel_gap`)*. In that case, today the user's call to HiGHS with
`--solver-options "mipgap=0.01"` was silently ignored; under the new
design, `0.01` actually takes effect. This is a fix, not a
regression, but it could surface differently in tests that compared
solver behavior.

### 6.2 The one user-visible behavior change

**Shipped:** `apply_solver_specs` now overlays per-spoke
solver-options on top of the global dict (replace-style was the
prior behavior, kept through phases 1–3 to minimize churn).

Concretely:

```
--solver-options "presolve=2 threads=4"
--lagrangian-solver-options "mipgap=0.01"
```

| Lagrangian-spoke effective dict | Today (replace)        | New (overlay)                         |
|---------------------------------|------------------------|---------------------------------------|
| `mipgap`                        | 0.01                   | 0.01                                  |
| `threads`                       | (only if `--max-solver-threads`) | 4                           |
| `presolve`                      | not set                | 2                                     |

The new dict is a *superset* of the old one in every case where the
spoke flag's parsed dict is a subset of the global flag's parsed dict,
which is the common pattern in existing scripts (spoke flag tightens
mipgap, leaves the rest alone). Users who relied on the spoke flag
*dropping* a global key recover that by re-spelling the spoke options
to include the keys they want, or by leaving the global flag unset.

Action items (all complete):

- ✔ Release notes call-out under "Behavior changes" — README.md
  has a 2026 NOTICE section pointing at the user docs;
  `doc/src/generic_cylinders.rst` has a `solver-options` section
  with a `.. warning::` directive carrying the migration recipe.
- ✔ Integration test asserting the new merge semantics — see
  `test_per_spoke_overlay_combines_global_and_spoke_keys` in
  `mpisppy/tests/test_solver_options_layers.py` (covers the
  worked example above directly).
- ✔ Audit `examples/` for any harness whose semantics depend on
  replace-style — no example uses per-spoke solver-options
  flags, so nothing needed updating.

### 6.3 Programmatic-API migration

| Surface                                                  | Today                              | New                                                                               | Removal       |
|----------------------------------------------------------|-------------------------------------|----------------------------------------------------------------------------------|---------------|
| `options["iter0_solver_options"]`, `options["iterk_solver_options"]` (input to PHBase) | active            | accepted; folded into `solver_options_layers`; one `DeprecationWarning` per run  | future, TBD   |
| `PHBase.iter0_solver_options`, `iterk_solver_options`, `current_solver_options` (attribute reads) | active            | property shims that compute the merged dict for that predicate; warn on first access per run | future, TBD |
| `option_string_to_dict`                                  | active                              | unchanged                                                                         | n/a           |
| `option_dict_to_string`                                  | active                              | unchanged                                                                         | n/a           |

Deprecation timelines are deliberately not committed in this redesign;
they get set when we do a separate cleanup pass on examples and
external callers. `examples/sslp/sslp.py:221` is the canonical caller
that will need migration.

### 6.4 Phased rollout

The redesign is large enough that it should land in review-sized
phases, each independently testable. Suggested order — each phase is
green-on-its-own:

1. **Layer data model (no behavior change) + this design document.**
   Add `solver_options_layers` to `PHBase` alongside the existing
   `iter0_solver_options` / `iterk_solver_options` attributes.
   `shared_options` and `apply_solver_specs` build the list *and*
   keep populating the iter0/iterk dicts. Internal consumers
   continue to read iter0/iterk; layer list is dormant. This
   document (`doc/designs/solver_options_redesign.md`) ships in the
   same PR so reviewers have the full context for the data-model
   choices.
2. **Switch consumption to layers.** `solve_loop` /
   `_effective_solver_options(k)` reads from layers; iter0/iterk
   dicts become derived properties (no warning yet). Existing tests
   pass unchanged.
3. **Solver-name-aware translation.** Add `translate_solver_options`
   helper; wire into `solve_one` before the options-write loop
   (`spopt.py:183-187`). Add the canonical-key table.
4. **Per-spoke overlay semantics.** Switch `apply_solver_specs` from
   replace to overlay (§5.5 / §6.2). The one user-visible behavior
   change. Add a test asserting overlay.
5. **Mipgap schedule integration.** `add_gapper` parses
   `--mipgaps-json` into static layers. `Gapper.set_mipgap` writes to
   the `dynamic_gapper` layer instead of mutating
   `current_solver_options`. Static-schedule branch in `mipgapper.py`
   becomes dead code.
6. **New options-file.** Register `--solver-options-file` (and per-
   spoke variants); add `load_solver_options_file`; plumb file layers
   in `shared_options` / `apply_solver_specs`.
7. **Lagranger deprecation warning** (§5.8). Single-line addition.
8. **Programmatic-API deprecation warnings** (§6.3). Final phase
   because it touches user code paths and can be done cleanly only
   once the new shape is stable.

Phases 1–2 land internally with no surface change. Phase 4 is the only
phase with a release-notes-worthy behavior change. Phases 6 and 7 add
new surface (new flag, new warning); phases 3 and 5 add new behavior
that improves on quietly-broken cases.

### 6.5 Test coverage

Per memory rule: any new `mpisppy/tests/test_*.py` file added in this
redesign must be wired into both `run_coverage.bash` and
`.github/workflows/test_pr_and_main.yml` in the same commit, or
codecov/patch reports 0%.

New unit tests:

- `test_solver_options_layers.py`: predicate matching, ordering,
  `_effective_solver_options(k)` for representative `k`. Covers
  axis-1 and axis-2 cases of §5.4 individually.
- `test_solver_options_translation.py`: round-trip mipgap/threads
  across each known `solver_name` in the canonical table.
- `test_solver_options_file.py`: schema acceptance, malformed JSON,
  per-spoke nesting, `starting_at_iter` int-coercion.

New integration tests (added to existing test files where natural):

- `--solver-options "..." --solver-options-file path.json` together:
  inline overlays file (§5.4 axis 2).
- `--solver-options "presolve=2" --lagrangian-solver-options
  "mipgap=0.01"` produces overlay (§6.2).
- `--mipgaps-json {"5": 1e-5} --iterk-mipgap 0.001`: at k=3, mipgap
  is 0.001; at k=7, mipgap is 1e-5 (§5.4 worked example, executed).

Existing tests to keep green without modification:

- `mpisppy/tests/test_ef_ph.py` (Gapper tests at 513, 535).
- `mpisppy/tests/test_ph_extensions.py` (Gapper tests at 58, 81 —
  programmatic `gapperoptions` dict).

### 6.6 Documentation

- One new RST page under `doc/src/` covering solver-options surface
  end-to-end: CLI flags, options-file schema, mipgap schedule,
  per-spoke overrides, translation table. Replaces whatever scattered
  notes exist today.
- `CHANGELOG` entry under "Behavior changes" pointing at §6.2.
- `CHANGELOG` entry under "New features" for `--solver-options-file`,
  per-spoke variants, and the per-spoke `--{name}-mipgaps-json` flags
  resolved in §4 q5.

### 6.7 Things this plan does *not* do

- Does not commit deprecation-removal timelines for any of the
  warning-only items (§5.8 lagranger, §5.10 programmatic-API). Those
  are tracked separately.
- Does not refactor lagranger's internal wiring beyond emitting the
  `DeprecationWarning` (§5.8). If lagranger is removed entirely
  before the wiring needs touching, the refactor is moot.
- Does not introduce CLI flags for individual non-mipgap-non-thread
  options (§3 non-goal).
