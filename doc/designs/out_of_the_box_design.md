# Out-of-the-box auto-configuration — design

**Status:** Design phase. Branch `outOfTheBox` (off Pyomo/mpi-sppy `main`), on
the DLWoodruff fork. Proceeding deliberately ("slowly"): requirements
confirmed; the decision-logic mechanism (§5) and instantiation effort tiers
(§5.2) are resolved; the first dated policy file is committed. No production
library code yet — only an uncommitted interpreter *sketch* (§7).
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-06-28

---

## 0. Vocabulary

**Out-of-the-box (OOTB)** mode: a CLI switch (`--out-of-the-box`, with a lighter
`--out-of-the-box-minus` and a heavier `--out-of-the-box-plus` variant — §5.2)
that lets a relatively naive user obtain a *sensible* mpi-sppy run with almost
no knowledge of the library's internals. The user supplies a model module (and
scenario data); mpi-sppy **introspects the environment and the model** and
**auto-assembles a defensible configuration** — algorithm, spokes, bundling,
solver — rather than requiring a hand-crafted hub/spoke command line.

The *spirit* is to lower the barrier to entry: the newcomer effectively says
"here is my model, go," and gets a reasonable decomposition plus a clear
explanation of what was chosen and how to do better.

---

## 1. Goals and non-goals

### Goals

1. A `--out-of-the-box` option (CLI entry `generic_cylinders.py`, implemented
   in the refactored `mpisppy/generic/` package) that sets run options
   automatically.
2. **User options always win** (requirement 0). OOTB only *fills gaps*; any
   option the user explicitly set is retained verbatim and never overridden.
3. **Environment + model introspection** (requirement 2). At minimum: the
   model module, the MPI rank count, and which solvers are actually installed.
   Where the OS / SLURM permits: core count and memory. Plus any options the
   user already supplied.
4. **Floor at 3 ranks** (requirement 3): with fewer than 3 ranks there is no
   useful cylinder configuration (hub + >= 2 spokes), so OOTB solves the **EF**
   instead.
5. **Transparency** (requirement 4): print (a) the **equivalent explicit
   command line** the choices imply, so the user can reproduce / learn from /
   tweak it, and (b) a short, prioritized list of **suggestions** (labelled
   "Suggestions"), **written after the run executes** so it can also reflect
   how the run went (e.g., a persistent solver, more ranks). The run proceeds
   regardless.
6. **Proper bundling is central** (requirement 5): auto-forming proper bundles
   from scenario count vs. available ranks is a first-class part of the
   decision, not an afterthought.
7. **Quick-start documentation** (requirement 1): OOTB becomes the
   recommended on-ramp in the quick start.

### Non-goals (initial cut)

- Replacing the explicit cylinder command line for expert users. OOTB is an
  *on-ramp*, and it deliberately emits the explicit command line it chose.
- Tuning convergence parameters to optimality. OOTB aims for *defensible*,
  not *optimal*, configurations.
- Running a solver to make decisions in the default path. Trial solves are
  confined to the opt-in `plus` tier (§5.2); minus and base never solve.

---

## 2. Confirmed scope decisions

These were raised as scoping questions and confirmed by the user (2026-06-28):

1. **Home + scope.** `generic_cylinders.py` is the primary (initial) entry
   point. Reachability via `Amalgamator` is possible later but not the first
   target. *(CONFIRMED as the assumed direction; revisit if Amalgamator
   coverage is wanted sooner.)*
2. **What we read from the module.** OOTB learns scenario count / names (hence
   two-stage vs. multistage structure) from the module. Whether OOTB also
   *instantiates scenarios* to gauge size/difficulty is **resolved** as a
   user-selectable effort axis (none / one probe / all) — see §5.2.
3. **"Dated data files."** Interpreted as a versioned, **date-stamped
   knowledge base** of heuristics/benchmarks ("for problems like X with
   resources like Y, configuration Z worked well") that *guides* the chooser
   and can be refreshed over time — **not** user run-config files.
4. **"What would help" output.** Echo the equivalent command line *and* a
   short prioritized **Suggestions** list — the command line up front, the
   suggestions after the run; the run proceeds anyway.

---

## 3. Inputs OOTB consults

| Source | Items | Notes |
|---|---|---|
| User-supplied options | anything already on the command line / Config | Highest precedence; never overridden |
| Model module | scenario count/names, stage structure (2-stage vs multistage); at base/plus also integrality, per-scenario size, nonant count | Depth set by the effort tier (§5.2): minus structural-only, base one probe, plus all |
| MPI environment | number of ranks | Drives EF-vs-cylinders and the 3-rank floor |
| Installed solvers | which solvers import / are licensed; persistent variants | Drives solver choice + a "get a persistent solver" hint |
| OS / SLURM (best effort) | core count, memory | OS-dependent; SLURM env vars when present |

---

## 4. Outputs OOTB produces

1. A fully-populated `Config` (or equivalent) that the normal driver path then
   executes.
2. A printed **equivalent explicit command line** (up front, before the run).
3. The run itself (EF when the EF gate trips — too few ranks, or the problem is
   small enough per §5.1/§5.2 — otherwise the chosen cylinder configuration);
   **skipped entirely under `--inspect-only`** (§5.4).
4. A printed prioritized **Suggestions** list, emitted **after the run**
   (labelled "Suggestions"). Mostly **computed** from live facts / decision /
   run outcome, not canned (§5.1).

---

## 5. Decision-logic mechanism (RESOLVED 2026-06-28)

The first design question — expert system vs. neural net vs. nested case/ifs,
all optionally driven by dated data files — is **resolved**:

**Authored, declarative, data-driven: a dated/versioned policy file holding the
knowledge, interpreted by a thin hand-written Python decision routine. No
rules-engine library, no neural net.**

Reasoning:

- The three candidates are really two categories: *authored* knowledge
  (expert system, nested ifs — the same thing at two points on a spectrum) vs.
  *learned* knowledge (neural net). The "dated data files" idea cuts across
  all three and is the real commitment: **knowledge lives in data, not frozen
  in code.**
- **Neural net rejected** on four independent grounds: cold start (no training
  corpus, and never NN-scale data), the hard transparency requirement (must
  explain *why* — req. 4), the contributor base (optimization researchers edit
  rules, not models), and testability.
- **Rules-engine libraries rejected** (pyke is abandonware ~2010; experta is
  stale ~2018 with `frozendict` pin problems; clipspy/CLIPS is maintained but
  heaviest — C extension + its own language — for our smallest need). A RETE
  engine earns its cost only with many interacting, re-firing rules; OOTB makes
  a handful of decisions once, in an obvious order. An engine also adds a heavy
  dependency into already-fragile MPI/solver installs and can make firing-order
  *harder* to explain.
- So: implement the "expert-system framing" (facts + declarative knowledge +
  thin matcher) as plain Python reading a JSON policy file (~order of 100
  lines). Conditions are **plain coded checks** over the facts — no expression
  language in v1.

**Migration path preserved:** when a benchmark corpus eventually exists it
*tunes the numbers inside the next dated policy file* (e.g. a regression that
sets the EF cutoff or bundle target); structure stays rule-based and
explainable. **Escape hatch:** if the knowledge base ever explodes into dozens
of forward-chaining rules, revisit **clipspy** (never experta/pyke) by porting
the policy file's contents into a CLIPS KB — the policy-as-data design keeps
that open. We expect never to need it.

Criteria this satisfies: transparency (every decision logs its reason →
equivalent command line + suggestions fall straight out), cold-start (authored v1
works day one), maintainability (Python + JSON), testability (deterministic
"facts X ⇒ config Y" unit tests).

## 5.1 Policy file: location, selection, and v1 schema

**Location.** Dated policy files ship with the library under
`mpisppy/generic/ootb_policies/` (co-located with the OOTB code, which lives in
the refactored `mpisppy/generic/` package — `parsing.py`, `ef.py`, `hub.py`,
`spokes.py`, `scenario_io.py`, … ; `generic_cylinders.py` remains the
user-facing CLI entry that delegates into this package). First file:
`ootb_policies/ootb_policy_2026-06-28.json`.

**Selection.** Each OOTB flag takes an **optional policy-file path** (see the
§5.2 mechanism): a bare flag uses the shipped default (the newest dated file
whose name carries no focus token, e.g. `ootb_policy_<date>.json`);
`--out-of-the-box PATH` uses `PATH`. There is **no separate policy-file flag** —
the optional value *is* the path. Multiple policy files with **different foci**
may ship side by side; the **focus is conveyed by the filename** (e.g.
`ootb_policy_quick_<date>.json`), which is human-facing documentation only — the
program does not parse focus from the name, and no machine-read `focus` field is
needed. A user selects a focus simply by passing that file's path. The run
**logs which policy file (and `policy_version`) it used**, for reproducibility.

**v1 schema** (see the file for the authoritative, self-documenting copy; every
threshold is flagged `_cold_start_guess` and is a placeholder to be tuned with
data):

| Key | Purpose |
|---|---|
| `ef_fallback` | when to solve the EF: rank floor `min_ranks_for_decomposition` (req. 3); else the **same effort model as bundles** on the whole problem — base `effort(num_scens) ≤ ef_effort_budget`, plus measured `ef_target_seconds`, minus count `ef_if_num_scens_at_most` |
| `solver` | `preference_order` (persistent-commercial → commercial → free QP-capable → LP/MIP-only), plus `commercial` / `qp_capable` / `lp_mip_only_force_linearize_prox` sets and `caveats` |
| `hub` | default hub factory (`ph_hub`, no flag) |
| `spoke_ladder` | ordered `rungs` of WIRED spoke flags (outer/inner), `core_roster_min` (≥1 outer + ≥1 inner = the 3-rank floor), `max_cylinders` |
| `rank_allocation` | small-core roster widened by ranks (add a rung only while each cylinder keeps ≥ `min_ranks_per_cylinder`, ≤ `max_cylinders`); ranks split **unbalanced** across cylinders by `rank_ratios` (xhatter 0.2) — crude cold-start (§5.5) |
| `effort_scaling` | shape of solve effort vs. size (continuous ~linear, integers superlinear via `int_exponent`); shared by bundle sizing (§5.3) |
| `bundle_sizing` | how big bundles are (base/plus only — **minus cannot bundle**): largest `spb` within an effort budget (base = relative M; plus = measured seconds); `--scenarios-per-bundle` divides `num_scens`; `#bundles ≥ #ranks` |
| `option_categories` | per-concern default options (`rho_setter` `--grad-rho`, `termination` `--rel-gap 0.01`, `max_iterations` `--max-iterations 100`, `dynamic_rho` `--dynamic-rho-primal-crit`), each skipped if the user set any flag in its `superseded_by` list; skipped on the EF path |
| `additional_options` | catch-all for other extra flags (each with an optional `superseded_by`, default = its own flag) |
| `suggestions` | toggles/tunes the **computed** suggestion generators (`disabled` suppresses specific ones); the prose lives in code, emitted after the run |

**Additional options, with per-concern override.** Beyond the structural
choices, the policy applies extra options grouped by **concern** in
`option_categories` — `rho_setter` (`--grad-rho`), `termination`
(`--rel-gap 0.01`), `max_iterations` (`--max-iterations 100`), `dynamic_rho`
(`--dynamic-rho-primal-crit`). Each carries a **`superseded_by`** list of user
flags that obviate OOTB's default for that concern, so OOTB backs off when the
user addresses the concern with *any* equivalent flag — not just the identical
one. This matters concretely: mpi-sppy allows **only one rho setter active**, so
adding `--grad-rho` when the user chose `--sensi-rho` would be a *hard error*, so
`rho_setter.superseded_by` lists all the rho setters. A leftover
`additional_options` catch-all handles miscellaneous flags (each with an optional
`superseded_by` that defaults to its own flag). Conditionality across problem
*classes* is still expressed by **focus** (which file ships which options). All
are decomposition-run options, skipped on the EF path. (`--dynamic-rho-primal-crit`
is a boolean — no argument; its threshold `dynamic_rho_primal_thresh` defaults to
0.1 — and needs an active rho setter, which OOTB's default `--grad-rho` provides.)

**Suggestions are computed, not canned.** The post-run **Suggestions** (req. 4,
§4) are produced by small Python *generators* that compute text from live
facts / decision / run-outcome; the policy only toggles/tunes them
(`suggestions.disabled`). This keeps the split clean: **decisions stay
data-driven (this policy); suggestions are the computed diagnostics layer.**

## 5.2 Effort tiers (instantiation depth) — RESOLVED 2026-06-28

OOTB exposes three tiers along an **effort axis** — how deeply it inspects the
model — selected by mutually-exclusive flags. They share ONE interpreter and
ONE policy file; the tiers differ only in how deeply `gather_facts()` populates
the `Facts` object. Every decision uses the **best fact available** and
**degrades gracefully** (to structural reasoning, or to a suggestion) when a
fact is absent. Requirement 0 (the user's explicit options win) is orthogonal
to the tier.

| Tier | Flag | Instantiates | New facts | What it can decide (vs. advise) |
|---|---|---|---|---|
| minus | `--out-of-the-box-minus` | nothing | scenario count, ranks, solvers, stage structure | EF gate by **count**; solver by availability; **cannot bundle**. Integrality/size unknown → only **advises** on prox linearization |
| base (default) | `--out-of-the-box` | **one** probe scenario | size profile: `vars_int`, `vars_cont`, `nonants_total`, `nonants_int` | EF gate **size-aware**; integrality **decides** ipopt/HiGHS/linearize-prox; effort-budgeted bundle sizing (§5.3) |
| plus (later) | `--out-of-the-box-plus` | **all** + brief solve | per-subproblem solve time, LP-relax / integrality gap | iteration/time-limit defaults, bundle sizing to amortize solve cost, "hard MIP" signals |

**Mechanism.** The three flags set one internal `ootb_effort` level
(`minus`/`base`/`plus`); at most one may be supplied. Each takes an **optional
policy-file path** — declared `domain=str, default=None,
argparse_args={'nargs': '?', 'const': <default-policy>}` — giving three states:
absent → `None` → OOTB off; bare flag → `const` → the default policy; `PATH` →
that policy file. (A `bool` domain cannot be used: pyomo's
`declare_as_argument` forces `store_true` for bool, which takes no value;
`str` + `nargs='?'` is the supported optional-value form, and `add_to_config`
forwards `argparse_args` straight to `argparse.add_argument`.) This keeps a
single code path — richer tiers merely turn suggestions into decisions.

**Why a probe, not reuse-all (the intervention seam).** The *structural*
choices — EF-vs-decomposition, bundling, cylinder/comm/rank layout — must be
fixed *before* models are built: they determine what is instantiated and the
MPI topology, and proper bundling builds a *new combined* Pyomo model, so
singleton instantiations cannot be reused as bundles (Pyomo components cannot
be re-parented). Instantiating everything first and then restructuring wastes
work and reshuffles comms. So **base** instantiates only a cheap *probe*
(discarded) for the structural decisions; the normal driver then does the real,
layout-correct instantiation — needed anyway — which also feeds any *parametric*
refinement (solver, spokes, rho, iterations) where intervening is free.
**plus** accepts more redundant work by design, in exchange for solve-time /
gap information.

**`probe_scenarios` knob (lands with the base tier).** How many representative
scenarios base/plus instantiate (`scenario_names[:probe_scenarios]`). Default
**1** — enough for structurally homogeneous scenarios; bump it for models whose
size or integrality varies by scenario. Added to the policy file when base is
implemented (PR1); not in the 2026-06-28 file.

## 5.3 Bundle sizing & effort scaling — how big, not just whether

The hard question is not *whether* to bundle but *how big* bundles should be —
unportable in raw variable counts (the same count is trivial for one model,
intractable for another). The design makes bundle size a **derived** quantity:
pick the **largest** `scenarios_per_bundle` (`spb`) that (a) divides
`num_scens`, (b) leaves at least `B_min = max(intra_ranks,
min_bundles_per_intra_rank · intra_ranks)` bundles, and (c) keeps a bundle's
**modeled solve effort** within a budget.

**Shared effort shape (`effort_scaling`).** One policy block models how solve
effort grows with sub-problem size, from the probe profile (`vars_cont`,
`vars_int`, `nonants_int`):

> `effort(spb) = cont_coeff·(spb·vars_cont) + int_weight·(spb·vars_int)^int_exponent + int_nonant_coeff·nonants_int`

Continuous content is ~linear; integers are **superlinear** (`int_exponent > 1`
captures branch-and-bound blow-up); integer nonants are a fixed per-bundle
coupling cost. (Second-stage integers scale with `spb`; integer *nonants* are
first-stage, shared once per bundle — hence a fixed term, not a `spb`
multiplier.)

**Two anchors, one shape.** Both tiers call the same `effort(spb)`; only the
budget differs:

- **base** (relative, no measurement): accept the largest `spb` with
  `effort(spb)/effort(1) ≤ base_max_hardness_vs_single_scenario` (**M**). M is
  unit-free and portable — "a bundle may be at most M× as hard as one
  scenario." Pure-continuous ⇒ allowed `spb ≈ M`; pure-integer ⇒
  `≈ M^(1/int_exponent)`, automatically smaller.
- **plus** (absolute, measured): measure `t₁` = a single-scenario solve (capped
  at `plus_probe_solve_time_cap_seconds`), predict
  `t(spb) ≈ t₁·effort(spb)/effort(1)`, accept the largest `spb` with
  `t(spb) ≤ plus_target_seconds_per_bundle`.
- **minus** (no profile): **cannot bundle** — with no model information there is
  no safe way to size a bundle, so minus always runs unbundled.

**Measurement does not remove the JSON assumptions.** `plus` only pins the
*scale* (`t₁`); the *shape* (`int_exponent`, weights) still comes from
`effort_scaling`. And MIP solve times are **noisy and non-monotone** (a bigger
MIP can solve faster), so a single timing must not drive the whole choice — the
JSON shape is a **prior/regularizer** that measurement calibrates. (Later
refinement: `plus` measures two points to nudge `int_exponent` locally.)

**EF gate uses the same effort model.** The EF is just all scenarios as one
model, so the EF gate reuses `effort()` on the whole problem: above the rank
floor, run the EF when `effort(num_scens)` is within an **EF budget**. Unlike
bundle sizing's *relative* budget (M× a single scenario), the EF budget is
**absolute** — the monolith has no single-scenario reference: **base**
`effort(num_scens) ≤ ef_effort_budget` (same effort units as bundle effort);
**plus** measured `t₁·effort(num_scens)/effort(1) ≤ ef_target_seconds`; **minus**
(no profile) falls back to the count rule `num_scens ≤ ef_if_num_scens_at_most`.
Because the units match bundle effort, the EF budget and bundle budgets are
mutually consistent: when the whole problem exceeds the EF budget, OOTB
decomposes and sizes each bundle within its own (smaller) effort budget.

**User-forced decomposition overrides the gate.** If the user explicitly set any
**decomposition flag** — a wired spoke or a non-default hub (`DECOMPOSITION_FLAGS`
in the interpreter) — and has at least the rank floor, OOTB **never** substitutes
the EF, even for a small problem (requirement 0). The rank floor is checked
first: below it the decomposition can't fit, so the EF is used regardless. (This
flag vocabulary is a *fact* about `generic_cylinders`, not a focus preference, so
it lives in code, not the policy; the validator checks it against the real CLI.)

**Status.** All `effort_scaling` / `bundle_sizing` numbers are
`_cold_start_guess`es; **foci** ship different shapes (a `mip-heavy` file with a
steeper `int_exponent`), and the dated-file migration path (§5) refines the
coefficients from benchmark data. The interpreter sketch implements the base
relative sizer (`_effort`, `_pick_spb_by_effort`); minus does not bundle; the
`plus` measure-and-scale hook is stubbed.

## 5.4 `--inspect-only` (dry run; shares OOTB's instantiation) — RESOLVED 2026-06-28

A general driver flag (not OOTB-specific, but documented here because it shares
code): do the inspection, **print the configuration + equivalent command line +
config-time suggestions, then stop before the production optimization run.**

- **Semantics:** "no *production* run," not "no solver call ever." Per the
  user's decision, **`--out-of-the-box-plus`'s brief calibration solves count as
  inspection** (resolution B) — they are bounded by
  `plus_probe_solve_time_cap_seconds` and are *how* `plus` forms its
  recommendation — so `plus` + `--inspect-only` still measures, then stops.
- **Standalone (no `--out-of-the-box`):** `--inspect-only` **verifies a scenario
  can be instantiated** (builds one and reports) — a cheap model smoke-test —
  **reusing OOTB's probe instantiation code** (`verify_instantiation`, shared
  with the base/plus probe).
- **× `minus`:** silly but allowed. `minus` says "instantiate nothing," yet
  `--inspect-only` must build one scenario to verify — so **`--inspect-only`
  takes priority**: a single verification instantiation happens, while the
  *decision* stays minus-level (structural, no size profile fed into choices).
- **Suggestions:** only the config-time ones (nothing ran to yield
  outcome-based ones).
- **Optional assumed rank count (HPC planning):** `--inspect-only N` plans as if
  `N` ranks were available — so a supercomputer user can get the recommended
  command line for, say, a 512-rank job *from a login node, without launching
  it*. Bare `--inspect-only` uses the actually-detected ranks. Everything else
  (solvers, model size) still comes from the real (possibly small) session; only
  the rank count is hypothetical, and the emitted `mpiexec -np N …` reflects it.

Ships in **PR1**: an **optional-value** flag (`domain=str, nargs='?'`, the value
being the assumed rank count) — *not* `store_true`, since it now takes a value
(the same bool-domain caveat as the effort flags). The driver short-circuits
after printing, before apply-to-`Config` / run.

## 5.5 Rank allocation — small core, widened, and unbalanced

Two parts (policy `rank_allocation`):

**Roster size — prefer width over weak spokes.** Start from the minimal core
(≥1 outer + ≥1 inner spoke = hub + `--lagrangian` + `--xhatshuffle`). Add
further ladder rungs only while each cylinder would still keep
`min_ranks_per_cylinder` ranks (a coarse uniform gate), up to `max_cylinders`
(now **7**, so all six ladder rungs are reachable at enough ranks — no dead
rung). So **6 ranks → hub + lagrangian + xhatshuffle (3 cylinders), widened** —
*not* 6 single-rank cylinders. Extra ranks buy subproblem throughput, which (so
far in practice) beats piling on lower-value bound spokes.

**Unbalanced distribution (flex-ranks).** Ranks are split across the chosen
cylinders by per-cylinder **`rank_ratios`**, *not* uniformly: cheaper cylinders
get a smaller share. v1 ships `--xhatshuffle` (and the xhat family) at **0.2**;
everything else uses `default_rank_ratio` 1.0. Ratios are normalized over the
chosen cylinders and floored at 1 rank each (e.g. 6 ranks → hub 3, lagrangian 2,
xhatshuffle 1). **This is a crude cold-start:** the right split is a much more
complicated calculation that depends on the *nature of the subproblems*
(relative solve cost), and is a natural place for the `plus` tier's measurements
to inform. The widest cylinder's rank count governs the bundling
`#bundles ≥ #ranks` floor.

---

## 6. Open details

- **Instantiation depth — RESOLVED** as the effort tiers (§5.2): minus (none),
  base (one probe, default), plus (all + brief solve, later).
- **Bundle sizing — RESOLVED** as an effort-budgeted rule (§5.3): policy
  `effort_scaling` shape + `bundle_sizing` budgets; interpreter `_effort` /
  `_pick_spb_by_effort` (base, relative M); minus does not bundle; stubbed
  `plus` measure-and-scale. Numbers are `_cold_start_guess`es.
- **EF gate — RESOLVED** (§5.3): reuses the bundle `effort()` model on the whole
  problem against an absolute **EF budget** — base `ef_effort_budget`, plus
  `ef_target_seconds`, minus the count rule `ef_if_num_scens_at_most`.
- **Still open:** how the dated data files are generated, versioned, and shipped
  (the §5 migration path anticipates data-tuned successors).
- **Still open:** Amalgamator reachability (§2.1) — `generic_cylinders` first.

---

## 7. Phased rollout

Per project convention, ship as review-sized phases, each green on its own. A
sketch of the interpreter already exists at `mpisppy/generic/out_of_the_box.py`
— the pure `recommend(facts, policy) → Decision` logic is complete and
smoke-tested; environment/model probing and apply-to-`Config` are stubbed.

- **PR1 — interpreter pipeline + `--out-of-the-box-minus` + `--out-of-the-box`
  (base).** These share everything except one probe instantiation, so they land
  together: fact-gathering (structural + one-scenario probe), the policy
  interpreter, EF-vs-cylinder / bundling / spoke selection, reporting
  (equivalent command line + post-run suggestions), apply-to-`Config`, the
  `probe_scenarios` knob, the `--inspect-only` dry run (§5.4, incl. the shared
  `verify_instantiation`), quick-start docs, and tests. Ships the default
  on-ramp and the no-instantiation escape hatch in one PR.
- **PR2 (later) — `--out-of-the-box-plus`.** Full instantiation + a brief timed
  solve, a probe-time budget, and handling for "doesn't solve quickly"; feeds
  iteration/time-limit defaults and solve-cost-aware bundling.

---

## 8. Policy-file validation (TODO — tool to build)

A **fully automated** validator that, given a policy file, checks it is correct
and produces sensible, *executable* configurations — using the mpi-sppy
**examples** as test models. Two layers:

**Static (schema) checks.** JSON parses; required keys/types present; every
referenced flag is real — `spoke_ladder` rungs are wired spokes, solver names
known, each `option_categories[*].flag` and every `superseded_by` entry is a
valid option, `DECOMPOSITION_FLAGS` match the `generic_cylinders` vocabulary;
`_cold_start_guess` entries name real keys; numbers in range.

**Behavioral checks (using examples).** Run `recommend()` against real example
models (farmer, aircond, sizes, …) under synthetic environments (varying ranks,
available solvers, problem sizes) and assert:

- **EF invoked when it should be:** small problem or `< min_ranks` ⇒ EF.
- **EF *not* invoked when it shouldn't be:** large / integer-heavy ⇒ decompose.
- **User-forced decomposition wins:** simulated user `--ph --lagrangian
  --xhatshuffle` with ≥ rank floor ⇒ never EF (§5.3).
- **Bundling validity:** when bundling, `scenarios_per_bundle` divides
  `num_scens` and `#bundles ≥ #ranks`.
- **No conflicting options:** `superseded_by` simulation ⇒ OOTB never stacks a
  second rho setter (which would be a hard error).
- **Round-trip executability (strongest):** actually *run* OOTB's emitted
  equivalent command line on the example (a short smoke run) and confirm it gets
  past setup / completes — both for OOTB's own choice *and* for forced
  decomposition, confirming decomposition "works more-or-less as expected."

**Status: design TODO** — this checklist is partial; more checks to add. The
validator will gate the shipped policy files (eventually in CI).
