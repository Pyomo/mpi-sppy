# Design: Pre-Pickle Preprocessing for Scenarios and Bundles

**Status:** Implemented — historical design notes. This document was
written before the feature landed and is retained for context on the
decisions made. For the user-facing reference, see
``doc/src/pickling.rst``. The "Current state" section below describes
the codebase *before* this design was implemented; it should be read as
historical background, not a description of the current code.
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-04-08

## Motivation

`mpisppy` supports pickling scenarios and proper bundles to disk so that
downstream PH / EF / MMW runs can unpickle them instead of rebuilding every
time. Today, the pickle captures exactly what `scenario_creator` returns:
a freshly constructed Pyomo model with no preprocessing applied and no
solver state.

Several expensive operations are re-done on every rerun that could have
been baked into the pickle once:

1. **Feasibility-based bounds tightening (FBBT).** `SPPresolve` runs on
   every PH / EF start when `cfg.presolve` is set. For large models this
   is non-trivial and deterministic — doing it at pickle time would pay
   the cost once.
2. **Optimization-based bounds tightening (OBBT).** Same story, but more
   expensive because it calls a solver; even more valuable to cache.
3. **iter0 solve.** In PH, iter0 solves each scenario / bundle with the
   original objective (W = 0, no prox). That solve is deterministic and
   its result is a natural warm start for every subsequent run. If we
   store the solution in the pickle, downstream PH runs either warm-start
   immediately or skip iter0 entirely.
4. **Model-specific cleanup.** Users sometimes know structural facts
   about their model (dominated constraints, known-tight bounds,
   redundant variables) that they want to apply once, outside of
   `scenario_creator`. Today there is no hook for this at pickle time.

## Current state (as of 2026-04-07)

### Pickling entry points

- `mpisppy/generic/scenario_io.py::write_bundles` — builds each bundle via
  `scenario_creator(bname, **kwargs)` and immediately calls
  `pickle_bundle.dill_pickle(bundle, fname)`.
- `mpisppy/generic/scenario_io.py::write_scenarios` — same pattern for
  individual scenarios.
- Both are dispatched from `mpisppy/generic_cylinders.py` when
  `cfg.pickle_bundles_dir` or `cfg.pickle_scenarios_dir` is set.

There is **no preprocessing hook** between `scenario_creator(...)` and
`dill_pickle(...)` today.

### Existing presolve machinery

`mpisppy/opt/presolve.py` provides:

- `SPPresolve` — the top-level object; bundles FBBT and optional OBBT.
- `SPFBBT` — wraps `pyomo.contrib.appsi.fbbt.IntervalTightener` per
  subproblem.
- `SPOBBT` — wraps `pyomo.contrib.alternative_solutions.obbt.obbt_analysis`
  per subproblem; requires a solver.
- The base `_SPIntervalTightenerBase.presolve()` loop is **distributed-aware**:
  it `Allreduce`s nonant lower/upper bounds across ranks via `opt.comms` so
  that scenarios on different ranks agree on tightened nonant bounds.

Currently invoked at `mpisppy/spopt.py:71` during SPOpt construction when
`options["presolve"]` is truthy — i.e., once per PH/EF run, paid every
time the job starts.

### Reference point: `write_scenario_lp_mps_files_only`

`mpisppy/generic/scenario_io.py::write_scenario_lp_mps_files_only` already
demonstrates the pattern of building an `SPBase` without solving, running
an extension (`Scenario_lp_mps_files.pre_iter0`) over it, and then doing
side-effect work. The pre-pickle preprocessing feature can reuse this
pattern closely.

## Proposed design

### Pipeline

At pickle time, the per-scenario / per-bundle pipeline becomes:

```
scenario_creator(sname)                         # build model (today)
  → SPPresolve (FBBT / optional OBBT)           # cfg.presolve_before_pickle
  → <pre_pickle_function>(model, cfg)           # user callback (if cfg.pre_pickle_function set)
  → solve iter0 (store values + duals)          # cfg.iter0_before_pickle
  → dill_pickle(model)                          # today
```

Each new stage is optional and independently controlled. The ordering is
significant:

- **Presolve before iter0** so the solver benefits from tightened bounds.
- **User callback between presolve and iter0** so users can tweak the
  presolved model (e.g., drop constraints whose redundancy was revealed
  by FBBT) before the solve sees it.
- **All three before dill** so downstream consumers get the fully
  processed model.

### Stage 1 — Reuse `SPPresolve` at pickle time (primary feature)

Modeled on `write_scenario_lp_mps_files_only`:

1. In `write_bundles` / `write_scenarios`, if `cfg.presolve_before_pickle`
   is set, build a transient `SPBase` over the rank-local scenarios
   (instead of directly calling `scenario_creator` one at a time).
2. Run `SPPresolve(sp, cfg.presolve_options).presolve()`.
3. Iterate over `sp.local_scenarios` (or `sp.local_subproblems` for
   bundles) and dill-pickle each one.
4. Release the `SPBase` so the C++ APPSI `_cmodel` memory is freed.

Benefits:

- Reuses all existing FBBT/OBBT code.
- Gets cross-scenario nonant bound synchronization "for free" via
  the MPI `Allreduce` machinery already in `_SPIntervalTightenerBase`.
- Behaves identically to an at-solve-time presolve — the user sees the
  same tightened bounds either way.

Cost:

- SPBase construction is heavier than calling `scenario_creator` once per
  bundle, so pickling time goes up. That is the tradeoff the user opted
  into by setting the flag.

### Stage 2 — User-supplied `pre_pickle_function`

Instead of discovering a fixed-name attribute on the module, the user
names the callback explicitly on the command line:

```
--pre-pickle-function my_module.my_pre_pickle_fn
```

The corresponding `cfg.pre_pickle_function` is a string. If set,
`write_bundles` / `write_scenarios` resolves the dotted name to a
callable and invokes it between the presolve stage and the iter0 stage:

```python
def my_pre_pickle_fn(model, cfg):
    """Called on each scenario or bundle just before it is pickled.
    Free to mutate the model in place. Return value is ignored.
    """
```

Why an explicit command-line flag rather than a magic module-level name?

- It is explicit: the user opts in by naming the function. There is no
  silent action just because a function happens to exist.
- The same module can expose multiple alternative pre-pickle functions
  (e.g., one that fixes a particular set of variables, one that does
  nothing) and the user picks per run.
- The function does not have to live in the model module — a generic
  cleanup utility can sit anywhere on the Python path.

Use cases:

- Apply `pyomo.contrib.preprocessing` transformations the user happens
  to trust for their model (coefficient tightening, redundant constraint
  removal, variable aggregation, zero-term elimination).
- Fix variables to known-tight values.
- Delete dominated constraints identified by domain knowledge.
- Rename / reorganize components for faster downstream access.

Rationale: users know their model. A callback is cheap to add and much
more flexible than trying to enumerate every useful transformation as
a flag.

### Stage 3 — Solve iter0 before pickle

New config flag `cfg.iter0_before_pickle`. When set, after presolve and
the user hook, call the solver on each scenario / bundle with its
original objective (no W, no prox — that is what iter0 means in PH).

Implementation notes:

- Reuse `cfg.solver_name` / `cfg.solver_options` by default. Add an
  optional `cfg.pickle_solver_name` override for users who want (e.g.)
  a fast LP solver at pickle time even though downstream uses a MIP
  solver.
- Use a persistent solver interface when available for speed, but
  *do not* try to serialize solver state — dill-pickle the Pyomo model
  only. Solver-side warm-start (basis, Gurobi WarmStart) stays inside
  the solver process and is lost, as expected.
- Variable `.value`s, and duals / reduced costs loaded into
  `Suffix(direction=IMPORT)`, **do** survive pickling and are what the
  downstream run consumes.
- For bundles, this solves the bundle EF deterministically. Potentially
  heavy. It is exactly what the user asked for: an optimal bundle
  solution baked into the pickle.

### Consuming the iter0 solution downstream (two tiers)

**Tier 1 — warm start only.** No PH-side changes required. PH still runs
iter0 on unpickle, but each subproblem solve starts with pre-populated
variable values. For MIPs this becomes a MIP start and is usually a big
win; for LPs it is less useful without a basis.

**Tier 2 — skip iter0 entirely.** Add `cfg.iter0_from_pickle` honored by
`phbase.py`: when set, PH reads variable values from the unpickled
scenarios, computes xbar directly, performs the first W update, and goes
straight to iter1. Eliminates iter0 wallclock entirely on every rerun.

Recommendation: ship tier 1 first (near-free once the pickle-time solve
hook exists). Tier 2 is a follow-up once tier 1 is validated.

## Configuration additions

Working names:

| Flag                               | Domain | Default | Purpose                                                      |
| ---------------------------------- | ------ | ------- | ------------------------------------------------------------ |
| `cfg.presolve_before_pickle`       | bool   | False   | Run `SPPresolve` during `write_bundles` / `write_scenarios`. |
| `cfg.pre_pickle_function`          | str    | None    | Dotted name of a `(model, cfg)` callback to invoke between presolve and iter0. |
| `cfg.iter0_before_pickle`          | bool   | False   | Solve each scenario / bundle before pickling.                |
| `cfg.pickle_solver_name`           | str    | None    | Optional pickle-time solver override.                        |
| `cfg.pickle_solver_options`        | dict   | None    | Optional pickle-time solver options override.                |
| `cfg.iter0_from_pickle`            | bool   | False   | (Tier 2) PH skips iter0 and uses pickled values as its iter0 result. |

Existing `cfg.presolve_options` is reused by the pickle-time presolve
(same options, same code path).

Alternative naming direction if you prefer one umbrella flag:
`cfg.pickle_with_preprocessing` with sub-options. Flat flags are probably
easier to reason about.

## Resolved design decisions

These items were originally open questions; the resolutions below are
authoritative for the v1 implementation.

### 1. Scope of the first iteration

**Decision: ship all three stages together.** Doing the pipeline in one
pass lets us design the ordering, error handling, and metadata story
once instead of unpicking it across follow-up PRs. Stage 1 (presolve),
Stage 2 (`pre_pickle_function`), and Stage 3 tier 1 (iter0 solve with
warm-start consumption) all land together. Tier 2 (PH-side iter0 skip)
remains a separate follow-up because it touches `phbase.py` rather than
the pickle pipeline.

### 2. Presolve granularity for bundles

**Decision: run presolve on the bundle (bundled EF).** That is what
`SPPresolve` already does over `local_subproblems`, so we get
intra-bundle propagation for free and avoid carrying a second code
path. Per-scenario presolve before bundling is not supported in v1.

### 3. OBBT at pickle time?

**Decision: documentation only.** OBBT is already gated behind its own
flag (`cfg.obbt`) inside `presolve_options`, so the existing machinery
already protects users with no solver installed. The pickling
documentation will call out explicitly that turning on OBBT at pickle
time introduces a solver dependency on the pickling job.

### 4. Symmetry between `write_scenarios` and `write_bundles`

**Decision: both paths accept the same pre-pickle flags.** No
asymmetry. The shared helper `_run_pre_pickle_pipeline(sp, cfg)` is the
single point where presolve / `pre_pickle_function` / iter0 run, and
both writers call it.

### 5. Failure handling for iter0

**Decision: if iter0 fails, shut down.** No "pickle anyway with a
warning" fallback. Pickling silently bad state would be worse than the
job stopping. The previously-proposed `cfg.allow_unsolved_pickle` flag
is dropped from the config table. The pickling driver propagates the
error and exits non-zero so a tuning workflow does not silently produce
corrupt pickles.

### 6. Duals and suffixes

**Decision: attach `IMPORT` suffixes for duals (and reduced costs)
before the iter0 solve, and document this behavior prominently.** Dual
values are then carried inside the pickle for downstream lagrangian /
lagranger / fixer consumers. Documentation in `pickling.rst` will state
clearly that pickles produced with `--iter0-before-pickle` carry duals
on a Pyomo `IMPORT` suffix.

### 7. Solver choice: pickle time vs. run time

**Decision: document, leave the choice to the user.** Pickling with an
LP-only solver and running with a MIP solver is allowed; the LP-relaxed
variable values are a fine warm start for many MIPs but not all. The
documentation will spell out the tradeoff and the cases where it can
hurt (notably integer-feasibility-sensitive solvers).

### 8. Interaction with ADMM wrappers

**Decision: this is an important case to test.** The pipeline runs on
the wrapped (ADMM) model that gets pickled, so presolve and iter0
operate on the model with consensus constraints and the
variable-probability adjustments already in place. Test plan: at least
one test under `test_pickle_bundle.py` exercises
`ProperBundler` → `admm` wrapper → pickle with both
`--presolve-before-pickle` and `--iter0-before-pickle` set, and
verifies that the unpickled model still solves to the same objective.

### 9. Interaction with unpickling

**Decision: tier 2 detection uses a flag on `_mpisppy_data`.** The
existing `proper_bundler.scenario_creator` unpickle path needs no
changes for tier 1 — variable values and tightened bounds are model
state and survive pickling automatically. For tier 2, the pickling
pipeline will set `model._mpisppy_data.iter0_already_done = True`
(rather than attaching a new attribute directly to the model). The
`_mpisppy_data` object is already attached by the time the pre-pickle
pipeline runs, so this avoids polluting the model namespace and keeps
mpisppy state in one place. `phbase.py` checks the same attribute on
unpickle and skips its iter0 solve when it is set.

### 10. File format / metadata

**Decision: attach metadata to the model via `_mpisppy_data`.** No
sidecar JSON. The metadata travels inside the pickle automatically and
survives file moves. Recorded fields include: which preprocessing
stages ran, the presolve options used, the solver name (and override,
if any), and a small version tag so future incompatible changes can be
detected on load. Concretely:
`model._mpisppy_data.pickle_metadata = {...}`.

## Implementation sketch

### Files to modify

- `mpisppy/generic/scenario_io.py`
  - `write_bundles`: optional SPBase build, optional presolve, optional
    `pre_pickle_function` invocation, optional iter0 solve, then pickle
    from `sp.local_subproblems`.
  - `write_scenarios`: symmetric.
  - New helper `_run_pre_pickle_pipeline(sp, cfg)` shared by both. The
    helper resolves `cfg.pre_pickle_function` (a dotted name) to a
    callable via `importlib`, validates it is callable, and invokes
    `fn(model, cfg)` per scenario / bundle.
- `mpisppy/utils/config.py`
  - Add the new flags in a new grouping method `pre_pickle_args` (or
    extend `pickle_scenarios_config` and `proper_bundle_config`):
    `presolve_before_pickle`, `pre_pickle_function`,
    `iter0_before_pickle`, `pickle_solver_name`, `pickle_solver_options`,
    `iter0_from_pickle` (tier 2).
- `mpisppy/generic_cylinders.py`
  - Ensure the bundle-wrapper / ADMM-wrapper path is still correct with
    the new pipeline. Add the ADMM + pickle test case.
- `mpisppy/phbase.py` (tier 2 only)
  - Detect `model._mpisppy_data.iter0_already_done` and branch the
    iter0 logic.
- `doc/src/pickling.rst`
  - New page; primary documentation for all pickling features (see
    documentation plan below).
- `doc/src/generic_cylinders.rst`, `doc/src/properbundles.rst`
  - Trim pickling content to a cross-reference to `pickling.rst`.
- Tests:
  - `mpisppy/tests/test_pickle_bundle.py` — add cases exercising each
    new flag independently and in combination, plus the ADMM + pickle
    case (resolved decision #8).
  - Farmer is probably the right smoke-test model; aircond for the
    multi-stage bundle path.

### Rough task breakdown

Per resolved decision #1, the v1 PR contains all three stages:

1. Config wiring for the new flags, including
   `cfg.pre_pickle_function`.
2. `_run_pre_pickle_pipeline` helper in `scenario_io.py` that runs
   presolve → user callback → iter0 solve, with hard-fail on iter0
   failure (resolved decision #5).
3. Dual / reduced-cost suffix attachment before iter0
   (resolved decision #6) and pickle metadata on `_mpisppy_data`
   (resolved decision #10).
4. ADMM + bundle + pickle test case (resolved decision #8) and
   per-flag tests on farmer / aircond.
5. Documentation: write `doc/src/pickling.rst` (new page) and trim the
   existing pickling sections in `generic_cylinders.rst` and
   `properbundles.rst` to cross-references. The new page must cover
   the tuning workflow and the "pickle iter0 to use all CPUs" angle.
6. Stage 3 tier 2 (PH-side iter0 skip via
   `_mpisppy_data.iter0_already_done`) as a separate follow-up PR.

## Notes and references

- Current presolve entry point: `mpisppy/spopt.py:64-71`.
- Current pickle entry points: `mpisppy/generic/scenario_io.py:75`
  (`write_bundles`), `mpisppy/generic/scenario_io.py:19`
  (`write_scenarios`).
- Existing no-solve SPBase construction pattern to copy:
  `mpisppy/generic/scenario_io.py:121` (`write_scenario_lp_mps_files_only`).
- Proper-bundle unpickle path:
  `mpisppy/utils/proper_bundler.py:108`.
- Existing `SPPresolve` API: `mpisppy/opt/presolve.py:491`.
