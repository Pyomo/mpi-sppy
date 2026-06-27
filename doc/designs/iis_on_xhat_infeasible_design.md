# Design: Emit an IIS when an xhatter hits an infeasible scenario

**Status:** Draft — up for discussion. Nothing is implemented yet.
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-06-14
**Resolves:** [#356](https://github.com/Pyomo/mpi-sppy/issues/356) — "Add IIS
emission to incumbent-finders" (jeanpaulwatson)

**Locked decisions:** enabling flag `--xhatter-write-iis`; default method
`auto`; emit for exactly **one** infeasible scenario per cylinder; the
feature gets its **own `doc/src/iis.rst`**, with the option help text
referencing it while staying fairly complete on its own.

## Motivation

An xhatter (incumbent-finder) fixes the first-stage (non-anticipative)
variables at a candidate `xhat` and solves every scenario subproblem. If
any scenario is infeasible at that `xhat`, the candidate is rejected and
the xhatter moves on — silently. That silence is exactly right during
normal operation: candidate rejection is a routine event and the search
recovers on its own.

But it is the *wrong* behavior when the model is supposed to have
(relatively) complete recourse and doesn't. As the issue puts it: this
"comes up a lot, e.g., for models that *should* have relatively complete
recourse (but of course don't, due to some weird model subtlety or data
issue)." When that happens the run can spin for a long time finding *no*
incumbent, with no diagnostic to explain why.

Pyomo ships `pyomo.contrib.iis`, which can compute an Irreducible
Infeasible Set (IIS) — the minimal conflicting set of constraints and
variable bounds — for an infeasible model. Surfacing that for the first
infeasible xhat subproblem turns a frustrating non-result into an
actionable answer: *"with these first-stage decisions, scenario X is
infeasible because of this handful of constraints."*

## What the user asked for

1. An **option** on the xhatters to **automatically write the IIS** when
   an xhatter gets an infeasible solution.
2. It should run **just once**, and that must be **clearly documented**.
   (Rationale: IIS computation is expensive — `write_iis` invokes the
   solver's IIS engine; `compute_infeasibility_explanation` does many
   relaxation solves. Doing it on every rejected candidate, every
   iteration, would be ruinous.)
3. **Output file naming should more-or-less match the solver-log-file
   naming convention** (per follow-up guidance).

## Non-goals

- IIS for routine PH/APH subproblem infeasibility. This feature is
  scoped to xhatters (incumbent-finders) only. Routine subproblem
  infeasibility during the main algorithm is a separate concern and is
  *not* in scope.
- Repeated / per-iteration IIS. By construction this fires at most once
  per cylinder (see §"Run-once semantics").
- Automatically repairing the model. We only diagnose.
- A new IIS algorithm. We use `pyomo.contrib.iis` as-is.

## Background: the xhat infeasibility-detection points

Every xhatter — both the **extensions** that run inside the PH hub
(`xhatlooper`, `xhatspecific`, `xhatxbar`, `xhatclosest`) and the
**spokes/cylinders** that own an `Xhat_Eval` (`xhatshufflelooper_bounder`,
`xhatlooper_bounder`, `xhatspecific_bounder`, `xhatxbar_bounder`) — fixes
nonants, calls `solve_loop(..., need_solution=False)` so infeasible solves
don't raise, and then tests for infeasibility. There are a small number of
distinct detection sites, and they all funnel through one object: the
`SPOpt` subclass at `self.opt` (a `PHBase` for extensions, an `Xhat_Eval`
for spokes).

| Site | File / method | Infeasibility test |
|---|---|---|
| Two-stage & regular multi-stage | `extensions/xhatbase.py :: XhatBase._try_one` | `self.opt.no_incumbent_prob() != 0` |
| Multi-stage stage2-EF | `extensions/xhatbase.py :: XhatBase._try_one` (stage2ef branch) | `not pyo.check_optimal_termination(results)` |
| Jensen's / feasible-xhat pre-loop | `cylinders/_preloop_xhat_mixin.py :: _PreLoopXhatMixin._evaluate_xhat` | `self.opt.no_incumbent_prob() != 0` |
| Steady-state incumbent eval | `utils/xhat_eval.py :: Xhat_Eval.calculate_incumbent` | `self.opt.no_incumbent_prob() != 0` |

After a `need_solution=False` solve, each infeasible local subproblem `s`
has `s._mpisppy_data.solution_available == False` (set in
`spopt.py::solve_one`). The nonants are still **fixed at the xhat values**
at the moment of detection — restore happens *after* — so the model is
sitting in exactly the infeasible configuration we want to explain.

The solver-log filename we are asked to mirror is built in
`spopt.py::solve_one`:

```python
file_name = f"{self._get_cylinder_name()}_{k}_{self._subproblem_solve_index[k]}.log"
# written under self.options["solver_log_dir"]
```

## Design overview

Add **one method on `SPOpt`** that does the IIS emission, and **call it
from the xhat infeasibility-detection sites** listed above. Centralizing
the emission keeps the IIS logic (option check, run-once guard, filename,
fail-soft) in one place; calling it explicitly from the xhat sites scopes
it to xhatters *by construction* — the PH/APH main loop never calls it, so
routine subproblem infeasibility is untouched. No global "armed" flag, no
risk of leaking IIS behavior into the main algorithm.

We deliberately do **not** put the trigger inside `solve_one` itself.
`solve_one` is shared with the PH/APH main loop, where infeasibility is
routine; gating it there would require a mutable "this is an xhat solve"
flag toggled around every xhat `solve_loop`, which is fragile state with
several arm/disarm sites. The explicit-call approach is simpler and
self-documenting. We pay a small price — re-deriving the filename stem
instead of reusing the one `solve_one` already built — which we address by
factoring a tiny shared stem helper (see §"File naming").

### The method (sketch)

Lives on `SPOpt` (`mpisppy/spopt.py`), so every xhatter's `self.opt` has
it:

```python
def write_iis_on_xhatter_infeasible(self, model=None, label=None):
    """Emit an IIS for an infeasible xhatter subproblem, at most once.

    No-op unless options['xhatter_write_iis'] is set. Fires at most once
    per cylinder (per MPI rank); a guard prevents any repeat. Never
    raises: any failure is reported and the guard is set anyway so the
    run is not perturbed and we do not retry.

    model/label: optionally target a specific model (e.g. a stage2-EF
    block). When omitted, the first infeasible local scenario is used.
    """
    if not self.options.get("xhatter_write_iis", False):
        return
    if getattr(self, "_xhatter_iis_written", False):
        return

    # Pick the target: explicit model, else first infeasible local sub.
    if model is None:
        target = next(((k, s) for k, s in self.local_scenarios.items()
                       if not s._mpisppy_data.solution_available), None)
        if target is None:
            return                      # nothing infeasible here on this rank
        label, model = target

    self._xhatter_iis_written = True    # set BEFORE the work: run-once even on error
    try:
        self._emit_iis(model, label)    # write_iis / compute_infeasibility_explanation
    except Exception as e:              # never perturb the run
        print(f"[{self._get_cylinder_name()}] IIS emission for {label} "
              f"failed: {e}")
```

`_xhatter_iis_written` is a per-object (hence per-rank) attribute, lazily
defaulted via `getattr`. Setting it *before* the emission work guarantees
"just once" even if `_emit_iis` throws — we never retry, and one bad IIS
attempt cannot turn into a per-iteration storm.

### Call sites

Each detection site gains one line, placed **before** any
`_restore_nonants()` (so the model still carries the fixed-nonant,
infeasible configuration):

- `XhatBase._try_one`, two-stage/multi-stage branch — right where
  `infeasP != 0` is detected, before the restore.
- `XhatBase._try_one`, stage2-EF branch — before its restore; passes the
  EF block explicitly: `self.opt.write_iis_on_xhatter_infeasible(model=self._EFs[ndn2], label=ndn2)`.
- `_PreLoopXhatMixin._evaluate_xhat` — where `no_incumbent_prob() != 0`.
- `Xhat_Eval.calculate_incumbent` — where `infeasP != 0`, before
  returning `None`.

All four are one-liners calling the same method on `self.opt`.

## Which Pyomo IIS facility

`pyomo.contrib.iis` offers two relevant entry points:

| Function | Output | Solver requirement |
|---|---|---|
| `write_iis(model, iis_file_name, solver=None, logger=...)` | a standard `.ilp` IIS file | a **commercial** solver with a native IIS engine: cplex, gurobi, or xpress |
| `compute_infeasibility_explanation(model, solver, tee=False, tolerance=1e-8, logger=...)` | a textual explanation **to a logger** (minimal infeasible system) | **any** Pyomo solver (uses constraint relaxation) |

The issue explicitly references `contrib.iis` and "write the IIS files,"
and mpi-sppy already requires a commercial MIP solver, so **`write_iis`
(`.ilp`) is the primary path.** `compute_infeasibility_explanation` is the
universal fallback — important for open-solver users and for CI, where a
commercial solver may not be present (it lets us test the feature without
cplex/gurobi/xpress).

Selection is controlled by `xhatter_iis_method` (see §Config):

- `ilp` — `write_iis` only; error (fail-soft) if no commercial solver.
- `explanation` — `compute_infeasibility_explanation` only.
- `auto` (default) — `write_iis` when the configured solver is
  cplex/gurobi/xpress, else `compute_infeasibility_explanation`.

The solver name passed to `write_iis` is derived from `solver_name` by
stripping the `_persistent` / `_direct` suffix (e.g. `gurobi_persistent`
→ `gurobi`), since the IIS engine wants the base solver. For
`compute_infeasibility_explanation`, we attach a `FileHandler` to the
`pyomo.contrib.iis` logger pointed at our chosen filename so the textual
explanation lands in a file rather than scrolling past in the console.

## File naming

We mirror the solver-log convention
(`{cylinder}_{scenario}_{index}.log`). Because the feature runs once,
the per-solve index is meaningless, so we drop it:

- `write_iis` path: `{cylinder_name}_{label}.ilp`
- `compute_infeasibility_explanation` path: `{cylinder_name}_{label}.iis.log`

where `cylinder_name` is `self._get_cylinder_name()` and `label` is the
scenario name (or the stage2-EF node name). To keep the two conventions
honestly in sync, factor the stem out of `solve_one`:

```python
def _subproblem_file_stem(self, k):
    return f"{self._get_cylinder_name()}_{k}"
```

`solve_one` then builds `f"{stem}_{idx}.log"`; the IIS path builds
`f"{stem}.ilp"`. Files are written under `xhatter_iis_dir` (default:
current working directory).

In a parallel run each rank owns different scenarios, so naming by
scenario means concurrent writers never collide.

## Run-once semantics (and how it's documented)

"Just once" is per **cylinder**, i.e. per **MPI rank**, enforced by the
`_xhatter_iis_written` guard on the opt object:

- The **first** time *this* xhatter detects an infeasible xhat
  subproblem, it emits an IIS for the **first infeasible local
  scenario** on this rank, then sets the guard. No further IIS is emitted
  by this cylinder for the rest of the run.
- In a multi-rank run, ranks have independent opt objects, so you may get
  **up to one IIS file per rank** that encountered an infeasible local
  scenario — each named by its offending scenario. This is the natural
  reading of "once" applied per worker, and it is what makes the feature
  useful in parallel (the infeasible scenario may not live on rank 0).

This will be stated prominently in the user docs and in the option's help
text. We considered a globally-once variant (an `Allreduce` to elect a
single rank to write) but rejected it: it adds a collective on an error
path, and it would suppress the diagnostic on exactly the rank that owns
the offending scenario. Per-rank-once is simpler and strictly more
informative.

An alternative — emit for *all* infeasible local scenarios in the first
triggering pass — is noted as a possible future knob
(`xhatter_iis_all_infeasible`), but the default stays "one," matching the
literal "just once" request and bounding cost.

## Configuration surface

Declared in `Config.popular_args()` (alongside `solver_log_dir`), so they
ride the existing global-option plumbing, and forwarded to spokes/hub in
`cfg_vanilla.shared_options` exactly as `solver_log_dir` is (gated the
same way — see `cfg_vanilla.py`). This is the proven path: the option
shows up in `self.opt.options` for every cylinder.

| Config key | CLI | Domain | Default | Meaning |
|---|---|---|---|---|
| `xhatter_write_iis` | `--xhatter-write-iis` | bool | `False` | Enable IIS emission on the first xhatter infeasibility. |
| `xhatter_iis_method` | `--xhatter-iis-method` | str (`auto`/`ilp`/`explanation`) | `auto` | Which IIS facility to use. |
| `xhatter_iis_dir` | `--xhatter-iis-dir` | str | `None` → cwd | Output directory for IIS files. |

The help text for `--xhatter-write-iis` is the most-read documentation of
the feature, so it must be fairly complete on its own — state that it
fires **at most once per cylinder (per MPI rank)**, name the default
method, and point to `doc/src/iis.rst` for the full story. A fourth knob,
`xhatter_iis_solver` (to override the IIS solver independently of
`solver_name`), is deferred unless wanted — `auto` derivation from
`solver_name` covers the common case.

## Testing

Mirror `mpisppy/tests/test_incumbent_writing.py`, which tests the three
surfaces of a similar feature without standing up MPI:

1. **Config registration** — `Config.popular_args()` registers
   `xhatter_write_iis` / `xhatter_iis_method` / `xhatter_iis_dir` with the
   right defaults.
2. **cfg_vanilla forwarding** — `shared_options(cfg)` forwards the keys
   into the options dict.
3. **The `SPOpt` method** — exercise `write_iis_on_xhatter_infeasible`
   against a tiny stub opt holding a deliberately infeasible Pyomo model
   (two constraints `x >= 1`, `x <= 0`), across the branches:
   - disabled (option off) → no file, guard untouched;
   - enabled, `explanation` method → file written (works with **any**
     solver, so CI without a commercial solver still covers this path);
   - enabled, called twice → exactly one emission (guard holds);
   - `_emit_iis` raises → fail-soft, guard still set, run continues.
   The `ilp` path is guarded by `tests/utils.get_solver()` and only runs
   when cplex/gurobi/xpress is available.

Per project convention, the new `mpisppy/tests/test_*.py` file must be
added to **both** `run_coverage.bash` and `.github/workflows/`
(`test_pr_and_main.yml`) in the same commit, or codecov/patch reports 0%.

An optional end-to-end check reuses `--xhat-from-file` (the file-supplied
xhat feature) to feed a known-infeasible first-stage vector to a model
with incomplete recourse and assert the `.ilp`/`.iis.log` appears exactly
once — a natural pairing already noted in
`doc/designs/xhat_from_file_design.md`.

## Documentation

The feature gets its **own dedicated page**, `doc/src/iis.rst`, since the
run-once semantics need real explanation and a casual reader will
otherwise be surprised by them. The page covers:

- **What an IIS is** and why an xhatter would want one (the
  complete-recourse-that-isn't story from the Motivation), with a pointer
  to `pyomo.contrib.iis`.
- **The three options** (`--xhatter-write-iis`, `--xhatter-iis-method`,
  `--xhatter-iis-dir`) with their defaults and the `auto`/`ilp`/
  `explanation` method semantics, including the commercial-solver
  requirement for `ilp` and the universal `explanation` fallback.
- **Run-once semantics, prominently** — at most once per cylinder (per
  MPI rank); in a parallel run, up to one file per rank that hit an
  infeasible local scenario, each named by its scenario; *why* it is
  bounded (cost) and why per-rank (the bad scenario may not be on rank 0).
- **Output file naming**, related to the `--solver-log-dir` convention.
- A short worked example (e.g. pairing with `--xhat-from-file` to feed a
  known-infeasible first-stage vector).

Add the page to `doc/src/index.rst`. Cross-reference it from
`doc/src/generic_cylinders.rst` near `--solver-log-dir`, and from the
related features it sits beside (`feasible_xhat.rst`,
`xhat_from_file.rst`).

The **option help text** is the most-read documentation of all, so it
must be fairly complete on its own — not a bare one-liner. Each of the
three options' `description=` strings states the essential behavior (the
`--xhatter-write-iis` help names the run-once-per-cylinder rule and the
default method) and then references `doc/src/iis.rst` for the full
treatment.

## Files touched (estimate)

- `mpisppy/utils/config.py` — declare the three options in
  `popular_args()`.
- `mpisppy/utils/cfg_vanilla.py` — forward them in `shared_options`.
- `mpisppy/spopt.py` — `write_iis_on_xhatter_infeasible`, `_emit_iis`,
  `_subproblem_file_stem`; use the stem in `solve_one`.
- `mpisppy/extensions/xhatbase.py` — two call sites in `_try_one`.
- `mpisppy/cylinders/_preloop_xhat_mixin.py` — one call site.
- `mpisppy/utils/xhat_eval.py` — one call site in `calculate_incumbent`.
- `mpisppy/tests/test_iis_on_infeasible.py` (new) + `run_coverage.bash` +
  `.github/workflows/test_pr_and_main.yml`.
- `doc/src/iis.rst` (new), `doc/src/index.rst`,
  `doc/src/generic_cylinders.rst`.

Small enough for a **single PR**; if reviewers prefer, it splits cleanly
into (1) config + plumbing + `SPOpt` method + tests, then (2) wiring the
call sites + docs.

## Decisions (locked)

1. **Enabling-flag name** — `--xhatter-write-iis`.
2. **Default method** — `auto` (`ilp` when the solver is
   cplex/gurobi/xpress, else `explanation`).
3. **One vs all infeasible local scenarios** in the first pass — **one**
   (matches "just once"); `xhatter_iis_all_infeasible` remains a possible
   future knob.
4. **Run-once scope** — per-cylinder (per MPI rank).
5. **Documentation** — dedicated `doc/src/iis.rst`; option help text
   references it but is fairly complete on its own.
