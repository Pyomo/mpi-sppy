# ADMM user-API automation — design

Status: draft, awaiting review.

Related: ongoing ADMM doc/usability work on the `admm-doc-crossref`
branch (cross-references to `generic_cylinders`, BFs flow-through,
xhatshuffle stage2ef hard error, `attach_root_node` contract
clarification).

---

## 0. Goals

Reduce the boilerplate and foot-guns a user must navigate when writing
an ADMM model module for `generic_cylinders --admm` or `--stoch-admm`.
Today the user has to:

1. Call `sputils.attach_root_node` in their `scenario_creator` — but
   only for `--stoch-admm`; for `--admm` the wrapper overwrites the
   call.  Asymmetric and easy to get wrong (#4 in our recent chat).
2. Stringly-type variable names in `consensus_vars_creator` so they
   exactly match `var.name`; typos / indexed-var formatting differences
   fail with cryptic errors.
3. Hand-write a `combining_names` / `split_admm_stoch_subproblem_scenario_name`
   inverse pair, with a documented underscore-collision warning.
4. Write `admm_stoch_subproblem_scenario_names_creator` and remember to
   nest the loops correctly (outer = stoch, inner = admm) so MPI rank
   assignment works.
5. Register `num_admm_subproblems` / `num_stoch_scens` in `inparser_adder`
   (already partially automated, but examples still register them).

Non-goals:

- Merging `AdmmWrapper` and `Stoch_AdmmWrapper` (item F; explicitly
  out of scope).
- Performance changes.
- Multistage-origin stoch-admm.  Phase A targets 2-stage origin only.
  Multistage origin is not currently exercised by any example or test
  (every caller passes `BFs=None`); extending the automation there
  belongs in a follow-up if/when a real multistage-origin model
  appears.

---

## 1. Phased plan

Five phases (A–E), each independently shippable, each its own
review-sized PR.  Each PR must:

- Keep the existing string-typed / hand-written API working (the old
  path is the back-compat fallback).
- Migrate the canonical example (`examples/distr/distr.py` for `--admm`,
  `examples/stoch_distr/stoch_distr.py` for `--stoch-admm`) to the
  new API so the doc has something to point at.
- Add a unit test exercising both the old and the new path.
- Update `doc/src/generic_admm.rst`.

---

## 2. Phase A — auto-build the scenario tree for `--stoch-admm`

### Today's contract

- `--admm`: `AdmmWrapper.assign_variable_probs` calls
  `sputils.attach_root_node(s, objfunc, varlist)` at
  `mpisppy/utils/admmWrapper.py:148`, overwriting any prior call.  User
  must not call it; if they do, it's silently a no-op.
- `--stoch-admm`: `Stoch_AdmmWrapper` reads
  `s._mpisppy_node_list[-1]` at `mpisppy/utils/stoch_admmWrapper.py:236`
  and appends an ADMM-consensus node.  User MUST call `attach_root_node`
  with the original problem's first-stage cost + varlist or the
  wrapper assertion-fails.

### Proposed

Add two optional module-level functions for `--stoch-admm`:

```python
def first_stage_cost(scenario):
    """Returns the Pyomo Expression for the original problem's
    first-stage cost (e.g., scenario.FirstStageCost)."""

def first_stage_varlist(scenario):
    """Returns the list of first-stage Pyomo Vars / VarDatas (e.g.,
    [scenario.y[n] for n in factory_nodes])."""
```

**Strict both-or-neither contract.**  `setup_stoch_admm` inspects the
module for both attributes.  If exactly one is present, raise
`RuntimeError` naming the missing function.  This catches users who
half-migrate.

When both are defined, `Stoch_AdmmWrapper` calls
`sputils.attach_root_node` itself for each local scenario *before*
`assign_variable_probs` runs (so the existing append-an-ADMM-stage
logic still works on top of the wrapper-attached node).

**Per-scenario error matrix** (checked after `scenario_creator`
returns, before the wrapper appends the ADMM stage):

| | Both hooks defined | Neither hook defined |
|---|---|---|
| `s._mpisppy_node_list` is set (user called `attach_root_node`) | **`RuntimeError`** — "hooks make `attach_root_node` redundant; remove the call from `scenario_creator`" | OK — legacy path; wrapper appends ADMM stage on top of the user's node list |
| `s._mpisppy_node_list` is **not** set | OK — wrapper attaches root via the hooks | **`RuntimeError`** — "scenario is missing `_mpisppy_node_list`. Either call `sputils.attach_root_node` in `scenario_creator`, or define `first_stage_cost` and `first_stage_varlist` module functions (recommended)" |

The check fires per-scenario, not just once, so inconsistent branching
in `scenario_creator` (e.g., calling `attach_root_node` only on some
scenario names) is also caught.  Both errors fire at
scenario-construction time, before `assign_variable_probs`, so users
get a clear message instead of a cryptic deep assertion.

The "both hooks + user call → error" choice is deliberately strict:
the wrapper's call would silently overwrite the user's, but erroring
instead teaches the user that the hooks obviate manual attachment.

### Files

- `mpisppy/utils/stoch_admmWrapper.py`: in `__init__`, after
  `scenario_creator` returns, apply the error matrix and (if hooks
  supplied) call `attach_root_node`.  Add `first_stage_cost` and
  `first_stage_varlist` parameters with both-or-neither validation.
- `mpisppy/utils/admm_bundler.py`: same hook plumbing for the
  bundled path (`AdmmBundler` constructs scenarios in its own
  `scenario_creator` method and runs the same error matrix
  per-scenario).  This was missed in the original design and only
  surfaced when the bundled-admm tests started failing during
  implementation.
- `mpisppy/generic/admm.py:setup_stoch_admm`: pass module hooks
  through.  Both-or-neither check at module level (so the error
  message can name the user's module).
- `mpisppy/generic/admm.py:setup_stoch_admm_with_bundles`: same
  hook-discovery and forwarding as the non-bundled path.
- `examples/stoch_distr/stoch_distr.py`: migrate.  Stash the
  first-stage varlist on the model in `scenario_creator` (so the
  hook can find it); remove the `attach_root_node` call; add the
  two new hook functions.  **Also refactor
  `consensus_vars_creator`**: it used to walk
  `model._mpisppy_node_list` to derive first-stage nonants, which
  now doesn't exist at that call site (the wrapper hasn't attached
  the root yet).  Read first-stage vars via `first_stage_varlist()`
  directly instead.
- `examples/stoch_distr/stoch_distr_admm_cylinders.py`: the custom
  driver constructs `Stoch_AdmmWrapper` directly; update it to
  forward the module's hooks so `stoch_distr_ef.py` (which goes
  through this driver) continues to work after the example
  migration.
- `doc/src/generic_admm.rst`: rewrite the "Extending to Stochastic
  ADMM" note around the new API; preserve the old `attach_root_node`
  call as a "legacy path" subsection.
- `mpisppy/tests/test_stoch_admmWrapper.py`: a new test class
  exercising both paths (hooks + legacy) and all four error-matrix
  cells, plus the `setup_stoch_admm`-level half-hooks error.

### Resolved

- **Hook signature: scenario form** (`first_stage_cost(scenario)` /
  `first_stage_varlist(scenario)`).  The function gets the constructed
  scenario and reads attributes directly.  Phase B will fix the
  parallel string-vs-Var issue in `consensus_vars_creator`.
- Naming: `first_stage_cost` / `first_stage_varlist`, matching
  `attach_root_node`'s parameter names (`firstobj`, `varlist`).

---

## 3. Phase B — accept Pyomo Var/VarData in `consensus_vars_creator`

### Today's contract

- `--admm`: `consensus_vars_creator` returns
  `{subproblem_name: [varname_str, ...]}`.  Wrapper calls
  `s.find_component(varname_str)`.
- `--stoch-admm`: returns `{subproblem_name: [(varname_str, stage), ...]}`.

Mismatches (e.g., `"flow[(DC1, DC2)]"` vs `"flow['DC1', 'DC2']"`) fail
with a list-of-pairs RuntimeError deep in `assign_variable_probs`.

### Proposed

Accept Pyomo Var / VarData objects *or* strings in the lists.  At wrapper
construction, normalize each item to a string via `obj.name` before the
existing `find_component` path.  Mixed lists OK (e.g., for migration).

Caveat: `consensus_vars_creator` for `--stoch-admm` is called *once*
with one stochastic scenario.  Vars passed in are tied to that
scenario instance; the wrapper still has to look them up by name in
the other scenarios via `find_component`.  The user benefit is "no
manual name formatting", not "skip the find_component step".

### Files

- `mpisppy/utils/admmWrapper.py:assign_variable_probs`: pre-normalize
  the lists.
- `mpisppy/utils/stoch_admmWrapper.py:assign_variable_probs`: same.
- Examples: migrate `consensus_vars_creator` in `distr.py` and
  `stoch_distr.py` to return Vars.
- Doc.
- Test: a unit test in `test_admm_bundler.py` or a new file that
  constructs a small model and verifies a Var-based
  `consensus_vars_creator` produces the same `varprob_dict` as the
  string-based form.

### Open questions

None significant.

---

## 4. Phase C — default `combining_names` / `split_admm_stoch_subproblem_scenario_name`

### Today's contract

User-defined inverse pair, e.g. from `stoch_distr.py`:

```python
ADMM_STOCH_PREFIX = "ADMM_STOCH_"

def combining_names(admm_sub, stoch_scen):
    return f"{ADMM_STOCH_PREFIX}{admm_sub}_{stoch_scen}"

def split_admm_stoch_subproblem_scenario_name(name):
    parts = name.split("_")
    return parts[2], parts[3]
```

`split_...` is fragile if any subproblem or stoch-scenario name
contains an underscore (today's doc has a Warning about this).

### Proposed

Provide a default pair in `mpisppy.utils.stoch_admmWrapper`:

```python
_DEFAULT_DELIM = "__ADMM__"   # verbose but unambiguous; safe for shells,
                              # filenames, and CSV writers

def default_combining_names(admm_sub, stoch_scen):
    return f"ADMM_STOCH{_DEFAULT_DELIM}{admm_sub}{_DEFAULT_DELIM}{stoch_scen}"

def default_split_admm_stoch_subproblem_scenario_name(name):
    parts = name.split(_DEFAULT_DELIM)
    return parts[1], parts[2]
```

In `setup_stoch_admm`, if the module does not define `combining_names`
*and* `split_admm_stoch_subproblem_scenario_name`, use the defaults.
If the module defines one, it must define the other (raise a clear
error otherwise — we cannot pair a default split with a custom combine
or vice versa).

### Files

- `mpisppy/utils/stoch_admmWrapper.py`: add defaults; export them.
- `mpisppy/generic/admm.py:setup_stoch_admm`: fallback logic.
- Examples: optionally drop the user-defined pair to demonstrate the
  default; leave at least one example using a custom pair to show how.
- Doc: move the underscore-collision warning into a "Customizing
  the naming convention" subsection; default-path users no longer
  hit it.
- Test: unit test that with no module-defined pair, the wrapper builds
  the same scenarios via the default pair as via the explicit pair.

### Resolved

- Default delimiter: `__ADMM__` (verbose but unambiguous; safe for
  shells, filenames, and CSV writers).

---

## 5. Phase D — default `admm_stoch_subproblem_scenario_names_creator`

### Today's contract

User function with required nesting order:

```python
def admm_stoch_subproblem_scenario_names_creator(admm_subproblem_names,
                                                  stoch_scenario_names):
    return [combining_names(sub, stoch)
            for stoch in stoch_scenario_names   # outer
            for sub in admm_subproblem_names]   # inner
```

If the user nests them backwards, MPI rank assignment goes wrong.

### Proposed

If the module does not define this function, use a default in
`mpisppy.utils.stoch_admmWrapper` that uses the module's
`combining_names` (or the phase-C default).

### Files

- `mpisppy/utils/stoch_admmWrapper.py`: add default.
- `mpisppy/generic/admm.py:setup_stoch_admm`: fallback logic.
- Examples: drop the user-defined version where appropriate.
- Doc.
- Test.

### Open questions

None.

---

## 6. Phase E — `inparser_adder` boilerplate

### Today's state

`mpisppy/generic/admm.py:admm_args` already registers `admm`,
`stoch_admm`, `num_admm_subproblems`, `num_stoch_scens`,
`branching_factors`, `stage2_ef_solver_name`, with `if not already
present` guards.  Example modules' `inparser_adder` typically also
register `num_admm_subproblems` / `num_stoch_scens`; the guards prevent
double-registration.

### Proposed

- Document `admm_args` as the canonical source for ADMM-specific args.
- Remove the redundant registrations from example modules.
- Optional: emit a one-time `DeprecationWarning` when a model module
  redundantly registers an ADMM arg.  (Maybe not — silent skip is
  fine.)

### Files

- Examples: drop registrations.
- Doc: update "Standard functions" / "Additional functions" lists.

### Open questions

Whether the DeprecationWarning is worth the noise.  Default: skip it.

---

## 7. Roll-out

After this design is approved:

1. Phase A as its own branch + PR off main.
2. After A merges, phase B branches off main (or off A if D depends on
   it — no expected dependency).
3. ...etc.

Each phase's PR title prefixes `stoch-admm:` (matching recent commits)
or `admm:`.  Each ships with green tests including the new automation
tests.

---

## 8. Risks and rollback

- Each phase preserves the old API as a fallback path; users with
  existing model modules are unaffected.
- If a phase reveals a deeper issue (e.g., phase A interacting with
  bundling), that phase can be reverted on its own without affecting
  earlier phases.
- The doc note added in commit `24bf5802` ("clarify attach_root_node
  contract differs for --admm vs --stoch-admm") will become obsolete
  when phase A lands; phase A's PR explicitly supersedes it.
