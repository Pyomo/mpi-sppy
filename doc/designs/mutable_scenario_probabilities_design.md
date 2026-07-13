# Mutable scenario probabilities

Status: phases 0–1 implemented and verified (addresses issue #797); phases
2–4 remain (§9). Covers the current behavior (§1), the use case and goals
(§2–3), the design (§4), the EF path in detail (§5), the PH/decomposition
path (§6), API and compatibility (§7), and open questions (§8).

Implemented so far (EF path, the issue's request):
`sputils.has_persistent_solve_api` (recognizes APPSI / `pyomo.contrib.solver`
as persistent for the EF workflow); `mutable_probability` option on
`_create_EF_from_scen_dict` and `ExtensiveForm` (option-B Param objective);
`ExtensiveForm.set_scenario_probabilities`; and a `reuse_instance` argument to
`solve_extensive_form`. Verified end-to-end on the farmer example against a
rebuild oracle — machine-precision agreement across a probability sweep, with
`set_instance` called once — for **both** `appsi_highs` (auto-tracks the
change) and `gurobi_persistent` (requires the explicit `set_objective`
re-push). Tests: `Test_mutable_probability` in `mpisppy/tests/test_ef_ph.py`.

---

## 0. Motivation (issue #797)

A user solves a two-stage stochastic program with the HiGHS *persistent*
solver in a rolling-horizon loop. The persistent model is built once, and
between solves only data changes. For each roll they need to change the
**scenario probabilities** and re-solve.

Today that is not possible without rebuilding: mpi-sppy bakes each
scenario's probability into the Extensive Form objective as a Python
**float constant** when the EF is constructed. Changing a probability
requires reconstructing the EF objective (and, for a persistent solver,
re-loading the instance), which defeats the point of a persistent model.

The request: represent scenario probabilities as **mutable Pyomo
parameters** so they can be updated in place, and the (persistent) solver
re-solved cheaply.

---

## 1. Current behavior

### 1.1 Where probabilities live

Each scenario model carries a scalar attribute `_mpisppy_probability`
(a Python `float`), set by the user's `scenario_creator` or defaulted to
uniform in `sputils.py`:

```python
# sputils.py ~281-288  (_check_get_default_probability / attach)
if probs_specified and all(... == "uniform" ...):
    ...
    scen._mpisppy_probability = 1 / total_number_of_scenarios
```

For multistage problems, `ScenarioNode.cond_prob` holds the conditional
probability at each tree node, and `spbase._compute_unconditional_node_probabilities`
derives, per scenario and node, an unconditional coefficient
`prob_coeff`:

```python
# spbase.py ~419-427
root.uncond_prob = 1.0
for parent, child in zip(node_list[:-1], node_list[1:]):
    child.uncond_prob = parent.uncond_prob * child.cond_prob
...
s._mpisppy_data.prob_coeff[node.name] = (s._mpisppy_probability / node.uncond_prob)
s._mpisppy_data.prob0_mask[node.name]  = 1.0
```

So there are **two** representations of "how much this scenario counts":

- `_mpisppy_probability` — the scenario's (leaf) probability; used to
  weight the **objective**.
- `_mpisppy_data.prob_coeff[ndn]` — per-node conditional coefficient;
  used to weight **xbar / W / rho / residuals** in PH-family algorithms.

Both are plain floats (or numpy float arrays, for variable probability).

### 1.2 Where the probability enters the EF objective

`sputils._create_EF_from_scen_dict` (the function behind
`ExtensiveForm` and behind PH bundles) builds the EF objective by
folding each scenario's float probability directly into the expression:

```python
# sputils.py 348-364
EF_instance._mpisppy_probability = 0
for (sname, scenario_instance) in scen_dict.items():
    EF_instance.add_component(sname, scenario_instance)
    ...
    obj_func = scenario_objs[0]
    EF_instance.EF_Obj.expr += scenario_instance._mpisppy_probability * obj_func.expr
    EF_instance._mpisppy_probability += scenario_instance._mpisppy_probability
# normalize (matters for bundles; a no-op weight-sum==1 for a full EF)
EF_instance.EF_Obj.expr /= EF_instance._mpisppy_probability
```

Because `_mpisppy_probability` is a `float`, the probabilities become
**constant coefficients** in the compiled objective. A persistent solver
that has already ingested this objective has no handle to update them;
the only recourse today is to rebuild the objective and re-set the
instance.

### 1.3 Where the probability enters PH

For PH/APH/subgradient/FWPH etc. the probability weighting is applied
numerically (not in a Pyomo expression) via `prob_coeff`:

- `phbase.py:94,164` — xbar and weighted averages
- `convergers/norms_and_residuals.py`, `convergers/primal_dual_converger.py`
- `extensions/dyn_rho_base.py:158`, `extensions/primal_dual_rho.py:75`,
  `extensions/grad_rho.py:113`
- `opt/aph.py:525`

These read `s._mpisppy_data.prob_coeff[ndn]` fresh each iteration, so they
are *already* update-friendly **provided** the stored value is refreshed.
The proximal/objective terms attached to each subproblem (`attach_Ws_and_prox`)
do **not** multiply by probability — PH's per-scenario objective is the raw
scenario objective plus W and prox terms; probability weighting happens in
the aggregation, not in the subproblem objective. That is important: for the
PH path, "mutable probabilities" is mostly a matter of recomputing
`prob_coeff`, not of touching any Pyomo expression.

---

## 2. Use case in scope

Primary (issue #797): **ExtensiveForm** with a persistent solver, probabilities
changed between solves, no rebuild. Two-stage is the concrete ask; the design
should not preclude multistage.

Secondary: keeping the PH/decomposition path consistent so that a user who
updates probabilities there gets correct xbar/W/rho weighting on the next
iteration.

## 3. Goals and non-goals

Goals:

1. Allow scenario probabilities to be updated after model construction and
   re-solved without rebuilding the Pyomo model.
2. For persistent solvers, make the update cheap (update param values +
   re-push the objective, not `set_instance`).
3. Keep the default path (probabilities fixed for the life of the run)
   behaving exactly as today, at no measurable overhead, and opt-in for the
   mutable path.
4. One source of truth: updating a probability must keep `_mpisppy_probability`,
   the EF objective, and `prob_coeff` mutually consistent.

Non-goals:

- Changing the **structure** of the scenario tree at runtime (adding/removing
  scenarios or nodes). Only the probability *values* on an existing tree.
- Re-deriving sample-average or confidence-interval machinery on the fly.
- Automatic re-solve orchestration in the rolling-horizon loop — that stays
  the user's driver code.

---

## 4. Design overview

Introduce an **opt-in mutable representation** of scenario probability, keyed
off a flag so the default stays a baked-in constant.

- A new option, `mutable_probability` (bool, default `False`), threaded to
  `ExtensiveForm` / `sputils._create_EF_from_scen_dict` (and available to
  `SPBase` for the PH path).
- When enabled, each scenario's probability is stored as a Pyomo **mutable
  `Param`** rather than folded in as a float, and the EF objective references
  that Param.
- A single public method, `set_scenario_probabilities(mapping)`, updates the
  Param values, refreshes the derived `prob_coeff`, and (for persistent
  solvers / EF) re-pushes the objective.

The flag keeps the common case free of any Param-indirection overhead and
avoids perturbing the many call sites that read `_mpisppy_probability`.

## 5. EF path (the issue's actual request)

### 5.1 Representation

In `_create_EF_from_scen_dict`, when `mutable_probability` is set, attach a
mutable Param per scenario on the EF's `_mpisppy_model` block and build the
objective against it:

```python
EF_instance._mpisppy_model.prob = pyo.Param(
    EF_instance._ef_scenario_names, mutable=True, within=pyo.NonNegativeReals,
    initialize={sname: scen._mpisppy_probability for sname, scen in scen_dict.items()},
)
# objective term
EF_instance.EF_Obj.expr += EF_instance._mpisppy_model.prob[sname] * obj_func.expr
```

Normalization (`/= sum`) is the subtlety: the current code divides the whole
objective by the accumulated probability sum. That division only does real
work for **bundles**, where member probabilities sum to <1 and the divisor
rescales the bundle objective into a proper within-bundle conditional
expectation. For a standalone **full EF** — the issue's use case — the
probabilities already sum to 1, so the division is a no-op.

The two considered options:

- **(A) Normalize inside the expression with a mutable divisor.** Store an
  unnormalized objective `sum_s prob[s] * obj_s` and a separate mutable Param
  `prob_sum`; write the objective as `(sum_s prob[s]*obj_s) / prob_sum`, and
  have `set_scenario_probabilities` update both `prob[*]` and `prob_sum`.
  General (handles sum != 1) but carries a division node and an extra Param.
- **(B) Require normalized input and drop the divisor.** Impose that the
  supplied probabilities sum to 1 whenever `mutable_probability` is set, and
  build the objective as just `sum_s prob[s] * obj_s` with no division.

**Recommendation: (B).** The requirement is cheap to impose *only* on the
mutable path because the mutable and needs-a-divisor regimes are disjoint in
practice: mutable probability is a full-EF feature, and bundles (the only
sum != 1 case) are built by this same function but never set the flag. So a
single branch at construction covers both without touching existing behavior:

```python
if mutable_probability:
    _validate_sum_to_one(scen_dict)            # raise on violation (see §8.5)
    EF_instance._mpisppy_model.prob = pyo.Param(
        EF_instance._ef_scenario_names, mutable=True, within=pyo.NonNegativeReals,
        initialize={sname: scen._mpisppy_probability for sname, scen in scen_dict.items()},
    )
    for (sname, scenario_instance) in scen_dict.items():
        ...
        EF_instance.EF_Obj.expr += EF_instance._mpisppy_model.prob[sname] * obj_func.expr
    # no /= sum
else:
    # unchanged float-coefficient path, including the /= accumulated-sum
    # normalization that bundles rely on
    ...
```

The same `_validate_sum_to_one` runs inside `set_scenario_probabilities`, so
the "mutable ⇒ normalized" contract holds at build time and on every update.

Why (B) is not just simpler but genuinely better here:

- **No `prob_sum` Param and no division node.** This matters for the
  persistent path (§5.2): re-pushing via `set_objective` becomes a plain
  linear combination of Params, with nothing for the solver to re-derive
  around a division.
- **Nothing existing changes.** The requirement is gated on the opt-in flag,
  so the float path — and bundle scaling in particular — is untouched.

Two guards close the one case where the disjointness assumption could be
violated:

1. **Bad input on the mutable path.** Reject (raise) rather than silently
   renormalizing — silent renormalization would secretly reintroduce the
   divisor B is trying to avoid. See §8.5.
2. **Mutable probabilities requested for a bundle.** Disallow explicitly
   (raise) rather than silently skipping the divisor a bundle needs.

### 5.2 Update + re-solve

```python
def set_scenario_probabilities(self, prob_map):
    # prob_map: {scenario_name: float}  (must keep the full set summing to 1)
    _validate_sum_to_one(prob_map_applied_to_all_scenarios)   # §5.1, §8.5
    for sname, p in prob_map.items():
        self.ef._mpisppy_model.prob[sname].value = p
        getattr(self.ef, sname)._mpisppy_probability = p      # §5.3
    if <persistent> and hasattr(self.solver, "set_objective"):
        self.solver.set_objective(self.ef.EF_Obj)   # re-push objective coefficients
```

Under option (B) there is no `prob_sum` to update — just the per-scenario
Params. `set_objective` re-extracts the objective coefficients from the
(already-updated) Param values without touching constraints or variables —
far cheaper than `set_instance`.

**Cross-solver behavior (verified by experiment, see §8.3).** Two persistent
families behave differently, and calling `set_objective` unconditionally on
the persistent path is correct for both:

- *APPSI / `pyomo.contrib.solver` (e.g. `appsi_highs`, the issue's solver).*
  The wrapper auto-tracks model changes on the next `solve()` — its
  `update_config.update_params` and `update_objective` default to `True` — so
  a mutable-Param objective change is picked up automatically even *without*
  `set_objective`. Calling `set_objective` is redundant but harmless.
- *Legacy `PersistentSolver` (e.g. `gurobi_persistent`, `cplex_persistent`).*
  These do **not** auto-track model changes; `set_objective` (or an explicit
  coefficient update) is **required**, or the solver re-solves the stale
  objective.

**Reuse guard.** `solve_extensive_form` currently re-instances on every call
for solvers it considers persistent (ef.py:117-118). For the rolling-horizon
reuse case we want to **skip** re-instancing when the instance is already
loaded and only the objective changed. *Resolved (§8.2):* add an explicit
`reuse_instance=False` argument to `solve_extensive_form`; when `True`, skip
`set_instance` and rely on the loaded instance plus the re-pushed objective.

**Detection caveat (found by experiment, see §8.3).** mpi-sppy does not currently
recognize `appsi_highs` as persistent at all: `ef.py:117` tests the substring
`"persistent" in solver_name` (false for `"appsi_highs"`), and
`sputils.is_persistent` tests `isinstance(solver, <legacy> PersistentSolver)`
(also false — APPSI solvers are `LegacySolver` wrappers, not that base class).
So today `ExtensiveForm` treats `appsi_highs` like a non-persistent solver and
never takes the `set_instance` / `load_vars` path. This feature must therefore
extend persistence detection to the APPSI / `pyomo.contrib.solver` interfaces
(e.g. duck-type on `set_instance`/`set_objective`/`load_vars`, or check for the
contrib base classes) — otherwise the `reuse_instance` path is unreachable for
exactly the solver in the issue. This is a prerequisite for phase 1, not a
nice-to-have.

### 5.3 Keeping `_mpisppy_probability` consistent

`set_scenario_probabilities` must also write `scen._mpisppy_probability = p`
on the underlying scenario models, so any downstream reader (solution
reporting, `get_objective_value`, zhat evaluation) sees the same numbers the
objective used.

## 6. PH / decomposition path

The PH path does not bake probabilities into a Pyomo objective, so no Param is
needed there. What it needs is a way to **recompute `prob_coeff`** after a
change. Today `_compute_unconditional_node_probabilities` only computes
`prob_coeff` if it is missing (`if not hasattr(...)`), so it will not pick up
an updated `_mpisppy_probability` on its own.

Design: add `SPBase.set_scenario_probabilities(mapping)` that

1. updates `_mpisppy_probability` (and, for multistage, the relevant
   `ScenarioNode.cond_prob`) on each local scenario,
2. forces recomputation of `uncond_prob` and `prob_coeff` (drop the
   `hasattr` short-circuit, or clear the cached dict first),
3. preserves any `has_variable_probability` overrides (re-apply
   `_use_variable_probability_setter` after the refresh).

Because xbar/W/rho all read `prob_coeff` fresh each iteration, refreshing that
dict is sufficient for the next PH iteration to be correct. This is a
lower-priority companion to the EF work — the issue is specifically about the
EF — but it belongs in the same design so the two representations stay in
sync and there is one method name across both.

## 7. API and backward compatibility

- New option `mutable_probability` (default `False`) — no behavior change for
  existing users; the objective is still a float-coefficient expression.
- New method `set_scenario_probabilities` on `ExtensiveForm` (and `SPBase`).
  Purely additive.
- `_mpisppy_probability` remains a float attribute and is still the read API;
  the Param is an internal, opt-in representation, not a replacement.
- CLI: expose `--mutable-probability` via a `config.py` arg if we want it
  reachable from `generic_cylinders.py`; not required for the programmatic
  use case in the issue.

## 8. Open questions

1. **Normalization** — design settles on option (B) (require sum-to-1 on the
   mutable path, drop the divisor; see §5.1). Remaining to confirm: the exact
   tolerance for the sum-to-1 check, and that no in-tree caller builds a
   *bundle* with `mutable_probability` set (the guard in §5.1 should make that
   an explicit error).
2. **Persistent re-instancing guard** — *resolved:* an explicit
   `reuse_instance=False` argument to `solve_extensive_form`. When `True`,
   skip the `set_instance` at ef.py:117 and rely on the already-loaded
   instance plus the updated objective. Explicit over an invisible internal
   flag so the reuse contract is visible at the call site.
3. **HiGHS `set_objective` support** — *resolved by experiment*
   (appsi_highs / highspy 1.15.1). Findings:
   (a) `set_objective(EF_Obj)` **does** re-extract objective coefficients from
   mutable Params — re-solve moved the shared first-stage var to the new
   optimum after only re-pushing the objective, no `set_instance`.
   (b) Stronger: appsi_highs **auto-tracks** the change on the next `solve()`
   even without `set_objective` (`update_config.update_params` /
   `update_objective` default `True`), and retains its instance across
   `.solve()` calls. So the mutable-Param objective "just works" for
   appsi_highs; `set_objective` is redundant-but-safe there and *required* for
   legacy persistent solvers (see §5.2).
   (c) *New prerequisite surfaced:* mpi-sppy classifies `appsi_highs` as
   **non-persistent** — `sputils.is_persistent` returns `False` and the
   `"persistent" in solver_name` check at ef.py:117 is also `False` — so the
   reuse path is currently unreachable for it. Persistence detection must be
   extended to APPSI / `pyomo.contrib.solver` interfaces (§5.2).
4. **Scope of a probability change** — do we ever need to change the *number*
   of scenarios or the tree structure between rolls? Declared out of scope
   here; if the user's rolling horizon changes the scenario set, that is a
   rebuild, not a probability update.
5. **Validation** — *resolved in the implementation.* Under option (B),
   `set_scenario_probabilities` **raises** if the resulting full probability
   vector does not sum to 1 (tolerance `1e-9`), with no silent renormalization
   (which would reintroduce the divisor B removes). Partial-mapping updates
   **are** allowed: scenarios omitted from the mapping keep their current
   probability, as long as the resulting total still sums to 1. The check runs
   *before* any value is written, so a rejected call leaves the model
   unchanged (transactional).

---

## 9. Suggested implementation phases

0. **[done]** Prerequisite: extend persistence detection to APPSI /
   `pyomo.contrib.solver` interfaces so `appsi_highs` (the issue's solver) is
   recognized as persistent (§5.2, §8.3c). Implemented as
   `sputils.has_persistent_solve_api`; `ExtensiveForm.solve_extensive_form`
   uses it in place of the old `"persistent" in name` / `is_persistent` checks.
1. **[done]** EF-only, two-stage: `mutable_probability` flag, Param objective
   with option-(B) normalization (require sum-to-1, no divisor),
   `set_scenario_probabilities`, explicit `reuse_instance` argument to
   `solve_extensive_form`. Closes the issue. Verified against a rebuild oracle
   on farmer for `appsi_highs` and `gurobi_persistent`.
2. PH path: `SPBase.set_scenario_probabilities` + `prob_coeff` refresh.
3. Multistage node probabilities and variable-probability interaction.
4. CLI exposure + docs + a rolling-horizon example under `examples/`.
