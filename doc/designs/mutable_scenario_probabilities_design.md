# Mutable scenario probabilities

Status: design proposal (addresses issue #797). Covers the current behavior
(§1), the use case and goals (§2–3), the design (§4), the EF path in detail
(§5), the PH/decomposition path (§6), API and compatibility (§7), and open
questions (§8).

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
objective by the probability sum as a constant. With mutable params the
normalizer must also be mutable so it tracks updates. Two options:

- **(A) Normalize outside the expression.** Store an unnormalized objective
  `sum_s prob[s] * obj_s` and a separate mutable Param `prob_sum`; write the
  objective as `(sum_s prob[s]*obj_s) / prob_sum`. `set_scenario_probabilities`
  updates both `prob[*]` and `prob_sum`. Keeps a single division node.
- **(B) Require normalized input.** Document that supplied probabilities must
  sum to 1 (full EF already expects this); skip the divide when
  `mutable_probability` and the caller asserts normalized. Simpler, but
  changes bundle semantics — bundles rely on the normalizer — so (A) is the
  safe default and (B) only for the full-EF case.

Recommendation: **(A)**, because it preserves the existing bundle scaling
behavior and is correct for both full EF and single-bundle EFs.

### 5.2 Update + re-solve

```python
def set_scenario_probabilities(self, prob_map):
    # prob_map: {scenario_name: float}
    for sname, p in prob_map.items():
        self.ef._mpisppy_model.prob[sname].value = p
    self.ef._mpisppy_model.prob_sum.value = sum(...)   # option (A)
    if sputils.is_persistent(self.solver):
        self.solver.set_objective(self.ef.EF_Obj)      # re-push only the objective
```

For a persistent solver this is the key win: `set_objective` re-extracts the
objective coefficients from the (already-updated) Param values without
touching constraints or variables — far cheaper than `set_instance`.
Non-persistent solvers need no special step; the next `solve` reads the new
Param values.

Note `solve_extensive_form` currently calls `self.solver.set_instance(self.ef)`
on every call for persistent solvers (ef.py:117-118). For the rolling-horizon
reuse case we want to **skip** re-instancing when the instance is already
loaded and only the objective changed. Add a guard (e.g. a
`self._instance_loaded` flag, or a `reuse_instance=True` argument to
`solve_extensive_form`) so the second and later solves take the cheap path.

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

1. **Normalization** — confirm option (A) is acceptable for bundles, and
   whether `prob_sum` should be exposed or kept internal.
2. **Persistent re-instancing guard** — is a `reuse_instance` argument to
   `solve_extensive_form` preferable to an internal `_instance_loaded` flag?
   The former is explicit; the latter is invisible to callers.
3. **HiGHS `set_objective` support** — verify the appsi/persistent HiGHS
   interface actually re-extracts objective coefficients from mutable Params
   on `set_objective` (Gurobi/CPLEX persistent do). If HiGHS does not, we may
   need `update_config` toggles or a targeted coefficient update.
4. **Scope of a probability change** — do we ever need to change the *number*
   of scenarios or the tree structure between rolls? Declared out of scope
   here; if the user's rolling horizon changes the scenario set, that is a
   rebuild, not a probability update.
5. **Validation** — should `set_scenario_probabilities` require the mapping to
   cover all scenarios / sum to 1, warn, or silently renormalize? Proposed:
   require full coverage, renormalize via `prob_sum`, warn if the raw sum is
   far from 1.

---

## 9. Suggested implementation phases

1. EF-only, two-stage: `mutable_probability` flag, Param objective with
   option-(A) normalization, `set_scenario_probabilities`, persistent-reuse
   guard in `solve_extensive_form`. Closes the issue.
2. PH path: `SPBase.set_scenario_probabilities` + `prob_coeff` refresh.
3. Multistage node probabilities and variable-probability interaction.
4. CLI exposure + docs + a rolling-horizon example under `examples/`.
