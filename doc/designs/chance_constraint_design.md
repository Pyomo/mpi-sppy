# Chance Constraints (PySP-style SAA) — Design

Status: draft for review. Branch `CC` (off Pyomo/mpi-sppy `main`).

## 1. Goal

Support a sample-average-approximation (SAA) **chance constraint** of the form

```
    P( risky constraint holds )  >=  1 - α
```

mirroring PySP's `runef --cc-indicator-var=NAME --cc-alpha=α`. As in PySP, the
user defines a binary **indicator variable** in each scenario model and the
big-M constraints linking it to satisfaction; mpi-sppy adds the single
aggregating constraint that turns the per-scenario indicators into a
probabilistic guarantee.

**Scope (decided): the extensive-form (EF) solve only.** Unlike CVaR, a chance
constraint does not separate across scenarios (see §4), so it is not inherited
by the decomposition cylinders. This matches PySP, which only solves the EF.
The not-decomposable caveat is the analogue of the CVaR design's
"not time-consistent" caveat and MUST appear verbatim in the docs — see §6.1.

## 2. Background: the indicator-variable SAA chance constraint

Let `z_s ∈ {0,1}` be a per-scenario indicator with the convention

```
    z_s = 1   ⇔   the risky constraint is SATISFIED in scenario s
```

The user enforces this link with their own big-M constraints (e.g.
`g(x, ξ_s) <= M·(1 - z_s)`, so `z_s = 1` forces `g <= 0`). Given that link, the
SAA chance constraint at confidence level `1 - α` is the single inequality

```
    Σ_s p_s · z_s  >=  1 - α
```

i.e. `E[z] >= 1 - α`, i.e. the probability mass of *satisfying* scenarios is at
least `1 - α`, i.e. the violation probability is at most `α`. `α = 0` forces
satisfaction in every scenario (a robust constraint); larger `α` buys a cheaper
objective by letting the worst (most expensive to satisfy) scenarios fail.

## 3. How PySP does it (reference: `Pyomo/pysp` → `pysp/ef.py`)

`create_ef_instance(..., cc_indicator_var_name=None, cc_alpha=0.0)`. When
`cc_indicator_var_name` is set, PySP modifies the *binding* (extensive-form)
model only — it creates **no** variables and **no** linking constraints (those
are the user's job in the ReferenceModel). For a scalar indicator it adds:

```python
cc_expression = 0
for scenario in scenario_tree._scenarios:
    scenario_instance = scenario_instances[scenario._name]
    cc_var = scenario_instance.find_component(cc_indicator_var_name)
    cc_expression += scenario._probability * cc_var

def makeCCRule(expression):
    def CCrule(model):
        return (1.0 - cc_alpha, cc_expression, None)   # 1-α <= Σ p_s z_s
    return CCrule

cc_constraint_name = "cc_" + cc_indicator_var_name
binding_instance.add_component(
    cc_constraint_name, Constraint(name=cc_constraint_name, rule=makeCCRule(cc_expression)))
```

For an **indexed** indicator var, PySP parses a string index template
(`extractVariableNameAndIndex`) and builds one chance constraint per matching
index, named `cc_<var>_<index>`. The constraint sense is always the same:
`Σ_s p_s z_s[i] >= 1 - α` for each index `i`.

Two PySP artifacts we deliberately do **not** copy:
- The string index-template parsing (`extractVariableNameAndIndex`,
  `extractComponentIndices`) is a CLI-string convenience. We take a real Pyomo
  component name and, if it is indexed, build one constraint per index of the
  component (cleaner; see §6.3).
- PySP's stray `print("multiple cc not yet supported.")` immediately before the
  code that does, in fact, support multiple indices. We just support it.

## 4. The structural insight (why this is the OPPOSITE of CVaR)

CVaR was a free win because `E[Cost] + β·CVaR` *distributes over scenarios*: it
became a per-scenario model transform (add η to the root node, a local δ_s, and
rewrite each objective), and EF / PH / APH / Lagrangian / xhat all inherited it
with **no algorithm changes** (see `cvar_design.md` §4).

A chance constraint is structurally the reverse. The aggregating inequality

```
    Σ_s p_s · z_s  >=  1 - α
```

is a **single constraint that couples a (binary) variable from every
scenario**. It is not separable, so there is no per-scenario transform that
captures it:

| Cylinder | What a coupling constraint means | Result |
|---|---|---|
| EF (`create_EF`) | all scenarios are sub-blocks of one model; add the one constraint over them | **reproduces PySP's EF-CC exactly** |
| PH / APH hub | each rank solves its scenarios *independently*; a constraint spanning all scenarios cannot be placed in any one subproblem | not supported (Phase 1) |
| Lagrangian / xhat spokes | same: the constraint is not local to a scenario | not supported (Phase 1) |

**Central design choice:** implement the chance constraint as a *post-EF-
construction transform on the assembled EF model*, NOT as a per-scenario
transform and NOT as new hub/spoke logic. The EF model exposes every scenario
as a named sub-block (`ef.ef._ef_scenario_names`, `getattr(ef.ef, sname)`), each
carrying `_mpisppy_probability`; that is exactly enough to write the aggregating
constraint over the indicator vars. (Decomposition would instead require
Lagrangian dualization of this coupling constraint with a scalar price μ ≥ 0 and
an outer 1-D search — a substantially harder, research-grade effort with an
integrality duality gap. Out of scope; see §6.1.)

Verified against the code: `_create_EF_from_scen_dict` adds each scenario via
`EF_instance.add_component(sname, scenario_instance)` and records
`EF_instance._ef_scenario_names`, and `ExtensiveForm` stores the assembled model
as `self.ef` — so reaching `getattr(self.ef, sname).<indicator>` for every
scenario is sufficient to add the constraint after construction.

## 5. Implementation surface

### 5.1 Core transform — new `mpisppy/utils/chance_constraint.py`

```python
def add_chance_constraint(ef_model, *, cc_indicator_var_name, cc_alpha):
    """Add a PySP-style SAA chance constraint to an ALREADY-BUILT EF model.

    For a scalar indicator var, adds to ``ef_model`` the single constraint

        Σ_s p_s · z_s  >=  (1 - cc_alpha) · Σ_s p_s

    where z_s = getattr(scenario_s, cc_indicator_var_name).  The Σ_s p_s factor
    on the right is the total probability of the scenarios in this model (1.0 for
    a full EF); carrying it makes the constraint correct even for normalized /
    bundled EFs, and reduces to PySP's plain ``>= 1 - α`` for the full EF.

    The user must define z_s (binary) and the big-M constraints linking it to
    satisfaction, exactly as in PySP.  This function adds only the aggregator.

    For an indexed indicator var, adds one such constraint per index.

    Adds to ef_model:
        _mpisppy_chance_constraint        (scalar case) or
        _mpisppy_chance_constraint[idx]   (indexed case)
    """
    if not (0.0 <= cc_alpha < 1.0):
        raise ValueError(f"cc_alpha must satisfy 0 <= alpha < 1 (got {cc_alpha})")

    snames = ef_model._ef_scenario_names
    total_prob = sum(getattr(ef_model, sn)._mpisppy_probability for sn in snames)

    # representative component decides scalar vs. indexed
    rep = getattr(ef_model, snames[0]).find_component(cc_indicator_var_name)
    if rep is None:
        raise ValueError(
            f"chance-constraint indicator '{cc_indicator_var_name}' not found "
            f"on scenario '{snames[0]}'")

    def _agg(index=None):
        expr = 0
        for sn in snames:
            s = getattr(ef_model, sn)
            z = s.find_component(cc_indicator_var_name)
            expr += s._mpisppy_probability * (z if index is None else z[index])
        return expr >= (1.0 - cc_alpha) * total_prob

    if rep.is_indexed():
        con = pyo.Constraint(list(rep.keys()), rule=lambda m, *idx: _agg(idx if len(idx) > 1 else idx[0]))
    else:
        con = pyo.Constraint(expr=_agg())
    ef_model.add_component("_mpisppy_chance_constraint", con)
    _warn_if_not_binary(ef_model, cc_indicator_var_name)   # rank-0-gated; see §6.4
```

Notes:
- **Post-construction, not a scenario_creator wrapper.** CVaR could wrap the
  scenario_creator because it was per-scenario; the chance constraint needs all
  scenarios present, so it operates on the assembled `ef.ef`.
- **Normalization.** Dividing nothing (PySP) is correct only when `Σ p_s = 1`.
  We multiply the RHS by `Σ p_s` so the constraint is also correct for
  bundle-style normalized EFs, and identical to PySP for a full EF.

### 5.2 EF wiring seam — `mpisppy/generic/ef.py`

In `do_EF`, immediately after the EF is constructed and before solve:

```python
ef = ExtensiveForm(...)
if cfg.cc_indicator_var is not None:
    from mpisppy.utils import chance_constraint
    chance_constraint.add_chance_constraint(
        ef.ef, cc_indicator_var_name=cfg.cc_indicator_var, cc_alpha=cfg.cc_alpha)
```

Programmatic users of `ExtensiveForm` call `add_chance_constraint(ef.ef, ...)`
themselves at the same point.

### 5.3 Config / CLI — `mpisppy/utils/config.py`

New `cfg.chance_constraint_args()` adding (flag names match PySP's `runef`):
- `--cc-indicator-var` (str) — name of the user's per-scenario binary indicator;
  presence of this flag enables the chance constraint.
- `--cc-alpha` (float, 0 <= α < 1; default 0.0) — allowed violation probability.

Validation (gated on rank 0 for warnings): `0 <= α < 1`; the indicator exists on
each scenario; warn if its domain is not Binary (exactness needs binary z, §6.4).

### 5.4 `generic_cylinders.py` guard

Chance constraints are EF-only in Phase 1. If `--cc-indicator-var` is set but
`--EF` is not, raise a clear error at parse/dispatch time:

```
chance constraints are currently supported only with --EF
(a chance constraint couples all scenarios and is not separable;
 see doc/designs/chance_constraint_design.md §4)
```

This prevents a silent no-op where a user requests a chance constraint under a
decomposition run and gets the risk-neutral answer.

## 6. Design decisions / deviations from PySP

1. **EF-only (decided).** See §4. Decomposition is out of scope and flagged for
   users in §6.1.
2. **User supplies the indicator + big-M link; we add only the aggregator.**
   Identical to PySP. We do not invent the indicator or guess the linking
   constraints — the modeler knows which constraint is "risky" and what M is.
3. **Indexed indicators by component, not by string template.** PySP parses a
   CLI index string; we take the component and, if indexed, emit one constraint
   per index of the component. Simpler and avoids the string-template parser.
4. **Binary check is a warning, not an error.** A continuous "indicator" yields a
   CVaR-like relaxation rather than a true chance constraint; we warn (rank 0)
   but proceed, matching PySP's permissiveness.
5. **Normalize the RHS by `Σ p_s`** so the constraint is correct for normalized
   EFs and identical to PySP (`>= 1 - α`) for a full EF.
6. **`α = 0` is allowed** (robust: satisfy every scenario), matching PySP's
   default. We require `α < 1` (`α = 1` is a vacuous constraint).

### 6.1 Documentation note (verbatim — to ship in the docs)

> **Scope: chance constraints are supported only for the extensive-form (EF)
> solve.** A chance constraint `Σ_s p_s z_s >= 1 - α` is a single constraint
> that couples a (binary) indicator variable from every scenario, so — unlike
> CVaR — it does not separate across scenarios and is **not** inherited by the
> PH / APH / Lagrangian / xhat decomposition cylinders. This matches PySP, which
> only solves the EF. Decomposing a chance constraint (e.g. Lagrangian
> dualization of the coupling constraint with a scalar price) is possible but is
> a substantially harder effort with an integrality duality gap. **If you need
> chance constraints under decomposition, please contact the mpi-sppy
> developers.**

## 7. Correctness / validation

- **Tiny closed-form instance.** N scenarios, known cost-to-satisfy per
  scenario; with `α` allowing the cheapest `k` to be satisfied, check the EF-CC
  optimum satisfies exactly the right set and `Σ p_s z_s >= 1 - α` is tight.
- **`α = 0` ⇒ robust.** Reduces to the EF with every indicator forced to 1;
  compare objective and solution.
- **`α` large enough ⇒ inactive.** Reduces to the risk-neutral EF (regression
  guard that the constraint is correctly slack).
- **Indexed indicator ⇒ one constraint per index** (count and senses).
- **mpi-sppy EF-CC vs. an independent PySP-style monolithic build** on the same
  instance (objective and chosen scenario set agree).

## 8. Rollout (a single review-sized PR)

EF-only chance constraints are small enough to land in one PR. It contains:

- `chance_constraint.py` (`add_chance_constraint`, scalar + indexed, normalized
  RHS, binary warning);
- `chance_constraint_args()` in config (`--cc-indicator-var`, `--cc-alpha`);
- `do_EF` wiring + the EF-only guard in `generic_cylinders`;
- `tests/test_chance_constraint.py` (tiny closed-form, `α=0` robust, `α`-inactive
  regression, indexed), wired into `run_coverage.bash` AND `test_pr_and_main.yml`
  in the same commit;
- a small risk-constrained example (e.g. a newsvendor / farmer variant with a
  service-level indicator) driven through
  `generic_cylinders --EF --cc-indicator-var ... --cc-alpha ...`;
- a docs section ("Chance Constraints") that MUST include the EF-only scope
  caveat verbatim from §6.1, plus the programmatic `add_chance_constraint`
  recipe.

## 9. Files touched

| Action | Path |
|---|---|
| new | `mpisppy/utils/chance_constraint.py` |
| new | `mpisppy/tests/test_chance_constraint.py` (+ `run_coverage.bash`, `test_pr_and_main.yml`) |
| edit | `mpisppy/utils/config.py` (`chance_constraint_args`) |
| edit | `mpisppy/generic/ef.py` (`do_EF` wiring) |
| edit | `mpisppy/generic_cylinders.py` (EF-only guard) |
| new | chance-constraint example / `--cc-*` pass-through |
| docs | `doc/source/…` Chance Constraints section (must include the §6.1 EF-only caveat) |
