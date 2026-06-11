# CVaR (Conditional Value-at-Risk) Risk Management — Design

Status: draft for review. Branch `CVaR` (off Pyomo/mpi-sppy `main`).

## 1. Goal

Support minimizing a risk-averse objective of the form

```
    E[Cost] + β · CVaR_α(Cost)
```

mirroring PySP's "weighted CVaR" (`runef --generate-weighted-cvar --cvar-weight=β --risk-alpha=α`),
but working uniformly across mpi-sppy's EF solve **and** its decomposition cylinders
(PH/APH hub, Lagrangian outer-bound spokes, xhat inner-bound spokes, FWPH, subgradient, …).

## 2. Background: the Rockafellar–Uryasev linearization

For a cost (loss) random variable `Y` and confidence level `α ∈ (0,1)`:

```
    CVaR_α(Y) = min_η { η + 1/(1-α) · E[(Y - η)_+] }
```

The minimizing `η*` is the Value-at-Risk (the α-quantile of `Y`). Linearize `(Y-η)_+`
with one auxiliary variable per scenario:

```
    minimize  η + 1/(1-α) · Σ_s p_s δ_s
    s.t.      δ_s ≥ Y_s - η ,   δ_s ≥ 0      for all scenarios s
```

## 3. How PySP does it (reference: `Pyomo/pysp` → `pysp/ef.py`)

When `generate_weighted_cvar` is set, PySP modifies the *binding* (extensive-form) model:

- `CVAR_ETA_ROOT` — a single root-node free `Var` (η, the VaR).
- `CVAR_EXCESS_<scenario>` — per-scenario `Var`, `NonNegativeReals` (min) / `NonPositiveReals` (max) (δ_s).
- `COMPUTE_SCENARIO_EXCESS` — `ConstraintList` adding `0 ≤ δ_s − cost_s + η` ⇒ `δ_s ≥ cost_s − η`.
- `CVAR_COST_ROOT` — `Expression` set to `η + 1/(1-α) · Σ_s p_s δ_s`.
- Objective `MASTER` = `EF_EXPECTED_COST + cvar_weight · CVAR_COST_ROOT`.

CVaR is taken over the **total** scenario cost (`scenario._instance_cost_expression`).

Two PySP quirks we deliberately do **not** copy:
- `cvar_weight == 0` is overloaded to mean "pure CVaR — drop the mean entirely." Surprising; see §6.1.
- A stray `cost_expr = 1.0` initializer adds a harmless constant `1/(1-α)` to the objective. We drop it.

## 4. The decomposition insight (why this is clean in mpi-sppy)

Because `Σ_s p_s = 1` and η is a single first-stage (shared, non-anticipative) variable, the
risk-averse objective distributes over scenarios:

```
    E[Cost] + β·CVaR_α(Cost) = Σ_s p_s · [ Cost_s + β·η + β/(1-α)·δ_s ]
```

So the *entire* risk measure is captured per scenario by:
- adding ONE new root-node (first-stage, non-anticipative) variable η,
- adding ONE scenario-local variable δ_s ≥ 0 with `δ_s ≥ Cost_s − η`,
- replacing the scenario objective with `Cost_s + β·η + β/(1-α)·δ_s`.

η is **"just another first-stage variable."** Everything mpi-sppy already does for first-stage
variables then handles CVaR for free:

| Cylinder | What happens automatically | Result |
|---|---|---|
| PH / APH hub | drives η to consensus (its VaR) with its own W and prox | no algorithm changes |
| EF (`create_EF`) | blends η into one reference var via existing NAC machinery; `Σ_s p_s·obj_s` | **reproduces PySP's EF-CVaR exactly** |
| Lagrangian / outer-bound spokes | relax NAC on η too | valid **lower** bound on the risk-averse objective |
| xhat / inner-bound spokes | fix η (a nonant) with the rest of stage 1; each scenario sets δ_s = (Cost_s−η)_+ | valid **upper** bound = E[Cost]+β·CVaR at that xhat |

**Central design choice:** implement CVaR as a *per-scenario model transform that adds η to the
root node*, NOT as new hub/spoke logic. One transform; every cylinder inherits it.

Verified against the code: nonants are enumerated from each root node's pre-built
`nonant_vardata_list` (`spbase._attach_nonant_indices`), and `_create_EF_from_scen_dict` builds
NAC reference variables generically over `node.nonant_vardata_list` — so appending η to the root
node is sufficient for both decomposition and EF. The objective is the active Pyomo `Objective`
(via `sputils.find_active_objective`); there is no separate `_mpisppy_objective_functional` to keep
in sync — we rewrite `obj.expr` directly (the same approach `admmWrapper` uses).

## 5. Implementation surface

### 5.1 Core transform — new `mpisppy/utils/cvar.py`

```python
def add_cvar(scenario, *, cvar_weight, cvar_alpha, cvar_mean_weight=1.0):
    """Mutate one scenario model in place to add Rockafellar–Uryasev CVaR terms.

    Adds  scenario._mpisppy_cvar_eta     (root-node nonant, free; the VaR η)
          scenario._mpisppy_cvar_excess  (>= 0; the excess δ_s)
          scenario._mpisppy_cvar_excess_con
    and rewrites the active objective to  λ·cost + β·η + β/(1-α)·δ_s,
    where λ = cvar_mean_weight (default 1.0) and β = cvar_weight.
    """
    obj  = sputils.find_active_objective(scenario)   # pristine user cost
    cost = obj.expr
    # sense from obj.sense; minimize first (maximize mirrors PySP — see §6.3)
    scenario._mpisppy_cvar_eta    = pyo.Var()
    scenario._mpisppy_cvar_excess = pyo.Var(domain=pyo.NonNegativeReals)
    scenario._mpisppy_cvar_excess_con = pyo.Constraint(
        expr=scenario._mpisppy_cvar_excess >= cost - scenario._mpisppy_cvar_eta)
    obj.expr = cvar_mean_weight*cost + cvar_weight*scenario._mpisppy_cvar_eta \
             + (cvar_weight/(1.0 - cvar_alpha))*scenario._mpisppy_cvar_excess
    root = scenario._mpisppy_node_list[0]
    root.nonant_list.append(scenario._mpisppy_cvar_eta)          # append to BOTH:
    root.nonant_vardata_list.append(scenario._mpisppy_cvar_eta)  # vardata list is pre-built in __init__
```

### 5.2 scenario_creator wrapper (lightweight, functional)

```python
def cvar_scenario_creator(scenario_creator, *, cvar_weight, cvar_alpha, cvar_mean_weight=1.0):
    def wrapped(sname, **kwargs):
        s = scenario_creator(sname, **kwargs)
        add_cvar(s, cvar_weight=cvar_weight, cvar_alpha=cvar_alpha,
                 cvar_mean_weight=cvar_mean_weight)
        return s
    return wrapped
```

No MPI/rank bookkeeping needed (unlike `AdmmWrapper`, which pre-creates local scenarios) — CVaR is
a pure per-scenario transform applied lazily inside the wrapped creator. Works for EF and all cylinders.

### 5.3 Config / CLI — `mpisppy/utils/config.py`

New `cfg.cvar_args()` adding:
- `--cvar` (bool) — enable
- `--cvar-weight` (float, β ≥ 0; default 1.0)
- `--cvar-alpha`  (float, 0 < α < 1; default 0.9)
- `--cvar-mean-weight` (float, λ ≥ 0; default 1.0) — weight on E[Cost]; λ=0, β=1 gives pure CVaR (§6.1)

Validation: β ≥ 0, λ ≥ 0, 0 < α < 1 (mirror PySP's checks, gated on rank 0 for warnings).

### 5.4 `generic_cylinders.py` wiring

At the existing scenario_creator seam (after the ADMM wrap block, ~lines 65–85):

```python
if cfg.cvar:
    scenario_creator = cvar.cvar_scenario_creator(
        scenario_creator, cvar_weight=cfg.cvar_weight, cvar_alpha=cfg.cvar_alpha,
        cvar_mean_weight=cfg.cvar_mean_weight)
```

A single insertion point → `do_EF`, `do_decomp`, and every spoke inherit the risk-averse objective.

## 6. Design decisions / deviations from PySP

1. **No `weight==0 ⇒ pure CVaR` overload (decided).** β=0 means risk-neutral (plain E[Cost]).
   A mean-free pure-CVaR objective is instead requested explicitly via `--cvar-mean-weight λ`
   (default 1.0): `obj = λ·Cost_s + β·(η + δ_s/(1-α))`. Pure CVaR = `λ=0, β=1`. Defaults preserve
   E[Cost]; no surprising overload. This is part of the core transform (Phase 1).
2. **CVaR on total cost only (PySP default).** η at the root node, over the full scenario objective.
   Per-stage / per-node (nested, time-consistent) CVaR is out of scope for v1.
3. **Minimize first.** Maximize handled by mirroring PySP (δ domain `NonPositiveReals`, negate the
   excess expression); ship in a later phase.
4. **η fixed during xhat evaluation** → valid but possibly loose inner bound. Optional later
   refinement: re-optimize the shared η given the fixed "real" first-stage vars.

## 7. Correctness / validation

- EF-CVaR vs closed form on a tiny instance: check `obj = E[Cost]+β·CVaR`, `η = empirical α-VaR`,
  `δ_s = (Cost_s−η)_+`.
- EF-CVaR (mpi-sppy) vs an independent PySP-style monolithic build.
- β=0 reduces to risk-neutral EF (regression guard).
- α sweep: α→0 ⇒ CVaR ≈ E[Cost]; α→1 ⇒ CVaR ≈ worst case.
- PH-on-CVaR converges to EF-CVaR objective; η consensus → VaR.
- Bound sandwich: Lagrangian outer ≤ EF-CVaR ≤ xhat inner.

## 8. Phased rollout (each phase a review-sized, green-on-its-own PR)

- **Phase 1 — core + EF.** `cvar.py` (`add_cvar` incl. `cvar_mean_weight` + wrapper);
  `tests/test_cvar.py` comparing EF-CVaR to closed form, the β=0 regression, and pure CVaR (λ=0).
  Programmatic API only.
- **Phase 2 — CLI + decomposition.** `cvar_args()` in config (`--cvar`, `--cvar-weight`,
  `--cvar-alpha`, `--cvar-mean-weight`); `generic_cylinders` wiring; a farmer risk-averse example; a
  cylinders test (PH hub + Lagrangian + xhat bound sandwich). Wire the new test into
  `run_coverage.bash` AND `test_pr_and_main.yml` in the same commit.
- **Phase 3 — polish.** maximize support; docs section ("Risk Management"); confidence-interval note
  (`zhat4xhat` evaluates the same risk-averse objective).

## 9. Files touched

| Action | Path |
|---|---|
| new | `mpisppy/utils/cvar.py` |
| new | `mpisppy/tests/test_cvar.py` (+ `run_coverage.bash`, `test_pr_and_main.yml`) |
| edit | `mpisppy/utils/config.py` (`cvar_args`) |
| edit | `mpisppy/generic_cylinders.py` (wrap seam) |
| new | farmer risk-averse example / `--cvar` pass-through |
| docs | `doc/source/…` Risk Management section |
