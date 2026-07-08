# What actually breaks a PH W-oscillation cycle? (experiments on `sizes`)

These experiments ask which interruption actually breaks a Progressive Hedging
W-vector oscillation, using the `sizes` model as a testbed. The finding drove
the decision to ship **slamming as the only interruption remedy** in the
W-oscillation extension (`mpisppy/extensions/w_oscillation.py`); the extension's
design doc (`doc/designs/w_oscillation_design.md`, §9) points here for the
evidence.

Nothing here is a product feature or wired into CI — it is a reproducible
research harness. It needs a MIP solver (gurobi_persistent by default).

## Background

Under plain PH, `sizes` settles into a **stable limit cycle**: ~11–20
nonanticipative variables have a sign-flipping W trajectory every iteration and
the run never converges. Watson–Woodruff (§2.1) attribute such cycling to the
dual weight `w` "shooting past" its optimum, which suggests dual-side remedies
(shrink the dual step, reduce rho). We test those and more.

## Two metrics (one is not enough)

* **`zc`** — how many nonants the zero-crossings detector still flags. It counts
  *sign changes*, so it is **scale-free** and can be fooled: anything that
  freezes W (e.g. driving rho → 0) makes `zc` fall even when nothing has
  converged.
* **`gap`** — the PH primal gap `sum_s p_s |x_s − xbar|`. This is the **ground
  truth**: it is low only when the scenarios actually agree. A large `gap` with
  a small `zc` means an intervention *decoupled* the scenarios rather than
  converging them.

## Running it

```bash
python run_experiments.py                          # gurobi_persistent, 60 iters
python run_experiments.py --solver-name cplex --iters 60 --outdir results
```

Each arm runs in its own subprocess (`_run_one_arm.py`) for clean state;
`w_osc_experiment_ext.py` is the PH extension that detects, intervenes, and
records the metrics. Results land in `--outdir` (`results/summary.md` and
`results/interventions_by_iteration.csv`).

## Results (sizes-3, 60 iterations, interventions from iter 10)

### 1. Interventions at the model's native (small) rho

| arm | zc | primal gap | reading |
|---|---|---|---|
| plain PH | 11.0 | 94 | still cycling |
| w_damping ×0.5 | 12.1 | 94 | still cycling |
| rho reduction (geometric ×0.7) | 9.1 | 23757 | decoupled (gap exploded) |
| W reset + rho ×0.5 | 13.9 | 243 | still cycling |
| **fix (slam analogue)** | 9.6 | **0** | **converges @ iter 35** |

### 2. rho level (uniform; native `_rho_setter` disabled)

| default_rho | zc | primal gap | reading |
|---|---|---|---|
| 0.001 | 18.2 | 60 | still cycling |
| 0.01 | 24.0 | 57 | still cycling |
| **0.1** | 24.0 | **6** | **converges** |
| **0.3** | 20.9 | **0** | **converges @ iter 25** |
| **1** | 15.9 | **0** | **converges @ iter 12** |

### 3. rho perturbation (native rho + jitter)

| perturbation | zc | primal gap | reading |
|---|---|---|---|
| native (ε 0) | 11.0 | 94 | still cycling |
| jitter ±50% per-variable | 11.0 | 68 | still cycling |
| jitter ±50% per-call | 11.8 | 80 | still cycling |

## Conclusions

1. **No state-perturbing move breaks the cycle.** W-damping (any factor),
   W-reset, and rho *reduction* all leave the primal gap high or make it worse.
   Geometric rho reduction is the cautionary case: `zc` *drops* (to 9.1) while
   the primal gap **explodes to ~24000** — driving rho → 0 freezes W (fooling
   the sign-based detector) while the scenarios fly apart. That is a false
   positive, not a fix.

2. **Randomly perturbing the rho values does not help either** — it is the rho
   *level*, not its *symmetry*, that matters.

3. **Only two things converge the primal gap:** `fix` (the slam analogue —
   changing the problem *structure* by fixing a cycling variable), and simply
   using a **larger rho**.

4. **Root cause:** the cycle is a **small-rho artifact**. `sizes`'s
   `_rho_setter` uses `rho = cost × 0.001`, squarely in the oscillating regime
   (rho ≤ 0.01 cycles; rho ≥ 0.1 converges, often early, with no interruption at
   all). Reducing rho pushes *further* into the bad regime — which is why the
   rho-reduction arm is counterproductive.

5. **Metric lesson:** an oscillation detector that watches W alone can be gamed
   by anything that freezes W. Confirming that an interruption actually
   *converged* (rather than *decoupled*) requires a primal-side check like the
   gap used here.

## Files

| file | role |
|---|---|
| `run_experiments.py` | orchestrator: runs every arm, writes `summary.md` + a per-iteration CSV |
| `_run_one_arm.py` | worker: builds a PH-only hub on `sizes` and runs one arm |
| `w_osc_experiment_ext.py` | PH extension: detect (reusing PR1 primitives), intervene, record `zc` + primal gap |
| `results/summary.md` | captured results (this run) |
| `results/interventions_by_iteration.csv` | per-iteration `zc` and `gap` for the intervention arms |
