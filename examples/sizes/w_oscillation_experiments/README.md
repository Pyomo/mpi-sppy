# What actually breaks a PH W-oscillation cycle? (experiments on `sizes`)

These experiments ask which interruption actually breaks a Progressive Hedging
W-vector oscillation, using the `sizes` model as a testbed. The harness began as
the evidence behind shipping **slamming as the only interruption remedy** in the
W-oscillation extension (`mpisppy/extensions/w_oscillation.py`; the design doc
`doc/designs/w_oscillation_design.md`, §9, points here). It has since been
extended to explore a family of **prox-penalty** schedules and mpi-sppy's
built-in **smoothed PH**, and to score not just whether a remedy *stops the
cycle* but whether the answer it produces is any *good* — separately in the
first-stage decision `x` and in the dual weights `W`.

Nothing here is a product feature or wired into CI — it is a reproducible
research harness. It needs a MIP solver (gurobi_persistent by default).

## Background

Under plain PH, `sizes` settles into a **stable limit cycle**: ~9–20
nonanticipative variables have a sign-flipping W trajectory every iteration and
the run never converges. Watson–Woodruff (§2.1) attribute such cycling to the
dual weight `w` "shooting past" its optimum, which suggests dual-side remedies
(shrink the dual step, reduce rho). We test those and more.

## Metrics (four of them, on two axes)

**Did it stop the cycle?** (per iteration)

* **`zc`** — how many nonants the zero-crossings detector still flags. It counts
  *sign changes*, so it is **scale-free** and can be fooled: anything that
  freezes W (e.g. driving rho → 0) makes `zc` fall even when nothing has
  converged.
* **`gap`** — the PH primal gap `sum_s p_s |x_s − xbar|`. This is the **ground
  truth** for consensus: low only when the scenarios actually agree.

**Was the answer any good?** (once, at the end, vs the monolithic EF optimum `z*`)

* **`x-gap`** — expected cost of *committing to* the consensus `xbar`, above `z*`.
  Small = a good first-stage decision.
* **`W-gap`** — the Lagrangian bound from the final `W`, below `z*`. Small = duals
  good enough to certify optimality. `sizes` is a MIP, so a duality gap keeps
  `W-gap > 0` even for a good `W` — read it *relative* across arms.

## The prox-boost idea

`prox_boost` scales **only** the quadratic proximal penalty — the `prox_on`
coefficient on `ProxExpr` in the objective — while leaving `rho` (and hence the
dual update `W += rho·(x−xbar)`) untouched. It is a *prox-only* lever, distinct
from the rho-level sweep, which moves the penalty and the dual step together. Four
schedules are tested:

| schedule | what it does |
|---|---|
| one-shot | boost for a few iterations on the first detected cycle, then revert |
| re-firing | re-open that window on each re-detected cycle (with a cooldown) |
| held | boost and never revert (near-permanent) |
| escalating | hold and *ramp the multiplier up* while the cycle persists |

## The W-average idea (and why it is a legal PH move)

`w_average` asks a different question from the state-perturbing arms: instead of
*shrinking* the dual step (`w_damping`) or *zeroing* it (`w_reset`), replace each
cycling nonant's `W`, in every scenario, with **its own mean over the cycle** — a
one-shot warm restart to the point the oscillation is *orbiting* rather than to
zero. It fires once, after a full detection `window` of history has buffered, so
the mean spans several cycle periods.

**Is that still a valid PH dual state?** Yes — and trivially so. PH maintains the
dual invariant `sum_s p_s W_s = 0` at every iteration (each update adds
`rho·(x_s − xbar)`, and `sum_s p_s (x_s − xbar) = 0` by definition of `xbar`). The
cycle average is a *per-scenario linear combination* of past iterates,
`W̃_s = (1/|K|)·Σ_{k∈K} W_s^k`, so

```
Σ_s p_s W̃_s = (1/|K|)·Σ_{k∈K} ( Σ_s p_s W_s^k ) = (1/|K|)·Σ_{k∈K} 0 = 0.
```

Every buffered `W^k` already has expectation zero and the expectation over
scenarios is linear, so their average inherits the property — nothing needs to be
renormalized. (The same one line licenses *any* weighted average of the iterates,
not only the uniform one.)

## Running it

```bash
python run_experiments.py                          # gurobi_persistent, 80 iters
python run_experiments.py --solver-name cplex --iters 120 --outdir results
```

Each arm runs in its own subprocess (`_run_one_arm.py`) for clean state;
`w_osc_experiment_ext.py` is the PH extension that detects, intervenes, records
the per-iteration metrics, and (at the end) measures the x/W solution quality.
Results land in `--outdir` (`results/summary.md` and
`results/interventions_by_iteration.csv`). Numbers below are representative;
solver tie-breaking shifts them run to run, but the qualitative picture is stable.

## Results (sizes-3, 80 iterations, interventions from iter 10)

### 1. Interventions at the model's native (small) rho

| arm | zc | primal gap | reading |
|---|---|---|---|
| plain PH | 9.0 | 34 | still cycling |
| w_damping ×0.5 | 9.0 | 33 | still cycling |
| rho reduction (geometric ×0.7) | 7.2 | 22261 | decoupled (gap exploded) |
| W reset + rho ×0.5 | 13.0 | 141 | still cycling |
| W average over cycle | 11.1 | 113 | still cycling |
| prox-boost (×10, 5 iters, one-shot) | 8.5 | 34 | still cycling |
| prox-refire (×10, 5 iters, cooldown 5) | 10.6 | 16 | damps to threshold (borderline) |
| prox-hold (×10, held to end) | 13.0 | 25 | damps to threshold (borderline) |
| **prox-escalate (×10 base, ×2 / 5 iters)** | 13.6 | **0** | **converges @ iter 32** |
| smoothing (r=0.1, b=0.2) | 17.8 | 968 | cycle amplified |
| **fix (slam analogue)** | 5.0 | **0** | **converges @ iter 48** |

### 2. rho level (uniform; native `_rho_setter` disabled)

| default_rho | zc | primal gap | reading |
|---|---|---|---|
| 0.001 | 18.1 | 73 | still cycling |
| 0.01 | 22.0 | 19 | converges (borderline) |
| **0.1** | 10.8 | **0** | **converges @ iter 52** |
| **0.3** | 17.2 | **0** | **converges @ iter 34** |
| **1** | 16.8 | **0** | **converges @ iter 24** |

### 3. rho perturbation (native rho + jitter)

| perturbation | zc | primal gap | reading |
|---|---|---|---|
| native (ε 0) | 9.0 | 34 | still cycling |
| jitter ±50% per-variable | 9.4 | 91 | still cycling |
| jitter ±50% per-call | 7.3 | 21 | borderline (noise) |

### 4. Solution quality vs the EF optimum (`z* = 224275`)

| arm | inner Ū | x-gap | outer L | W-gap | reading |
|---|---|---|---|---|---|
| plain PH (cycling) | 224872 | **+0.27%** | 222992 | **+0.6%** | good x, good W |
| W average over cycle | 224885 | +0.27% | 222684 | +0.7% | good x, good W |
| prox-boost (one-shot) | 224878 | +0.27% | 222948 | +0.6% | good x, good W |
| prox-hold | 224944 | +0.30% | 223039 | +0.6% | good x, good W |
| smoothing (r=0.1, b=0.2) | 225735 | +0.65% | 223699 | +0.3% | good x, good W |
| **prox-escalate** | 224967 | **+0.31%** | 216655 | **+3.4%** | good x, **loose W** |
| **fix (slam analogue)** | 225060 | +0.35% | 213836 | +4.7% | good x, **loose W** |
| rho=0.1 | 229380 | +2.28% | 212513 | +5.2% | poor x, loose W |
| rho=0.3 | 230894 | +2.95% | 180390 | +19.6% | poor x, loose W |
| rho=1 | 231089 | **+3.04%** | 209067 | +6.8% | poor x, loose W |

## Conclusions

1. **Stopping the cycle: only three things converge the primal gap** —
   `fix` (change the problem *structure* by fixing a cycling variable), an
   **escalating** prox boost (keep raising the penalty until it forces
   consensus), and a **larger rho**. Every *fixed*-magnitude perturbation
   (W-damping, W-reset, W-average, rho reduction/jitter, a one-shot or re-firing
   prox boost) leaves the cycle intact or only *damps* it to a residual near the
   convergence threshold. `w_average` is the sharpest illustration: restarting
   `W` at the cycle's centre kicks the primal gap *up* (≈3 → ≈5200 as the duals
   jump off their orbit), after which it slowly re-damps but re-enters the same
   oscillation — a one-shot move to a good centre cannot hold without also
   changing rho or the problem structure. `prox-hold` and `prox-refire` sit
   exactly on that boundary and read
   "converged" or "cycling" depending on the budget — a warning against calling
   the cycle beaten from a short run. mpi-sppy's built-in **smoothing** goes the
   *wrong* way: it *amplifies* the cycle (gap ~968 at ratio 0.1, worse as the
   ratio grows, numerically catastrophic by ratio 100), because it anchors each
   scenario to its own lagged EMA `z` — a per-scenario momentum that fights
   consensus. Its center still orbits the optimum (the rounded average is a fine
   decision), but it is strictly worse than plain PH — bigger swings, no
   convergence — at this pathological small rho.

2. **A prox boost is prox-only, and that matters.** Scaling `prox_on` tightens
   the anchor to `xbar` without inflating the dual step (`rho`). A temporary or
   re-firing boost is therefore a transient anesthetic — the small-rho cycle
   snaps back the instant the penalty relaxes. Only *escalating* it — cranking
   the multiplier (×10 → ×160 here) while the gap persists — is strong enough to
   force primal consensus outright.

3. **Convergence and solution quality are different axes.** The arm that
   converges *fastest* (a large uniform rho) lands the *worst* decision
   (x-gap ~3%) and an overshot dual bound (rho=0.3 → W-gap ~20% — the literal
   "shooting past"). Meanwhile the native small-rho **cycle that never converges
   is near-optimal in both** x (~0.3%) and W (tightest bound, <1%): it is quietly
   orbiting the optimum.

4. **So what gives better x, and better W?**
   * **Better x (the decision):** keep the dual step small. `prox-escalate`
     (0.31%) and `fix` (0.35%) match the cycle's near-optimal decision and beat a
     larger rho (2–3%) by an order of magnitude, because they settle near the
     cycle's centre instead of a rho-distorted point.
   * **Better W (the bound):** the **cycle itself** — the very thing we are trying
     to fix. Every cycle-breaker moves `W` away from optimal: `prox-escalate` and
     `fix` *freeze* it off-optimum (loose bound ~3–5%), and a large rho
     *overshoots* it. A strong prox forces consensus regardless of the duals (the
     `W`'s cancel in aggregate), which is exactly why the decision is good while
     the certificate is weak.
   * **Net:** for a good first-stage **decision**, escalating prox beats raising
     rho; for a tight dual **bound**, nothing beats letting the small-rho cycle
     run.

5. **Root cause of the cycle:** it is a **small-rho artifact**. `sizes`'s
   `_rho_setter` uses `rho = cost × 0.001`, squarely in the oscillating regime
   (rho ≤ 0.01 cycles; rho ≥ 0.1 converges early). Reducing rho pushes *further*
   into the bad regime — which is why the rho-reduction arm is counterproductive.

6. **Metric lesson:** a detector that watches `W` alone can be gamed by anything
   that freezes `W`; confirming an interruption actually *converged* (not merely
   *decoupled* or *damped*) needs a primal-side check like the gap — and judging
   whether the result is *good* needs the x/W bounds in §4, not the cycle metrics.

## Files

| file | role |
|---|---|
| `run_experiments.py` | orchestrator: runs every arm, solves the EF for `z*`, writes `summary.md` + a per-iteration CSV |
| `_run_one_arm.py` | worker: builds a PH-only hub on `sizes` and runs one arm |
| `w_osc_experiment_ext.py` | PH extension: detect (reusing PR1 primitives), intervene (incl. the prox-boost schedules), record `zc`/gap, and measure end-of-run x/W quality |
| `results/summary.md` | captured results (this run) |
| `results/interventions_by_iteration.csv` | per-iteration `zc` and `gap` for the intervention arms |
