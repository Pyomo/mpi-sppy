# What breaks the sizes-3 W-oscillation cycle?

Budget 80 iters; interventions from iter 10. Metrics averaged over the last 10 iterations (unless the run converged and stopped early).

## 1. Interventions (native rho)

| arm | zc | primal gap | reading |
|---|---|---|---|
| plain PH | 9.0 | 34 | still cycling |
| w_damping x0.5 | 9.0 | 33 | still cycling |
| rho reduction (geom x0.7) | 7.2 | 22261 | decoupled (gap exploded) |
| W reset + rho x0.5 | 13.0 | 141 | still cycling |
| prox-boost (x10, 5 iters, one-shot) | 8.5 | 34 | still cycling |
| prox-refire (x10, 5 iters, cooldown 5) | 10.6 | 16 | **converges (cycle broken)** |
| prox-hold (x10, held to end) | 13.0 | 25 | **converges (cycle broken)** |
| prox-escalate (x10 base, x2/5 iters) | 13.6 | 0 | **converges (cycle broken)** (converged @ iter 32) |
| smoothing (r=0.1, b=0.2) | 17.8 | 968 | cycle amplified |
| fix (slam analogue) | 5.0 | 0 | **converges (cycle broken)** (converged @ iter 48) |

## 2. rho level (uniform; native `_rho_setter` disabled)

| default_rho | zc | primal gap | reading |
|---|---|---|---|
| 0.001 | 18.1 | 73 | still cycling |
| 0.01 | 22.0 | 19 | **converges (cycle broken)** |
| 0.1 | 10.8 | 0 | **converges (cycle broken)** (converged @ iter 52) |
| 0.3 | 17.2 | 0 | **converges (cycle broken)** (converged @ iter 34) |
| 1 | 16.8 | 0 | **converges (cycle broken)** (converged @ iter 24) |

## 3. rho perturbation (native rho + jitter)

| perturbation | zc | primal gap | reading |
|---|---|---|---|
| native (eps 0) | 9.0 | 34 | still cycling |
| jitter +/-50% per-var | 9.4 | 91 | still cycling |
| jitter +/-50% per-call | 7.3 | 21 | **converges (cycle broken)** |

* **zc**: nonants the detector still flags (scale-free; fooled by frozen W).
* **primal gap** `sum_s p_s |x_s - xbar|`: low only when scenarios actually agree.

Only `fix` (changing the problem structure), an escalating prox boost, and a larger rho converge the gap. A *fixed*-magnitude state-perturbing move (W-damping, W-reset, rho reduction, rho jitter, a one-shot or re-firing prox boost) leaves the cycle intact, decouples the scenarios, or only damps it to a residual. `smoothing` *amplifies* the cycle (it anchors each scenario to its own lagged EMA, fighting consensus), worse as the ratio grows. The sizes cycle is a small-rho artifact -- its `_rho_setter` uses cost x 0.001.

## 4. Solution quality vs EF optimum (z* = 224275)

Sections 1-3 say whether an arm stopped *moving*; this says whether what it produced is any *good*. **x-gap** = expected cost of committing to the consensus xbar, above z* (small = a good decision). **W-gap** = Lagrangian bound from the final W, below z* (small = duals good enough to certify optimality). sizes is a MIP, so a duality gap keeps W-gap > 0 even for good W -- read it *relative* across arms.

| arm | inner Ū | x-gap | outer L | W-gap | reading |
|---|---|---|---|---|---|
| plain PH | 224872 | +0.27% | 222992 | +0.6% | good x, good W |
| w_damping x0.5 | 224878 | +0.27% | 223150 | +0.5% | good x, good W |
| rho reduction (geom x0.7) | 225107 | +0.37% | 223586 | +0.3% | good x, good W |
| W reset + rho x0.5 | 224868 | +0.26% | 222325 | +0.9% | good x, good W |
| prox-boost (x10, 5 iters, one-shot) | 224878 | +0.27% | 222948 | +0.6% | good x, good W |
| prox-refire (x10, 5 iters, cooldown 5) | 224891 | +0.27% | 222967 | +0.6% | good x, good W |
| prox-hold (x10, held to end) | 224944 | +0.30% | 223039 | +0.6% | good x, good W |
| prox-escalate (x10 base, x2/5 iters) | 224967 | +0.31% | 216655 | +3.4% | good x, **loose W** |
| smoothing (r=0.1, b=0.2) | 225735 | +0.65% | 223699 | +0.3% | good x, good W |
| fix (slam analogue) | 225060 | +0.35% | 213836 | +4.7% | good x, **loose W** |
| rho=0.001 | 225215 | +0.42% | 222607 | +0.7% | good x, good W |
| rho=0.01 | 225635 | +0.61% | 214655 | +4.3% | good x, **loose W** |
| rho=0.1 | 229380 | +2.28% | 212513 | +5.2% | poor x, loose W |
| rho=0.3 | 230894 | +2.95% | 180390 | +19.6% | poor x, loose W |
| rho=1 | 231089 | +3.04% | 209067 | +6.8% | poor x, loose W |

**Convergence and solution quality are different axes.** The native small-rho cycle never meets a convergence *criterion*, yet it orbits the optimum: its average xbar is near-optimal (x-gap ~0.3%) and its W gives the *tightest* Lagrangian bound (W-gap <1%). Raising rho uniformly converges fast but to a *worse* consensus -- both gaps blow up (rho=1: x ~3%, W loose; rho=0.3: W-gap ~20%) -- so "a larger rho converges" costs real solution quality. An **escalating prox boost is the sweet spot for the decision**: it converges (primal gap -> 0) and keeps x near-optimal, because it retains the small native dual step and settles near the cycle's centre rather than a distorted large-rho point. But a strong prox forces primal consensus regardless of the duals (the W's cancel in aggregate), so it *freezes* W off-optimum -- a loose dual bound (W-gap ~3%). `fix` shares that profile. Net: for a good first-stage **decision**, escalating prox beats raising rho; for a tight dual **bound**, none of the cycle-breakers beats letting the small-rho cycle run.
