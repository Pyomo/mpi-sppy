# What breaks the sizes-3 W-oscillation cycle?

Budget 60 iters; interventions from iter 10. Metrics averaged over the last 10 iterations (unless the run converged and stopped early).

## 1. Interventions (native rho)

| arm | zc | primal gap | reading |
|---|---|---|---|
| plain PH | 11.0 | 94 | still cycling |
| w_damping x0.5 | 12.1 | 94 | still cycling |
| rho reduction (geom x0.7) | 9.1 | 23757 | decoupled (gap exploded) |
| W reset + rho x0.5 | 13.9 | 243 | still cycling |
| fix (slam analogue) | 9.6 | 0 | **converges (cycle broken)** (converged @ iter 35) |

## 2. rho level (uniform; native `_rho_setter` disabled)

| default_rho | zc | primal gap | reading |
|---|---|---|---|
| 0.001 | 18.2 | 60 | still cycling |
| 0.01 | 24.0 | 57 | still cycling |
| 0.1 | 24.0 | 6 | **converges (cycle broken)** |
| 0.3 | 20.9 | 0 | **converges (cycle broken)** (converged @ iter 25) |
| 1 | 15.9 | 0 | **converges (cycle broken)** (converged @ iter 12) |

## 3. rho perturbation (native rho + jitter)

| perturbation | zc | primal gap | reading |
|---|---|---|---|
| native (eps 0) | 11.0 | 94 | still cycling |
| jitter +/-50% per-var | 11.0 | 68 | still cycling |
| jitter +/-50% per-call | 11.8 | 80 | still cycling |

* **zc**: nonants the detector still flags (scale-free; fooled by frozen W).
* **primal gap** `sum_s p_s |x_s - xbar|`: low only when scenarios actually agree.

Only `fix` (changing the problem structure) and a larger rho converge the gap. Every state-perturbing move leaves the cycle intact, decouples the scenarios, or worsens it. The sizes cycle is a small-rho artifact -- its `_rho_setter` uses cost x 0.001.
