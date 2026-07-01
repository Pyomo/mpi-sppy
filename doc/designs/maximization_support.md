# Maximization support in mpi-sppy

Status: record of the maximization-correctness audit and the per-feature support
matrix it produced.

## Summary

mpi-sppy supports both minimization and maximization problems
(`sense=pyo.minimize` / `sense=pyo.maximize` on the scenario objective). There
is a single source of truth for the sense: `SPBase.is_minimizing`, set once in
`SPBase._set_sense` from `sputils._models_have_same_sense`, which **raises a
`RuntimeError` if the scenarios do not all share the same sense**.

This document records the result of an audit whose guiding principle was: every
feature must either *work correctly* for a maximization problem or *raise a
clear error* — never silently produce a wrong answer. "Runs without error" is
not the bar; correctness (e.g. bounds bracketing the optimum on the correct,
sense-dependent side) is.

The core was found to be far more sense-aware than a "min-biased" reading would
suggest. The real risks were untested paths (now tested) and two clusters of
genuinely-silent wrongness (the confidence-interval pipeline and some agnostic
guests), now fixed or guarded.

## Support matrix

| Feature | Maximization status | Tested for max |
|---|---|---|
| Extensive form (`ExtensiveForm`, `create_EF`) | works (objective assembled with native sense; mixed sense rejected) | yes |
| Progressive Hedging (`PH`) | works (prox/W augmentation sign flips on sense, `phbase.py`) | yes |
| Subgradient | works (inherits PH) | yes |
| FWPH | works (negates objective to a minimize QP) | yes |
| L-shaped (`LShapedMethod`) | works (negates objective internally; bound negated back) | yes |
| Cylinders bound/convergence (hub, spoke, spcommunicator) | works (inner/outer init, comparators, gaps flip on sense) | yes (Lagrangian outer bound) |
| xhat / incumbent selection + rho setters / fixers / reduced-cost | works (funnels through sense-aware primitives; rho heuristics use `abs`) | partial (via PH/cylinders) |
| CG / DualCG | works | yes (pre-existing) |
| MMW confidence interval (`mmw_ci`, `gap_estimators`) | works (gap sign + bound magnitude corrected by sense) | yes |
| CVaR (`utils/cvar.py`) | works (lower/reward-tail Rockafellar-Uryasev mirror) | yes (closed-form) |
| Scenario bundling (`proper_bundler`) | works (delegates to the EF builder; no sense logic of its own) | by construction |
| ADMM wrappers (`admmWrapper`, `stoch_admmWrapper`) | works (sense-preserving scaling; consensus penalty applied by PH) | by construction |
| Chance constraints (`utils/chance_constraint.py`) | works (adds only a feasibility aggregator; sense-independent) | by construction |
| Linearized prox (`utils/prox_approx.py`) | works (cuts are lower bounds on `x^2`; objective sign-flip drives them; placement heuristic uses `abs`) | via PH max test |
| `generic_cylinders.py`, `problem_io/mps_reader.py` | works (sense passed through / mapped from the model) | by construction |
| Agnostic Pyomo guest (`agnostic/pyomo_guest.py`, loose example) | works (reads the model's sense) | yes |
| Agnostic GAMS guest (`agnostic/gams_guest.py`) | runs (negates a `maximizing` model to a minimize internally) — but its only test is a smoke test that does not assert objective/bound values, so the sign-correctness of reported bounds for max is **unverified** | smoke only |

## Minimization-only (raise a clear error or are not a maximization surface)

| Feature | Behavior on max |
|---|---|
| Sequential sampling (`seqsampling.SeqSampling`, `multi_seqsampling.IndepScens_SeqSampling`) | **raises `RuntimeError`** — the BM/BPL stopping criteria and sample-size rules assume a non-negative, shrinking optimality gap. A vetted gap-magnitude formulation would be needed to support max. |
| Agnostic AMPL guest (`agnostic/ampl_guest.py`) | **raises `RuntimeError`** — the guest splices the PH term into a `minimize` objective; minimize-only. |
| Agnostic gurobipy guest (`examples/farmer/agnostic/farmer_gurobipy_agnostic.py`) | **raises `RuntimeError`** on `GRB.MAXIMIZE` — minimize-only (PH prox/weight terms use a fixed minimize sign). |

### Note on the agnostic guests

By project guidance the agnostic guests are a minimize-only surface; there is no
plan to support maximization in them. The AMPL and gurobipy guests raise a clear
`RuntimeError` on a maximization model rather than silently mis-signing it. The
Pyomo guest happens to handle maximization (its sense was simply being read
instead of hardcoded). The GAMS guest predates this guidance and *runs* a
`maximizing` model by negating it to a minimize internally; its test is only a
smoke test (it does not assert the reported objective/bounds), so the
sign-correctness of those reported values under maximization is unverified. None
of the guests should be relied on as a maintained maximization surface.

## Tests added by the audit

- `mpisppy/tests/test_maximization.py` — EF, PH, subgradient, L-shaped, FWPH on
  a maximize farmer; each solves both senses and checks `max == -min` and that
  bounds bracket the optimum on the correct side.
- `mpisppy/tests/test_with_cylinders.py::test_lagrangian_max` — hub→spoke outer
  bound under maximization (a valid upper bound);
  `test_wheel_sign_flip_equivalence` — the same PH-hub + xhatshuffle-spoke wheel
  run min and max, asserting every reported bound is the exact negation of its
  counterpart and brackets the optimum on the correct side.
- `mpisppy/tests/test_ciutils.py` / `test_conf_int_farmer.py` — `correcting_numeric`
  sense handling, and end-to-end maximize `gap_estimators` / MMW, plus a
  sequential-sampling-rejects-max test.
- `mpisppy/tests/test_cvar.py` — closed-form maximize CVaR (lower-tail).
- `mpisppy/tests/test_agnostic.py::test_agnostic_pyomo_PH_maximize` — the Pyomo
  guest under maximization; `test_agnostic_gurobipy_maximize_raises` — the
  gurobipy guest rejects maximization. (The AMPL min-only guard is exercised by
  the CI agnostic job, which has that backend installed.)
