# Value of the Stochastic Solution (VSS) design (two-stage first)

Status: design approved by DLW (all §10 decisions resolved). V1 implemented
(`mpisppy/generic/vss.py`, `--vss` flag, `mpisppy/tests/test_vss.py`,
`doc/src/vss.rst`).

Related: `doc/designs/pysp_but_not_mpisppy.md` §A7 (PySP `ef_vss.py` —
`create_expected_value_instance` + `fix_ef_first_stage_variables` — has no
mpi-sppy equivalent). This design closes that gap by *reusing* machinery
that already shipped, rather than porting PySP's code.

---

## 0. What VSS is (and why anyone cares)

The **Value of the Stochastic Solution** measures how much you gain by
actually modeling the uncertainty instead of collapsing it to a single
"average" future and solving that deterministic problem.

The comparison is between two *first-stage* (here-and-now) decisions:

- the decision you get from the full stochastic program, and
- the decision you get from the **mean-value problem** — replace every
  random parameter by its expected value, solve the resulting
  deterministic model, and take its first-stage solution.

The mean-value decision is cheap and intuitive, and practitioners reach
for it constantly ("just plan for the average demand"). VSS answers the
question *"what does that shortcut cost me, in expectation, once the real
uncertainty shows up?"* A large VSS is the empirical justification for
having built a stochastic model at all; a near-zero VSS says the
deterministic shortcut was almost as good and the modeling effort bought
little.

### 0.1 The three numbers (minimization convention)

Write the two-stage stochastic program with first-stage vector `x`,
random data `ξ`, and second-stage value function `Q(x, ξ)`:

```
RP  = min_x { c·x + E_ξ[ Q(x, ξ) ] }
```

- **RP** — *Recourse Problem* value: the optimal objective of the full
  stochastic program. This is the here-and-now solution; **the run already
  computes it** (see §3 on where it comes from and how exact it is).

- **EV** — *Expected-Value (mean-value) problem*: replace `ξ` by its mean
  `ξ̄` and solve the deterministic model
  `EV = min_x { c·x + Q(x, ξ̄) }`.
  Call its optimal first-stage solution `x̄`. Note `EV` is an *objective
  value*; `x̄` is the *decision* we actually reuse. In the SAA world
  mpi-sppy lives in, `ξ̄` is the average of the data over the explicit
  scenario set the run uses — exactly what `average_scenario_creator`
  already builds for Jensen's bound.

- **EEV** — *Expected result of the EV solution*: pin the first stage at
  `x̄`, then honestly evaluate expected cost across **every real
  scenario**:
  `EEV = c·x̄ + E_ξ[ Q(x̄, ξ) ]`.

Then

```
VSS = EEV − RP        (minimization; VSS ≥ 0 always)
```

`VSS ≥ 0` because `x̄` is feasible for the stochastic program but not
necessarily optimal, so evaluating it can only do worse than `RP`.

For a **maximization** model every inequality flips and the definition is
`VSS = RP − EEV` (still `≥ 0`). The implementation reads
`is_minimizing` off the scenarios and picks the sign; VSS is always
reported as a non-negative "cost of using the average."

### 0.2 Not to be confused with EVPI

VSS is frequently confused with **EVPI** (Expected Value of Perfect
Information), `EVPI = RP − WS`, where `WS` ("wait-and-see") is the
expected value of solving each scenario *with perfect foresight*. EVPI
measures the value of *knowing the future*; VSS measures the value of
*modeling that you don't*. They are different numbers and answer
different questions. This design implements **VSS only**; EVPI is a
natural sibling (it reuses the same evaluation plumbing) and is noted as
future work in §8.

---

## 1. Front-and-center warning: VSS can be expensive

**This is a required, prominent note in the user-facing docs and in the
`--vss` help string.**

Computing VSS is *not* free relative to the run that produced `RP`. The
three pieces cost very differently:

| Quantity | Work | Typical cost |
| :--- | :--- | :--- |
| `RP` | already done by the run | free |
| `EV` / `x̄` | one deterministic solve of the average scenario | cheap |
| **`EEV`** | **fix `x̄`, solve the recourse subproblem for every scenario** | **an extra pass over all scenarios** |

`EEV` is the added piece: `N` second-stage solves with the first stage
fixed. How much wall-clock that adds is genuinely hard to state in general,
and the naive "it's a second solve, so ~2x" intuition is usually wrong.
Fixing `x̄` *decouples* the scenarios and removes the here-and-now decisions,
so each `EEV` subproblem is smaller and easier than the original coupled
solve:

- vs. an **EF** run, `EEV` solves `N` decoupled fixed-first-stage
  subproblems instead of one big coupled model — typically much cheaper,
  and dramatically so for a MIP (fixing integer first-stage vars is exactly
  what makes the recourse easy).
- vs. a **decomposition** run, `EEV` is roughly *one iteration's* worth of
  subproblem solves, against the run's many iterations — a small fraction.

So `--vss` adds real but usually sub-linear work; it becomes noticeable
mainly with very many scenarios or genuinely expensive recourse. The docs
should say this honestly rather than promise a fixed multiplier.

**Solver options / mipgap.** The `EV` and `EEV` solves reuse the run's
solver and `solver_options` (via `solver_specification`) — `EF_solver_options`
after an `--EF` run, `solver_options` after a decomposition run — so all
three numbers are solved consistently, and any `mipgap` in that option
string applies to the VSS solves too. This matters because `VSS = EEV - RP`
is a *difference* of two optimized values: with a loose gap a small VSS is
gap-noise. V1 threads only the solver-options *string*, not the separate
`EF_mipgap` / `*_iter*_mipgap` knobs or a `*_solver_options_file` (a
possible later refinement).

A second, subtler cost: if `RP` came from a decomposition run that did
**not** close the optimality gap, then `RP` is only known to lie in a
bracket, and so is VSS (§3.2). Getting a tight point value for VSS may
require solving the extensive form for an exact `RP`, which can be far
more expensive than the decomposition run itself. `--vss` never does that
automatically; it reports the bracket instead.

---

## 2. User-facing surface: the `--vss` flag

The request calls it `-vss`; in mpi-sppy's Pyomo-`ConfigDict` CLI
convention that is the boolean flag **`--vss`** (single-dash `-vss` would
argparse-split into `-v -s -s`). Default `False`.

Behavior: after the main algorithm finishes and `RP` is known, if
`cfg.vss` is set, `generic_cylinders` computes `EV`, `x̄`, `EEV`, and
`VSS` and prints a **VSS report** at the end of the run. It is
**report-only** — it never changes the solution, the bounds, or any
written solution file. It is an optional add-on to *either* an `--EF`
run or a decomposition (cylinders) run.

Help string (drafted, includes the cost warning):

```
--vss   After the run, report the Value of the Stochastic Solution
        (VSS = EEV - RP). Requires the scenario module to define
        average_scenario_creator. WARNING: computing EEV re-solves every
        scenario with the first stage fixed and can roughly double run
        time on large/integer models. Two-stage only (see docs).
```

### 2.1 Module contract and hard-fail

`--vss` reuses the **same contract as Jensen's bound**: the scenario
module must define

```python
def average_scenario_creator(scenario_name, **kwargs):
    """Return a single deterministic scenario built from the sample-mean
    of the random data (probability 1.0)."""
```

This already exists in `examples/farmer/farmer.py`,
`examples/sizes/sizes.py`, and `examples/netdes/netdes.py`, and is
discovered with the established `getattr(module, "average_scenario_creator",
None)` pattern.

If `--vss` is set and the module has no `average_scenario_creator`, fail
**early and loudly at setup** (not after the whole run) with a message
that names the missing function and points at the Jensen's docs — the
same fail-fast posture used by the xhat-feasibility-cuts and Jensen's
paths. Wasting a long solve only to discover at the report step that VSS
cannot be computed is the failure mode to avoid.

### 2.2 V1 scope restrictions (explicit, hard-error)

To keep V1 honest and small, `--vss` refuses to combine with transforms
that change what "the objective" means, mirroring how `--cvar` already
refuses proper bundles / ADMM (`generic_cylinders.py`):

- **Two-stage only.** Multistage is out of scope for V1 (§7).
- **No proper bundles, no ADMM, no CVaR** in V1. Each of these rewrites
  the scenario objective or the scenario/first-stage structure, so `RP`
  and `EEV` would have to be defined against the *transformed* problem to
  be comparable. That is meaningful but subtle (what is "the average
  scenario" of a risk measure?) and is deferred. Combining `--vss` with
  any of them is a clear setup-time error in V1.

Each restriction is a one-line guard with an explanatory message, not a
silent no-op.

---

## 3. Where `RP` comes from, and how exact it is

`RP` is the objective of the stochastic program, and its exactness
depends on which driver path produced it.

### 3.1 `--EF` path — exact `RP`

`do_EF` already solves the extensive form and has
`ef.get_objective_value()`. That *is* `RP`, exactly (up to the solver's
own gap). VSS from an EF run is a clean point value.

### 3.2 Decomposition path — bracketed `RP`

`do_decomp` returns the `WheelSpinner`, which exposes
`wheel.BestInnerBound` and `wheel.BestOuterBound`. For a minimization
run the true `RP` satisfies

```
BestOuterBound ≤ RP ≤ BestInnerBound
```

so, with `EEV` fixed,

```
VSS  =  EEV − RP     ∈  [ EEV − BestInnerBound ,  EEV − BestOuterBound ].
```

**Resolved (§10.2):** on a decomposition run the report therefore always
does both:

- prints the **point value** `VSS = EEV − BestInnerBound` (using the
  incumbent — the honest here-and-now value the run actually achieved),
  clearly labeled as *conservative* (it underestimates the true VSS,
  since the incumbent is `≥ RP`), and
- prints the **bracket** `[EEV − BestInnerBound, EEV − BestOuterBound]`
  whenever the run's gap is nonzero, so the reader sees exactly how much
  of the uncertainty in VSS is just the unclosed optimality gap.

We report the incumbent-based point value rather than refusing one,
because the incumbent is the decision the run actually stands behind; the
always-printed bracket keeps that honest. The bracket falls out for free
from bounds the wheel already carries; no extra solve. The maximization
case swaps inner/outer roles accordingly.

---

## 4. Where `EV`, `x̄`, and `EEV` come from — reuse, don't reinvent

All three map onto functions that already exist and are tested.

### 4.1 `EV` and `x̄` — the mean-value solve

`mpisppy/utils/xhat_helpers.py::average_xhat_nonants` already:

1. calls `average_scenario_creator(...)`,
2. asserts two-stage,
3. solves it, and
4. returns the ROOT first-stage values as a 1-D `np.ndarray`.

That array **is** `x̄`. For VSS we additionally want the *objective* of
that solve to report `EV`, which `average_xhat_nonants` does not return.
**Resolved (§10.1):** `do_vss` builds and solves the average scenario
**inline** — a few lines that call `average_scenario_creator`, solve, and
read both `pyo.value(<objective>)` (→ `EV`) and the root nonants (→ `x̄`).
This keeps `xhat_helpers.average_xhat_nonants` untouched; we do not add an
objective-returning sibling unless a second caller ever needs one.

`EV` is reported for context (and to sanity-check `EEV ≥ EV`), but it is
**not** part of the VSS arithmetic — only `x̄` is.

### 4.2 `EEV` — honest cross-scenario evaluation

This is exactly what `Xhat_Eval.evaluate` does
(`mpisppy/utils/xhat_eval.py:257`): fix a nonant cache across all local
scenarios, `solve_loop`, and return the probability-weighted
`Eobjective`. Construct an `Xhat_Eval` over **all** scenarios (the
canonical pattern is in `confidence_intervals/ciutils.py:403` and
`zhat4xhat.py:94`), pack `x̄` as `{"ROOT": x̄}`, and call `evaluate`:

```python
ev = Xhat_Eval(options, all_scenario_names, scenario_creator,
               scenario_creator_kwargs=scenario_creator_kwargs,
               all_nodenames=all_nodenames)
EEV = ev.evaluate({"ROOT": xbar})     # xbar is x̄ from §4.1
```

`Xhat_Eval` already distributes scenarios across the MPI comm and reduces
the expected objective, so `EEV` is computed in parallel with no new
communication code. Crucially, `evaluate` uses the **same
`scenario_creator`** the run used, so `EEV` and `RP` are the same
objective measured two ways — apples to apples.

### 4.3 Infeasibility of the mean-value solution is a *real answer*

If `x̄` (built for the average scenario) is infeasible when fixed into
some real scenario — i.e. the model lacks relatively complete recourse —
then `Q(x̄, ξ) = +∞` for that scenario, `EEV = +∞`, and `VSS = +∞`. This
is not a bug; it is the strongest possible statement that the
deterministic shortcut is *unusable*. The report must therefore:

- detect per-scenario infeasibility from the evaluate pass rather than
  crashing,
- list which scenarios were infeasible, and
- print `EEV = +inf`, `VSS = +inf` with a one-line explanation.

This is the same "fix a candidate, it may not be feasible everywhere"
concern documented for feasible-xhat (`doc/src/feasible_xhat.rst`); VSS
does not attempt any model-specific repair — a repaired `x̄` would no
longer be *the mean-value decision*, so repairing it would silently
change the quantity being measured.

---

## 5. The report

Printed once, on `cylinder_rank == 0` (per the rank-gating convention),
via `global_toc`, after the run's normal output. Sketch:

```
================= VSS report =================
  RP  (stochastic solution, here-and-now)   :   108900.0    [EF, exact]
  EV  (mean-value problem objective)         :   107240.0
  EEV (EV first stage over all scenarios)    :   115405.6
  VSS = EEV - RP                             :     6505.6    (5.98% of |RP|)
=============================================
```

Decomposition variant adds the bracket:

```
  RP  (stochastic solution, here-and-now)   :   108900.0    [decomposition incumbent]
      optimality bracket [outer, inner]      : [108640.0, 108900.0]
  ...
  VSS = EEV - RP (point, conservative)       :     6505.6
      VSS bracket [EEV-inner, EEV-outer]     : [  6505.6,   6765.6]
```

Infeasible variant:

```
  EEV : +inf   (EV solution infeasible in scenarios: Scenario7, Scenario12)
  VSS : +inf   (mean-value first stage is not usable across all scenarios)
```

The VSS percentage is `VSS / |RP|` guarded against `RP == 0`.

---

## 6. Code layout

Mirror the MMW post-run report exactly.

- **New `mpisppy/generic/vss.py`** with `do_vss(module, cfg, ef=None,
  wheel=None, scenario_creator=..., scenario_creator_kwargs=...)`,
  paralleling `mpisppy/generic/mmw.py::do_mmw`. It sources `RP` from `ef`
  or `wheel`, computes `EV`/`x̄`/`EEV`, and prints the report. A
  `vss_requested(cfg)` predicate mirrors `mmw_requested`.
- **`mpisppy/generic_cylinders.py`**: after `do_EF` (pass the returned
  `ef`) and after `do_decomp` (pass the returned `wheel`), call `do_vss`
  when `cfg.get("vss")`. Two call sites, symmetric with the existing
  `do_mmw` calls (`generic_cylinders.py:141,146`).
- **`mpisppy/utils/config.py`**: one `add_to_config('vss', ...)` boolean,
  in whichever driver-level args group owns post-run report options
  (alongside the MMW flags), so it is available to `generic_cylinders`.
- **`mpisppy/utils/xhat_helpers.py`**: untouched. `do_vss` solves the
  average scenario inline to get both `EV` and `x̄` (§4.1); no
  objective-returning sibling is added in V1.

No spoke edits, no hub edits, no changes to the algorithms — VSS is
strictly a post-processing consumer of results the drivers already
return, exactly like MMW.

---

## 7. Multistage (future, but the plumbing is already here)

Two-stage is V1. The multistage generalization is well-defined and the
building blocks exist:

- The mean-value analogue is the **expected-value tree** — the real
  branching structure with the random data at each node pinned to its
  conditional mean. Solving that EF yields a first-stage-through-last
  decision at *every* non-leaf node.
- `mpisppy/utils/xhat_helpers.py::ef_xhat_nonants` already solves an EF
  and returns the whole `{node_name: np.ndarray}` nonant tree — the exact
  cache form `Xhat_Eval._fix_nonants` / `fix_nonants_upto_stage`
  consume. So `EEV` for the multistage case is "fix the whole EV tree,
  evaluate across real scenarios" and needs no new evaluation code.

What is missing for multistage is a **module contract for building the
expected-value tree** (an `ev_tree` creator, or a documented convention
for driving `average_scenario_creator` per node). That contract, plus
the choice among the several textbook multistage-VSS variants, is why
multistage is deferred rather than shipped in V1. The `--vss` two-stage
guard (§2.2) is the placeholder that later lifts.

---

## 8. Testing plan

New `mpisppy/tests/test_vss.py` (and wire it into `run_coverage.bash`
**and** `test_pr_and_main.yml` in the same commit, per the coverage-harness
convention):

1. **Farmer EF, exact RP.** Run `--EF --vss`. Independently, in the test:
   solve the average scenario for `x̄`, `Xhat_Eval.evaluate({"ROOT": x̄})`
   for `EEV`, and `ef.get_objective_value()` for `RP`. Assert the
   reported VSS equals `EEV − RP` to tolerance, and that `VSS ≥ 0`.
2. **Farmer decomposition, bracket.** Run cylinders `--vss`; assert the
   point VSS uses `BestInnerBound` and the printed bracket endpoints
   match `EEV − BestInnerBound` / `EEV − BestOuterBound`.
3. **Maximization sign.** Sign-flipped farmer variant; assert
   `VSS = RP − EEV ≥ 0` and that the report labels it correctly (reuse
   the max-support test fixtures).
4. **Missing `average_scenario_creator`.** A module without it + `--vss`
   ⇒ `RuntimeError` at setup, before any long solve.
5. **Infeasible mean-value solution.** A small model with no relatively
   complete recourse where `x̄` is infeasible in ≥1 scenario ⇒ report
   shows `EEV = +inf`, `VSS = +inf`, names the offending scenarios, and
   does not crash.
6. **V1 restriction guards.** `--vss` with `--EF`... plus `--cvar` (or a
   proper-bundle / ADMM config) ⇒ clear setup-time error.

---

## 9. Documentation

- New `doc/src/vss.rst`: the §0 "what VSS is" explanation (RP/EV/EEV/VSS,
  the minimization convention, and the EVPI distinction), the §1 cost
  warning **at the top**, a worked farmer example, and the decomposition
  bracket + infeasibility semantics. Cross-link `doc/src/jensens.rst`
  (shared `average_scenario_creator` contract) and
  `doc/src/feasible_xhat.rst` (shared fix-a-candidate feasibility caveat).
- Add `--vss` to the generic_cylinders option reference.
- One line in `doc/designs/pysp_but_not_mpisppy.md` §A7 noting the gap is
  addressed (two-stage) once this ships.

---

## 10. Resolved decisions

All resolved; implementation can proceed.

1. **`EV` objective plumbing (§4.1/§6):** solve the average scenario
   **inline** in `do_vss` and read both its objective (`EV`) and root
   nonants (`x̄`). `xhat_helpers.average_xhat_nonants` stays untouched; no
   objective-returning sibling until a second caller needs one.
2. **Decomposition point value (§3.2):** report the **incumbent-based
   point value** `VSS = EEV − BestInnerBound`, labeled conservative, and
   **always** print the `[EEV − BestInnerBound, EEV − BestOuterBound]`
   bracket when the gap is open. Do not refuse a point value.
3. **EVPI (§0.2):** **VSS only** in V1, for a review-sized PR. EVPI is a
   natural follow-on (it reuses the same evaluate plumbing, with WS =
   per-scenario perfect-foresight solves) and is left to a later PR.
```
