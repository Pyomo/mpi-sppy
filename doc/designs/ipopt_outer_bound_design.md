# `ipopt_outer_bound` — a certified outer-bound cylinder for convex NLP subproblems

Status: draft for review. Branch `ipopt-outer-bound` (off Pyomo/mpi-sppy `main`).

The mathematics is set out separately, with proofs, in
[`ipopt_outer_bound_certificate.tex`](ipopt_outer_bound_certificate.pdf) —
weak duality, the box underestimator, exactness at a KKT point, the canonical-form
sign condition, and the aggregation hypothesis. This document covers the design
decisions and the implementation; that one covers why the bound is valid.

## 1. Goal

Give mpi-sppy an outer bound for stochastic programs whose scenario subproblems are
**convex NLPs solved with Ipopt**. Today there is none: the Lagrangian spoke's bound
comes from the solver's dual bound, Ipopt is not a branch-and-bound solver and reports
no dual bound, and the spoke says so out loud:

```python
# mpisppy/cylinders/lagrangian_bounder.py
if "ipopt" in self.opt.options["solver_name"]:
    print("\n WARNING: An ipopt solver will not give outer bounds\n")
```

Measured on this branch, a converged Ipopt solve through Pyomo returns:

| field | value |
|---|---|
| `results.problem.lower_bound` / `upper_bound` | `-inf` / `inf` (Pyomo's untouched defaults) |
| `results.solution[0].objective` | `{}` — empty, even with `load_solutions=False` |
| `results.solver.message` | `'Ipopt 3.13.2\x3a Optimal Solution Found'` |

So `Ebound()` sums `-inf` and a user running a convex stochastic NLP gets no bound at
all. This design supplies one that is *certified* — valid by a theorem, not by an
assertion that the solver converged.

Scope is **Ipopt only**, by decision. The mechanism generalizes to any NLP solver
returning duals, but nothing here is written to be solver-neutral, and the dual sign
conventions in §5 are measured from Ipopt.

## 2. Why this cannot just read a number off the solver

For a minimization subproblem the solver's returned objective value is an *inner*
bound: it is the value at a point, hence ≥ the subproblem optimum. An outer bound
needs a number ≤ the optimum. Ipopt supplies the former and never the latter.

Ipopt's `tol` does not close the gap either. It is a **relative KKT-residual**
tolerance (the binary's own `-=` listing: "Desired convergence tolerance (relative)"),
not an objective-error tolerance; converting one to the other needs multiplier
magnitudes and curvature. And the two error sources point opposite ways: optimality
error stops you slightly *above* the optimum (unsafe for an outer bound), while
constraint violation — `constr_viol_tol` defaults to `1e-4`, four orders looser than
`tol` — can put you slightly *below* it. A single scalar cushion covers neither
honestly.

What *is* rigorous is Lagrangian weak duality, which needs no convergence assumption
at all. That is what this cylinder computes.

## 3. The bound

### 3.1 Setting

The hub relaxes non-anticipativity with weights `W`. Provided the weights satisfy the
usual condition `Σ_s p_s W_s = 0`, the scenario-separable Lagrangian dual value

```
    D(W) = Σ_s p_s L_s(W_s)  ≤  optimum
```

is an outer bound, where the scenario subproblem — exactly the problem the Lagrangian
spoke already builds and solves, prox term off — is

```
    L_s(W_s) = min  f_s(v) + W_sᵀ x(v)
               s.t. g_s(v) ≤ 0,  h_s(v) = 0,  v ∈ B = [lo, hi]
```

`v` is every variable of the scenario model (non-anticipative `x` plus recourse), and
`B` is its box of variable bounds.

**Convexity assumption (load-bearing, unverifiable in general):** `f_s` convex, each
component of `g_s` convex, `h_s` affine, all over `B`. §6 lists the parts of this that
*are* mechanically checkable; the rest is a user assertion.

Note that the requirement is on the **canonical** `g`, which negates the body of a `>=`
row, so it is not the same as "the constraint body is convex":

| as written | canonical `g` | requirement on the body |
|---|---|---|
| `body ≤ upper` | `body − upper` | convex |
| `body ≥ lower` | `lower − body` | **concave** |
| `lo ≤ body ≤ up` | both rows | affine |
| `body == rhs` | `body − rhs` | affine |

The theorem applies to `x² ≤ 4` but not to `x² ≥ 1`, though both are written with a
convex body — and the feasible set of the second is not convex at all. This is worth
stating loudly because it is easy to read the wrong way: a code review of this branch
turned up that the guard, the unit test and `spokes.rst` had all settled on "only
equalities need to be affine", which let `min x s.t. x² ≥ 1, x ∈ [0, 1.5]` through and
certified 1.25 for a problem whose optimum is 1.0 — an outer bound above the optimum.

### 3.2 The dual function and the trap

For multipliers `λ ≥ 0` and free `μ`, define

```
    φ_s(v) = f_s(v) + W_sᵀ x(v) + λᵀ g_s(v) + μᵀ h_s(v)
    q_s(λ, μ) = inf_{v ∈ B} φ_s(v)                        ≤ L_s(W_s)   (weak duality)
```

Weak duality holds for **any** `λ ≥ 0` and **any** `μ` — no optimality of the
multipliers is required. That is the whole reason this approach is rigorous where
"trust the objective value" is not.

The trap: `q_s` is defined by an *infimum*. Handing `φ_s` to Ipopt and solving it
returns a point, and the value at that point is ≥ the infimum — an upper bound on
`q_s`, which is the wrong direction again. **A second NLP solve does not by itself
produce a certificate.** This is the one place where the original sketch for this
cylinder (solve, take λ, solve the dual, report) does not close.

### 3.3 What does close it: a linear underestimator over the box

`φ_s` is convex on `B`, so for any point `v̂ ∈ B` the tangent at `v̂` lies below it:

```
    φ_s(v) ≥ φ_s(v̂) + ∇φ_s(v̂)ᵀ (v − v̂)       for all v
```

Minimizing the right-hand side over the box is separable and closed-form, so

```
    q_s(λ, μ)  ≥  φ_s(v̂) + Σ_i  min_{v_i ∈ [lo_i, hi_i]}  ∂_i φ_s(v̂) · (v_i − v̂_i)
                              ╰──────────────────────────────────────────────────╯
                              = ∂_i φ · (lo_i − v̂_i)   if ∂_i φ > 0
                                ∂_i φ · (hi_i − v̂_i)   otherwise
```

Call the right-hand side `q̂_s`. Then

```
    q̂_s  ≤  L_s(W_s)      for ANY v̂ ∈ B, ANY λ ≥ 0, ANY μ
```

and the cylinder reports `Σ_s p_s q̂_s`. One gradient evaluation and a loop over
variables — no second solve, no tolerance argument, no feasibility requirement on `v̂`.

**Looseness has a closed form, and it is the box width that sets it.** Each term is
`|∂_i φ|` times the distance from `v̂_i` to the far end of its interval, so

```
    L_s(W_s) − q̂_s  ≈  Σ_i |∂_i φ(v̂)| · (width of the box in the descending direction)
```

Confirmed numerically: on the running example a converged solve leaves
`|∂_x φ| = 5.1e−11` over a box of width 9, and the observed gap to the analytic optimum
is 4.56e−10 ≈ 5.1e−11 × 9. The practical consequence is the one worth telling users:
**this cylinder is only as tight as the variable bounds are.** A model whose bounds come
back from `fbbt` as ±1e10 will produce valid but useless numbers — 5e−11 × 1e10 is half a
unit of objective — which is the same failure the unbounded case in §6.1 reaches in the
limit, not a different one.

### 3.4 Why the correction term is ~0 in normal operation

At an exact KKT point of the scenario subproblem, with its multipliers `(λ, μ)` and
bound multipliers `z_L, z_U ≥ 0`, stationarity says exactly

```
    ∇φ_s(v̂) = z_L − z_U
```

Componentwise: an interior `v̂_i` has `z_L,i = z_U,i = 0`, so `∂_i φ = 0` and the term
is 0. A `v̂_i` sitting at `lo_i` has `∂_i φ = z_L,i ≥ 0`, so the minimizing `v_i` is
`lo_i = v̂_i` and the term is again 0; symmetrically at `hi_i`. **The correction
vanishes at an exact KKT point**, where `q̂_s = φ_s(v̂) = f_s(v̂) + W_sᵀx̂ = L_s(W_s)`
by complementarity — strong duality, recovered exactly.

So the correction term is precisely the price of inexactness, and it measures itself.
Measured on a 3-variable convex NLP with one convex `≤`, one active `≥`, one equality,
and finite bounds:

| solve | `φ(v̂)` | correction | certified bound | vs. reference |
|---|---|---|---|---|
| converged | 15.93325207 | −3.6e−09 | 15.93325207 | tight to 9 digits |
| `max_iter=5` | 15.93321 | −3.8e−02 | 15.89485 | valid, 0.04 loose |
| `max_iter=3` | 14.35348 | −2.6e+01 | −11.85 | valid, useless |
| `max_iter=2` | 18.86760 | −6.9e+01 | −49.97 | valid, useless |

Note the `max_iter=2` row: the naive "use the objective value" bound would have been
18.87, which is **above** the optimum — an invalid outer bound. The certificate
returns −49.97 instead: worthless, but sound. That is the trade this design makes
everywhere.

### 3.5 Consequence for cost: one solve, not two

Because `v̂` from the ordinary subproblem solve is already the minimizer of `φ_s` over
the box (§3.4), the anticipated second solve per iteration is very nearly a no-op at a
converged first solve. **The cylinder costs the same as the existing Lagrangian spoke
— one solve per scenario per iteration** — and lags the hub no more than that spoke
does. The re-solve retains value only when the first solve is sloppy, so it becomes an
opt-in tightening pass (Phase 4) rather than the mechanism.

### 3.6 Which tightenings of the box are admissible

The box is not fixed. Two separate mechanisms shrink it, they shrink it for different
reasons, and they are **not** sound for the same reason — which is worth stating,
because the tempting one-line justification ("tightening the box only removes points,
and fewer points can only raise an infimum") is an argument that the bound gets
*tighter*, not an argument that it stays *valid*. Raising `q̂_s` is exactly the
direction that could break it.

What the aggregate bound actually needs is easy to state. Let `x*` be an optimal
nonanticipative solution of the full problem. For each scenario `s`, if `x* ∈ B_s`
then

```
    q̂_s  ≤  inf_{v ∈ B_s} φ_s(v)  ≤  φ_s(x*)  ≤  f_s(x*) + W_sᵀx*
```

— the last step because `λ ≥ 0`, `g_s(x*) ≤ 0` and `h_s(x*) = 0` — and summing with
weights `p_s` under `Σ_s p_s W_s = 0` gives `Σ_s p_s q̂_s ≤ OPT`, which is §8. So the
requirement on the boxes is precisely:

> **every `B_s` must contain one common optimal solution of the full problem.**

Note "common": preserving a *different* optimizer in each scenario would not do, since
the sum telescopes only at a single `x*`.

**fbbt (setup, once).** `unbounded_variables(s, do_fbbt=True)` runs feasibility-based
bounds tightening before any certificate is computed. This satisfies the requirement
in its strong form: fbbt removes no point that satisfies the scenario's constraints,
so `B_s` still contains *every* feasible point of subproblem `s`, `x*` among them. No
appeal to optimality is needed, which is why this one is safe to describe as "shrinks
`B` without removing a feasible point".

**The nonant-bounds channel (every iteration).** `receive_nonant_bounds()` narrows the
nonanticipative variables' bounds from `Field.NONANT_LOWER/UPPER_BOUNDS`, and in the
current code base only `reduced_costs_spoke` sends that field. Reduced-cost fixing is
*optimality*-based: it discards points that are provably no better than an incumbent,
which does remove feasible points. It therefore does **not** satisfy the strong form,
and the weak form is what carries it — its contract is that at least one optimal
solution survives, and because the bounds are broadcast and applied identically to
every scenario's nonants, the solution that survives is the same one everywhere. That
is the "common `x*`" the requirement asks for.

Two practical notes. First, this is close to moot today: reduced-cost fixing wants
discrete variables, which this cylinder rejects as a hard error at setup, so in
practice the field is not being sent to it. Second, fbbt is deliberately not re-run
after nonant bounds arrive. That leaves tightness on the table rather than risking
anything — re-running it would still be sound, since anything fbbt infers from
constraints and bounds that all hold at `x*` also holds at `x*` — but the setup-time
pass is where the unbounded-variable warning in §6.1 belongs, and repeating it every
iteration would buy little.

**Anything else that narrows the box needs one of these two arguments made for it.**
A tightening that preserves neither all feasible points nor a common optimum breaks the
bound, and nothing in the code can detect that.

## 4. Getting the gradient

`∇φ_s(v̂)` comes from Pyomo's reverse-mode differentiation:

```python
from pyomo.core.expr.calculus.derivatives import differentiate, Modes
grad = differentiate(phi, wrt_list=vlist, mode=Modes.reverse_numeric)
```

Measured on this branch, a 500-variable / 500-constraint model: **7.6 ms** for the
whole gradient — negligible against an NLP solve, and one reverse sweep rather than
one pass per variable.

This deliberately avoids PyNumero. `PyomoNLP` would also serve, but it needs the
compiled `pynumero_ASL` library, and `AmplInterface.available()` is `False` here — it is
not an mpi-sppy dependency and nothing installs it. The `differentiate` route adds no
dependency at all.

Worth recording precisely, because it is easy to misread: the idaes-ext bundle §10.1
installs *does* ship `libpynumero_ASL.so`. That does not make PyNumero available, because
Pyomo looks for it under `PYOMO_CONFIG_DIR` (`~/.pyomo`), not on `PATH`. So the choice
here is not "PyNumero is unobtainable" — it is one deliberate copy away — but rather that
taking it would add an install step and a second way for the gradient to be unavailable
at runtime, in exchange for nothing this calculation needs.

## 5. Multipliers: canonical form and Ipopt's sign conventions

Constraints are canonicalized so every inequality reads `g(v) ≤ 0`. The mapping from
Pyomo's `dual` suffix to the canonical multiplier is orientation-dependent; measured
against analytically-known multipliers (`min (x−3)²` with the constraint active, true
multiplier 4):

| Pyomo constraint | canonical form | Pyomo `dual` | canonical multiplier |
|---|---|---|---|
| `body <= upper` | `g = body − upper` | −4.0 | `λ = max(−d, 0)` |
| `body >= lower` | `g = lower − body` | +4.0 | `λ = max(+d, 0)` |
| `body == rhs` | `h = body − rhs` | −4.0 | `μ = −d` |

The equality row agrees with the existing repo constant
`solver_dual_sign_convention['ipopt'] = -1` in `mpisppy/utils/lshaped_cuts.py`.

Ranged constraints (`lower ≤ body ≤ upper`, both finite) carry one dual for two rows;
they are split into both inequalities and the two rules above are applied, which lands
the magnitude on the active side and zero on the other. Verified in both directions: a
range active at its upper bound returns `d = −4` (`λ_upper = 4`, `λ_lower = 0`), and one
active at its lower bound returns `d = +4` (`λ_lower = 4`, `λ_upper = 0`). An inactive
inequality returns `d = 0`, so both clipped multipliers are 0 and the constraint drops
out of `φ` — which is what complementarity requires.

Bound multipliers `z_L`/`z_U` are **not needed** — the box is kept explicit in the
certificate rather than dualized, so `ipopt_zL_out`/`ipopt_zU_out` need not be
imported at all.

### 5.1 The robustness property that makes this safe

Weak duality holds for any `λ ≥ 0` and any `μ`. Therefore **a sign-convention error, a
stale dual, or a clipped multiplier can only make the bound loose, never wrong.** The
observed failure mode of a wrong sign is a bound of −34.7 where 1.2476 was available:
obviously broken, harmless. Combined with the hub keeping the best outer bound seen
(`_outer_bound_update` in `spcommunicator.py`), a weak bound is silently ignored.

The exception is convexity, which is genuinely load-bearing: if the model is not
convex, the tangent in §3.3 is not an underestimator and the bound is simply wrong.
Hence the guards in §6.

### 5.2 Floating point

At a converged solve the certified bound exceeded `f(v̂)` by 1.5e−06. This is not a
demonstrated violation — `v̂` carries up to `constr_viol_tol` of infeasibility, so
`f(v̂)` understates the true optimum — but it shows the margin sits at the level of the
solver's feasibility tolerance. A small relative cushion is cheap insurance and costs
nothing that matters in a PH outer bound.

**Decided: the cushion is on by default.** The engine reports

```
    q̂ − ε_rel·(1 + |q̂|),      ε_rel = 1e-9 by default
```

settable through `--ipopt-outer-bound-cushion` (0 disables it). Note honestly what this
does and does not buy: 1e-9 is an order of magnitude *smaller* than the 1.5e−06 margin
measured above, so it is last-bit hygiene, not a proof-carrying margin. The margin
itself is not a soundness problem — the theorem in §3.3 needs no tolerance argument —
and a user who wants to absorb `constr_viol_tol`-scale infeasibility can raise `ε_rel`.

**Where the cushion does not scale with the risk.** `φ(v̂)` is evaluated in double
precision as `f + Σᵢ λᵢgᵢ + Σⱼ μⱼhⱼ`, and `q̂` adds `−Σᵢ |∂ᵢφ|·dᵢ`, where `dᵢ` is the
distance from `v̂ᵢ` to the far end of its interval. Rounding gives

```
    |q̂_computed − q̂_exact|  ≲  u·( |f| + Σᵢ|λᵢgᵢ| + Σⱼ|μⱼhⱼ| )  +  u·Σᵢ (terms of ∂ᵢφ)·dᵢ
    u = 2⁻⁵³ ≈ 1.1e−16
```

Both error terms are governed by the size of the **summands**. The cushion
`ε_rel·(1+|q̂|)` is governed by the size of the **result**. On a well-scaled model those
track each other and 1e−9 is generous by seven orders of magnitude. Where cancellation
is severe they come apart: a constraint row scaled by 1e−8 drives its multiplier to
~1e8 against an objective of order one, so the error floor is ~1e−8 absolute while the
cushion is ~1e−9.

Note that the second error term is the one that can push `q̂` *up*. The correction is
`−Σᵢ|∂ᵢφ|dᵢ`, so a gradient component computed slightly too small in magnitude makes
the correction slightly less negative. Unlike a multiplier error, which §5.1 shows is
one-directional, a gradient error is not. It is not amplified by anything — see §5.3 —
but it is not self-protecting either.

**Practical consequence: on a model known to be badly scaled, raise
`--ipopt-outer-bound-cushion`.** This is the one respect in which numerics bear on
validity rather than on tightness, and it is the reason the flag exists as a knob
instead of a constant.

**Non-finite results are screened.** A diverged solve can leave NaN in the point or in
the duals, and NaN propagates silently through every expression above. An infinite
multiplier produces NaN (`inf·0`) or `±inf`. `certified_lower_bound` therefore checks
`math.isfinite(q̂)` before applying the cushion and returns `None` otherwise, reusing the
word it already has for "no bound this time". Without the check the safety would be
accidental: NaN loses every comparison, so the hub's `new > old` test in
`_outer_bound_update` happens to reject it — but `+inf` would compare as an
*improvement* and be latched as an outer bound that is not one.

### 5.3 Ill-conditioning: why it costs tightness and not validity

Ill-conditioning is the obvious worry for a bound read out of an NLP solver, so it is
worth saying precisely where it does and does not enter.

**"Sub-optimal" has only one meaning here.** The model is convex by hypothesis, so it
has no non-global local minima. Ipopt returning a sub-optimal answer can therefore only
mean it stopped short of converging — an inexact iterate with inexact multipliers. That
is exactly the case §3.3 was constructed to cover: the corollary holds for *any* `v̂`,
optimal or not, feasible or not, and for *any* `λ ≥ 0` and *any* `μ`.

**The bounding plane is immune because there is no solve.** A condition number measures
how much a *linear solve* amplifies error — `κ(H)` bounds the relative error in `d`
from `Hd = −g`. The certificate inverts nothing. Its entire computation is: evaluate
`φ` at `v̂`, evaluate `∇φ` at `v̂` by reverse-mode AD, compare each component to zero,
select an endpoint of that variable's interval, multiply, and add. Every one of those
is a forward evaluation carrying relative error `O(u)` per operation. `κ` has no route
in. Correspondingly, the inequality the plane rests on,

```
    φ(v) ≥ φ(v̂) + ∇φ(v̂)ᵀ(v − v̂)      for all v,
```

is a pointwise consequence of convexity. It holds exactly, at every `v̂`, with no error
constant and no dependence on how well-conditioned anything is. Conditioning changes
how *loose* the plane is away from `v̂`; it cannot change which side of `φ` it lies on.

**Where conditioning does land: the correction term.** A badly conditioned problem
leaves `v̂` further from optimal and `∇φ(v̂)` correspondingly larger, and the shortfall
`Σᵢ|∂ᵢφ|·dᵢ` grows with it. It also gives worse multipliers, which weakens `φ` itself.
Both effects move the bound *down*. This is visible rather than theoretical: in the
study below the well-scaled Hilbert QP closed to 1e−9 relative, while the same problem
with a 1e−8 row scaling stalled at 1.8e−2 relative and never closed further. Loose,
valid, and correctly reported as such.

**Empirical check.** Hilbert-matrix QPs (`H_ij = 1/(i+j+1)`, `cond(H)` ≈ 1.6e13 at
n=10 and 3.5e17 at n=16), with row scalings of 1e±8 and an objective offset of 1e12,
solved at `max_iter ∈ {1,2,3,5,10,∞}` and certified at each. Every bound was compared
against `f` at a point *constructed* to be feasible — not against `f(v̂)`, which is the
trap: `v̂` carries up to `constr_viol_tol` of infeasibility, so `f(v̂)` can sit below the
true optimum and manufacture an apparent violation where there is none. No violations at
any conditioning, scaling, or truncation level. This is `TestIllConditioning` in
`mpisppy/tests/test_dual_certificate.py`.

**What ill-conditioning *can* break is convexity, not the certificate.** A Hessian that
is nearly singular and slightly indefinite is non-convex, the tangent is then not an
underestimator, and §5.1's exception applies with full force. That is a property of the
model as written, and no guard in §6 catches it.

## 6. Guards

Checkable at setup, hard error (following the repo's fail-loudly convention):

- **Any integer or binary variable** in a subproblem. A convexity claim is
  definitionally false there, and this is the failure mode most likely to be reached by
  accident.
- **Any equality constraint with `polynomial_degree() != 1`.** A nonlinear equality
  makes `μᵀh` non-convex for one sign of `μ`, breaking §3.3.
- **Any two-sided (ranged) constraint with `polynomial_degree() != 1`.** It splits into
  `g = body − upper` *and* `g = lower − body`, so its body would have to be both convex
  and concave — affine. Like the equality case this is decidable, so it is enforced
  rather than assumed. One-sided nonlinear rows are *not* rejected: whether the body is
  convex (for `≤`) or concave (for `≥`) is not decidable here, and is the user's
  assertion. See the table in §3.1 — the direction of that assertion flips with the
  orientation, which is the part most likely to be got wrong.
- **Prox term attached.** With a prox term the subproblem is not the Lagrangian
  relaxation and the bound is not a Lagrangian bound. `lagrangian_prep` already passes
  `attach_prox=False`; assert it.
- **Nonanticipative variables fixed by an extension.** Same failure as the prox term and
  less obvious: PH's variable-fixing extensions (`fixer.py`, the reduced-cost fixers)
  restrict the subproblem, which can only *raise* its minimum, so the resulting number
  is a bound on the restricted problem and not on the original. The certificate engine
  cannot detect this — a fixed variable is simply a constant to it — so the check
  belongs to the cylinder, in Phase 2. This is a real interaction: fixing extensions are
  commonly on.
- **Maximization.** Phase 1 is minimize-only with a hard error; §10 Phase 5 adds max
  through the concave mirror.
- **Solver is not Ipopt.** Scope decision; the sign conventions in §5 are measured from
  Ipopt only.

### 6.1 Unbounded variables: warn and stand down, do not fail

The closed-form box minimization in §3.3 returns −∞ as soon as an unbounded direction
has a nonzero gradient component.

How often that happens is **model-dependent, and the first draft of this design
overstated it.** Measured on the running example at a converged solve, one component is
−5.1e−11 (nonzero, as claimed) while the component belonging to the variable that was
left unbounded is *exactly* 0.0 — the cancellation `2(ŷ−2) + λ` is exact there — so a
bound is still available despite the missing bounds. Truncate the same solve to one
iteration and that component becomes −1.9e−01 and the certificate correctly returns
`None`. The honest statement: at a converged solve these components are small but
generically nonzero, so an unbounded variable *may* defeat the certificate and reliably
does whenever the solve is not converged. Neither outcome can be counted on in advance,
which is what makes warn-and-stand-down the right policy rather than either
fail-at-setup or assume-it-is-fine.

`fbbt` runs first (precedent: the CVaR `eta` bound in `utils/cvar.py`), which recovers
bounds implied by the constraints. **Decided: a variable still unbounded after `fbbt`
does not raise.** Setup emits a warning naming the offending variables, and the
cylinder then reports no bound — `outer_bound = None` per scenario, so `Ebound()`
returns `None` collectively and nothing is sent to the hub.

Rationale for warning rather than erroring: this cylinder is an *optional* source of a
bound. A model that is merely under-bounded is not a broken model, and killing a whole
parallel run over a spoke that could have sat quiet is the wrong trade — unlike the
guards above, where the model genuinely violates an assumption and any number the
cylinder produced would be wrong. The warning is what keeps the quiet case from being
silent.

Per repo convention the warning is emitted on one rank only (`cylinder_rank == 0`),
once at setup rather than once per iteration.

Not checkable, documented as the user's assertion: convexity of `f_s` and `g_s`.

## 7. Solver scoping: Ipopt for this cylinder, anything for everyone else

This cylinder pins **its own** solver to Ipopt. The hub and every other spoke keep
whatever `--solver-name` selects — gurobi, cplex, xpress, glpk — and nothing here
constrains them. Each cylinder builds its own copy of the scenario models and solves
them with its own solver; the only coupling between cylinders is the numeric exchange
of `W` and bounds, which is solver-agnostic.

The mechanism already exists. `apply_solver_specs(name, spoke, cfg)` overlays a
per-cylinder solver name onto the spoke's options dict:

```python
# mpisppy/utils/cfg_vanilla.py
if _hasit(cfg, name+"_solver_name"):
    options["solver_name"] = cfg.get(name+"_solver_name")
```

So the factory declares `ipopt_outer_bound_solver_name` with default `"ipopt"` and
calls `apply_solver_specs("ipopt_outer_bound", ...)` like every other spoke factory.
The guard in §6 rejects a value that does not name Ipopt.

### 7.1 Global solver options must not leak into this cylinder

This part is not free. `apply_solver_specs` deliberately layers per-spoke options *on
top of* the global `--solver-options` dict, and the global keys remain in place. That
is right for spokes sharing the global solver and wrong here, because **Ipopt
hard-fails on an unrecognized keyword** rather than ignoring it. Observed on this
branch:

```
ERROR: Solver log: Ipopt 3.13.2: ... Unknown keyword "acceptable_iter"
pyomo.common.errors.ApplicationError: Solver (ipopt) did not exit normally
```

So an entirely ordinary run — `--solver-name gurobi --solver-options "mipgap=0.01"`
for the hub, this cylinder attached alongside — would kill the cylinder on its first
solve, with an error naming Ipopt rather than the option routing.

Decision: this cylinder does **not** inherit the global solver-options layer. It starts
from an empty base, and Ipopt-specific settings arrive through
`--ipopt-outer-bound-solver-options`. Filtering the global dict against a list of
Ipopt-known keywords was considered and rejected: the list would have to track Ipopt
releases, and silently dropping an option the user set is worse than never applying it.

### 7.2 Convexity is a property of the model, not of the routing

Worth stating because the flexibility above invites the wrong inference: the §6 model
guards apply to the shared scenario model, not to this cylinder's private copy of it.
If the model has integer variables this cylinder is inapplicable no matter what solver
anything else runs — and "another cylinder needs a MIP solver" is usually the signal
that it does. The routing lets Ipopt coexist with a MIP solver on a *convex* model (a
hub pushing an LP through gurobi, say); it does not let it certify a non-convex one.

## 8. Combining across the cylinder's ranks

This needs no new machinery, but it depends on an existing invariant that must not be
"optimized away", so it is recorded here.

`SPOpt.Ebound()` already does exactly the required reduction — `Allreduce(..., MPI.SUM)`
of `p_s · outer_bound_s` over `self.mpicomm` — and already returns `None`, collectively,
if *any* scenario on *any* rank is missing its bound.

**Every scenario in the sum must use the same `W`.** This is not fussiness. Let `W'`
mix generations across scenarios. At the true optimum `(x*, x̄*)`,

```
    Σ_s p_s L_s(W'_s) ≤ Σ_s p_s [f_s(x*) + W'_sᵀ x*] = OPT + (Σ_s p_s W'_s)ᵀ x̄*
```

which bounds `OPT` only when `Σ_s p_s W'_s = 0` — a *joint* condition on the whole
weight vector that a mixture of generations does not satisfy. A cross-rank mixture
produces a number that is not a bound at all, and nothing downstream would detect it.

Cross-rank agreement is already enforced: `get_receive_buffer(..., synchronize=True)`
routes through `_write_ids_agree`, which `Allreduce`s the write id and rejects a
mixed-generation read for retry. The new cylinder relies on this and must not pass
`synchronize=False`.

The same argument forbids the stale-bound fallback used on the `outer_bound_only` path
(`spopt.py`: "Leave outer_bound at its previous value"), which is sound only if *every*
scenario is stale together. This cylinder instead sets `outer_bound = None` for any
scenario whose certificate fails, letting `Ebound()` return `None` and sending nothing.
Whether the existing Lagrangian spoke has the same exposure is a separate question,
noted in §12.

## 9. Implementation sketch

| File | Change |
|---|---|
| `mpisppy/utils/dual_certificate.py` | **Landed (Phase 1).** Pure function of a solved Pyomo model + its `dual` suffix → certified lower bound. No MPI, no cylinder, no PH. Named neutrally because only the §5 sign table is Ipopt-specific; the convention is a `sign_convention="ipopt"` argument rather than a hard-coded assumption. API: `check_model_is_certifiable`, `unbounded_variables`, `certified_lower_bound`, `CertificateError`. |
| `mpisppy/tests/test_dual_certificate.py` | **Landed (Phase 1).** Wired into `run_coverage.bash` and the `unit-tests` CI job. |
| `mpisppy/cylinders/ipopt_outer_bound.py` | **Landed.** `IpoptOuterBound(_LagrangianMixin, OuterBoundWSpoke)`; `outer_bound_only = False` (duals and primals are both needed); per-scenario certificate → `_mpisppy_data.outer_bound` → `Ebound()`. |
| `mpisppy/utils/config.py` | **Landed.** `ipopt_outer_bound_args()`. |
| `mpisppy/utils/cfg_vanilla.py` | **Landed.** `ipopt_outer_bound_spoke()` factory, alongside `lagrangian_spoke()`. |
| `mpisppy/generic/spokes.py` | **Landed.** Spoke registry. Note this is *not* `generic_cylinders.py`, where an earlier draft of this table put it — the driver delegates spoke construction to `generic/spokes.py`, and the arg registration to `generic/parsing.py`. |
| `mpisppy/generic/parsing.py` | **Landed.** Registers `ipopt_outer_bound_args()`. |
| `mpisppy/tests/test_ipopt_outer_bound.py` | **Landed.** Wiring tests (no solver, no MPI) plus an end-to-end run against the EF optimum. |
| `doc/src/spokes.rst` | **Landed.** User-facing page. |

Prep attaches a `dual` Suffix (IMPORT) to each scenario. The objective expression after
`PH_Prep(attach_prox=False)` is already `f_s + W_sᵀx`, so `φ_s` builds directly on top
of it.

The certificate engine is deliberately a standalone utility rather than a method on the
spoke: it is the part with the interesting math, and it is fully testable serially.
The *cylinder* keeps the Ipopt name, because the scope decision in §1 is real — only
Ipopt's conventions are measured and only Ipopt is accepted by the §6 guard.

## 10. Phased rollout

Each phase is its own review-sized PR and is green on its own.

- **Phase 1 — certificate engine. DONE, in this design branch.**
  `utils/dual_certificate.py` plus the §6 model guards plus
  `tests/test_dual_certificate.py`. No cylinder, no MPI, no config surface.
- **Phase 2 — the cylinder. DONE.** `IpoptOuterBound`, config surface, `cfg_vanilla`
  factory, driver wiring.
- **Phase 3 — parallel + docs. DONE.** Two-rank test exercising the `Ebound`
  reduction; `doc/src/spokes.rst`; a driver command-line smoke run.

**Phases 1–3 ship as a single PR**, revising the "each phase its own PR" plan above.
The reason is review latency rather than principle: with no reviewer available in this
area for some time, splitting buys nothing and costs coherence — and Phase 1 on its own
is arguably *harder* to review, being a mathematical argument with no consumer. Together,
a reviewer can watch the bound get produced against a known EF optimum.

Deferred, and possibly forever — neither is needed for the feature to be complete, so
both are better filed as issues than kept as planned phases:

- **Optional tightening re-solve.** Re-solve `min_{v∈B} φ_s(v)` when the correction
  exceeds a threshold, and re-certify at the new point. By §3.4's own argument the
  correction is ~0 at a converged solve, so this may never be worth building.
- **Maximization** through the concave mirror. A hard error today, which is a fine
  permanent state unless someone actually needs it.

### 10.1 Ipopt in CI — done, and it is the good build

There was no Ipopt anywhere in `.github/workflows/`. Phase 1 was unaffected, since its
tests are deliberately solver-free (§11), but Phase 2 and Phase 3 could not have been
meaningfully green: their tests would have *skipped* rather than failed, which is the
worst of both. So this landed with Phase 1 rather than being left to Phase 3.

**Job `ipopt-tests`**, in `test_pr_and_main.yml`. It pulls the release bundle from
`IDAES/idaes-ext` — the same source Pyomo's and Egret's own CI use, both of which were
read directly rather than reconstructed from memory:

- Pyomo: `.github/workflows/test_pr_and_main.yml`, step "Install Ipopt"
- Egret: `.github/workflows/egret.yml`, step "Install Solvers"

Why this source and not pip or conda: **pip/conda Ipopt is built against MUMPS only.**
The idaes-ext bundle additionally carries the HSL linear solvers, which is what makes
Ipopt usable on real NLPs. Measured on the extracted bundle: `ma27`, `ma57`, `ma97` all
solve; `ma86` does not ship. The job asserts those three actually work before running
any test, because a MUMPS-only fallback would otherwise show up much later as an
abnormal exit rather than as a clear "wrong build" message.

Verified before committing, by rehearsing the whole step locally rather than trusting
the YAML: the tag extraction returns `3.4.2` from the live API and the hardcoded
fallback correctly triggers on a junk response; the tarball unpacks **flat** (so the
extraction directory itself goes on `PATH`); the extracted binary runs; `ma27/ma57/ma97`
all solve from it; and the Phase 1 suite passes against that exact binary.

One non-obvious prerequisite: **the idaes-ext binary carries no RPATH** and resolves
`libgfortran`, `liblapack` and `libblas` from the system. A bare runner has none of
them, and the failure surfaces as an unhelpful load error at the first solve rather
than at install time, so the job `apt-get install`s the three before downloading.
Pyomo's workflow does the same, for the same reason.

Two facts worth keeping:

- `3.4.2` (2024-08-12) is still the current idaes-ext release, and is also what is
  installed on the development machine — so local results and CI results are from the
  same build, not merely from "some Ipopt".
- The bundle targets `ubuntu2204` while `ubuntu-latest` runners are 24.04. This is fine:
  the same tarball is what runs on the 24.04 / glibc 2.39 development machine.

## 11. Test plan

**Phase 1's tests need no solver, by construction.** The certificate is a pure function
of the point the variables hold and the values in the `dual` suffix, so both are set by
hand and every expected number is exact analytic arithmetic rather than a recorded
observation. Given §10.1 this is not a stylistic preference: a solver-gated suite would
skip in CI and report nothing. The handful of assertions that genuinely require Ipopt —
that its reported dual *signs* are what the table assumes — are isolated in one
`skipUnless` class.

The running example is `min (x−3)² + (y−2)²  s.t.  x + y ≤ 1, x,y ∈ [−10,10]`, with
optimum 8 at `(1, 0)` and multiplier 4, all analytic.

- **Analytic**: `q̂ = 8` *exactly* at the KKT point — the correction term is 0 there, so
  this is an equality assertion, not a tolerance.
- **Sign table**: one test per orientation (`<=`, `>=`, `==`, ranged), each with the
  dual Ipopt reports for that orientation, all four required to return exactly 8. A lost
  minus sign shows up as a wrong number, not a near-miss.
- **Degradation**: validity from points that are non-optimal, and from points that are
  outright *infeasible* — the certificate requires neither. Under real truncated solves
  (`max_iter ∈ {1,2,3,5,8}`, Ipopt-gated) validity is asserted at every level and the
  converged bound is asserted to be at least as tight as each truncated one. Pairwise
  monotonicity across `max_iter` is deliberately **not** asserted: the iterate path is a
  solver detail that can differ by linear-solver build, and a test that fails on a
  machine with HSL but passes on MUMPS is worse than no test.
- **Wrong-sign robustness**: feeding the `>=` dual to a `<=` constraint must yield a
  *loose but valid* bound (−68 against an optimum of 8), pinning §5.1 down as a test
  rather than a claim.
- **Box width**: widening the box from ±10 to ±1000 must loosen the bound and must not
  invalidate it, pinning the §3.3 scaling law.
- **Guards**: integer var, binary var, nonlinear equality, maximize, no objective,
  missing `dual` suffix, missing dual for a constraint, unknown sign convention — each
  raises. Plus the negative cases that matter as much: a *fixed* discrete variable is
  allowed (it is a constant, so it carries no convexity claim) and a *nonlinear
  inequality* is allowed (convex inequalities are the entire point; only equalities must
  be affine). The prox and non-Ipopt-solver guards are cylinder-level and land in
  Phase 2.
- **Unbounded variable**: asserts the §6.1 path — `None` rather than `-inf`, and only
  when the unbounded direction actually carries a nonzero gradient component. Includes
  the half-open case, where whether the bound survives depends on which way the gradient
  points. Also asserts the `fbbt` pre-pass rescues a variable whose bounds are implied
  by the constraints.
- **Cushion**: `ε_rel = 0` reproduces `q̂` exactly; the default shaves it by
  `1e-9·(1+|q̂|)` and never turns a valid bound invalid.
- **Non-finite results** (no solver): a NaN dual, a NaN in the point, and an infinite
  dual must each yield `None` or a valid number, never a non-finite one. The `+inf` case
  is the one that matters — NaN is rejected downstream by accident, `+inf` would be
  latched as an improvement — so it is asserted explicitly rather than left to the
  general check.
- **Ill-conditioning** (Ipopt-gated): Hilbert-matrix QPs at `cond(H)` ≈ 1.6e13 and
  3.5e17, with row scalings of 1e±8 and an objective offset of 1e12, certified at
  `max_iter ∈ {1,2,3,5,10,∞}`. Each bound is checked against `f` at a point *constructed*
  to be feasible by bisecting a uniform downshift — comparing against `f(v̂)` from the
  solver instead is the trap described in §5.3 and produces false violations. Two
  supporting assertions keep it honest: that the 1e−8 row scaling really does drive the
  multiplier past 1e6 (or the cancellation variant would silently stop exercising
  cancellation), and that the well-scaled case closes to 1e−6 relative while the badly
  scaled one only has to stay valid — pinning §5.3's "tightness, not validity" as a test.
  Tolerances are a few thousand ulps relative, which is the §5.2 arithmetic allowance;
  an exact comparison at an offset of 1e12 would be testing the FPU.
- **Integration**: `farmer` is linear, hence convex, and Ipopt solves it; compare the
  cylinder's bound against the EF optimum.
- **MPI**: 2-rank cylinder, same comparison, exercising `Ebound`'s reduction. The hub
  runs Ipopt too by default so the test needs no MIP solver; a second case runs the hub
  on a MIP solver and skips when none is available, which is what actually demonstrates
  the §7 routing claim.
- **Option routing** (no solver, no MPI): that the global `--solver-options` layer does
  *not* reach this spoke while the per-spoke layer does, and — the assertion that makes
  that meaningful — that `lagrangian_spoke` still *does* inherit the global layer. This
  is the §7.1 decision, and it is the part most likely to break silently.

**No `run_all.py` entry.** The usual home for "the documented command line still works"
is `run_all.py`, but `do_one` has no skip machinery, so an entry there would force an
Ipopt install into both `run_all` CI jobs, which have no other use for one. The same
coverage is bought far more cheaply by a driver smoke run inside the `ipopt-tests` job,
which already has Ipopt. Worth revisiting if Ipopt ever lands in those jobs for another
reason.

New `mpisppy/tests/test_*.py` files go into `run_coverage.bash` **and**
`.github/workflows/test_pr_and_main.yml` in the same commit, or codecov reports 0% on
the patch.

## 12. Decisions

Every question this design opened is now settled; item 4 turned out to be a bug in
shipped code rather than a design choice.

1. **Cushion default — on.** `ε_rel = 1e-9`, `--ipopt-outer-bound-cushion` to change,
   0 to disable. See §5.2 for what it is and is not worth.
2. **Unbounded variables — warn, then report no bound.** `fbbt` first; anything still
   unbounded produces a rank-0 setup warning naming the variables and a `None` bound,
   not an exception. See §6.1.
3. **Naming — neutral engine, Ipopt cylinder.** `utils/dual_certificate.py` takes the
   sign convention as an argument; `cylinders/ipopt_outer_bound.py` is the Ipopt-scoped
   consumer. See §9.
4. **Pre-existing stale-bound exposure — confirmed, and fixed separately.** The
   question was whether the existing stale-`outer_bound` fallback admits the §8
   mixed-`W` case in practice. It does. `solve_one` left a subproblem's previous
   bound in place when a solve produced no bound, so `Ebound` could form a sum
   mixing one scenario's stale bound with the others' fresh ones — not a bound at
   all, and invisible to `Ebound`'s missing-bound check, which tests for `None` and
   sees a number. The same exposure existed on the ordinary failed-solve path.
   Reproduced and fixed in **PR #839**, based on `main` rather than stacked here,
   since it is a soundness bug in shipped code and independent of this design.
   Ipopt itself never reaches that path (it reports `Lower_bound = -inf`, not
   `None`), so nothing in this design depends on the outcome.
