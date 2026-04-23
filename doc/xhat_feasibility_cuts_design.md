# Design: Optional Feasibility Cuts from Xhatters

**Status:** Draft — up for discussion. Nothing is implemented yet.
**Addresses:** [issue #601](https://github.com/Pyomo/mpi-sppy/issues/601)
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-04-23

## Motivation

For two-stage problems **without complete recourse**, an xhatter (an xhat
spoke such as `xhatlooper`, `xhatshufflelooper`, `xhatspecific`) can
propose a first-stage candidate `xhat` that is infeasible in one or more
of the rank-local scenarios. Today the xhatter just discards `xhat` and
moves on — but nothing prevents the hub (or a later xhatter call) from
trying the same `xhat` again a few iterations down the road.

Issue #601 proposes: when an xhatter discovers infeasibility, **optionally
generate a feasibility cut** and push it into **every scenario** so the
same `xhat` (and a neighborhood of it, for continuous variables) does
not get revisited. This is a classic no-good / Farkas-dual
feasibility-cut pattern from Benders-style decomposition, applied
inside the mpi-sppy xhat evaluation loop.

## Non-goals

- Replacing the main `lshaped` / `fwph` cut machinery. We add feasibility
  cuts on the PH side; we do not change how L-shaped itself produces
  its optimality cuts.
- Changing default behavior. The feature is off by default
  (`--xhat-feasibility-cuts-count 0`).

## Stage support

The two-stage-only restriction in `cross_scen_extension` and
`lshaped_cuts` is about the L-shaped **cut structure**
(`eta_s >= const + coef · x_root`, one scalar recourse cost per
scenario) — that formulation breaks when the recourse cost is spread
across a tree of per-node contributions.

A **no-good cut** (first-milestone scope here) carries none of that
structure: it is just

```
sum_{i: xhat_i = 1} (1 - x_i) + sum_{i: xhat_i = 0} x_i >= 1
```

over whichever binary nonants the xhatter fixed. Nothing in it depends
on the recourse-cost formulation, so it is valid for multi-stage as
well as two-stage. Mechanically, multi-stage only changes three
things:

- **Nonant keys** become `(node_name, i)` rather than `("ROOT", i)`.
  Existing code that loops over `s._mpisppy_data.nonant_indices` already
  handles this.
- **Cut reach**. An xhatter in a multi-stage run proposes a value for
  every non-leaf decision it touches, so the cut excludes the whole
  tree-path xhat, not just the root. That is strictly stronger than
  the two-stage case.
- **Bundle structure**. Multi-stage proper bundles cover entire
  second-stage nodes; the "install cut into every per-scenario block
  inside the bundle EF" recipe is identical.

So: **no-good cuts work in multi-stage from day one.** The
**Farkas / optimality-style feasibility-cut path** (follow-up
milestone; see below) keeps the two-stage restriction because that is
where the L-shaped formulation matters.

## Blueprint: cross-scenario cuts

We already have a working precedent for "spoke generates cuts, hub
installs them into every scenario". The relevant pieces:

### Flat serialization over the shared-memory window

- `Field.CROSS_SCENARIO_CUT = 300` in `mpisppy/cylinders/spwindow.py`,
  pre-sized as `nscen × (nonant_len + 2)`.
- Row format: `[constant, eta_coef, nonant_coefs...]` — one row per
  target scenario, fixed width `1 + 1 + nonant_len`.
- The spoke writes the whole `all_coefs` buffer per iteration;
  the hub-side consumer checks `is_new()` and unpacks.

### Dedicated spoke: `CrossScenarioCutSpoke`

- Declares `send_fields = (..., CROSS_SCENARIO_CUT)` and
  `receive_fields = (..., NONANT, CROSS_SCENARIO_COST)`.
- Builds a pseudo-root `ConcreteModel` with first-stage variable copies,
  attaches an `LShapedCutGenerator` (thin wrapper over
  `pyomo.contrib.benders.benders_cuts.BendersCutGeneratorData`), adds
  every scenario as a Benders subproblem, and reuses the Benders
  `generate_cut()` output.
- On each iteration it reads the current nonants and etas broadcast by
  the hub, computes the "farthest" xhat, and emits cuts.

### Hub-side extension: `CrossScenarioExtension`

- `sync_with_spokes()` pulls the cut buffer, calls `make_cuts(coefs)`.
- `make_cuts` iterates rows, builds a `pyomo.core.expr.numeric_expr
  .LinearExpression`, and appends to
  `s._mpisppy_model.benders_cuts[outer_iter, scen_name]` (an
  `IndexedConstraint`).
- Persistent-solver awareness: `persistent_solver.add_constraint(...)`
  for every added cut, so the underlying solver state stays current.
- Tracks a best-bound constraint (`inner_bound_constr`) alongside the
  cuts; that part is cross-scen-specific and does **not** apply here.

## The pyomo Benders gap

`pyomo.contrib.benders.benders_cuts.BendersCutGeneratorData.generate_cut()`
**hard-errors** when a subproblem is not optimal:

```python
if res.solver.termination_condition != pyo.TerminationCondition.optimal:
    raise RuntimeError('Unable to generate cut because subproblem failed
                        to converge.')
```

That is, it generates **optimality cuts only** from optimally-solved
subproblems. `mpisppy/utils/lshaped_cuts.py` inherits the same
limitation — its `LShapedCutGenerator` is a thin subclass of pyomo's
`BendersCutGeneratorData`.

Issue #601 is fundamentally about the **infeasibility** case — that's
where we want the cut. So "reuse `lshaped_cuts`" is necessary but not
sufficient. We need one of:

1. **Upstream extension** — add feasibility-cut support to
   `pyomo.contrib.benders.benders_cuts` so that an infeasible
   subproblem yields a Farkas-dual cut. This is the most correct
   option; it is also a Pyomo PR with its own review cycle.
2. **mpi-sppy-side extraction** — drop down to solver-specific APIs
   (`gurobipy.Model.FarkasProof`, `cplex.solution.get_status`,
   `xpress.getdualray`) inside mpi-sppy and build the cut ourselves,
   bypassing pyomo Benders for the feasibility branch. Faster to land,
   but fragile across solver versions and adds a new solver-sniffing
   surface.
3. **Scope-limited first version** — start with **no-good cuts,
   valid only when every first-stage (nonant) variable is binary**.
   That needs no dual information at all: the infeasible `xhat_bin`
   yields the cut
   `sum_{i: xhat_i = 1} (1 - x_i) + sum_{i: xhat_i = 0} x_i >= 1`.
   Trivial to code, covers UC / scheduling / lot-sizing. If any
   first-stage nonant is integer or continuous, the no-good cut is
   invalid — so enabling the feature on such a model is a hard
   error, not a silent no-op (see "Startup check" below). The
   integer case needs a different cut form (e.g. `|x - xhat| >= 1`
   with auxiliary binaries) and the continuous case needs Farkas
   duals; both are out of scope for the first milestone.

**Recommendation:** pursue (1) as the north star but ship (3) as the
first deliverable so the plumbing lands independently of the Pyomo
review cycle. (2) is a trap — avoid.

## Proposed architecture

Match the cross-scen split: one new shared-memory field, spoke-side
generation, hub-side extension for installation.

### New shared-memory field

`Field.XHAT_FEASIBILITY_CUT` in `spwindow.py`, pre-sized as
`cuts_per_iter × (1 + nonant_len) + 1`:

- Row format: `[rhs_constant, nonant_coef_1, ..., nonant_coef_N]`.
  No eta column — feasibility cuts do not involve the recourse-cost
  variable.
- Trailing slot: the number of cuts actually written this round
  (so unused rows are ignored).

`cuts_per_iter` is set by `cfg.xhat_feasibility_cuts_count` at buffer
registration time.

### CLI

```
--xhat-feasibility-cuts-count INT   (default 0, meaning off)
```

Interpretation: maximum number of feasibility cuts generated per
xhatter iteration. Zero disables the whole feature. The flag
doubles as on/off switch and buffer sizer.

### Spoke-side generation

Modify the **existing** xhatter spokes (`xhatlooper_bounder`,
`xhatshufflelooper_bounder`, `xhatspecific_bounder`). They already
know when their candidate is infeasible in a scenario; we add a
small path:

1. If `cfg.xhat_feasibility_cuts_count > 0`, register
   `Field.XHAT_FEASIBILITY_CUT` as a send field at startup.
2. On a `solve_loop` termination whose `termination_condition` is one
   of `infeasible` or `infeasibleOrUnbounded` for some scenario,
   build a cut row for that scenario:
   - **Version 1 (binary no-good)**: produce the no-good row directly
     from the `xhat` values. The "all nonants are binary" precondition
     is enforced at setup time (see Startup check below), so by the
     time we reach this path we can rely on it — no per-iteration
     re-check, no silent skip.
   - **Version 2 (pyomo-upstream Farkas)**: call the (to-be-added)
     infeasibility branch of pyomo Benders to get Farkas duals and
     pack them into the same row format.
3. Pack up to `cuts_per_iter` rows into the send buffer; write the
   actual count into the trailing slot; `put_send_buffer(...)`.

Most of the "set up a pseudo-root model with first-stage copies" code
from `CrossScenarioCutSpoke.prep_cs_cuts` is reusable.

### Hub-side extension

New `XhatFeasibilityCutExtension` in
`mpisppy/extensions/xhat_feasibility_cut_extension.py`, modeled on
`CrossScenarioExtension` but stripped of the eta / inner-bound
bookkeeping:

- `register_send_fields` / `register_receive_fields`: register
  receive of `Field.XHAT_FEASIBILITY_CUT` from the xhatter spoke.
- `setup_hub`: on each local scenario, attach an empty
  `pyo.ConstraintList` (or `IndexedConstraint` keyed by
  `(cut_batch_iter, cut_row_idx)`) named
  `s._mpisppy_model.xhat_feasibility_cuts`.
- `sync_with_spokes`: read the buffer. If `is_new()`, unpack the
  count-header and the rows, build `LinearExpression` per row,
  append the **same cut to every local scenario's**
  `xhat_feasibility_cuts`, and call
  `persistent_solver.add_constraint(...)` as cross-scen does. Because
  the cut is on first-stage (nonant) variables, nonanticipativity
  makes it valid globally — there is no version that cuts only the
  infeasible scenario.
- Bundle-safe: when a local "scenario" is a proper bundle, push the
  cut into each per-scenario block inside the bundle (the nonants are
  duplicated across the scenario blocks inside the bundle EF, so the
  cut has to land on each). This is the same class of gotcha PR #669
  is addressing on the `nonant_cost_coeffs` side — solve it once
  here, up front.

### Startup check: first-stage must be fully binary

The first-milestone no-good cut is only valid when **every** first-stage
nonant is binary. A mix of binary and continuous nonants is not safe
to treat as "no-good on the binary part only": the continuous part can
be perturbed, so such a cut does not actually exclude the infeasible
xhat — it would silently produce an invalid relaxation.

To prevent misuse, `XhatFeasibilityCutExtension.setup_hub` scans
`s._mpisppy_data.nonant_indices` on an arbitrary local scenario and
confirms that every variable is binary (`v.is_binary()` or
`v.domain in {Binary, BooleanSet}`). For multi-stage, the scan covers
every node in `_mpisppy_node_list`. If any non-binary nonant is
present, raise

```
RuntimeError(
    "--xhat-feasibility-cuts-count > 0 requires every first-stage "
    "(nonant) variable to be binary; found non-binary nonant "
    f"{v.name!r} with domain {v.domain}. The first-milestone "
    "feasibility-cut generator is no-good-only. Support for integer "
    "and continuous first-stage variables is planned as a follow-up "
    "milestone (pyomo Benders / Farkas extension)."
)
```

Fail fast at setup time rather than at the first infeasibility — the
user sees the error before any work is done.

### Plumbing touchpoints

- `mpisppy/utils/config.py` — add `xhat_feasibility_cuts_count` to an
  appropriate config bundle (likely `cfg.spoke_xhat_args()` or a new
  `cfg.xhat_feasibility_cut_args()`).
- `mpisppy/utils/cfg_vanilla.py` — flag in the xhatter spoke dicts so
  the extension and the send-field are registered together.
- `mpisppy/cylinders/spwindow.py` — new `Field` enum value and
  length formula.
- `mpisppy/extensions/xhat_feasibility_cut_extension.py` — new hub
  extension.
- `mpisppy/cylinders/xhatlooper_bounder.py` (etc.) — add the cut
  emission path gated on `cfg.xhat_feasibility_cuts_count > 0`.

## Code sharing / refactoring opportunities

The user asked explicitly: can we share code with the cross-scen path?
Three candidates, in descending value:

1. **Cut installation into scenarios, with persistent-solver and
   bundle awareness.** Today this logic is inline in
   `CrossScenarioExtension.make_cuts`; the xhat-feas version will want
   the same behavior. Factor into
   `mpisppy.utils.cut_installer.install_linear_cut(scenario, expr,
   key, persistent_solver=...)`. Both extensions call it. High
   payoff: it's the trickiest bit (persistent-solver management plus
   the bundle duplication).

2. **Pseudo-root model construction.** `CrossScenarioCutSpoke
   .prep_cs_cuts` builds a `ConcreteModel` with first-stage-variable
   copies keyed by an "index against later" map. Xhat-feas spoke side
   needs the same kind of harness if/when we pursue the pyomo-Benders
   route. Factor into
   `mpisppy.utils.pseudo_root.build_root_from_nonants(opt) -> (root,
   root_vars, nonant_vid_to_copy_map)`.

3. **Flat-buffer packer/unpacker.** Both cross-scen and xhat-feas use
   `[constant, ...coefs]` rows. A small
   `mpisppy.utils.cut_buffer.pack_rows(...) / unpack_rows(...)`
   helper de-duplicates the row arithmetic. Lower payoff — the
   arithmetic is straightforward and inlined it is arguably more
   readable — but it's essentially free to write if we do (1).

Recommended refactoring sequence: land the new xhat-feas feature first
(without factoring), **then** do a follow-up PR that extracts the
shared helpers and updates cross-scen to use them. This avoids
bundling a behavior change with a refactor.

## Open questions

- **Cut pool management.** Every xhatter iteration can emit cuts;
  without pruning the subproblems accumulate constraints indefinitely
  and solver times eventually bloat. `CrossScenarioExtension` has the
  same unbounded-growth behavior today (its `benders_cuts` are keyed
  by `(outer_iter, scen_name)` and never removed). Tracked as a
  separate item, scoped to both features so they share one strategy:
  **[issue #670](https://github.com/Pyomo/mpi-sppy/issues/670)**. For
  this milestone we rely on the per-iter cap in
  `--xhat-feasibility-cuts-count` to keep growth linear.

## First-milestone scope (what a PR would actually deliver)

1. CLI flag + buffer registration + send/receive field — wired end to
   end but with the emit path producing zero cuts.
2. No-good-cut emission for problems whose first-stage (nonant)
   variables are **all binary**. Setup-time check raises `RuntimeError`
   otherwise, so the feature never silently produces an invalid cut.
3. Hub-side install of cuts into every local scenario / bundle block,
   keyed by `(node_name, i)` so it works for two-stage and multi-stage
   alike.
4. Regression tests:
   (a) a two-stage model with a binary first-stage where one specific
       xhat is infeasible — verify the cut is added and the infeasible
       xhat is not revisited;
   (b) a bundled version of the same;
   (c) a small multi-stage model with binary nonants to verify nothing
       in the plumbing assumes `("ROOT", i)` keys;
   (d) a negative test: enabling the feature on a model with a
       continuous nonant raises the expected `RuntimeError` at setup.
5. Documentation: a **new** user-facing page at
   `doc/src/xhat_feasibility_cuts.rst`, added to `doc/src/index.rst`'s
   toctree next to the other cylinder / extension pages. The page
   must cover:
   - What the feature does and when to use it (non-complete-recourse
     problems with binary first-stage).
   - The CLI flag `--xhat-feasibility-cuts-count`.
   - The binary-only restriction, with the exact `RuntimeError` text
     users will see if they enable the feature on a non-binary model.
   - A brief note on interaction with proper bundles.
   - Forward pointer to issue #670 for cut-pool management.

   Also add a one-sentence cross-reference from each of the xhatter
   cylinder docs (wherever those live today) pointing at the new
   page, so readers browsing xhatter options discover it.

That is a discrete, reviewable PR. The pyomo-Benders Farkas extension
(which lifts the binary-only restriction, two-stage-only) and the
cross-scen refactor are natural follow-ups.
