# Multistage `feasible_xhat_creator`

## Summary

The `feasible_xhat_creator` convention (see `doc/src/feasible_xhat.rst`)
lets a model author supply a candidate incumbent — a point that is
*feasible to fix in every real scenario's per-scenario subproblem* — that
an xhat spoke pins and evaluates once before its main loop, via
`--<xhatter>-try-feasible-xhat-first`. Today it is documented as
**two-stage only**.

The headline finding of this design: **the in-cylinder mechanism is
already fully multistage.** A multistage model could define a
`feasible_xhat_creator` that returns a full-tree cache today and the
existing spoke code would pin and evaluate it correctly, with **zero
library changes**. What is missing is (a) a reusable *engine* that
produces a multistage candidate (the two shipped engines are hard-coded
two-stage), (b) a worked multistage **example**, and (c) the
documentation, including the one genuinely new conceptual point —
inter-stage feasibility coupling.

So this is a small change: one new helper function, one example
auxiliary file, a doc revision, and a test. No changes to any spoke, to
`_fix_nonants`, or to the discovery/attach wiring.

## Scope: this is *not* "read the xhat from a file"

A candidate xhat can already be read from a CSV nonant tree
(`--xhat-from-file`, merged in #760). That is a *separate input
mechanism*: the user has the values already and the framework loads them.
`feasible_xhat_creator` is the opposite — the model author writes code
that *constructs* the candidate (solve a proxy, run a heuristic, …). This
design deliberately builds the example around an in-memory construction,
**not** a file read, so the two paths stay distinct.

## Background: how the two-stage path works

The cache form every consumer speaks is `{node_name: sequence-of-values}`,
one entry per non-leaf node, each value list in
`node.nonant_vardata_list` order.

- The spoke entry point `_PreLoopXhatMixin._try_feasible_xhat`
  (`mpisppy/cylinders/_preloop_xhat_mixin.py`) calls the user's
  `creator(...)`, checks the result is a dict containing `"ROOT"`, then
  hands it to `_evaluate_xhat`.
- `_evaluate_xhat` calls `self.opt._fix_nonants(cache)`, runs
  `solve_loop(need_solution=False)`, and returns `Eobjective()` — or
  `None` if any scenario was infeasible at the pinned point.
- The two shipped engines in `mpisppy/utils/xhat_helpers.py`
  (`average_xhat_nonants`, `lp_xbar_nonants`) produce the ROOT array; the
  model author wraps it as `{"ROOT": arr}` and applies a model-specific
  rounding/repair rule.

The two-stage example creators are tiny — e.g. `examples/farmer/farmer_auxiliary.py`:

```python
def feasible_xhat_creator(*, solver_name, solver_options=None, **scenario_creator_kwargs):
    arr = average_xhat_nonants(average_scenario_creator, solver_name=solver_name,
                               scenario_creator_kwargs=scenario_creator_kwargs,
                               solver_options=solver_options)
    return {"ROOT": arr}
```

## What is already multistage (evidence)

`SPOpt._fix_nonants` (`mpisppy/spopt.py:715`) loops over **every** node in
each scenario's `_mpisppy_node_list`, and for each node requires
`cache[ndn]` to exist with exactly `nlens[ndn]` values, in
`nonant_vardata_list` order:

```python
for node in s._mpisppy_node_list:
    ndn = node.name
    if ndn not in cache: raise ...
    if len(cache[ndn]) != nlens[ndn]: raise ...
    for i in range(nlens[ndn]):
        node.nonant_vardata_list[i].fix(cache[ndn][i])   # (binary/int rounded)
```

This is already a multi-node loop. It is the same routine PH uses to
restore nonants every iteration, and it has always been multistage.

- `_try_feasible_xhat` validates only `"ROOT" in cache` — a multistage
  cache contains `"ROOT"` plus the deeper nodes, so the check passes
  unchanged.
- `_evaluate_xhat` → `_fix_nonants` / `solve_loop` / `no_incumbent_prob`
  / `Eobjective` are all already multistage (xhatshuffle runs multistage
  today).
- The CLI flags, discovery (`cfg_vanilla._find_feasible_xhat_creator`),
  and attach (`_maybe_attach_feasible_xhat`) are tree-agnostic.

**Conclusion:** no spoke or framework code needs to change. The "two-stage
only" label is a property of the *engines and docs*, not the mechanism.

## What is not multistage

`mpisppy/utils/xhat_helpers.py`. Both engines call `_check_two_stage`
(raises unless `len(_mpisppy_node_list) == 1`) and
`_solve_and_extract_root`, which reads only `_mpisppy_node_list[0]`. They
stay as-is; a multistage model uses a new engine.

## The one genuinely new idea: inter-stage feasibility coupling

In two stages there is a single decision point: one ROOT vector, and
recourse cleans up the rest per scenario. The feasibility question
factors scenario-by-scenario.

In multiple stages the candidate is a **policy over the whole tree** —
a vector at every non-leaf node. "Feasible to fix in every scenario" now
means: for each root-to-leaf path, fixing *all* of that path's node
vectors simultaneously leaves the path feasible. The node vectors are not
independent — a stage-3 decision lives downstream of the stage-1 and
stage-2 decisions on the same path, through the model's staircase
constraints.

This is why you cannot assemble a multistage candidate by choosing each
node's vector in isolation (e.g. averaging each node's values across
scenarios independently): the per-node choices may be individually
reasonable yet jointly infeasible along a path.

**The construction principle that keeps it sound:** derive *all* node
vectors from **one coherent feasible solution of a single deterministic
proxy whose scenario tree has the same node structure as the real
problem.** Because the node values then come from one feasible point of
(a relaxation/proxy of) the same staircase system, they are jointly
feasible along every path by construction — provided recourse is
relatively complete (or a repair rule preserves path feasibility, below).

This is the multistage analog of "solve the average *scenario*": solve
the average **tree**.

### Complete recourse vs. coupled recourse (the repair story)

- **Relatively complete recourse** (aircond, hydro): any node-consistent
  point from one proxy solve is feasible to pin on every path. The
  shipped engine + example cover this cleanly; no rounding needed for
  continuous nonants.
- **Integer or tightly-coupled recourse**: a feasible *path* is a
  stronger guarantee than the two-stage per-variable rounding rules give.
  Per-node monotone rounding does **not** generally preserve path
  feasibility, because rounding a stage-t decision can render a later
  stage infeasible on some path. mpi-sppy will **not** ship an automatic
  multistage repair; as in the two-stage doc ("the rounding rule is
  yours"), this is the model author's responsibility, now escalated. The
  doc will say so plainly and the shipped example will be a
  complete-recourse model so it needs no repair.

## New code

### 1. Engine: `ef_xhat_nonants` (in `mpisppy/utils/xhat_helpers.py`)

A multistage sibling of `average_xhat_nonants`. The caller supplies the
scenario names + kwargs that define a deterministic proxy tree (same
branching structure as the real problem, deterministic data). The engine
builds the proxy EF, optionally LP-relaxes it, solves it, and returns the
full node tree as the cache.

```python
def ef_xhat_nonants(scenario_creator, scenario_names, *, solver_name,
                    scenario_creator_kwargs=None, solver_options=None,
                    relax_integrality=False):
    """Solve one EF over the supplied (proxy) scenario set and return
    {node_name: np.ndarray} over all non-leaf nodes, each array in that
    node's nonant order -- the cache form _fix_nonants consumes.

    The proxy tree must have the same node structure as the real
    problem so every real non-leaf node has a counterpart here.
    """
    ef = sputils.create_EF(scenario_names, scenario_creator,
                           scenario_creator_kwargs=scenario_creator_kwargs or {})
    if relax_integrality:
        pyo.TransformationFactory("core.relax_integer_vars").apply_to(ef)
    # ... solve ef with solver_name/solver_options (mirror _solve_and_extract_root) ...
    by_node = {}
    for (ndn, i), var in ef.ref_vars.items():          # ref_vars: see sputils.ef_nonants
        by_node.setdefault(ndn, []).append((i, pyo.value(var)))
    return {ndn: np.array([v for _, v in sorted(pairs)], dtype="d")
            for ndn, pairs in by_node.items()}
```

This reuses the merged-#760 machinery honestly: `sputils.ef_nonants`
already iterates `(ndn, var, value)` over `ef.ref_vars`, the representative
nonant per `(node, index)`. The engine groups those into per-node arrays.
(This is the *real, modest* relationship to #760 — a reused extraction
helper, not a "foundation": see the corrected note in
`doc/designs/multistage_xhat_write_design.md`'s memory and the
`feasible_xhat` two-stage path, which needs no files at all.)

Two-stage callers can keep using `average_xhat_nonants`; this function is
additive.

### 2. Example: `examples/aircond/aircond_auxiliary.py`

aircond is the right vehicle: it is the canonical multistage test model
(3–4 stages in the #760 MPI test), and its recourse is relatively
complete and continuous.

- **Nonants per non-leaf node:** `[RegularProd, OvertimeProd]`
  (`MakeNodesforScen`), both `NonNegativeReals`.
- **Recourse:** the material-balance constraint
  `prev_Inventory + RegularProd + OvertimeProd - Inventory == Demand`
  with `Inventory` a free var split into penalized pos/neg parts. So for
  *any* `(RegularProd, OvertimeProd)` with `RegularProd <= Capacity`,
  every downstream stage is feasible (`Inventory` absorbs the imbalance as
  penalized backorder). With `start_ups=False` (default) there are no
  binaries. **Relatively complete recourse → no rounding needed.**

```python
import numpy as np
from mpisppy.utils.xhat_helpers import ef_xhat_nonants
from aircond import scenario_creator, scenario_names_creator

def feasible_xhat_creator(*, solver_name, solver_options=None,
                          branching_factors=None, **scenario_creator_kwargs):
    # Expected-value proxy: same tree, deterministic mean demand (sigma_dev=0).
    proxy_kwargs = dict(scenario_creator_kwargs)
    proxy_kwargs["branching_factors"] = branching_factors
    proxy_kwargs["sigma_dev"] = 0.0
    snames = scenario_names_creator(np.prod(branching_factors))
    return ef_xhat_nonants(scenario_creator, snames, solver_name=solver_name,
                           scenario_creator_kwargs=proxy_kwargs,
                           solver_options=solver_options)
```

The expected-value tree has the same node structure as the real problem
(so every real node has a counterpart), but with `sigma_dev=0` it is a
trivial deterministic LP. Honest caveat to put in the doc: the EV tree has
the same *node count* as the real tree, so for aircond (already an LP)
it is not dramatically cheaper than the real EF — its value here is to
demonstrate the convention and to hand back a feasible deterministic
policy. The real speedup from this pattern appears when the true EF is a
hard MIP and the proxy is its LP relaxation. Reducing the proxy's
branching for genuine size savings is possible but needs a node-name
mapping back onto the real tree, and is left to the model author.

Mention (don't ship) an alternative that showcases the "use any method"
clause: a closed-form myopic rule (set each node's `RegularProd =
min(Capacity, max(0, expected_demand - incoming_inventory))`, `OvertimeProd`
covers the rest) builds a feasible policy with **no solve and no file** —
illustrating that the engine is a convenience, not a requirement.

### 3. Doc revision: `doc/src/feasible_xhat.rst`

Replace the "Two-stage only, for now." caveat (lines 37–39) with a
multistage section: the cache is `{node_name: ndarray}` over *all*
non-leaf nodes; the inter-stage coupling principle (one proxy solve, not
independent per-node points); the complete- vs coupled-recourse repair
story; and the aircond worked example. Note explicitly that the mechanism
required no spoke changes.

### 4. Test: `mpisppy/tests/test_feasible_xhat.py`

- Unit: call `aircond_auxiliary.feasible_xhat_creator` on a small tree
  (e.g. `branching_factors=[3,3]`), assert the returned dict has a key for
  every non-leaf node (`ROOT`, `ROOT_0..`, …), each array of length 2
  (`RegularProd`, `OvertimeProd`), values within `[0, bigM]` and
  `RegularProd <= Capacity`.
- End-to-end (optional, cheap LP): pin the cache on each real scenario via
  `_fix_nonants` and confirm every scenario solves feasibly (mirrors the
  existing two-stage `_evaluate_xhat`-style test).
- Per repo convention, wire the new test entry into **both**
  `run_coverage.bash` and `.github/workflows/test_pr_and_main.yml` in the
  same commit.

## Non-goals

- Automatic multistage rounding/repair for coupled-recourse models (model
  author's responsibility; flagged in docs).
- Any change to `--xhat-from-file` / the CSV nonant-tree path (separate
  mechanism).
- Generalizing the two-stage engines; `average_xhat_nonants` /
  `lp_xbar_nonants` stay two-stage.
- An outer-bound (Jensen's) multistage story — Jensen's path is explicitly
  two-stage and out of scope here.

## Honest relationship to the merged multistage-file work (#760)

#760 produced in-memory node-tree extraction helpers
(`SPBase.gather_nonant_tree_to_rank0`, `sputils.ef_nonants`). The new
engine reuses `ef_nonants`. That is the whole connection: a reused
helper. The file read/write is **not** a substrate for this feature — the
two-stage `feasible_xhat_creator` already exists and never touches a file,
and the multistage version relates the same way.

## Why one PR (not phased)

The change is small and cohesive — one helper, one example, one doc
section, one test — and there is no intermediate state worth shipping
separately. Mirrors the chance-constraint decision: a single
review-sized PR.

## Implementation checklist

1. `ef_xhat_nonants` in `mpisppy/utils/xhat_helpers.py` (+ reuse the
   existing solve/termination-check pattern from `_solve_and_extract_root`).
2. `aircond_auxiliary.py` with `feasible_xhat_creator`. Placed at
   `mpisppy/tests/examples/aircond_auxiliary.py` — beside the model
   (`mpisppy.tests.examples.aircond`) so the `<module>_auxiliary`
   discovery in `_find_feasible_xhat_creator` resolves it; the
   `examples/<model>/` doc convention assumes the model lives under
   `examples/`, which aircond does not.
3. Revise `doc/src/feasible_xhat.rst` (multistage section + aircond worked
   example; drop the two-stage-only caveat).
4. Tests in `mpisppy/tests/test_feasible_xhat.py`; wire into
   `run_coverage.bash` + `test_pr_and_main.yml` (same commit).
5. `ruff check --no-cache` clean; add file headers (`addheader -c addheader.yml`).
</content>
</invoke>
