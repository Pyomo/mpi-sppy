# Design: Proper Bundles for ADMM Problems

Status: In Progress (design phase)
Date: 2026-03-14

## Goal

Enable proper bundling for stochastic ADMM problems, where stochastic scenarios
within the same ADMM subproblem are grouped into EF bundles. Deterministic ADMM
bundling does not apply (each subproblem is unique).

## Background

### How ProperBundler Works

`ProperBundler` (`mpisppy/utils/proper_bundler.py`) wraps a model module:

1. `bundle_names_creator()` generates names like `Bundle_1_4` (scenarios 1-4)
2. `scenario_creator("Bundle_1_4")` calls `sputils.create_EF()` on constituent
   scenarios, then `sputils.attach_root_node()` to make it a 2-stage problem
3. From PH's perspective, each bundle is a "big scenario" with summed probability
4. For multi-stage, `bunBFs` (bundle branching factors) ensure consistent tree shape
5. `nonant_for_fixed_vars=False` is used in `create_EF()`

Key files: `proper_bundler.py:86` (name format), `proper_bundler.py:125` (create_EF),
`proper_bundler.py:145` (attach_root_node)

### How Stochastic ADMM Works

`Stoch_AdmmWrapper` (`mpisppy/utils/stoch_admmWrapper.py`):

1. Virtual scenarios = cross product: `admm_subproblems × stochastic_scenarios`
   - Named like `ADMM_STOCH_Region1_StochasticScenario1`
2. **Pre-creates** all local scenarios in `__init__` (lines 81-87)
3. `assign_variable_probs()` (lines 130-227):
   - Adds dummy fixed variables (prob=0) for consensus vars not in a subproblem
   - Records per-variable probabilities in `varprob_dict` (keyed by model object)
   - Augments scenario tree: adds ADMM leaf node, multiplies node costs
   - Divides scenario probability by `num_admm_subproblems`
4. `admmWrapper_scenario_creator(sname)` returns pre-created scenario with
   objective multiplied by `num_admm_subproblems`
5. `var_prob_list(s)` returns `varprob_dict[s]` — list of `(id(var), prob)`

Key constraint: all scenarios from the same ADMM subproblem have the same
real/dummy consensus variable pattern, so variable probabilities are uniform
within a subproblem group.

### Current State

`mpisppy/generic/admm.py:59-60` explicitly blocks bundling:
```python
if cfg.get("scenarios_per_bundle") is not None:
    raise RuntimeError("Proper bundles are not supported with ADMM")
```

## Architecture: Two Paths, Shared Code

```
Path 1 (standard):   module → [ProperBundler] → PH
Path 2 (ADMM):       module → Stoch_AdmmWrapper → [AdmmBundler] → PH
```

### Shared code (already in sputils)
- `create_EF()` — builds EF model from scenarios
- `attach_root_node()` — converts to 2-stage for PH
- Config option: `scenarios_per_bundle`

### What may need factoring from ProperBundler
- Multi-stage `bunBFs` computation logic
- Pickle/unpickle infrastructure
- The `scenario_creator` dispatch pattern (detect bundle name → build EF)

### New ADMM-specific code
- Bundle name generation grouped by subproblem (e.g., `Bundle_Region1_1_4`)
- Splitting Stoch_AdmmWrapper flow: pre-create → assign_variable_probs → bundle → scale objectives
- `var_prob_list` for bundle models (lookup by subproblem, not by model object)

## Design Options

### Option B: Separate AdmmBundler after wrapper

A new `AdmmBundler` class that wraps `Stoch_AdmmWrapper`:
- Groups pre-created scenarios by ADMM subproblem
- Creates EF bundles within each group
- Provides `var_prob_list` for bundle models
- Handles objective scaling on bundle EF_Obj

Pro: Modular, wrapper unchanged.
Con: Wrapper must expose unscaled scenarios (objective scaling currently
happens in `admmWrapper_scenario_creator`).

### Option C: Bundle inside the wrapper

Factor `Stoch_AdmmWrapper` so bundling happens between pre-creation and
objective scaling:

1. Pre-create scenarios
2. `assign_variable_probs()` (adds dummy vars, computes prob vectors)
3. Group by subproblem, create EF bundles (NEW)
4. Scale bundle objectives
5. Compute bundle-level variable probabilities

Pro: Cleanest handling of objective scaling.
Con: Larger refactor of Stoch_AdmmWrapper.

## Key Technical Details

### Variable probability for bundles
All scenarios in a same-subproblem bundle have identical real/dummy consensus
variable patterns. So `var_prob_list` for a bundle = same as any constituent
scenario. Implementation: lookup by subproblem name, not by model object.

### Objective scaling
Currently `admmWrapper_scenario_creator` multiplies individual scenario
objectives by `num_admm_subproblems`. With bundling, `create_EF` does its own
probability-weighted aggregation, so the multiplication must happen on the
bundle's `EF_Obj` instead of on individual scenarios.

### Tree structure
Stochastic ADMM adds an ADMM leaf node to the tree. Bundling collapses
stochastic stages within a subproblem. The ADMM leaf node would attach to the
bundle. Analogous to how ProperBundler uses `bunBFs` for multi-stage.

### Bundle naming
Proposed: `Bundle_Region1_0_3` — encodes subproblem name for easy grouping
validation and debugging.

## Open Design Questions

1. Option B (separate bundler) vs Option C (inside wrapper)?
2. Should ProperBundler be factored into a base class, or keep shared logic in sputils?
3. Bundle naming convention: encode subproblem name or flat numbering?
4. Keep the `_check_admm_compatibility` guard for `--admm` but lift it for `--stoch-admm`?

## Files Involved

- `mpisppy/utils/proper_bundler.py` — existing bundler (pattern to follow / factor)
- `mpisppy/utils/stoch_admmWrapper.py` — stochastic ADMM wrapper (must be modified)
- `mpisppy/utils/admmWrapper.py` — deterministic ADMM (no bundling, keep guard)
- `mpisppy/generic/admm.py` — generic ADMM setup (lift guard for stoch_admm)
- `mpisppy/generic/parsing.py` — `name_lists()`, ADMM-aware bundle name generation
- `mpisppy/generic_cylinders.py` — entry point, compose bundler with ADMM
- `mpisppy/spbase.py` — variable_probability handling, probability checking
- `mpisppy/phbase.py` — PH iteration (prob0_mask for zero-prob vars)
- `mpisppy/utils/sputils.py` — `create_EF()`, `attach_root_node()` (shared code)
