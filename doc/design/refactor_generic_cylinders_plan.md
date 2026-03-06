# Refactoring Plan: `generic_cylinders.py` → `mpisppy/generic/`

## Goal

Split `mpisppy/generic_cylinders.py` (~860 lines) into a package `mpisppy/generic/`
with focused modules. This makes the code easier to maintain and sets the stage for
`admm_generic_cylinders` support (via `--admm` / `--stoch-admm` flags).

## Dependency

Wait for the `remove_loose_bundles` PR to merge first. That PR will remove or
simplify bundle-related code (`_write_bundles`, `_proper_bundles`, `bundle_wrapper`
parameters, `pickle_bundles_dir` config, etc.), reducing what we need to move.

The plan below describes the **post-merge** state. Lines/functions that get removed
by `remove_loose_bundles` are marked with **(RLB)** — skip those during the refactor.

## New Package Structure

```
mpisppy/generic/
    __init__.py          # re-exports main entry points
    parsing.py           # CLI arg parsing, module loading
    hub.py               # hub dict construction
    extensions.py        # extension wiring onto hub_dict
    spokes.py            # spoke dict construction
    ef.py                # _do_EF (extensive form solve)
    solution_io.py       # scenario/bundle pickling, solution writing
    mmw.py               # do_mmw (MMW confidence intervals)
    admm.py              # (future) ADMM wrapper creation
```

## Module-by-Module Breakdown

### `mpisppy/generic/__init__.py`

Re-export the public API so existing imports still work:

```python
from mpisppy.generic.decomp import do_decomp
from mpisppy.generic.ef import do_EF
from mpisppy.generic.mmw import do_mmw
```

### `mpisppy/generic/parsing.py`

**Moved from `generic_cylinders.py`:**
- `_model_fname()` (lines 615-637)
- `_parse_args(m)` (lines 32-104)
- `_name_lists(module, cfg, bundle_wrapper=None)` (lines 106-133)

**Notes:**
- `_parse_args` registers all cfg arg groups. When ADMM is added later,
  `admm.py` will call a helper to add ADMM-specific args before `parse_command_line`.
- `_name_lists` handles `branching_factors` and bundle logic. Post-RLB, bundle
  parts may be simplified or removed.

### `mpisppy/generic/hub.py`

**New function:** `build_hub_dict(module, cfg, beans, scenario_creator_kwargs, rho_setter, all_nodenames, ph_converger)`

Extracted from `do_decomp` lines 177-225. Contains the if/elif chain:
- `cfg.APH` → `vanilla.aph_hub(...)`
- `cfg.subgradient_hub` → `vanilla.subgradient_hub(...)`
- `cfg.fwph_hub` → `vanilla.fwph_hub(...)`
- `cfg.ph_primal_hub` → `vanilla.ph_primal_hub(...)`
- else → `vanilla.ph_hub(...)`

Plus line 225: `hub_dict['opt_kwargs']['options']['cfg'] = cfg`

**ADMM hook:** The function will accept an optional `variable_probability=None`
parameter and pass it through to the vanilla factory calls.

### `mpisppy/generic/extensions.py`

**New function:** `configure_extensions(hub_dict, module, cfg)`

Extracted from `do_decomp` lines 228-323. Handles:
- `ext_classes` list construction
- Gapper, fixer, rc_fixer, relaxed_ph_fixer, integer_relax_then_enforce
- GradRho, WXBarReader, WXBarWriter
- User-defined extensions
- sep_rho, coeff_rho, sensi_rho, reduced_costs_rho
- MultiExtension assembly
- primal_dual_converger_options
- NormRhoUpdater, PrimalDualRho

### `mpisppy/generic/spokes.py`

**New function:** `build_spoke_list(cfg, beans, scenario_creator_kwargs, rho_setter, all_nodenames)`

Extracted from `do_decomp` lines 326-441. Creates spoke dicts for each enabled
cylinder and returns `list_of_spoke_dict`.

Handles: fwph, lagrangian, ph_dual, relaxed_ph, subgradient, xhatshuffle,
xhatxbar, reduced_costs.

**ADMM hook:** Accept optional `variable_probability=None` and pass to spokes
that support it (xhatxbar, etc.).

### `mpisppy/generic/decomp.py`

**New function:** `do_decomp(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement, bundle_wrapper=None)`

This becomes a thin orchestrator (~40 lines):

```python
def do_decomp(module, cfg, scenario_creator, scenario_creator_kwargs,
              scenario_denouement, bundle_wrapper=None):
    rho_setter = _get_rho_setter(module, cfg)
    ph_converger = _get_converger(cfg)
    all_scenario_names, all_nodenames = name_lists(module, cfg, bundle_wrapper)

    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)
    variable_probability = cfg.get("variable_probability")  # None unless ADMM

    hub_dict = build_hub_dict(module, cfg, beans, scenario_creator_kwargs,
                              rho_setter, all_nodenames, ph_converger,
                              variable_probability=variable_probability)
    configure_extensions(hub_dict, module, cfg)
    list_of_spoke_dict = build_spoke_list(cfg, beans, scenario_creator_kwargs,
                                          rho_setter, all_nodenames,
                                          variable_probability=variable_probability)

    # Model callback hook
    if hasattr(module, 'hub_and_spoke_dict_callback'):
        module.hub_and_spoke_dict_callback(hub_dict, list_of_spoke_dict, cfg)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    _write_solutions(wheel, module, cfg)
    return wheel
```

Plus small helpers:
- `_get_rho_setter(module, cfg)` — lines 153-158
- `_get_converger(cfg)` — lines 160-169
- `_write_solutions(wheel, module, cfg)` — lines 451-468

### `mpisppy/generic/ef.py`

**Moved from `generic_cylinders.py`:**
- `_do_EF(module, cfg, scenario_creator, scenario_creator_kwargs, scenario_denouement, bundle_wrapper=None)` (lines 558-613)

Renamed to `do_EF` (public).

### `mpisppy/generic/solution_io.py`

**Moved from `generic_cylinders.py`:**
- `_write_scenarios(...)` (lines 473-505) **(RLB: may be simplified)**
- `_read_pickled_scenario(sname, cfg)` (lines 509-513)
- `_write_bundles(...)` (lines 517-555) **(RLB: likely removed)**
- `_write_scenario_lp_mps_files_only(...)` (lines 645-687)

### `mpisppy/generic/mmw.py`

**Moved from `generic_cylinders.py`:**
- `_mmw_requested(cfg)` (lines 690-708)
- `do_mmw(module_fname, cfg, wheel=None)` (lines 711-778)

**External import note:** `test_conf_int_farmer.py:278` imports `do_mmw` from
`mpisppy.generic_cylinders`. Update to import from `mpisppy.generic.mmw` (or
from `mpisppy.generic` via `__init__.py` re-export).

### `mpisppy/generic/admm.py` (future — not part of this refactor)

Will contain:
- `setup_admm(module, cfg, n_cylinders)` — creates `AdmmWrapper`, returns
  modified `scenario_creator`, attaches `variable_probability` to cfg
- `setup_stoch_admm(module, cfg, n_cylinders)` — same for `Stoch_AdmmWrapper`
- ADMM-specific arg registration

### `mpisppy/generic_cylinders.py` (after refactor)

Becomes a slim CLI entry point (~60 lines):

```python
"""Generic cylinder driver for mpi-sppy."""
import sys
from mpisppy.generic.parsing import model_fname, parse_args, proper_bundles
from mpisppy.generic.decomp import do_decomp
from mpisppy.generic.ef import do_EF
from mpisppy.generic.solution_io import (write_bundles, write_scenarios,
                                          write_scenario_lp_mps_files_only)
from mpisppy.generic.mmw import mmw_requested, do_mmw
# future: from mpisppy.generic.admm import setup_admm, setup_stoch_admm

if __name__ == "__main__":
    # load module
    # parse args
    # set up scenario_creator (bundles, pickled, or standard)
    # dispatch to EF, decomp, pickle, or lp/mps writing
    # optionally run MMW
```

## `sputils.count_cylinders(cfg)`

Add to `mpisppy/utils/sputils.py`:

```python
# Canonical list of cfg flags that create a spoke
SPOKE_CFG_FLAGS = [
    "lagrangian", "xhatshuffle", "xhatxbar", "fwph",
    "ph_dual", "relaxed_ph", "subgradient", "reduced_costs",
]

def count_cylinders(cfg):
    """Count the number of cylinders (1 hub + enabled spokes)."""
    count = 1  # hub
    for flag in SPOKE_CFG_FLAGS:
        if cfg.get(flag, False):
            count += 1
    return count
```

## Migration Steps

1. **Create `mpisppy/generic/__init__.py`** with re-exports
2. **Create each module** by moving functions (not copy-paste — actually move)
3. **Update `generic_cylinders.py`** to import from the new package
4. **Update `test_conf_int_farmer.py`** import of `do_mmw`
5. **Add `count_cylinders` to `sputils.py`**
6. **Run tests:** `python -m pytest mpisppy/tests/test_ef_ph.py` and any ADMM-related tests
7. **Verify MPI tests:** `mpiexec -np 3 python -m mpi4py mpisppy/generic_cylinders.py --module-name <some_model> ...`

## Risks and Notes

- **Import cycles:** `generic/` modules import from `mpisppy.utils.cfg_vanilla`,
  `mpisppy.utils.sputils`, etc. No risk of cycles since `generic/` is a new leaf package.
- **`remove_loose_bundles` impact:** The `bundle_wrapper` parameter flows through
  `do_decomp`, `_name_lists`, `_do_EF`, and `__main__`. Post-RLB, these will be
  simplified. Do the refactor after that PR merges to avoid moving code that's about
  to be deleted.
- **Backwards compatibility:** Keep `generic_cylinders.py` as the CLI entry point.
  Re-export `do_decomp` and `do_mmw` from `mpisppy.generic` so library users can
  import from either path during a transition period.
