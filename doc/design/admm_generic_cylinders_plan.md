# Design Plan: ADMM in `generic_cylinders`

## Goal

Add `--admm` and `--stoch-admm` CLI flags to `generic_cylinders.py` so that
ADMM-based decomposition can be used with any compatible model module, without
requiring a bespoke `*_admm_cylinders.py` driver per problem.

## Current State

The `mpisppy/generic/` package refactor is complete. `generic_cylinders.py` is now
a slim `__main__` entry point that delegates to:

- `mpisppy/generic/parsing.py` — CLI arg parsing, module loading, `name_lists()`
- `mpisppy/generic/hub.py` — `build_hub_dict()`
- `mpisppy/generic/spokes.py` — `build_spoke_list()`
- `mpisppy/generic/decomp.py` — `do_decomp()` orchestrator
- `mpisppy/generic/ef.py` — `do_EF()`
- `mpisppy/generic/extensions.py` — `configure_extensions()`
- `mpisppy/generic/mmw.py` — `do_mmw()`
- `mpisppy/generic/scenario_io.py` — pickle/LP/MPS I/O

The new ADMM module will be `mpisppy/generic/admm.py`.

## CLI Interface

```bash
# Deterministic ADMM (replaces distr_admm_cylinders.py pattern)
mpiexec -np 6 python -m mpi4py mpisppy/generic_cylinders.py \
    --module-name distr --admm --num-scens 3 \
    --default-rho 1.0 --max-iterations 100 --solver-name cplex \
    --lagrangian --xhatxbar

# Stochastic ADMM (replaces stoch_distr_admm_cylinders.py pattern)
mpiexec -np 6 python -m mpi4py mpisppy/generic_cylinders.py \
    --module-name stoch_distr --stoch-admm --num-admm-subproblems 3 \
    --num-stoch-scens 3 --default-rho 1.0 --max-iterations 100 \
    --solver-name cplex --lagrangian --xhatxbar
```

## Model Module Interface

### Standard (non-ADMM) — already required by `generic_cylinders`

- `scenario_creator(scenario_name, **kwargs)`
- `scenario_names_creator(num_scens, ...)`
- `scenario_denouement(rank, scenario_name, scenario)`
- `kw_creator(cfg)`
- `inparser_adder(cfg)`

### Additional for `--admm` (deterministic)

- `consensus_vars_creator(num_scens, all_scenario_names, **scenario_creator_kwargs)` → returns `consensus_vars` dict
  - Keys: subproblem names (= scenario names)
  - Values: list of consensus variable name strings
  - `all_scenario_names` is passed explicitly (not part of `scenario_creator_kwargs`)
  - Remaining kwargs come from `kw_creator(cfg)` output via `**scenario_creator_kwargs`

### Additional for `--stoch-admm`

- `consensus_vars_creator(admm_subproblem_names, stoch_scenario_name, **scenario_creator_kwargs)` → `consensus_vars` dict
  - Keys: ADMM subproblem names
  - Values: list of `(var_name, stage)` tuples
  - Remaining kwargs come from `kw_creator(cfg)` output via `**scenario_creator_kwargs`
- `admm_subproblem_names_creator(num_admm_subproblems)` → list of subproblem name strings
- `stoch_scenario_names_creator(num_stoch_scens)` → list of stochastic scenario name strings
- `admm_stoch_subproblem_scenario_names_creator(admm_subproblem_names, stoch_scenario_names)` → list of composite names
- `split_admm_stoch_subproblem_scenario_name(name)` → `(admm_subproblem_name, stoch_scenario_name)`

## New Config Args

Added when `--admm` or `--stoch-admm` is present:

| Arg | Domain | Description |
|---|---|---|
| `--admm` | bool | Enable deterministic ADMM decomposition |
| `--stoch-admm` | bool | Enable stochastic ADMM decomposition |
| `--num-admm-subproblems` | int | Number of ADMM subproblems (stoch-admm only) |
| `--num-stoch-scens` | int | Number of stochastic scenarios (stoch-admm only) |

## Implementation: `mpisppy/generic/admm.py`

```python
"""ADMM setup for generic_cylinders."""

from mpisppy import MPI
from mpisppy.utils.admmWrapper import AdmmWrapper
from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper


def admm_args(cfg):
    """Register ADMM-specific config args."""
    cfg.add_to_config("admm", description="Use ADMM decomposition",
                      domain=bool, default=False)
    cfg.add_to_config("stoch_admm", description="Use stochastic ADMM decomposition",
                      domain=bool, default=False)


def setup_admm(module, cfg, n_cylinders):
    """Create AdmmWrapper for deterministic ADMM.

    Modifies cfg by attaching variable_probability.
    Returns modified scenario_creator, scenario_creator_kwargs,
    all_scenario_names, all_nodenames.
    """
    all_scenario_names = module.scenario_names_creator(cfg.num_scens)
    scenario_creator_kwargs = module.kw_creator(cfg)
    consensus_vars = module.consensus_vars_creator(
        cfg.num_scens, all_scenario_names, **scenario_creator_kwargs
    )

    admm = AdmmWrapper(
        options={},
        all_scenario_names=all_scenario_names,
        scenario_creator=module.scenario_creator,
        consensus_vars=consensus_vars,
        n_cylinders=n_cylinders,
        mpicomm=MPI.COMM_WORLD,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    cfg.quick_assign("variable_probability", object, admm.var_prob_list)

    return (admm.admmWrapper_scenario_creator, None,
            all_scenario_names, None)


def setup_stoch_admm(module, cfg, n_cylinders):
    """Create Stoch_AdmmWrapper for stochastic ADMM.

    Modifies cfg by attaching variable_probability.
    Returns modified scenario_creator, scenario_creator_kwargs,
    all_scenario_names, all_nodenames.
    """
    admm_subproblem_names = module.admm_subproblem_names_creator(cfg.num_admm_subproblems)
    stoch_scenario_names = module.stoch_scenario_names_creator(cfg.num_stoch_scens)
    all_names = module.admm_stoch_subproblem_scenario_names_creator(
        admm_subproblem_names, stoch_scenario_names)

    scenario_creator_kwargs = module.kw_creator(cfg)
    stoch_scenario_name = stoch_scenario_names[0]
    consensus_vars = module.consensus_vars_creator(
        admm_subproblem_names, stoch_scenario_name, **scenario_creator_kwargs)

    admm = Stoch_AdmmWrapper(
        options={},
        all_admm_stoch_subproblem_scenario_names=all_names,
        split_admm_stoch_subproblem_scenario_name=module.split_admm_stoch_subproblem_scenario_name,
        admm_subproblem_names=admm_subproblem_names,
        stoch_scenario_names=stoch_scenario_names,
        scenario_creator=module.scenario_creator,
        consensus_vars=consensus_vars,
        n_cylinders=n_cylinders,
        mpicomm=MPI.COMM_WORLD,
        scenario_creator_kwargs=scenario_creator_kwargs,
        BFs=None,  # could be extracted from cfg if multi-stage
    )

    cfg.quick_assign("variable_probability", object, admm.var_prob_list)

    return (admm.admmWrapper_scenario_creator, None,
            all_names, admm.all_nodenames)
```

## Integration Points

### 1. `mpisppy/generic/parsing.py` — register ADMM args

In `parse_args()`, add a call to `admm_args(cfg)` alongside the other arg
registration calls (after `cfg.xhatxbar_args()`, etc.).

### 2. `mpisppy/generic/decomp.py` — pass `variable_probability`

`do_decomp()` reads `variable_probability` from cfg and passes it through to
`build_hub_dict()` and `build_spoke_list()`. Both `build_hub_dict` (in `hub.py`)
and `build_spoke_list` (in `spokes.py`) forward it to the vanilla factory
functions, which already accept `variable_probability` parameters.

```python
# In do_decomp(), after getting beans:
variable_probability = cfg.get("variable_probability")  # None unless ADMM

hub_dict = build_hub_dict(cfg, beans, scenario_creator_kwargs,
                          rho_setter, all_nodenames, ph_converger,
                          variable_probability=variable_probability)

list_of_spoke_dict = build_spoke_list(cfg, beans, scenario_creator_kwargs,
                                      rho_setter, all_nodenames,
                                      variable_probability=variable_probability)
```

### 3. `mpisppy/generic/hub.py` — accept and forward `variable_probability`

Add `variable_probability=None` parameter to `build_hub_dict()` and pass it
through to each `vanilla.*_hub()` call.

### 4. `mpisppy/generic/spokes.py` — accept and forward `variable_probability`

Add `variable_probability=None` parameter to `build_spoke_list()` and pass it
through to each `vanilla.*_spoke()` call.

### 5. `mpisppy/generic_cylinders.py` `__main__` block

Add ADMM setup between scenario_creator resolution and the EF/decomp dispatch:

```python
from mpisppy.generic.admm import setup_admm, setup_stoch_admm
import mpisppy.utils.sputils as sputils

# ... after scenario_creator/kwargs are set up ...

if cfg.get("admm", ifmissing=False) or cfg.get("stoch_admm", ifmissing=False):
    n_cylinders = sputils.count_cylinders(cfg)
    if cfg.admm:
        scenario_creator, scenario_creator_kwargs, _, _ = \
            setup_admm(module, cfg, n_cylinders)
    else:
        scenario_creator, scenario_creator_kwargs, _, _ = \
            setup_stoch_admm(module, cfg, n_cylinders)
    # scenario_creator is now the ADMM wrapper's creator
    # variable_probability is attached to cfg
```

Note: With ADMM, `name_lists()` in `do_decomp()` still works because
`module.scenario_names_creator` returns the ADMM-wrapped scenario names.
For stoch-admm, `name_lists()` may need adjustment since the "scenarios"
are actually composite ADMM-stochastic names — this needs careful testing.

## Scope

This design targets **models that already conform to the `generic_cylinders` module
interface** — i.e., they provide `scenario_creator`, `scenario_names_creator`,
`kw_creator(cfg)`, `inparser_adder(cfg)`, and `scenario_denouement`.

Models with highly customized ADMM setups will continue to use bespoke drivers.

## Constraints and Limitations

- **FWPH spoke does not work with `variable_probability`** — should raise an error
  if both `--admm`/`--stoch-admm` and `--fwph` are enabled.
- **EF mode with ADMM** — the existing `distr_ef.py` pattern (EF of the wrapped
  scenarios) should work, but needs testing. Could be supported via `--EF --admm`.
- **Bundle support** — ADMM + proper bundles interaction is untested and likely
  unsupported initially. Should raise an error if both are specified.
- **`n_cylinders` must be computed before wrapper creation** — uses
  `sputils.count_cylinders(cfg)`.

## Testing Plan

1. **Unit test:** Verify `setup_admm` and `setup_stoch_admm` produce correct
   wrapper objects with known inputs (distr, stoch_distr models).
2. **Integration test:** Run existing distr/stoch_distr examples through
   `generic_cylinders.py --admm` and compare results to existing bespoke drivers.
3. **Error cases:** Test that `--admm --fwph`, `--admm --stoch-admm`,
   `--admm --scenarios-per-bundle` all raise clear errors.

## Migration Path for Existing Examples

Once working, the bespoke drivers can be simplified to thin wrappers or deprecated:

```python
# examples/distr/distr_admm_cylinders.py (simplified)
# Now just: mpiexec -np 6 python -m mpi4py mpisppy/generic_cylinders.py \
#   --module-name distr --admm --num-scens 3 ...
```
