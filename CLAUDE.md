# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

mpi-sppy is a parallel stochastic programming library built on [Pyomo](https://pyomo.org). It implements a hub-and-spoke MPI architecture for solving large-scale scenario-based optimization under uncertainty.

## Installation

```bash
pip install -e .[mpi]
```

Requires a working MPI installation (OpenMPI or MPICH) and mpi4py. Also requires a Pyomo-compatible MIP solver (cplex, gurobi, or xpress).

## Commands

### Running Tests

Most tests use `unittest` and can be run serially:
```bash
python -m pytest mpisppy/tests/test_ef_ph.py
python -m pytest mpisppy/tests/test_conf_int_farmer.py
```

Tests that require MPI must be launched with mpiexec:
```bash
mpiexec -np 2 python -m mpi4py mpisppy/tests/test_with_cylinders.py
```

Smoke tests that spawn mpiexec internally:
```bash
python mpisppy/tests/straight_tests.py
```

Run a single unittest test method:
```bash
python -m pytest mpisppy/tests/test_ef_ph.py::Test_sizes::test_ef_2stage
```

### Linting

```bash
ruff check .
```

Configuration is in `.ruff.toml`. Old Pyomo model files (e.g., `ReferenceModel*.py`) are excluded.

### File Headers

Every non-empty Python file must have the mpi-sppy copyright header. Add missing headers with:
```bash
addheader -c addheader.yml
```

Header text is in `file_header.txt`. The `test_headers.py` pytest test enforces this.

### Building Docs

```bash
cd doc && make html
```

## Architecture

### Core Class Hierarchy

```
SPBase (spbase.py)           — base for all optimization objects; holds local_scenarios, comms
  └── SPOpt (spopt.py)       — adds Pyomo solve methods
        └── PHBase (phbase.py) — implements xbar computation and PH proximal terms
              └── opt/ph.py   — full Progressive Hedging hub
              └── opt/aph.py  — Asynchronous PH hub
```

### Optimization Algorithms (`mpisppy/opt/`)

| File | Class | Role |
|---|---|---|
| `ef.py` | `ExtensiveForm` | Solve all scenarios as one monolithic model |
| `ph.py` | `PH` | Progressive Hedging (synchronous) |
| `aph.py` | `APH` | Asynchronous Progressive Hedging |
| `fwph.py` | `FWPH` | Frank-Wolfe based PH |
| `lshaped.py` | `LShapedMethod` | L-shaped (Benders) decomposition |
| `subgradient.py` | `Subgradient` | Subgradient method |

### Hub-and-Spoke System (`mpisppy/cylinders/`)

`WheelSpinner` (`spin_the_wheel.py`) is the main entry point for parallel runs. It takes a `hub_dict` and `list_of_spoke_dict` describing which hub and spoke classes to instantiate on which MPI ranks.

- `hub.py`: Abstract `Hub` base class
- `spoke.py`: Abstract `Spoke` / `_BoundSpoke` base classes
- Concrete spokes provide inner bounds (xhat heuristics: `xhatlooper_bounder.py`, `xhatshufflelooper_bounder.py`, etc.) or outer bounds (`lagrangian_bounder.py`, `lagranger_bounder.py`, `subgradient_bounder.py`, etc.)

### Configuration System (`mpisppy/utils/config.py`)

`Config` (extends `pyomo.common.config.ConfigDict`) manages all options and CLI argument parsing. Usage pattern:
```python
cfg = config.Config()
cfg.popular_args()   # common args like solver_name, max_iterations
cfg.ph_args()        # PH-specific args
cfg.lagrangian_args()
cfg.parse_command_line("myprog")
```

### Standard Hub/Spoke Dictionaries (`mpisppy/utils/cfg_vanilla.py`)

`cfg_vanilla` provides factory functions (`ph_hub`, `lagrangian_spoke`, `xhatshufflelooper_spoke`, etc.) that build the dicts consumed by `WheelSpinner`.

### High-Level Wrappers

- **`Amalgamator`** (`utils/amalgamator.py`): Wraps `WheelSpinner`; assembles hub/spoke objects from a `Config` object and a list of cylinder names.
- **`generic_cylinders.py`** (top-level in `mpisppy/`): Command-line driver; loads any model module dynamically and runs the cylinder system.

### Extensions (`mpisppy/extensions/`)

Extensions plug into the PH iteration loop. They inherit from `Extension` (`extensions/extension.py`) and override hooks like `pre_iter0`, `post_iter0`, `miditer`, `enditer`, `post_everything`. Common ones: xhat evaluators (`xhatlooper.py`, `xhatspecific.py`), rho updaters (`mult_rho_updater.py`, `norm_rho_updater.py`, `grad_rho.py`), convergence fixers.

### Confidence Intervals (`mpisppy/confidence_intervals/`)

- `mmw_ci.py`: MMW (Mak-Morton-Wood) confidence interval estimator
- `seqsampling.py`: Sequential sampling
- `zhat4xhat.py`: Zhat evaluation for a given xhat

### Agnostic (Non-Pyomo) Models (`mpisppy/agnostic/`)

Allows coupling mpi-sppy to models written in AMPL (`ampl_guest.py`), GAMS (`gams_guest.py`), gurobipy, or another Pyomo model (`pyomo_guest.py`). The guest provides scenario creation and solve callbacks; `agnostic.py` coordinates with the hub/spoke system.

### Scenario Tree (`mpisppy/scenario_tree.py`)

`ScenarioNode` defines nodes in the scenario tree. Each scenario model must be annotated by `scenario_creator` with:
- `_mpisppy_probability`: scenario probability
- `_mpisppy_node_list`: list of `ScenarioNode` objects from root to leaf
- `_mpisppy_objective_functional`: the objective expression

### Key Utilities (`mpisppy/utils/`)

- `sputils.py`: Core helpers (e.g., `extract_num` scrapes scenario index from name, `attach_Ws_and_prox` adds PH weight/prox terms)
- `wxbarutils.py` / `wxbarwriter.py` / `wxbarreader.py`: W and xbar serialization
- `proper_bundler.py`: Scenario bundling support
- `admmWrapper.py` / `stoch_admmWrapper.py`: ADMM-based decomposition wrappers

### `mpisppy/MPI.py`

Thin shim: imports from `mpi4py.MPI` when available, otherwise provides a mock single-rank stub so the library works without MPI installed.

## Key Conventions

- **Scenario naming**: Scenarios are named with a string prefix followed by digits (e.g., `Scenario1`, `Scenario10`). `sputils.extract_num` extracts the integer suffix.
- **Solver detection in tests**: `mpisppy/tests/utils.py::get_solver()` tries cplex, gurobi, xpress (persistent variants first).
- **Options vs. Config**: Old code uses `options` dicts; new code uses the `Config` object. When adding features, prefer `Config`.
- **MPICH on HPC**: Set `MPICH_ASYNC_PROGRESS=1` for correct non-blocking behavior on MPICH clusters.
