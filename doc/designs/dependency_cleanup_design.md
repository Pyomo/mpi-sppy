# Dependency cleanup and packaging extras — Design

Status: reviewed; open questions resolved (see §6). Motivated by a developer request: dependencies are
"all over the place"; even without more explicit hard dependencies, there
should be an extra like `[test]` or `[extras]` that pulls in everything one
could want, which would also make the CI runners cleaner.

## 1. Current state (the mess, itemized)

### 1.1 Two packaging files, out of sync

- `pyproject.toml` is authoritative (PEP 621 `[project]` table; `pip install -e .`
  and the publish workflow both use it; installed metadata reports its version,
  0.14.1.dev0).
- `setup.py` still exists with **stale** metadata: version 0.13.2.dev0, the old
  `find_packages()` call, and its own copy of `install_requires`/`extras_require`.
  It was not updated when 0.14.0 was cut. Nothing uses it; it can only mislead.

### 1.2 A hard dependency that is never imported

`sortedcollections` is in `dependencies` (both files) but is **not imported
anywhere in the repository** — not in `mpisppy/`, not in `examples/`, not in
tests. It is a leftover from code that has since been removed.

### 1.3 Extras that don't carve the space usefully

Current extras: `doc` (sphinx trio), `mpi` (mpi4py), and three single-package
aliases — `scipy`, `pandas`, `plot` (= matplotlib). The aliases save no typing
over `pip install scipy`, are used by no CI job and no documentation, and there
is no extra at all for "everything", for testing, or for developing.

### 1.4 What the code actually needs (import survey, 2026-07)

Hard requirements (unguarded top-level imports throughout the core):
**pyomo** (~100+ files) and **numpy** (~40 files). Correctly declared.

Optional, properly guarded (missing package never blocks `import mpisppy.*`):

| Package | Where used | Guard mechanism | Enables |
|---|---|---|---|
| mpi4py | `mpisppy/MPI.py` only | try/except with mock fallback | real parallelism |
| scipy | `confidence_intervals/` (3 files), `utils/prox_approx.py`, `utils/kkt/` | `pyomo.common.dependencies` | CI intervals, prox approx |
| pandas | `utils/rho_utils.py`, `w_utils/wtracker.py`, `convergers/primal_dual_converger.py`, `extensions/phtracker.py` | `pyomo.common.dependencies` | rho CSV, W tracking |
| dill | `utils/pickle_bundle.py` | `attempt_import` | pickled bundles |

Optional, **unguarded** at module top level — importing that one module crashes
without the package (acceptable "opt-in module" pattern, but the tracebacks are
unfriendly):

| Package | Crashing module | Enables |
|---|---|---|
| mip (python-mip) | `problem_io/mps_reader.py` (and transitively `smps_module.py`) | MPS/LP/SMPS file input |
| gurobipy | `extensions/timed_mipgap.py` | timed mipgap extension |
| parapint | `opt/sc.py` | Schur-complement |
| gams + gamspy_base | `agnostic/gams_guest.py` | GAMS agnostic guest |

Tests and examples additionally use: **egret** (UC tests/examples; PyPI name
`gridx-egret`), **matplotlib** (examples only), **pytest**/**coverage** (CI),
**amplpy** (agnostic examples only).

### 1.5 CI runners hand-roll everything

`test_pr_and_main.yml` has ~25 jobs, each with its own `pip install` list, and
they have drifted: test jobs install sphinx (never used there); some install
`sphinx-copybutton`, some don't; `dill` appears in most lists but not all;
`matplotlib scipy` appear in exactly the jobs where someone once hit a missing
import; egret comes from `gridx-egret` (PyPI) in most jobs but
`git+https://github.com/grid-parity-exchange/egret.git` in others. None of the
jobs use the package extras. Adding a test dependency today means editing a
dozen YAML stanzas (this has already bitten us — see the standing rule that new
test files must be wired into both `run_coverage.bash` and the workflow).

## 2. Goals / non-goals

Goals:

1. One authoritative packaging file with truthful dependencies.
2. A small set of purposeful extras, including a "kitchen sink" one, so
   `pip install -e ".[dev,mpi]"` gives a contributor everything.
3. CI jobs install via extras; per-job deltas shrink to genuinely job-specific
   items (solvers, conda MPI, git checkouts).

Non-goals:

- No new hard dependencies. `dependencies` stays `numpy`, `pyomo>=6.4`.
- Solvers (cplex, xpress, gurobipy) stay **out of all extras** — they are
  licensing decisions, and CI's community editions are a CI concern.
- Proprietary-bridge packages (gams, gamspy_base, amplpy) stay out — useless
  without a licensed installation.
- No import-graph restructuring; guarding the four crash-on-import modules is
  a small optional phase, not a rewrite.

## 3. Proposed extras

In `pyproject.toml` `[project.optional-dependencies]` (extras may reference
other extras of the same project; pip has supported this for years):

```toml
[project.optional-dependencies]
mpi = ["mpi4py>=3.0.3"]                      # unchanged
doc = ["sphinx", "sphinx-copybutton", "sphinx_rtd_theme"]   # unchanged
# Every pip-safe optional feature of the library itself:
extras = [
    "scipy",        # confidence intervals, prox_approx, kkt
    "pandas",       # rho CSV utilities, wtracker, phtracker
    "matplotlib",   # plotting in convergers/extensions/examples
    "dill",         # pickled scenario bundles
    "mip",          # MPS/LP/SMPS file input (problem_io)
]
# Everything the test suite and examples need beyond a solver:
test = [
    "mpi-sppy[extras]",
    "pytest",
    "coverage",
    "addheader",    # imported by tests/test_headers.py
    "gridx-egret",  # UC tests and examples
]
# One-stop shop for contributors:
dev = [
    "mpi-sppy[test]",
    "mpi-sppy[doc]",
    "ruff",
]
```

Notes and judgment calls:

- **Names**: `extras` and `test` singular-style to match existing `mpi`/`doc`.
  (`all` was considered for the kitchen-sink extra but is misleading: it cannot
  responsibly include mpi4py — `pip install mpi4py` fails outright on machines
  without an MPI toolchain — nor solvers. `extras` says "the optional features",
  which is what it is.)
- **`parapint`** is deliberately not in `test`: only the schur-complement CI
  job uses it, and that job needs conda-side pieces (pymumps, openmpi) anyway.
  It stays an explicit install in that one job.
- **egret source**: standardize on the `gridx-egret` PyPI release everywhere in
  `test_pr_and_main.yml`; only `test_uc_weekly.yml` keeps the git checkout
  (its purpose is tracking upstream).
- **Old alias extras** `scipy`, `pandas`, `plot`: dropped immediately. Nothing
  in the repo, docs, or CI uses them; pip only warns (does not error) on an
  unknown extra, so a stale external `pip install mpi-sppy[scipy]` still
  installs mpi-sppy itself, and each alias's package is trivially installable
  by its own name.

Contributor happy path, documented in README and `quick_start.rst`:

```bash
pip install -e ".[dev,mpi]"     # or ".[test,mpi]" without doc/lint tools
```

## 4. Changes, phased (one review-sized PR each, each green on its own)

### Phase 1 — packaging truth

- Delete `setup.py`. (`pyproject.toml` + setuptools backend fully covers build,
  editable install, and the console scripts; the publish workflow uses `build`.)
- Remove `sortedcollections` from `dependencies`.
- Add the `extras`, `test`, `dev` extras; keep `mpi`, `doc`; drop the
  `scipy`/`pandas`/`plot` alias extras outright.
- Update install docs: README.md, `doc/src/quick_start.rst`,
  `doc/src/install_mpi.rst`, CLAUDE.md (test/lint commands unchanged, install
  line gains the extras mention). Add a short "what each extra enables" table
  to the docs (essentially §1.4's first table).
- Sanity check: fresh venv, `pip install -e ".[dev]"`, `python -c "import
  mpisppy"`, `ruff check .`, run a no-solver unit test file.

### Phase 2 — CI runners consume the extras

In `test_pr_and_main.yml` (and `pyotracker.yml`):

- Replace each job's hand-rolled feature-package list with
  `pip install -e ".[test]"` at the existing `pip install -e .` step (conda
  jobs keep `conda install mpi4py ... setuptools` first; pip then sees mpi4py
  satisfied and the `[mpi]` extra is unnecessary in CI).
- Keep explicit, per job: solver installs (`cplex`, `xpress`, `gurobipy`),
  `pybind11` (pyomo extension builds), git installs (parapint, pyutilib),
  and anything pinned for a specific job.
- Drop the sphinx installs from non-doc test jobs; the docs job uses `[doc]`.
- Net effect: adding a test dependency becomes a one-line `pyproject.toml`
  change instead of ~12 YAML edits.
- `test_uc_weekly.yml`: same treatment, keeping its git egret.
- Verify by letting the full matrix run on the PR (it already runs on PRs).

### Phase 3 (behavioral) — friendly errors for opt-in modules

Convert the four unguarded opt-in modules (`problem_io/mps_reader.py`,
`extensions/timed_mipgap.py`, `opt/sc.py`, `agnostic/gams_guest.py`) to
`pyomo.common.dependencies.attempt_import`, so a missing package produces
"module X requires package Y (pip install ...)" at first *use* instead of an
ImportError traceback at import. Purely a UX improvement; no behavior change
when the package is present.

## 5. Risks

- **Deleting `setup.py`**: any downstream tooling that shells out to
  `python setup.py ...` breaks — but such invocations are deprecated
  setuptools-wide, and the file's metadata is already wrong, which is worse.
- **Alias-extra users**: none found in repo, docs, or CI. An external user of
  a dropped alias gets a pip warning and a missing (guarded) feature package,
  not a broken install; the fix is `pip install <package>` or `[extras]`.
- **`test` extra pulls egret** even for developers who never run UC tests
  (~small pure-python install). Judged acceptable for the "everything you
  could want" semantics; anyone who objects can install `[extras]` instead.

## 6. Resolved review decisions (DLW, 2026-07-23)

1. Kitchen-sink extra is named `extras`.
2. The `scipy`/`pandas`/`plot` alias extras are dropped immediately (no
   deprecation window) — see §3 for rationale.
3. Phase 3 is in scope and will be done as the third PR.
