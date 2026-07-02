# Merging boot-sp into mpi-sppy — design

**Status:** design captured and decisions ratified 2026-07-02; no code yet.
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-07-02

Related work:

- **boot-sp** (https://github.com/boot-sp/boot-sp; local clone
  `~/Documents/Research/boot-sp`) — the package being merged. Data-based,
  two-stage stochastic programming using bootstrap and bagging for
  confidence intervals. Two papers describe it:
  Chen & Woodruff, *Software for data-based stochastic programming using
  bootstrap estimation*, INFORMS Journal on Computing, 2023; and
  Chen & Woodruff, *Distributions and Bootstrap for Data-based Stochastic
  Programming*, Computational Management Science, 2024.
- **`mpisppy/confidence_intervals/`** — the existing confidence-interval
  package (MMW, sequential sampling, zhat4xhat). Bootstrap CIs are a
  sibling method and land inside this package. boot-sp already imports
  `ciutils`, `mmw_ci`, `xhat_eval`, `sputils`, `config`, and the
  `mpisppy.MPI` shim, so the merge direction is natural.
- **statdist** (`bootsp/statdist/`) — a statistical-distribution library
  (distribution fitting, epi-splines, kernels, copulas) that DLW drove the
  development of under separate funding, always intended to be open-source.
  It shares lineage with the GOSM/Prescient scenario-generation tools. The
  smoothed bootstrap methods use its univariate distributions.

---

## 1. Goals and non-goals

### Goals

1. All boot-sp functionality that users need — the end-user CI tool
   (`user_boot`), the z*/xhat prep tool (`boot_general_prep`), the
   coverage-experiment harness (`simulate_boot`), and all eleven
   `BootMethods` (empirical and smoothed) — available from mpi-sppy proper,
   with tests, docs, and examples.
2. **No new hard dependencies.** mpi-sppy's core deps stay
   numpy/pyomo/sortedcollections; scipy and matplotlib remain optional
   extras. Anything that needs them imports lazily and fails with a
   friendly message.
3. Importable on current and future scipy (the `scipy.stats.mvn` removal
   in scipy 1.14 must not break mpi-sppy).
4. mpi-sppy conventions throughout: standard file headers, `Config`-based
   options, `mpisppy.tests.utils.get_solver` in tests, tests wired into
   both `run_coverage.bash` and `test_pr_and_main.yml` in the same commit
   as the test files.
5. One canonical copy: after the merge, the boot-sp GitHub repo is
   archived with a pointer to mpi-sppy.

### Non-goals

- **Copulas and vines.** Only the univariate statdist distributions come
  over (see §3). The multivariate/copula/vine code stays in the archived
  boot-sp repo, recoverable if a future project needs it. The `gosm`
  import hook goes with it.
- **paper_runs.** The scripts reproducing the papers' experiments stay in
  the archived repo, where the citations point.
- **generic_cylinders integration.** MMW has `--mmw-*` flags in the
  generic driver; giving bootstrap CIs the same treatment is a possible
  follow-on, not part of this merge.
- **Multistage.** boot-sp is two-stage by construction; that does not
  change here.

---

## 2. Ratified decisions (2026-07-02)

1. **Placement:** subpackage `mpisppy/confidence_intervals/bootsp/`, with
   the trimmed `statdist/` nested inside it.
2. **statdist:** trimmed port — univariate distributions and their
   support modules only (~3,000 of ~5,300 lines).
3. **simulate_boot:** ported (it is the coverage-validation tool and the
   most heavily CI-exercised part of boot-sp).
4. **Old repo:** archive with a README pointer after the final phase; no
   shim release.

---

## 3. Target layout

```
mpisppy/confidence_intervals/bootsp/
    __init__.py
    boot_sp.py              # estimator engine: classical, extended,
                            #   subsampling, bagging (+ *_resample helpers)
    smoothed_boot_sp.py     # smoothed bootstrap/bagging          (PR-B)
    boot_utils.py           # BootMethods enum, cfg_for_boot, module
                            #   loading, cfg_from_json, compute_xhat
    user_boot.py            # end-user CLI
    boot_general_prep.py    # z*/xhat precompute CLI
    simulate_boot.py        # coverage-experiment harness          (PR-C)
    statdist/               # trimmed distribution library         (PR-B)
        __init__.py
        README.md           # provenance + what was trimmed and where
                            #   the full version lives (archived repo)
        base_distribution.py
        distributions.py    # univariate classes only
        distribution_factory.py
        splines.py          # epi-spline fitting (builds a Pyomo model)
        utilities.py
        sampler.py

examples/bootsp/
    farmer/                 # PR-A; cvar/, multi_knapsack/, schultz/ in PR-C

mpisppy/tests/
    test_boot_sp.py                 # PR-A (empirical, user_boot, prep)
    test_boot_sp_smoothed.py        # PR-B (smoothed + statdist univariate)
    test_boot_sp_simulate.py        # PR-C (coverage harness)

doc/src/boot_sp.rst         # toctree: after seqsamp.rst in index.rst
doc/designs/bootsp_merge_design.md   # this file
```

CLI invocations become:

```
python -m mpisppy.confidence_intervals.bootsp.user_boot <module> --options
python -m mpisppy.confidence_intervals.bootsp.boot_general_prep <json>
python -m mpisppy.confidence_intervals.bootsp.simulate_boot <json>
```

### statdist trim details

Kept: `base_distribution.py`, `distributions.py` (univariate classes:
normal, uniform, discrete, empirical, kernel, epi-spline, student-t),
`distribution_factory.py`, `splines.py`, `utilities.py`, `sampler.py`.

Dropped: `copula.py`, `vine.py`, `bicop.py` (~2,150 lines; `bicop.py` was
unreachable even in boot-sp) and the multivariate classes in
`distributions.py`. Dropping the multivariate classes removes the
`from scipy.stats import mvn` import — the scipy≥1.14 breakage — with no
replacement code needed. The `gosm` hook lives in `copula.py` and goes
with it.

Required edits to the survivors:

- `__init__.py`: currently star-imports `distributions`, `copula`, and
  `vine`; reduce to the univariate surface.
- `distribution_factory.import_all_classes()`: drop the `copula` import;
  while there, stop re-importing on every factory call.
- `base_distribution.py`: move the top-level `matplotlib.pyplot` /
  `mpl_toolkits` imports inside the plotting methods so matplotlib stays
  an optional extra.

---

## 4. Reconciliations and fixes during the port

Behavior-preserving unless noted.

1. **`xhat_generator` naming.** boot-sp looks up
   `xhat_generator_<module_name>` dynamically; mpi-sppy's CI code uses the
   fixed name `xhat_generator`. The port looks for the fixed name first
   and falls back to the dynamic one; a miss raises an error naming both
   spellings.
2. **MPI boilerplate.** `boot_sp.py` and `smoothed_boot_sp.py` each
   define module-level `n_proc/my_rank/comm` plus a per-rank singleton
   `rankcomm = comm.Split(...)`. Consolidate into one place
   (`boot_utils`); likewise the empirical-vs-smoothed dispatch block
   duplicated between `user_boot.py` and `simulate_boot.py`.
3. **Latent bug (behavior change, from broken to working):**
   `simulate_boot.smoothed_main_routine` calls
   `fit_resample_utils.compute_xhat`, but no such module is imported or
   exists — the no-`xhat_fname` branch of the smoothed simulation cannot
   run today. Port it calling `boot_utils.compute_xhat` (the evident
   intent) and cover the branch with a test.
4. **Unused/heavy imports.** Delete the unused `matplotlib.pyplot` import
   in `smoothed_boot_sp.py`; scipy imports in the bootsp modules follow
   the guarded/lazy pattern `mmw_ci.py` already uses
   (`from pyomo.common.dependencies import scipy`).
5. **Headers.** Every ported file gets the standard mpi-sppy header
   (`addheader -c addheader.yml`); `test_headers.py` enforces. Both
   projects are BSD-3 with DLW holding the boot-sp copyright, so
   relicensing under the mpi-sppy header is clean. The statdist README
   records the provenance described at the top of this document.
6. **Config.** Options stay on `mpisppy.utils.config.Config` with their
   boot-sp names (`nB`, `sample_size`, `boot_method`, ...); the drivers
   build their own cfg, so there is no namespace collision with
   `confidence_config.py`. The JSON path (`cfg_from_json`) is retained
   for `simulate_boot` and `boot_general_prep`. Any `warnings.warn` added
   or ported must gate on rank 0.
7. **Tests hardening.** boot-sp's tests assert exact solver-specific
   digits and assume `cwd == tests dir`. Ported tests use the
   `test_conf_int_farmer.py` template: `get_solver()`, `skipIf`,
   `round_pos_sig` comparisons, paths relative to `__file__`.

---

## 5. Tests, CI, and docs wiring

- New test files join the **`confidence-intervals` job** in
  `test_pr_and_main.yml` and get matching `run_phase` lines in
  `run_coverage.bash` — in the same commit that adds each test file.
- Tests need a MIP/LP solver (community cplex/xpress in CI); the
  epi-spline fit additionally builds a small Pyomo NLP at runtime. Skip
  guards follow the existing conf-int tests. Anything community-solver
  oversized is skipped the way boot-sp already skips `Extended`.
- `doc/src/boot_sp.rst` ports the boot-sp readthedocs content (methods
  table, module contract, JSON/CLI usage, citations) and enters the
  `index.rst` toctree under "Solutions and Confidence Intervals" after
  `seqsamp.rst`. Each PR ships the doc section for what it merges.
- Examples: `examples/bootsp/farmer` in PR-A; whether to also register a
  small run in `examples/run_all.py` is decided in PR-A (open question
  §7).

---

## 6. Phased PRs

Each phase is a review-sized PR, green on its own.

- **PR-A — empirical core.** `bootsp/` subpackage with `boot_sp.py`,
  `boot_utils.py`, `user_boot.py`, `boot_general_prep.py`;
  `examples/bootsp/farmer`; `test_boot_sp.py` (+ CI/coverage wiring);
  `boot_sp.rst` covering the empirical methods. No statdist, no scipy
  requirement. The `BootMethods` enum ships complete; dispatch of a
  `Smoothed_*` method raises a clear "smoothed methods not yet merged —
  use the boot-sp package meanwhile" error.
- **PR-B — smoothed methods.** Trimmed `statdist/` (per §3) +
  `smoothed_boot_sp.py`; smoothed dispatch replaces the PR-A error;
  `test_boot_sp_smoothed.py` including direct univariate-distribution
  tests (statdist's own test covered one class; extend to the
  distributions the smoothed methods actually request); doc section.
- **PR-C — simulation harness and remaining examples.**
  `simulate_boot.py` (with the §4.3 fix), `examples/bootsp/{cvar,
  multi_knapsack, schultz}`, `test_boot_sp_simulate.py`, doc completion.
- **Post-merge (not a PR).** Update the boot-sp README to point at the
  mpi-sppy docs and archive the GitHub repo. `paper_runs/`, the copula/
  vine/bicop code, and the multivariate classes remain there.

---

## 7. Open questions

1. Do the empirical estimators need scipy at all (e.g. `norm.ppf` for the
   gaussian CI)? Checked during PR-A; if yes, they use the same guarded
   import and PR-A's "no scipy" claim softens to "scipy only for the
   gaussian CI path".
2. Register a bootstrap example in `examples/run_all.py`, or rely on the
   unit tests for CI coverage? (The MMW example has a dedicated
   `do_one_mmw` helper there; a `do_one_boot` would mirror it.)
3. Multi-rank smoke test: boot-sp's own CI never ran under `mpiexec`.
   Batch-parallel bootstrap with `Gatherv` deserves at least one
   `mpiexec -np 2` test — probably in `straight_tests.py` or a
   `test_with_cylinders.py`-style guard. Decide in PR-A.
