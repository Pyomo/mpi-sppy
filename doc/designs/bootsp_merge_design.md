# Merging boot-sp into mpi-sppy — design

**Status:** design captured and decisions ratified 2026-07-02; open
questions resolved and PR structure set to two PRs 2026-07-02; PR-1 example
scope narrowed to the statdist-free example while implementing PR-1
(2026-07-02, see §2.5 and §8).
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
  generic driver; giving bootstrap CIs the same treatment is a follow-on,
  not part of this merge. The intended shape (a possible third PR) is
  richer than mirroring the MMW flags: a *data-splitting* workflow where,
  given a dataset, `generic_cylinders` finds `xhat` from part of the data
  and computes a bootstrap confidence interval from the rest — a hold-out
  split that maps onto boot-sp's `candidate_sample_size` (M) /
  `sample_size` (N) partition of a `max_count` dataset. This does not
  change the two-PR plan for the merge itself.
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
5. **Two PRs** (was three): PR-1 = empirical core + simulation harness +
   the statdist-free example (schultz); PR-2 = statdist + smoothed methods
   + the three examples that need statdist (farmer, cvar, multi_knapsack).
   The seam sits on the one real fault line — the dependency boundary
   between the empirical code (numpy-only) and the smoothed code (statdist
   + scipy + epi-spline Pyomo fits) — and folding the thin
   `simulate_boot.py` (~215 lines) into PR-1 lets the §4.2 dispatch
   consolidation happen once instead of being built in one PR and touched
   again in another.

   **Example placement follows the same fault line (revised 2026-07-02
   during PR-1 implementation).** The original plan put all four examples
   in PR-1, but three of them (farmer, cvar, multi_knapsack) build their
   scenario data with statdist univariate distributions and `Sampler`
   *on the empirical path too* — `farmer` calls
   `distribution_factory('univariate-unif')`, `multi_knapsack` calls
   `'univariate-normal'`, `cvar` samples a fitted distribution — so
   importing them pulls in `statdist.distributions`, whose top of file has
   `from scipy.stats import mvn` (the scipy≥1.14 breakage this merge
   removes). Those examples therefore cannot import, let alone run, in a
   statdist-free/scipy-free PR-1, and PR-1 must be green on its own. Only
   `schultz` generates its data with plain numpy. So PR-1 ships `schultz`
   and its tests; farmer/cvar/multi_knapsack — and their empirical *and*
   smoothed tests — land in PR-2 with statdist, each example landing
   exactly once. See §8 for the full record.

---

## 3. Target layout

```
mpisppy/confidence_intervals/bootsp/
    __init__.py
    boot_sp.py              # estimator engine: classical, extended,
                            #   subsampling, bagging (+ *_resample helpers)
    smoothed_boot_sp.py     # smoothed bootstrap/bagging          (PR-2)
    boot_utils.py           # BootMethods enum, cfg_for_boot, module
                            #   loading, cfg_from_json, compute_xhat
    user_boot.py            # end-user CLI
    boot_general_prep.py    # z*/xhat precompute CLI
    simulate_boot.py        # coverage-experiment harness
    statdist/               # trimmed distribution library         (PR-2)
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
    schultz/                # PR-1 (numpy-only data generation)
    farmer/                 # PR-2 (needs statdist univariate + Sampler)
    cvar/                   # PR-2
    multi_knapsack/         # PR-2

mpisppy/tests/
    test_boot_sp.py                 # PR-1 (empirical, user_boot, prep; schultz)
    test_boot_sp_simulate.py        # PR-1 (coverage harness + MPI; schultz)
    test_boot_sp_smoothed.py        # PR-2 (smoothed + statdist univariate;
                                    #   also the empirical farmer/cvar tests)

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
   intent) and cover the branch with a test. Although `simulate_boot.py`
   is ported in PR-1, this branch is behind the smoothed dispatch, so the
   fix becomes live (and gets its test) in PR-2.
4. **Unused/heavy imports.** Delete the unused `matplotlib.pyplot` import
   in `smoothed_boot_sp.py`. The only scipy use in the empirical modules
   is `scipy.stats.norm.ppf(1-cfg.alpha/2)` in `boot_sp.py` (two call
   sites, the gaussian CI half-width); replace it with the numerically
   identical `statistics.NormalDist().inv_cdf()` from the standard
   library, so the empirical core is scipy-free outright. Any scipy
   imports remaining in the PR-2 modules follow the guarded/lazy pattern
   `mmw_ci.py` already uses
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
- Examples: `examples/bootsp/schultz/` in PR-1 (the other three examples
  need statdist and land in PR-2, §2.5). A `do_one_boot` helper
  (mirroring `do_one_mmw`) registers a small bootstrap run on schultz in
  part 1 of `examples/run_all.py`, also in PR-1.
- MPI testing: batch-parallel bootstrap with `Gatherv` gets real
  `mpiexec -np 2` tests with value assertions (not just smoke) — the
  `test_with_cylinders.py` pattern, wired into CI. schultz is a small
  integer program with fully deterministic discrete data, so its EF
  optimum and bootstrap draws are solver- and rank-independent, which
  makes it a good vehicle for cross-rank value assertions. Empirical
  coverage in PR-1; the smoothed methods get the same treatment in PR-2.

---

## 6. Phased PRs

Two PRs, each review-sized and green on its own, split on the dependency
boundary (see ratified decision §2.5).

- **PR-1 — empirical core and simulation harness.** `bootsp/` subpackage
  with `boot_sp.py`, `boot_utils.py`, `user_boot.py`,
  `boot_general_prep.py`, and `simulate_boot.py`; `examples/bootsp/schultz/`
  (the statdist-free example) plus the `do_one_boot` entry in part 1 of
  `run_all.py`; `test_boot_sp.py` and `test_boot_sp_simulate.py`
  including the `mpiexec -np 2` Gatherv tests (+ CI/coverage wiring);
  `boot_sp.rst` covering the empirical methods. No statdist, no scipy
  (§4.4). The `BootMethods` enum ships complete; dispatch of a
  `Smoothed_*` method — in `user_boot` or `simulate_boot` alike — raises
  a clear "smoothed methods not yet merged — use the boot-sp package
  meanwhile" error.
- **PR-2 — smoothed methods.** Trimmed `statdist/` (per §3) +
  `smoothed_boot_sp.py`; the farmer, cvar, and multi_knapsack examples
  (which need statdist, §2.5); smoothed dispatch replaces the PR-1 error,
  which activates the smoothed simulation path and its §4.3 fix;
  `test_boot_sp_smoothed.py` including direct univariate-distribution
  tests (statdist's own test covered one class; extend to the
  distributions the smoothed methods actually request), the empirical
  farmer/cvar tests that PR-1 could not host, and the smoothed MPI test;
  doc completion.
- **Post-merge (not a PR).** Update the boot-sp README to point at the
  mpi-sppy docs and archive the GitHub repo. `paper_runs/`, the copula/
  vine/bicop code, and the multivariate classes remain there.

---

## 7. Resolved questions (2026-07-02)

1. **Does the empirical core need scipy?** Checked: the only use is
   `scipy.stats.norm.ppf` at two call sites in `boot_sp.py`. Resolved by
   replacing it with the stdlib `statistics.NormalDist().inv_cdf()`
   (§4.4), so PR-1 needs no scipy at all.

2. **`run_all.py` registration?** Yes — a `do_one_boot` helper
   (mirroring `do_one_mmw`) in part 1 of `examples/run_all.py`,
   shipped in PR-1 (§5).

3. **MPI testing?** Yes, and more than smoke: `mpiexec -np 2` tests with
   value assertions on the Gatherv-based batch parallelism, wired into
   CI — empirical in PR-1, smoothed in PR-2 (§5).

---

## 8. Revision during PR-1 implementation (2026-07-02)

While implementing PR-1 it turned out that the example/PR boundary and the
code/PR boundary are the *same* fault line, and the original "all examples
in PR-1" plan crossed it.

- **Discovery.** farmer, cvar, and multi_knapsack build their scenario
  data with statdist univariate distributions and `Sampler` on the
  empirical path — not just the smoothed path. Importing any of them
  imports `bootsp.statdist.distributions`, which begins with
  `from scipy.stats import mvn` (removed in scipy 1.14). So these three
  examples cannot be imported in a statdist-free, scipy-free PR-1, and a
  PR must be green on its own.
- **Resolution.** PR-1 ships only `schultz` (plain-numpy data) and its
  tests. farmer, cvar, and multi_knapsack — and their empirical *and*
  smoothed tests — move to PR-2, where statdist is present, so each
  example is added exactly once. The consequential edits to §2.5, §3,
  §5, and §6 above were made at the same time.
- **Not chosen:** pulling a scipy-lazy univariate subset of statdist
  forward into PR-1 so all four examples could ship there. It would have
  worked but blurred the dependency seam that motivates the two-PR split
  and grown PR-1 with statdist code the "statdist lands in PR-2" decision
  (§2.2) deliberately kept out.
