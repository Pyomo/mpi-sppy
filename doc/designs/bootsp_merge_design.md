# Bootstrap/bagging for data-based stochastic programming in mpi-sppy — design

**Status:** design captured and decisions ratified 2026-07-02; PR-1
implemented (empirical core + schultz, incl. a data-file example) and pushed
to a fork branch; extended 2026-07-03 to state the end goal
(`generic_cylinders` integration) and a stacked, multi-PR roadmap (§6, §9).
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-07-03

**Ultimate goal.** The end state this design builds toward is *bootstrap and
bagging confidence intervals, computed from a given dataset, available
directly in `generic_cylinders`*: point the driver at a data file and it finds
a candidate solution (`xhat`) from part of the data and a confidence interval
on the optimality gap from the rest — the data-based analog of the driver's
existing MMW option. Merging boot-sp into mpi-sppy (the bulk of this document)
is the enabling step; the `generic_cylinders` integration is the payoff, is
designed in §9, and lands in later PRs of a stacked, multi-PR effort (§6).

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

1. **The end goal — data-based bootstrap/bagging in `generic_cylinders`.**
   Given a dataset (a data file for the first examples), `generic_cylinders`
   finds `xhat` from part of the data and a bootstrap/bagging confidence
   interval on the optimality gap from the rest, using the driver's normal
   solve machinery. This is the data-based analog of the driver's MMW option;
   everything else in this document enables it. Designed in §9.
2. All boot-sp functionality that users need — the end-user CI tool
   (`user_boot`), the z*/xhat prep tool (`boot_general_prep`), the
   coverage-experiment harness (`simulate_boot`), and all eleven
   `BootMethods` (empirical and smoothed) — available from mpi-sppy proper,
   with tests, docs, and examples.
3. **No new hard dependencies.** mpi-sppy's core deps stay
   numpy/pyomo/sortedcollections; scipy and matplotlib remain optional
   extras. Anything that needs them imports lazily and fails with a
   friendly message.
4. Importable on current and future scipy (the `scipy.stats.mvn` removal
   in scipy 1.14 must not break mpi-sppy).
5. mpi-sppy conventions throughout: standard file headers, `Config`-based
   options, `mpisppy.tests.utils.get_solver` in tests, tests wired into
   both `run_coverage.bash` and `test_pr_and_main.yml` in the same commit
   as the test files.
6. One canonical copy: after the merge, the boot-sp GitHub repo is
   archived with a pointer to mpi-sppy.

### Non-goals

- **Copulas and vines.** Only the univariate statdist distributions come
  over (see §3). The multivariate/copula/vine code stays in the archived
  boot-sp repo, recoverable if a future project needs it. The `gosm`
  import hook goes with it.
- **paper_runs.** The scripts reproducing the papers' experiments stay in
  the archived repo, where the citations point.
- **Non-file data sources.** The `generic_cylinders` integration (§9)
  targets a dataset in a *data file* for its first examples. Pulling data
  from a database, a live sampler, or a model's own generator is a later
  concern, not part of the initial integration.
- **Multistage.** boot-sp is two-stage by construction; that does not
  change here.

(The `generic_cylinders` integration was a non-goal in the first draft of
this document; it is now the stated end goal — see the goals above and §9.)

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
5. **The *move* is two PRs** (was three): PR-1 = empirical core + simulation
   harness + the statdist-free example (schultz); PR-2 = statdist + smoothed
   methods + the three examples that need statdist (farmer, cvar,
   multi_knapsack). This decision is about Stage 1 (the merge); the overall
   effort then adds the Stage-2 `generic_cylinders` integration PR(s) (§6,
   §9). The seam for the move sits on the one real fault line — the
   dependency boundary between the empirical code (numpy-only) and the
   smoothed code (statdist + scipy + epi-spline Pyomo fits) — and folding the
   thin `simulate_boot.py` (~215 lines) into PR-1 lets the §4.2 dispatch
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

## 6. Phased, stacked PRs

This is a multi-PR effort in two stages, and the PRs are **stacked**: each
branches off its predecessor rather than off `main`, so reviewers see a
review-sized, self-contained diff and each PR is green on its own. As each
lands, the next is rebased down onto the new `main`. (Stacking is used
deliberately here for review/testing granularity; it is the exception to the
usual "branch off main" default.)

- **Stage 1 — move boot-sp into mpi-sppy** (PR-1, PR-2). Split on the
  dependency boundary (ratified decision §2.5).
- **Stage 2 — integrate into `generic_cylinders`** (PR-3, and possibly a
  PR-4). The payoff; designed in §9. Depends on Stage 1 being in.

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
- **PR-3 — `generic_cylinders` integration (the end goal).** A `--boot-*`
  option group in `generic_cylinders` that takes a dataset (a data file for
  the first examples), finds `xhat` from part of it and a bootstrap/bagging
  CI on the gap from the rest, reusing the driver's solve machinery. Includes
  the positional name/sample layer (§9), mutual-exclusion guards against the
  sampling-based CI options (§9), at least one worked data-file example
  (building on `examples/bootsp/schultz_data`), tests (serial + `mpiexec`),
  and docs. If it grows past review size, the additional examples/features
  split into a PR-4. Full design in §9.
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

---

## 9. Integrating bootstrap/bagging into generic_cylinders (the end goal)

This is the payoff described in the goals and lands in Stage 2 (§6). Stage 1
delivers the estimators as a library and via `user_boot`/`simulate_boot`;
Stage 2 makes them a first-class capability of the everyday driver.

### 9.1 What it adds

A `--boot-*` option group on `generic_cylinders` (a sibling of the existing
`--mmw-*` group) that, given a **dataset**, reports a bootstrap/bagging
confidence interval on the optimality gap. The first examples take the
dataset from a **data file** (as in `examples/bootsp/schultz_data`). The
workflow is a hold-out split: a candidate solution `xhat` is found from part
of the data (the `candidate_sample_size` M records) and the confidence
interval is estimated by resampling the rest (the `sample_size` N records),
for the chosen `boot_method`.

**Decided: the split is strictly disjoint in `generic_cylinders`.** The M
records used to find `xhat` and the N records resampled for the CI do not
overlap. This is a correctness requirement, not a convenience: a bootstrap CI
on the optimality gap of `xhat` is only meaningful when it is estimated on
data that did *not* choose `xhat` — reusing the candidate records would make
the gap estimate in-sample and optimistic. This is stricter than boot-sp's
standalone classical bootstrap, which draws its pool from the whole dataset
and can overlap the candidate records. The positional layer (§9.2) makes the
disjointness natural: the driver reserves one block of positions for the
candidate and a separate block for the resampling pool.

The intended synthesis is that `xhat` is produced with the driver's *own*
solve machinery — EF for small instances, or PH and the cylinder hub/spoke
system for large ones — rather than the direct sub-EF solve boot-sp uses in
isolation. The per-batch resample solves remain small (sub-EF-sized) and stay
within the estimator code.

### 9.2 Scenario names vs. positions (the key reconciliation)

mpi-sppy's convention is that `scenario_creator` is **name-based**: it is
handed a scenario *name* and builds that scenario, and `scenario_names_creator`
produces the canonical, ordered list of names. The bootstrap/bagging logic,
however, is inherently **positional**: it partitions the dataset (the M/N
split) and *resamples* it by index into the list of all scenarios. The
integration reconciles the two as follows.

1. The driver treats the dataset as the ordered list of scenario names from
   `scenario_names_creator` (one name per data record). Record *k* is
   *position k* in that list; there is no need to parse an integer out of a
   name.
2. Bootstrap/bagging operate on **positions** `0..N-1`: the hold-out split is
   a positional partition, and a resample is a multiset of positions (drawn
   with replacement for bootstrap, as size-limited subsets for bagging).
3. Because an mpi-sppy extensive form needs **distinct** scenario names but a
   resample can select the same record more than once, the driver builds a
   fresh, unique *sample* name for each draw and keeps a mapping from each
   sample name back to the position (hence to the underlying record's
   canonical name). `scenario_creator` is always called with a name that
   resolves to exactly one record. This generalizes boot-sp's existing
   `Scenario{i}` / `SampleScenario{i}`→mapping trick (§4 / `boot_sp.py`), but
   keyed on **list position** instead of an integer scraped from the name.

The practical consequence: the positional layer (the canonical name list and
the sample-name→position mapping) lives in the driver/estimator, so a model
only has to follow the standard mpi-sppy naming (`scen0`, `scen1`, …) and map
its own name to its own data — it does not have to embed dataset addressing in
`extract_num`. The current ported `boot_sp.py` still addresses records via
`extract_num`; PR-3 introduces the positional layer so the integrated path
does not depend on that convention.

### 9.3 Features that do not apply

Not every `generic_cylinders` capability is meaningful for a data-based
bootstrap run, and the driver should refuse incompatible combinations with a
clear message rather than silently ignoring them:

- **MMW (`--mmw-*`) and sequential sampling.** These are *distribution-
  sampling* confidence-interval methods; they assume a scenario generator that
  can draw fresh samples, not a fixed dataset. They are mutually exclusive
  with the bootstrap CI — you would not run MMW and bootstrap on the same run.
- **Anything that assumes unbounded scenario generation.** The dataset is
  finite; options that presume the model can mint arbitrarily many new
  scenarios do not apply.
- **Multistage.** Two-stage only, as everywhere else in this document.

### 9.4 Open questions for PR-3

- **Dataset contract.** Whether to standardize a `--data-file` convention plus
  a light "data model" contract (a model advertises its dataset size and how a
  position maps to data), versus per-model loading like `schultz_data`.
- **Solve path for xhat and for batches.** Confirm whether `xhat` uses the
  full cylinder machinery while the per-batch resample solves stay as direct
  EF solves, and how the bootstrap `Gatherv` batching coexists with cylinder
  rank layouts.
- **Flag names.** The concrete `--boot-*` option spelling (kept close to the
  boot-sp option names in §4.6 for continuity).
