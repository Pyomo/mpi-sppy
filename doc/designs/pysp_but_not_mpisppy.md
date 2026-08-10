# Features in PySP but not in mpi-sppy

Status: reference note. Branch `pysp-but-not-mpisppy` (off Pyomo/mpi-sppy `main`).

## Purpose

mpi-sppy is the successor to PySP (Watson, Woodruff & Hart, "PySP: modeling and
solving stochastic programs in Python," *Math. Prog. Comp.* 4(2):109–149, 2012).
Most of PySP's capabilities were re-implemented — usually in a better form — but
some were not carried over. This note catalogs **features that exist in PySP and
do not (yet) have an equivalent in mpi-sppy**. It deliberately ignores the
reverse direction (the large set of things mpi-sppy does that PySP never did).

The intent is to make parity gaps visible so we can decide, feature by feature,
whether each is worth reproducing, is obsolete, or was intentionally dropped. A
gap here is *not* a claim of regression: mpi-sppy is a deliberate
re-architecture (MPI hub-and-spoke in place of Pyro client/server), so several
"absences" are replacements rather than losses — those are separated out in
section D.

## Method / provenance

The inventory was built from three independent sources and then each "absent"
claim was checked against the current mpi-sppy tree rather than assumed:

- **PySP source** at the last commit before it was removed from Pyomo
  (`f097a97bf`, package root `pyomo/pysp/`), read directly from git history.
- **The 2012 PySP paper** and the archived standalone `Pyomo/pysp` repo
  (README, `setup.py` entry points, `doc/pysp.rst`).
- **The current mpi-sppy source** in this repository.

Confidence is high for section A/B items (verified by grep against mpi-sppy);
section C is minor tooling; section D lists things that look absent but are
actually present in another form.

## A. Whole capabilities that are absent

### A1. Branch-and-Bound Progressive Hedging (`bbph`)
PySP `pyomo/pysp/bbph/brancher.py` wraps an outer branch-and-bound loop around
PH: it branches on non-anticipativity-violating variables using PH's bounds and
hands nodes to a B&B driver. mpi-sppy has no branch-and-bound-over-PH.

### A2. Interfaces to external structured / decomposition solvers
PySP had a pluggable `SPSolver` framework (`pyomo/pysp/solvers/`) that wrote
problem files and shelled out to external solvers:

| PySP driver / writer | External solver | mpi-sppy equivalent |
|---|---|---|
| `runddsip`, `convert.ddsip` | DDSIP (dual decomposition) | none |
| `runsd`, `solvers/sd.py` | Sen's Stochastic Decomposition (SD) | none |
| `runschuripopt`, `convert.schuripopt` | SchurIpopt / PIPS-style interior point (writes per-subproblem `.nl`) | in-process Schur complement only (`opt/sc.py`, via parapint, continuous-only) — no external `.nl` export |

mpi-sppy solves everything in-process; it has no bridges to DDSIP, SD, or an
external PIPS-style solver.

### A3. SMPS *export*
PySP writes full SMPS (`convert.smps` → `.cor` / `.tim` / `.sto`, core in MPS or
LP), driven by stochastic-data annotations. mpi-sppy only **reads** SMPS
(`--smps-dir`), and that reader is two-stage + `SCENARIOS DISCRETE` only. There
is no SMPS writer.

### A4. EmbeddedSP and the parametric distribution library
PySP `embeddedsp.py` plus `TableDistribution`, `UniformDistribution`,
`NormalDistribution`, `LogNormalDistribution`, `GammaDistribution`,
`BetaDistribution` (each with `sample()` / `expectation()`), together with the
stochastic-data annotations (`StochasticConstraintBounds/Body/Objective`,
`StochasticVariableBounds`) that feed the SMPS/DDSIP writers. mpi-sppy has none
of this — all sampling is the user's job inside `scenario_creator`.

### A5. Lagrangian chance-constraint / PR-curve tooling
`lagrangeParam.py` (`LagrangeParametric` sweeps λ to trace the optimal
cost-vs-probability efficient frontier → `PRoptimal.csv`), `lagrangeMorePR.py`,
`drive_lagrangian_cc.py`, `lagrangeutils.py`. mpi-sppy supports SAA chance
constraints on the EF (`--cc-indicator-var`, `--cc-alpha`) but **not** the
parametric Lagrangian efficient-frontier machinery.

### A6. Convex-hull / dual cutting-plane bound
`convexhullboundextension` + `DualPHModel` build a cutting-plane convex-hull dual
master that overrides PH's W updates to tighten the dual bound. mpi-sppy's FWPH
spoke is the spiritual successor (inner linearization) but the convex-hull
dual-bound extension itself is not reproduced.

### A7. Value of the Stochastic Solution (VSS) evaluation
`ef_vss.py` (`create_expected_value_instance` + `fix_ef_first_stage_variables`)
computes the VSS / evaluates the expected-value solution across scenarios.
mpi-sppy has `average_scenario_creator` (an EV building block) but no dedicated
VSS/EEV computation utility.
**Addressed (two-stage):** `generic_cylinders --vss` now reports RP/EV/EEV/VSS
after a run (`mpisppy/generic/vss.py`; see `doc/src/vss.rst` and
`doc/designs/vss_design.md`). Multistage VSS is still a gap.

## B. PH algorithmic knobs that are missing

### B1. PH over-relaxation
PySP `runph --overrelax` / `--nu`. mpi-sppy PH has no over-relaxation step (the
`nu` / `gamma` parameters in the code are APH's projective-step parameters,
unrelated).

### B2. "voting" xhat recovery
PySP `--xhat-method {closest-scenario, voting, rounding}`. mpi-sppy has
closest (`XhatClosest`) and rounding-style repair (`rounding_bias`), but no
**voting** scheme.

### B3. Free-discrete-count convergence + restricted "PH-then-EF" solve
PySP `--enable-free-discrete-count-convergence` (stop once the number of
un-fixed discretes drops below a threshold) followed by fixing the converged
discretes and solving the small residual EF (`--write-ef` / `--solve-ef` from
`runph`). mpi-sppy has `IntegerRelaxThenEnforce` (a different idea) but not this
terminate-then-solve-restricted-EF path.

### B4. Rich Watson-Woodruff per-variable annotations
mpi-sppy reimplements the WW *mechanics* — `Fixer`, `Slammer` (with slam
priorities and directions), W-hashing cycle detection, MIP-gap scheduling
(`Gapper`). What it does **not** reproduce is the economic / decision annotation
layer from the WW annotation file: `going_price`, `obj_effect_family_name` /
`obj_effect_family_factor`, `decision_hierarchy_level`, `feasibility_direction`,
`relax_int` / `reasonable_int` / `low_int`. mpi-sppy's slam/fix directives are
simpler and carry no domain-economic semantics.

## C. Minor / tooling gaps

- **Scenario-tree Graphviz export** — PySP `ScenarioTree.save_to_dot`. No native
  tree visualization in mpi-sppy.
- **Native random-bundle creation + tree downsampling** — PySP
  `--create-random-bundles`, `--scenario-tree-downsample-fraction`. These exist
  in mpi-sppy but only inside the PySP-compatibility reader
  (`utils/pysp_model/instance_factory.generate_scenario_tree`), not on the
  native `scenario_creator` path.
- **Full PH iteration-history dump** — PySP `phhistoryextension` serializes every
  iteration's weights and solutions to `ph_history.json` / shelve. mpi-sppy's
  `phtracker` / `wtracker` cover part of this but not the same complete history.

## D. Present in PySP, absent by design (not real gaps)

These look like PySP features mpi-sppy lacks, but each is replaced or already
covered:

- **Pyro-based distributed solve** (`phsolverserver`, `scenariotreeserver`,
  `pyro_mip_server`, name-server tooling, `--solver-manager=phpyro`).
  Intentionally replaced by the MPI hub-and-spoke architecture.
- **Declarative `ScenarioStructure.dat` modeling** (abstract `ReferenceModel.py`
  + per-scenario / per-node `.dat`). Supported in mpi-sppy through the
  compatibility layer `utils/pysp_model/` (`PySPModel`), which converts a PySP
  tree into `ScenarioNode`s. It is a compat shim rather than a first-class
  native path, but the capability is present.
- **Eckstein-Combettes asynchronous projective hedging.** This *is* mpi-sppy's
  APH (`opt/aph.py` implements the same projective algorithm).
- **ADMM / consensus.** PySP `runadmm`; mpi-sppy has `admmWrapper` /
  `stoch_admmWrapper`.
- **MMW confidence intervals and `evaluate_xhat`.** PySP `computeconf` /
  `evaluate_xhat`; mpi-sppy has `confidence_intervals/mmw_ci.py` and
  `utils/xhat_eval.Xhat_Eval`.

## Priorities (suggested)

If we choose to close gaps, the highest-value candidates are the
interoperability and analysis features that have no substitute today:

1. **A2 / A3** — external-solver interoperability (DDSIP, SD) and SMPS export.
2. **A5** — the Lagrangian chance-constraint PR-curve tools.
3. **A7** — VSS computation.
4. **A1** — branch-and-bound PH.

The section B knobs are small, self-contained additions to PH if wanted.
