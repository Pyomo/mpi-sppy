# Out-of-the-box auto-configuration — design

**Status:** Design only; no library code yet. Branch `outOfTheBox`
(off Pyomo/mpi-sppy `main`), on the DLWoodruff fork. Proceeding deliberately
("slowly"): requirements captured and confirmed; the core *decision-logic
mechanism* is still an open question (§5).
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-06-28

---

## 0. Vocabulary

**Out-of-the-box (OOTB)** mode: a single CLI switch (`--out-of-the-box`) that
lets a relatively naive user obtain a *sensible* mpi-sppy run with almost no
knowledge of the library's internals. The user supplies a model module (and
scenario data); mpi-sppy **introspects the environment and the model** and
**auto-assembles a defensible configuration** — algorithm, spokes, bundling,
solver — rather than requiring a hand-crafted hub/spoke command line.

The *spirit* is to lower the barrier to entry: the newcomer effectively says
"here is my model, go," and gets a reasonable decomposition plus a clear
explanation of what was chosen and how to do better.

---

## 1. Goals and non-goals

### Goals

1. A `--out-of-the-box` option (primary home: `generic_cylinders.py`) that
   sets run options automatically.
2. **User options always win** (requirement 0). OOTB only *fills gaps*; any
   option the user explicitly set is retained verbatim and never overridden.
3. **Environment + model introspection** (requirement 2). At minimum: the
   model module, the MPI rank count, and which solvers are actually installed.
   Where the OS / SLURM permits: core count and memory. Plus any options the
   user already supplied.
4. **Floor at 3 ranks** (requirement 3): with fewer than 3 ranks there is no
   useful cylinder configuration (hub + >= 2 spokes), so OOTB solves the **EF**
   instead.
5. **Transparency** (requirement 4): print (a) the **equivalent explicit
   command line** the choices imply, so the user can reproduce / learn from /
   tweak it, and (b) a short, prioritized **"to improve, get..."** list
   (e.g., a persistent solver, more ranks). The run proceeds regardless.
6. **Proper bundling is central** (requirement 5): auto-forming proper bundles
   from scenario count vs. available ranks is a first-class part of the
   decision, not an afterthought.
7. **Quick-start documentation** (requirement 1): OOTB becomes the
   recommended on-ramp in the quick start.

### Non-goals (initial cut)

- Replacing the explicit cylinder command line for expert users. OOTB is an
  *on-ramp*, and it deliberately emits the explicit command line it chose.
- Tuning convergence parameters to optimality. OOTB aims for *defensible*,
  not *optimal*, configurations.
- (TBD, see §6) Deep per-model profiling / trial solves beyond what is needed
  to pick a configuration.

---

## 2. Confirmed scope decisions

These were raised as scoping questions and confirmed by the user (2026-06-28):

1. **Home + scope.** `generic_cylinders.py` is the primary (initial) entry
   point. Reachability via `Amalgamator` is possible later but not the first
   target. *(CONFIRMED as the assumed direction; revisit if Amalgamator
   coverage is wanted sooner.)*
2. **What we read from the module.** OOTB learns scenario count / names (hence
   two-stage vs. multistage structure) from the module. Whether OOTB may
   *instantiate a scenario* to gauge size/difficulty vs. staying purely
   structural is itself an open detail — see §6.
3. **"Dated data files."** Interpreted as a versioned, **date-stamped
   knowledge base** of heuristics/benchmarks ("for problems like X with
   resources like Y, configuration Z worked well") that *guides* the chooser
   and can be refreshed over time — **not** user run-config files.
4. **"What would help" output.** Echo the equivalent command line *and* a
   short prioritized "to improve, get..." list; both printed; the run proceeds
   anyway.

---

## 3. Inputs OOTB consults

| Source | Items | Notes |
|---|---|---|
| User-supplied options | anything already on the command line / Config | Highest precedence; never overridden |
| Model module | scenario count/names, stage structure (2-stage vs multistage) | Structural read; trial instantiation is TBD (§6) |
| MPI environment | number of ranks | Drives EF-vs-cylinders and the 3-rank floor |
| Installed solvers | which solvers import / are licensed; persistent variants | Drives solver choice + a "get a persistent solver" hint |
| OS / SLURM (best effort) | core count, memory | OS-dependent; SLURM env vars when present |

---

## 4. Outputs OOTB produces

1. A fully-populated `Config` (or equivalent) that the normal driver path then
   executes.
2. A printed **equivalent explicit command line**.
3. A printed prioritized **"to improve" advisory** list.
4. The run itself (EF if < 3 ranks, otherwise the chosen cylinder
   configuration).

---

## 5. OPEN DESIGN QUESTION — the decision-logic mechanism

> This is the **first design question** and is intentionally **not yet
> decided.** Candidates raised, in no order:
>
> - **Expert system** (rules engine over facts about environment + model)
> - **Neural net** (learned mapping from features to configuration)
> - **Nested case statements / ifs** (hand-coded decision tree)
>
> Crosscutting all three: **dated data files** (§2.3) that *drive or guide*
> the decision process, so recommendations can evolve and be tuned over time
> without rewriting code.

Evaluation criteria to apply once we take this up (placeholder; to be filled
in during the design discussion): transparency/explainability (OOTB must emit
*why* it chose what it chose), maintainability, the cold-start problem (what
do we do before we have data), how the dated data files are produced and
consumed, and testability.

---

## 6. Open details (deferred)

- May OOTB instantiate a single scenario to estimate size/difficulty, or stay
  purely structural for the first cut?
- Exact bundling heuristic (scenarios per bundle vs. ranks) — depends on §5.
- How the dated data files are generated, versioned, and shipped.
- Amalgamator reachability (§2.1).

---

## 7. Phased rollout (placeholder)

To be drafted once §5 is settled. Per project convention, sizable redesigns
ship as review-sized phases, each green on its own.
