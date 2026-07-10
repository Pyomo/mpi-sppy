# Checkpoint / Resume for mpi-sppy — Design

Status: **draft** (framework and dill-reload backend both PoC-validated — the
latter on a serial MIP; multi-rank and cylinders pending). Scope: checkpoint a
running mpi-sppy job so it can be stopped and resumed later. Must work on multiple
MPI ranks and for cylinder (hub-and-spoke) runs.

---

## 1. Goals and non-goals

**Primary use case.** A long (multi-day) run that is **intentionally stopped and
resumed** on a schedule — e.g. a three-day study that ends each day and picks up
the next morning on the same cluster. Checkpoints are **infrequent** (a small
number over the whole run; roughly twice as many writes as resumes) and resumes
are planned, not crash-driven. The scenarios are **large MIPs**.

That regime drives the design decisions below (dill the scenario models, restore
a MIP warm start), and it is worth stating up front because a *different* use
case — frequent checkpoints purely for hard-kill safety — would push toward a
lighter, leaf-data-only checkpoint (still supported; §4, §11 Phase 6).

**Goals**

- Resume a Progressive Hedging (PH) run — serial, multi-rank, or full cylinders
  (hub + spokes) — after a planned stop (and, as a bonus, after a crash).
- **Continue the optimization as if it had not stopped**, warm-started from the
  last iterate, without losing the **best feasible solution found so far (the best
  xhat), not just the best bound.**
- Survive a hard kill (`kill -9`, node failure, walltime) as a secondary benefit,
  via atomic publication (§9).
- Add no measurable overhead when checkpointing is off, and — because checkpoints
  are infrequent here — tolerate a heavier per-checkpoint cost in exchange for a
  complete, warm-startable restore.

**Non-goals (initially)**

- Resuming across a *different* rank count or scenario-to-rank distribution
  (cross-geometry remap). The first cut requires identical geometry and refuses a
  mismatch with a clear error.
- **Bit-identical reproduction for MIPs.** Multi-threaded MIP solves are not
  deterministic and admit multiple optima, so a resumed MIP run is *not*
  bit-reproducible against a hypothetical uninterrupted run. The guarantee is
  correct, warm-started continuation with the incumbent preserved (§7). (A
  deterministic LP/QP solve *can* be bit-identical under the leaf-rebuild backend
  — that is what the PoC showed — but it is not the target here.)
- Bit-identical reproduction of *bounds* (§7 — bounds are async and not
  reproducible; carried forward as best-so-far).
- Robustness across a library/env upgrade between stop and resume. The use case
  resumes in the **same environment** the next day, so a dill-based checkpoint
  (welded to the current Pyomo/mpi-sppy/model code) is acceptable. Cross-version
  resume is out of scope.
- APH. A C++ APH is expected to replace the Python `opt/aph.py`; PH-family only.

---

## 2. What to serialize, and what not to

There are two very different "just dill it" ideas, and they get opposite answers.

### 2.1 Do NOT dill the opt/hub object graph

`dill.dump()` of the whole opt/hub object and reload is not viable — the core
objects are built around **live, non-serializable OS/MPI/solver handles**, not
data:

- **MPI communicators** — `SPBase.mpicomm`, `SPBase.comms` (`spbase.py`), and the
  `fullcomm`/`strata_comm`/`cylinder_comm` on the spcomm
  (`cylinders/spcommunicator.py`). Handles into a running MPI runtime; meaningless
  once the process exits.
- **MPI RMA windows and `MPI.Alloc_mem` buffers** — `SPWindow`
  (`cylinders/spwindow.py`) and the `FieldArray` send/receive buffers. Kernel
  shared-memory regions.
- **Persistent solver handles** — `s._solver_plugin` for gurobi/cplex/xpress
  persistent interfaces (`spopt.py`). A C handle + license session.

A resumed run launches a **fresh process** (new MPI job) anyway, so these must be
**reconstructed via normal startup** regardless of how anything else is restored.
That is not negotiable and not something a checkpoint can carry.

### 2.2 DO dill the scenario models (the recommended backend here)

The narrower idea — dill each **scenario Pyomo model** — is not only viable, it is
the right backend for this use case. The repo already dills *clean* scenario
models for the scenario-pickle path (§4); the only extension is to dill them
**mid-run**. The single non-serializable attribute on a scenario model is
`s._solver_plugin`, which we drop before writing and rebuild with `set_instance`
on resume (a dance we already do — the reconstruct step needs it regardless);
`_mpisppy_data` is a `pyo.Block` on the model with **no** back-references to the
comms or opt object.

Dilling the mid-run model captures, in one shot and mutually consistent,
everything that lives *on* the scenario model:

- the dual weights `W`, `rho`, `xbars` (on `s._mpisppy_model`);
- nonant values **and fixedness** (from variable-fixing/forcing extensions);
- **second-stage (recourse) variable values → a MIP warm start** (§5.2);
- the proximal-approximation `xsqvar` and its accumulated `xsqvar_cuts`, plus the
  `ProxApproxManager` bookkeeping on `_mpisppy_data` (§5.3) — the linearized prox
  is likely in use for large MIPs (a MIQP prox is often intractable), and dill
  brings the cuts back for free instead of replaying them;
- model-attached extension state, e.g. `fixer`'s per-variable `conv_iter_count`
  on `s._mpisppy_data`.

The costs that argued against this backend elsewhere — **version fragility** (dill
serializes by class/closure reference) and **per-checkpoint overhead** (full
models are large) — are both moot for this use case: the resume is same-environment
and the checkpoints are infrequent, so a heavy write paid a handful of times over
three days is negligible. And it *avoids re-running an expensive `scenario_creator`*
on every resume, which for large models is itself a real saving.

The alternative — rebuild each model via `scenario_creator` and overlay the state
as leaf data (arrays/name→value maps) — remains available as the **low-cost
backend**:
its checkpoints are tiny and fast (a handful of `O(first-stage)` arrays, no model
structure) and version-robust (plain numbers, not pickled classes). That makes it
the right choice for small scenarios, cheap creators, or *frequent* kill-safety
checkpoints where a per-write model dump would hurt — the mirror image of this
use case. It is also what the PoC validated (§6, §11 Phase 6). The two backends
share the same framework and manifest; only the scenario-model restore step
differs.

---

## 3. Approach: reconstruct the scaffolding, restore the state

A checkpoint is **not** a snapshot of the object graph. On resume:

1. **Reconstruct the scaffolding** via the normal startup path — comms, RMA
   windows, and persistent solvers. This is exactly what a fresh run already does.
2. **Restore the scenario-model state** via the chosen backend:
   - **dill-reload (recommended here):** load each rank's dilled mid-run scenario
     models; rebuild `_solver_plugin` (`set_instance`) and mark
     `solution_available` so the first solve warm-starts (§5.2). Because the
     reloaded model already carries the spliced W/prox objective and the prox
     cuts, startup must **skip** re-attaching them (§9, item 2).
   - **leaf-rebuild (alternative):** rebuild each model via `scenario_creator`,
     then overlay W / rho / nonant values+fixedness / prox `cut_values` from the
     checkpoint.
3. **Restore the non-model state** — the pieces that do *not* live on a scenario
   model and so are never captured by dilling models: the global iteration
   counter, hub bounds/incumbent objective, the spoke incumbent (best xhat), the
   internal state of extension *objects*, and cursor/RNG. These are always small
   leaf data (§5.4–5.6).

---

## 4. Building blocks already in the repo

- **Scenario-model pickling (dill):** `utils/pickle_bundle.py`
  (`dill_pickle`/`dill_unpickle`) and `generic/scenario_io.py` pickle each
  scenario Pyomo model *alone*, driven by `--pickle-scenarios-dir` /
  `--unpickle-scenarios-dir` (with `iter0_before_pickle` baking an iter-0 solve
  into the pickle). The dill-reload backend (§2.2) is exactly this path
  **generalized from iter-0 to iter-k** — pickle the model as it stands at the
  checkpoint, not only after iter 0.
- **Warm-start plumbing:** `spopt.py` already supports warm-starting subproblem
  solves — the `warmstart_subproblems` option plus `WarmstartStatus.PRIOR_SOLUTION`
  use a warm start when `s._mpisppy_data.solution_available` is set
  (`spopt.py:301-305`). Restoring the model's variable values and setting
  `solution_available=True` feeds the restored MIP solution straight into this
  path — no new solver code.
- **W / xbar persistence:** `utils/w_utils/wxbarwriter.py` (writes in
  `post_everything`) / `wxbarreader.py` (reads in `pre_iter0`) round-trip `W` and
  `xbar` as CSV. A precedent for the leaf-rebuild backend; unnecessary under
  dill-reload (W/xbar ride in the model).
- **Incumbent-to-disk (spokes):** `cylinders/spoke.py`
  `_maybe_write_incumbent_on_improvement`
  (`--incumbent-on-improvement-filename-prefix`) already writes the first-stage
  solution on each improvement — the reference for serializing the incumbent.

---

## 5. State inventory

For each piece: is it **reconstructed** (rebuilt by startup, no save), **carried
in the dilled model** (dill-reload backend), or **restored as non-model leaf
data** (always)? And if restored, is it **carried forward** as a valid
best-so-far value or (LP/QP only) potentially bit-reproducible?

### 5.1 Hub PH primal state — *in the dilled model*

The hub's primal trajectory is pure synchronous PH, independent of the spokes
(lagrangian only contributes an outer bound; xhatshuffle only an incumbent).
Per local scenario, all of the following live **on the scenario model** and are
therefore captured by dilling it:

| State | Where it lives | Note |
|---|---|---|
| `W[ndn_i]` (accumulated duals) | `s._mpisppy_model.W` | `Update_W` accumulates — not recomputable; must be preserved |
| nonant values | nonant vardata `_value` | drive `Compute_Xbar`; `xbar` itself need not be separately saved |
| nonant **fixedness** (+ fixed value) | nonant vardata `.fixed` | `fixer`/`slammer` leave nonants fixed; must survive (§5.5) |
| `rho[ndn_i]` | `s._mpisppy_model.rho` | rho-updaters mutate it |
| `xbars[ndn_i]` | `s._mpisppy_model.xbars` | consensus target |
| smoothing `z/p/beta` | `s._mpisppy_model` | only if `--smoothing` |

Under **leaf-rebuild**, this set is instead gathered/restored explicitly — helpers
already exist: `_populate_W_cache`/`W_from_flat_list` (`phbase.py`),
`_save_nonants`/`_restore_nonants` (`spopt.py`, which already captures fixedness in
`fixedness_cache`).

**Restore point:** after the scaffolding exists and models are loaded, before the
`iterk` loop. In serial this can be a driver call; in cylinders the hub runs
`ph_main` internally, so restore happens in the **`post_iter0_after_sync`**
extension hook (end of `Iter0`, `phbase.py:1079`).

### 5.2 Recourse variable values — *warm start, in the dilled model*

The second-stage (recourse) variable values are the bulk of a large scenario and
are **not** part of the algorithmic primal state (only the nonants and params
are). Their value is as a **MIP warm start**: restoring them and setting
`s._mpisppy_data.solution_available = True` makes the first resumed subproblem
solve start from the last iterate's solution via the existing
`warmstart_subproblems` path (§4). For large MIPs this can save substantial
branch-and-bound time on the first solve after each resume.

Because they ride in the dilled model, they cost nothing extra here. Under the
leaf-rebuild backend they would be an **optional** all-var snapshot (off by
default): a per-checkpoint O(scenario) cost that only pays off for MIP/simplex
warm starts and is pure overhead for barrier solves — a bad trade when
checkpoints are frequent, which is why it is opt-in there.

**Caveat (both backends):** an xhat/incumbent evaluation fixes the first stage and
re-solves, leaving recourse vars in the *eval* state; `_restore_nonants` restores
only nonants. So the model must be checkpointed (dilled) at a point where its
recourse values reflect the true last subproblem solve, not a mid-eval state —
i.e. snapshot before an eval corrupts them, or evaluate on a copy (§9, item 4).

### 5.3 Proximal-approximation cuts (`--linearize-proximal-terms`) — *in the dilled model*

When the linearized prox is on, `attach_PH_to_objective` builds, per scenario
(`phbase.py:892-894`):

- `s._mpisppy_model.xsqvar` — the epigraph var for `x²`;
- `s._mpisppy_model.xsqvar_cuts` — a `Constraint` that accumulates one linear cut
  per visited x-location (`prox_approx.py:232,278,291`);
- `s._mpisppy_data.xsqvar_prox_approx[ndn_i]` — a `ProxApproxManager` whose
  bookkeeping (`cut_index`, the sorted `cut_values` array; `prox_approx.py:46-47`)
  decides when a new cut is redundant.

The cut *constraints* live on the model; the manager's bookkeeping lives on
`_mpisppy_data` (a Block on the model). **Dilling the model captures both, and
keeps them consistent** (the manager's references and the constraint set come back
in lockstep) — no replay, no re-binding.

Under **leaf-rebuild**, neither survives (`attach_PH_to_objective` rebuilds
`xsqvar_cuts` empty). Each cut is fully determined by its x-location (continuous:
`xsqvar ≥ 2v·x − v²`; discrete: integer-keyed), so the checkpoint stores only the
per-nonant `cut_values` arrays and **replays** `add_cut` into the fresh model on
restore. Skipping this leaves resume correct (cuts regenerate lazily via
`check_tol_add_cut`) but coarser initially — fine for MIPs (not bit-reproducible
anyway), relevant only if an LP/QP run wants bit-identity.

### 5.4 Hub bounds + incumbent, and the spoke incumbent — *non-model leaf data, carried forward*

None of this lives on a hub scenario model, so it is restored as leaf data under
**both** backends:

- `spcomm.BestInnerBound`, `spcomm.BestOuterBound`; `opt.best_bound_obj_val`,
  `opt.best_solution_obj_val`. Products of **async** spoke interaction — their
  timing is not reproducible, so they are carried forward as best-so-far. They
  stay valid: a restored looser bound is improved again; a restored incumbent
  objective is never regressed because `update_best_solution_if_improving`
  (`spbase.py:578`) only accepts improvements. In cylinders the hub's
  `best_solution_obj_val` is often `None` — the inner bound arrives as a scalar via
  `receive_innerbounds` (`spcommunicator.py:1010`) into `spcomm.BestInnerBound`.
- **The best xhat SOLUTION values live on the xhat spoke**, in
  `spoke.opt.best_solution_cache` (a `ComponentMap` over all vars) +
  `spoke.best_inner_bound`; `InnerBoundSpoke.finalize()` (`spoke.py:293`) loads
  them back. So **"keep the best xhat" requires checkpointing the spoke
  incumbent**, not just hub bounds. The spoke checkpoints its own cache **on its
  own schedule** — on each improvement, reusing
  `_maybe_write_incumbent_on_improvement`, independent of the hub checkpoint (§9,
  item 6). Serialize the `ComponentMap` **by variable name** (`{var.name: value}`)
  and rebuild by name lookup on the reconstructed model.

### 5.5 Stateful extensions — *split: object state is leaf data, model state rides in the dill*

Several extensions hold trajectory-driving state and **must** be restored or resume
diverges:

- rho updaters (`mult_rho_updater`, `norm_rho_updater`, `grad_rho`), convergers —
  multiplier / gradient / convergence history, kept on the **extension object**.
- variable-fixing/forcing extensions — `fixer.py` and `slammer.py` pin nonants and
  then **skip what they already pinned**, so their tracking *is* the trajectory.
  They span both storage locations: `slammer._slammed` is on the **extension
  object**, while `fixer`'s per-variable `conv_iter_count` is on the **scenario
  model** (`s._mpisppy_data`).

Consequences:

- **Model-attached tracker state (`fixer`) rides in the dilled model** for free —
  consistent with the nonant fixedness it pairs with (§5.1). Under leaf-rebuild it
  must be gathered explicitly.
- **Extension-object state is never on a model**, so it needs a serialization
  contract regardless of backend. The `Extension` base has none today; add
  `checkpoint_state()` / `restore_state()` (no-ops by default; implemented by rho
  updaters, `fixer`, `slammer`, convergers), aggregated by the `Checkpointer`
  (§9, item 3). The same contract serves hub and xhatter extensions
  (`MultiExtension`).
- **The tracker and the actual variable state must agree.** Restoring "already
  fixed X" without X's real `.fixed`/value (§5.1) makes the extension skip X while
  the solver frees it — worse than no tracking. dill-reload gives this for free
  (both come back together); leaf-rebuild must restore fixedness and the tracker as
  one unit.

### 5.6 RNG and spoke cursor — *non-model leaf data, partially restored*

- xhatshuffle seeds its stream to a fixed `42` and samples **once**
  (`xhatshufflelooper_bounder.py:88,94`) — deterministic, no RNG state to save.
- The `ScenarioCycler` cursor and `xh_iter` are **local variables inside `main()`**
  — unreachable. Exact spoke-cursor resume needs them hoisted onto `self` (Phase
  5). Without it the spoke restarts its cursor; this only changes *which* scenario
  it tries next, not the preserved best (restored from §5.4).
- lagrangian / lagranger spokes use **no RNG**; their bound is deterministic given
  the hub's `W`. State to carry: `_PHIter`, `trivial_bound`, last `bound`, received
  `localWs`.

### 5.7 Geometry / cfg fingerprint — *checkpoint metadata*

Each per-rank file records `{n_proc, rank, local scenario list}` and a cfg hash.
Resume verifies the current layout matches and **refuses a mismatch with a clear
error** (validated — §6).

---

## 6. PoC evidence (what is validated, and what is not)

A throwaway PoC (serial + multi-rank + cylinders, farmer LP, gurobi_persistent)
validated the **framework and the leaf-rebuild backend**:

- **Serial:** resume-from-iter-6 reproduced a full 12-iteration run with
  `max|diff| = 0.000e+00` for W, nonants, rho (bit-identical — LP, deterministic
  solver). Persistent solver survives the rebuild (Iter0 re-creates +
  `set_instance`).
- **Multi-rank:** `-np 3` (1 scenario/rank) and uneven `-np 2` (2+1) resume
  bit-identical on every rank; per-rank rank-tagged files, barrier + atomic
  temp-then-rename write. Geometry mismatch fails with a clear error.
- **Cylinders (PH hub + lagrangian + xhatshuffle):** hub primal resumes
  bit-identical inside `WheelSpinner`; the best xhat *solution* (on the spoke) is
  preserved exactly; `BestInnerBound` carried exactly; `BestOuterBound` differed
  run-to-run (async) but stayed valid.

A second PoC then validated the **dill-reload backend on a MIP** (`sizes` SIZES3,
`gurobi_persistent`, single-thread `Threads=1`/`Seed=1`/`MIPGap=0` for a
deterministic solve — the §7 validation crutch):

- **Mid-run model round-trip.** After a few PH iterations, a scenario model was
  stripped of `_solver_plugin`, dilled, and reloaded **both in-process and in a
  fresh process**; a new solver was attached with `set_instance` and the
  subproblem re-solved. The reloaded model reproduced the original solve's
  objective and **every decision variable exactly** — including the hardest case,
  **linearized prox** (176 KB carrying **845 `xsqvar_cuts` + 65
  `ProxApproxManager`s** on `_mpisppy_data`), which came back structurally
  identical and self-consistent. The only difference was the x² epigraph auxiliary
  `xsqvar` wobbling ~1.5e-6 at solver feasibility tolerance (immaterial; MIQP was
  exact). This is the load-bearing assumption — that a mid-run MIP model, cuts and
  all, survives dill — and it **holds**.
- **Stop → reload → continue, bit-identical.** Stopping PH at iteration 3, dilling
  the mid-run models, then rebuilding the scaffolding and continuing through the
  reload branch reproduced an uninterrupted 6-iteration run with
  `max|dW| = max|d nonant| = 0.0` — for **both** quadratic and linearized prox.
  Under the deterministic single-thread solve this is exact bit-identity, the
  strong "nothing was lost" check.

Still to prove in later phases (this PoC was serial and focused on the model
round-trip + continuation): the dill-reload backend under **multi-rank** and
**cylinders**; carrying the **incumbent** across a dill-reload stop; a measured
warm-start speedup; and the disk/time footprint at true model scale.

---

## 7. Determinism contract (what resume guarantees)

- **For the target MIP use case:** resume **continues the optimization correctly
  and warm-started**, and **never loses or regresses the best xhat**. It is *not*
  bit-reproducible — multi-threaded MIP solves are nondeterministic and admit
  multiple optima, so the resumed iterates may differ from a hypothetical
  uninterrupted run. That is expected, not a bug.
- **Bounds and incumbent:** valid and best-so-far, not bit-reproducible (async,
  timing-dependent). Resume never reports a *worse* best-so-far than the
  checkpoint.
- **Leaf-rebuild on a deterministic LP/QP solver:** the primal trajectory (W,
  nonants, rho, xbar) *can* be bit-identical — this is what the PoC showed — but it
  is a bonus, not the target guarantee.

State this in user docs so a differing (but valid) trajectory or bound after
resuming a MIP is not mistaken for a bug.

---

## 8. Configuration and semantics

Checkpointing is **opt-in** and adds nothing when off.

- **Trigger — end-of-run / on-signal (primary).** The use case stops on a
  schedule, so the natural trigger is "checkpoint and exit cleanly at the end of
  the run or on a signal" — after a fixed number of iterations
  (`--max-iterations`), on a wall-clock budget, or on `SIGTERM`/`SIGUSR1`. This
  writes one complete, resumable checkpoint per planned stop.
- **`--checkpoint-every k` (optional insurance).** Also checkpoint every `k` PH
  iterations for unplanned-crash coverage. `k` unset/`0` ⇒ disabled (the
  `Checkpointer` extension is not attached; zero overhead, no files). `k ≥ 1` ⇒
  every `k` iterations **and always at the end of iteration 0** (which establishes
  the resume baseline — solvers created, trivial bound computed, initial `W = 0`
  solve done — and composes with `--iter0-from-pickle`). Given infrequent planned
  stops, most runs will leave this off or large.
- **`--checkpoint-dir <dir>`** — where per-rank files and the manifest are written
  (§10).
- **`--checkpoint-backend {dill-model, leaf}`** — how scenario-model state is
  restored (§2.2). Default `dill-model` (captures the warm start + cuts, dodges an
  expensive `scenario_creator` re-run). `leaf` is the **low-cost** option — tiny,
  fast, version-robust checkpoints — for small/cheap-creator runs or frequent
  kill-safety writes.
- **`--resume-from <dir>`** (or `--resume`, auto-selecting the latest *complete*
  checkpoint from the manifest) — reconstruct the wheel and restore. Resume
  requires identical geometry (§5.7); a mismatch is refused with a clear error.

Each checkpoint is published atomically (§9, item 7; §10), so a kill *during* a
write leaves the previous complete checkpoint intact and referenced — never a
half-written one.

### 8.1 Bundles

mpi-sppy has **only proper bundles** now — loose bundling was removed in 2026
(`spbase.py`; `doc/src/properbundles.rst`). A proper bundle is a **first-class
subproblem**: it appears in `local_scenarios` with its own `nonant_indices`, and
is itself a Pyomo model. So checkpointing **applies uniformly** — dilling
`local_scenarios` dills bundles exactly as it dills plain scenarios, and the
leaf-rebuild path iterates `nonant_indices` identically. Holds whether bundles are
in memory (`--scenarios-per-bundle`) or pickled (`--pickle-bundles-dir` /
`--unpickle-bundles-dir`).

One cleanup: `_restore_nonants` still carries a 2019 comment that it "will not work
on bundles" (`spopt.py`). That predates proper bundles and refers to the removed
loose mechanism; re-verify and refresh it when bundle checkpointing is validated
(Phase 2).

---

## 9. Core changes required

Touch-points an implementation needs beyond the PoC's extension/subclass hacks:

1. **Global iteration counter / resume offset.** `iterk_loop` hardcodes
   `for _PHIter in range(1, max+1)` (`phbase.py:1156`), so a resumed run renumbers
   from 1 and its checkpoints collide with the pre-crash ones. Add a resume offset
   so checkpoint numbering is the global iteration and termination honors the
   original `max_iterations`.
2. **A reload-model resume branch.** When restoring via dill-reload, startup must
   reconstruct comms/windows/solvers but **skip both `attach_Ws_and_prox` and
   `attach_PH_to_objective`** — the reloaded model already carries the W/rho/xbars
   params, the spliced objective, and the prox cuts, so re-running either would
   duplicate components or double the terms. Then strip/rebuild `_solver_plugin`
   (`set_instance`) and set `solution_available` for the warm start (§5.2). Two
   details the PoC surfaced: **refresh `saved_objectives[sname]`** for each
   reloaded model — `Eobjective` reads those objective handles and they otherwise
   dangle to the discarded fresh model — and note the reload targets
   `local_scenarios` only (there is no `local_subproblems`; the solve path
   iterates `local_scenarios`). This is a distinct branch from the leaf-rebuild
   "build fresh, overlay values" path; the `Checkpointer` picks the branch from
   `--checkpoint-backend`.
3. **Extension `checkpoint_state` / `restore_state` contract** on `Extension`
   (no-ops by default; implemented by rho updaters, `fixer`, `slammer`,
   convergers). Covers **extension-object** state under both backends;
   model-attached state (`fixer`'s `conv_iter_count`) rides in the dill under
   dill-reload but must be gathered explicitly under leaf-rebuild (§5.5). The
   `Checkpointer` aggregates the dicts into the per-rank file.
4. **Clean-point model snapshot (xhat/incumbent eval).** Evaluating an xhat fixes
   the first stage and re-solves, corrupting recourse vars (§5.2). The model must
   be dilled (or its values gathered) when recourse values reflect the true last
   solve — snapshot before an eval, or evaluate on a copy.
5. **Geometry / cfg fingerprint** (§5.7) with a clear refusal on mismatch.
6. **Async per-spoke incumbent checkpoints — no hub↔spoke coordination.** Each
   spoke serializes its *own* best incumbent (the best xhat solution values, §5.4)
   and bound whenever its incumbent improves — reusing
   `_maybe_write_incumbent_on_improvement` — to its own rank-tagged file with the
   same atomic write (item 7). Spokes are **not** synchronized to the hub's
   checkpoint iteration: the determinism contract (§7) makes bounds/incumbent
   best-so-far, not bit-reproducible, so a globally-consistent "snapshot at
   iteration `k`" across cylinders is unnecessary. On resume the hub restores its
   primal state while each spoke reloads its latest incumbent/bound, all accepted
   only if improving (`update_best_solution_if_improving`, `spbase.py:578`). This
   also avoids a hub-triggered snapshot barrier and its stall/deadlock risk.
7. **Atomic writes with a single published generation.** Each rank writes only its
   local state (dilled models + leaf non-model data) to rank-tagged temp files and
   renames them into place; the set of per-rank files is then published as one
   checkpoint by atomically rewriting `manifest.json` (itself temp-then-rename) to
   point at the new complete generation (§10). That flip is the single commit
   point, so **one committed generation is enough**: a kill before it keeps the
   previous checkpoint, a kill after it keeps the new one. The prior generation can
   be deleted once the manifest is in place. Keeping a few older generations is
   optional convenience, not a kill-safety requirement.
8. **A `Checkpointer` extension** (write at `enditer` / on trigger, restore at
   `post_iter0_after_sync`). For spokes, the xhatter `main()` loop calls no
   per-iteration extension hook — add a single `self.opt.extobject.enditer()` (or a
   dedicated checkpoint hook) inside it so **one `Checkpointer` serves hub and
   xhatter uniformly** (restore already has a home: `pre_iter0`/`post_iter0` fire
   once in `xhat_prep`, `xhatbase.py:36,49`).

---

## 10. File layout (proposed)

```
<ckpt_dir>/
  manifest.json                       # cfg hash, n_proc, backend, cylinder map, latest complete hub generation
  hub/
    gen_<NNNN>/                        # NNNN = global PH iteration at the checkpoint
      hub_rank_<RRRR>.pkl             # non-model leaf state: iter counter, bounds, extension-object state
      hub_rank_<RRRR>_scen_<S>.dill   # dilled scenario model(s) for this rank (dill-model backend)
  spokes/
    spoke_<name>_rank_<RRRR>.pkl      # each spoke's latest incumbent (best xhat, by name) + bound,
                                      #   overwritten asynchronously on improvement (§9, item 6)
```

The hub writes iteration-tagged generations under `hub/`; each spoke keeps a
single latest-wins file under `spokes/` that it overwrites atomically on
improvement — the two are deliberately *not* aligned (§9, item 6).
`manifest.json` is the single commit point: it names the latest *complete* hub
generation and records the backend so resume loads the right way. Under the `leaf`
backend the `.dill` model files are replaced by numeric arrays inside the
`hub_rank_*.pkl`. Use plain `pickle` for the numeric/leaf state; `dill` for the
scenario models.

---

## 11. Phased rollout

Each phase is a review-sized PR that is green on its own and adds user-visible
value. New tests are wired into `run_coverage.bash` **and**
`test_pr_and_main.yml` in the same commit.

- **Phase 1 — Serial hub checkpoint/resume, dill-model backend.** `Checkpointer`
  extension; global iteration counter / resume offset; reload-model resume branch
  (skip `attach_PH_to_objective`, rebuild `_solver_plugin`, set warm start);
  geometry+cfg fingerprint; atomic per-rank writes + manifest; end-of-run/on-signal
  trigger (+ optional `--checkpoint-every k`); CLI flags `--checkpoint-dir`,
  `--checkpoint-backend`, `--resume-from`/`--resume`. Test: serial **MIP** (e.g.
  `sizes`) stop+resume — run continues correctly, incumbent preserved, warm start
  taken (mid-run model dill round-trip proven, §6).
- **Phase 2 — Multi-rank + bundles.** Barriers, rank-tagged files, single-generation
  atomic publish. Validate with **proper bundles** (§8.1) and refresh the stale
  `_restore_nonants` comment. Test: `mpiexec` MIP stop+resume on every rank, incl.
  uneven distribution and `--scenarios-per-bundle`; mismatch refusal.
- **Phase 3 — Extension-object state contract.** `checkpoint_state`/`restore_state`
  on `Extension`; implement for rho updaters, `fixer`, `slammer`, convergers.
  (Model-attached `fixer` counter and nonant fixedness ride in the dill.) Test: PH
  + norm-rho-updater, PH + `fixer`, PH + `slammer` each resume with state intact
  and consistent with variable fixedness.
- **Phase 4 — Cylinders / spokes.** One-line xhatter write hook; unified
  `Checkpointer` on spoke opts; each spoke checkpoints its own **best xhat** (by
  name) asynchronously on improvement — no hub↔spoke coordination (§9, item 6).
  Test: farmer/`sizes` cylinders (hub+lagrangian+xhatshuffle) stop+resume — run
  continues, best xhat preserved.
- **Phase 5 — Exact spoke continuity (optional).** Hoist `ScenarioCycler`/`xh_iter`
  onto `self`; checkpoint the cursor (+ RNG getstate if a stream becomes stateful).
- **Phase 6 — Leaf-rebuild backend + broader coverage.** The lighter,
  version-robust `--checkpoint-backend leaf` path (rebuild via `scenario_creator`,
  overlay W/rho/nonants/fixedness, replay prox `cut_values`, optional all-var warm
  start); this is what the PoC prototyped. Plus lagranger, FWPH, subgradient
  spokes.

---

## 12. Open questions / risks

Resolved (given the §1 use case):

- **Backend choice.** dill the scenario models (§2.2): overhead is negligible at a
  few checkpoints, version robustness is unneeded (same-environment resume next
  day), and it captures the warm start + prox cuts + model-attached state for free
  while avoiding an expensive `scenario_creator` re-run.
- **Warm start.** Worthwhile for MIPs (branch-and-bound benefits), free via the
  dilled model, fed through the existing `warmstart_subproblems` /
  `solution_available` path.
- **Checkpoint retention** (§9, item 7): a single manifest-published generation
  suffices; older generations are optional history.
- **Spoke snapshot coordination** (§9, item 6): resolved by *not* coordinating.
- **Mid-run MIP model dill round-trip** — was the load-bearing unvalidated
  assumption; **validated by the MIP dill-reload PoC** (§6), including the
  linearized-prox cuts, in-process and cross-process, with serial stop→reload→
  continue bit-identical under a deterministic solver.

Still open:

- **Disk footprint.** Dilled large MIP models × scenarios/rank × a few generations
  can be large; the single-generation policy (§9, item 7) keeps only one live, but
  document the peak (two generations during a publish).
- **variable_probability / surrogate vars.** mpi-sppy masks `W` (not prox) for
  zero-probability nonants and assumes each surrogate var is fixed at 0. Restore
  must **reproduce that invariant**: dill-reload preserves the mask and fixedness
  automatically, but verify; leaf-rebuild must re-apply the mask and re-fix the
  surrogate at 0 (§5.1 fixedness), not just reload raw `W`. Among the spokes only
  `xhatxbar` supports variable_probability today.
- **Cross-geometry resume** is explicitly deferred; revisit if HPC users need to
  resume on a different node count.
```
