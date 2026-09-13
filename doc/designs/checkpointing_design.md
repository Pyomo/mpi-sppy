# Checkpoint / Resume for mpi-sppy — Design

Status: **phase 1a implemented** (serial PH hub; see §11). Phases 2 onward are
still design. Where this document and the shipped code have disagreed, the code
is authoritative and this document has been corrected — §8 in particular records
a design that was tried, failed, and was replaced. Scope: checkpoint a
running mpi-sppy job so it can be stopped and resumed later. Must work on multiple
MPI ranks and for cylinder (hub-and-spoke) runs.

**Companion notes** (`notes/`):

- [`checkpointing_decisions.md`](notes/checkpointing_decisions.md) — what was
  tried and rejected, with the measurements. Read this before proposing a
  simplification; several of the obvious ones were built and removed.
- [`pyomo_configvalue_pickle.md`](notes/pyomo_configvalue_pickle.md) — the
  upstream Pyomo bug that makes some scenario models unserializable, and why
  it is not a dill quirk.

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
lighter, leaf-data-only checkpoint — the leaf-rebuild backend, kept in this design
as a future option but **not currently planned** (§4, §11 Phase 6).

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
- **Catching external OS signals** (SIGTERM/SIGUSR1 from a scheduler, Ctrl-C) to
  trigger a checkpoint. Checkpoints are written at iteration boundaries (§8)
  termination — including hitting `--time-limit` — which covers the planned-stop
  use case. A scheduler that hard-kills mid-solve is covered only insofar as the
  previous checkpoint is preserved by atomic publication (§9), not by catching the
  signal and checkpointing in response. Where a known walltime is the hazard,
  `--checkpoint-before-seconds` (§8) is the answer instead of a signal handler:
  it writes at the last iteration boundary preceding a user-supplied deadline.

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
**mid-run**. The non-serializable attribute mpi-sppy itself puts on a scenario
model is `s._solver_plugin`, which we drop before writing and rebuild with
`set_instance` on resume (a dance we already do — the reconstruct step needs it
regardless); `_mpisppy_data` is a `pyo.Block` on the model with **no**
back-references to the comms or opt object.

**But the model can also be made undillable by the user's own modeling code,**
and that is not something checkpointing can strip. The known case: a Pyomo rule
written as a nested function that closes over the `Config` object (`cfg`). The
closure drags the `Config` into the model's serialization graph, and Pyomo's
`ConfigDict` whose entries still hold unresolved defaults fails the *first*
time it is serialized — under dill and the standard `pickle` module alike. The
cause is upstream: Pyomo's `UninitializedMixin` sets `self.__class__` while
`__getstate__` is reading `_data`, so the pickler has already captured the old
class when the consistency check runs. Each failed attempt resolves one entry,
so a real model can need many attempts; it is intra-process only, so a fresh
run fails deterministically on the first.
`examples/stoch_distr/stoch_distr.py` has exactly this shape, so its scenario
models cannot be dilled at all, wrapper or no wrapper (issue #828). This is not
specific to checkpointing: the existing `--pickle-scenarios-dir` path would fail
the same way on such a model. Two consequences for this design:

- The dill-reload backend cannot promise to checkpoint an *arbitrary* model; it
  requires a dill-serializable one. The implementation therefore **probes one
  local scenario at setup** and fails immediately with an actionable message
  rather than discovering the problem at the first write, hours in.
- The workaround lives in the model (hoist the value out of the closure before
  defining the rule), so the user-facing docs must say so.

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
on every resume, which for large models is itself a real saving. (Exception:
the ADMM paths, where the wrapper re-runs the creator at startup regardless —
§8.2, item 2.)

The alternative — rebuild each model via `scenario_creator` and overlay the state
as leaf data (arrays/name→value maps) — is the **low-cost backend**, designed here
but **not currently planned** (§11 Phase 6):
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
     models and swap them in ahead of solver creation, so `_solver_plugin` is
     attached (`set_instance`) to the reloaded model rather than to one about to
     be discarded; mark `solution_available` so the first solve warm-starts
     when `warmstart_subproblems` is on -- it is off by default, and the flag
     is inert without it (§5.2). Because the reloaded model already carries the spliced W/prox
     objective and the prox cuts, the deferred objective attach must be disarmed
     so it does not run again downstream (§9, item 2).
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
  (`warmstart_subproblems` in `spopt.py`). Restoring the model's variable values and setting
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

**Restore point: a resume branch inside `Iter0`, in place of the iter-0 solve.**
`Iter0` (`phbase.py`) already contains the structural precedent: the
`iter0_from_pickle` option replaces the iter-0 `solve_loop` with
`_iter0_use_pickled_solution()`. Resume is a third branch in the same method,
*instead of* the solve loop, that loads the checkpoint and proceeds to
`iterk_loop`. This matters for the target use case: restoring in a *post*-iter0
extension hook (what the PoC did, to avoid core changes) would first solve every
fresh model with `W = 0` and then throw those solutions away — one discarded
full MIP solve per scenario per resume, plausibly hours. The design therefore
specifies the core branch, not the hook (§9, item 2): the PoC's
`post_iter0_after_sync` restore was a zero-core-change validation crutch and is
**not** the design.

**The model swap goes *before* `_create_solvers()`, not after.**
`_create_solvers` (`spopt.py`) walks `local_scenarios` attaching a
`_solver_plugin` and, for a persistent solver, calling `set_instance_retry` —
which builds the whole model inside the solver. If the reload happened after it,
every scenario would pay `set_instance` **twice** per resume: once on the fresh
model that is about to be discarded, once on the reloaded one. At this design's
target scale that is precisely the cost §2.2 sets out to avoid. Reloading first
lets the existing `_create_solvers` call attach the solver to the right model,
once, with no strip-and-rebuild dance. Nothing between the top of `Iter0` and
`_create_solvers` depends on the fresh models' identity.

On the resume branch, the rest of `Iter0` adjusts as follows: the feasibility
check reads the restored per-scenario feasibility flags; `trivial_bound` /
`best_bound_obj_val` are restored from the checkpoint's leaf data rather than
recomputed via `Ebound`; the spoke sync still runs (publishing the *restored*
W/xbar/nonants to the spokes — they start from checkpointed state
immediately); the `rho_setter` is skipped, as are the `post_iter0` rho
recomputations of the rho-setting extensions (rho rides in the reloaded model,
and recomputing it at the splice would clobber whatever adaptation had happened
by the checkpoint); the converger is constructed as usual, and since no
converger state rides in a checkpoint, the resume warns that a
history-accumulating converger restarts empty (the `checkpoint_state` /
`restore_state` contract of §9, item 3, has not shipped). `pre_iter0` fires
*after* the splice, so extension hooks act on the models the run will actually
iterate rather than on fresh models the splice discards.

**The deferred objective attach must be disarmed, not "skipped".** Iteration-0
deferral (`_deferred_ph_attach`) splices the W/prox terms into the objective at
the *end* of `Iter0` — after the resume branch has already run — and re-runs
`set_instance` on persistent solvers. On a reloaded model that would duplicate
the prox components and double the W terms, so the resume branch clears the flag
rather than relying on control flow to miss it. Note that the *other* attach,
`attach_Ws_and_prox`, runs earlier still, in `PH_Prep`, and therefore cannot be
skipped from a branch inside `Iter0` at all; it is harmless (it decorates fresh
models that the swap discards), which is why disarming the deferred attach is
the requirement and the earlier call is not (§9, item 2).

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
(in `attach_PH_to_objective`, `phbase.py`):

- `s._mpisppy_model.xsqvar` — the epigraph var for `x²`;
- `s._mpisppy_model.xsqvar_cuts` — a `Constraint` that accumulates one linear cut
  per visited x-location (`add_cut` in `prox_approx.py`);
- `s._mpisppy_data.xsqvar_prox_approx[ndn_i]` — a `ProxApproxManager` whose
  bookkeeping (`cut_index`, the sorted `cut_values` array; `ProxApproxManager` in `prox_approx.py`)
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
  (`spbase.py`) only accepts improvements. In cylinders the hub's
  `best_solution_obj_val` is often `None` — the inner bound arrives as a scalar via
  `receive_innerbounds` (`spcommunicator.py`) into `spcomm.BestInnerBound`.
- **The best xhat SOLUTION values live on the xhat spoke**, in
  `spoke.opt.best_solution_cache` (a `ComponentMap` over all vars) +
  `spoke.best_inner_bound`; `InnerBoundSpoke.finalize()` (`spoke.py`) loads
  them back. So **"keep the best xhat" requires checkpointing the spoke
  incumbent**, not just hub bounds. The spoke checkpoints its own cache **on its
  own schedule** — on each improvement, reusing
  `_maybe_write_incumbent_on_improvement`, independent of the hub checkpoint (§9,
  item 6). Serialize the `ComponentMap` **by variable name** (`{var.name: value}`)
  and rebuild by name lookup on the reconstructed model.
- **The initially-fixed-nonant baseline** (`opt._initial_fixed_varibles`), which
  gates whether the outer bound may be updated at all. It lives on the opt object
  and is keyed by variable *identity*, so it neither rides in the dill nor
  survives the model swap; it is checkpointed **by variable name** and rebuilt
  against the reloaded models. A plain PH hub happens to be insulated from
  getting this wrong, but `Subgradient` and `FWPH` are not — see §9, item 11.

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
  (`main()` in `xhatshufflelooper_bounder.py`) — deterministic, no RNG state to save.
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
warm-start speedup; the disk/time footprint at true model scale; and the
mid-run dill round-trip of a **stoch-ADMM wrapper-mutated model** (§8.2, item
4), which is structurally stranger than anything this PoC dilled. Note also
that both PoCs restored in the `post_iter0_after_sync` hook; the design
replaces that with the in-core resume branch (§5.1), which is itself unproven.

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

Checkpointing is **opt-in** and adds nothing when off. It is enabled by
`--checkpoint-dir`; with a directory set, the triggers below decide *when* a
checkpoint is written. They **compose** — whichever fires writes a checkpoint, all
sharing the same atomic publish (§9). With no `--checkpoint-dir` the `Checkpointer`
extension is not attached at all — zero overhead, no files.

**When a checkpoint is written**

**At the end of every completed PH iteration, and only there.** This replaced an
earlier design in which a single checkpoint was written at termination, from
`post_everything`. That approach does not work, and the reason is worth
recording because it is not obvious and it cost three review rounds to settle.

`iterk_loop` computes xbar, updates `W`, runs the `miditer` extension hook, and
*may break* — user converger, convergence threshold, `--time-limit` — and only
then solves. A run ending through one of those breaks leaves the models
describing half an iteration: `W` at iteration *k*, nonants still at *k−1*'s
solve. `--time-limit`, the planned-stop trigger this design exists for, exits
that way every time. Resuming from such a state applies the dual update to the
same iterate twice and skips a solve outright (measured: 37.8 divergence on
farmer against an uninterrupted run).

Reconstructing a coherent iterate from that state is unbounded work.
`miditer` gives every extension a chance to change `rho`, fix nonants, relax
domains or add cuts, so any list of things to rewind is a list of the extensions
someone has thought about so far — an implementation attempt unwound `W`, then
needed `rho`, then nonant fixedness, with domains and cuts next.

Writing at `enditer`, after the solve, sidesteps all of it: the checkpoint
always describes a *completed* iteration, whatever extensions are loaded and
whatever they touched. The invariant is one sentence and holds by construction.

Consequences, all deliberate:

- **The cost is one model serialization per iteration** rather than one per
  run, paid only when `--checkpoint-dir` is given (with no checkpoint directory
  the `Checkpointer` is never constructed and none of its hooks exist).
  Measured: 38–48% on MIP instances, and 262% on a 50-scenario farmer where the
  solves are trivially cheap. For the large-MIP target this is negligible; for
  many cheap scenarios it dominates. Retention is a single generation, so disk
  does not grow with the iteration count.
- **A run that ends before completing iteration 1 publishes nothing.** No
  iteration completed, so there is no iterate to resume from.
- **Iteration 0 is not a checkpoint point.** `Iter0` splices the `W`/prox terms
  into the objective *after* the last extension hook available
  (`post_iter0_after_sync`), so a checkpoint taken during it captures a model
  whose objective is not the one PH goes on to iterate — and the resume branch
  disarms the deferred attach, so a resume from it would have no prox term at
  all (measured: 330 divergence). Preserving iteration 0's work would need a new
  core hook at the true end of `Iter0`; that was considered and declined.
- **A mid-run write failure warns and continues; it does not kill the run.**
  Conditions detectable at setup (unwritable directory, undillable model,
  unknown backend, colliding sanitized scenario names) fail loudly before any
  solving. A *transient* failure at an iteration boundary — disk full, an NFS
  hiccup — is different: the previously published generation is untouched and
  remains resumable, while the optimization progress a raise would destroy
  lives only in memory. So `Checkpointer.enditer` catches the write error,
  reports it loudly, and retries at the next iteration boundary.
- The **incidental benefit**: because every write now precedes any
  `post_everything`, an xhat evaluation can no longer contaminate a checkpoint,
  which closes §9 item 4 without separate machinery.

**Triggers.** The periodic and anticipated triggers below were designed against
the terminal-checkpoint model. Writing at completed iterations subsumes most of
what they were for — a checkpoint from a recent completed iteration always
exists, so `--checkpoint-every-seconds` and the anticipatory
`--checkpoint-before-seconds` have no gap left to fill, and neither is
implemented. What remained genuinely useful was the opposite of insurance: a
way to write **less** often, to buy back the per-iteration cost on models with
many cheap scenarios.

`--checkpoint-every-iterations K` **is implemented** and is that control; its
meaning inverted along the way — it is a cost control, not a safety net. Writes
happen at every K-th completed iteration by *absolute* iteration number, so the
cadence is unaffected by a resume (which continues the global counter). An
unplanned stop therefore loses up to K−1 completed iterations, which is the
whole trade. It moves *which* boundaries are checkpoint points and never moves
a write off a boundary, so the coherence argument above is untouched.

The one iteration written regardless of K is the last one of an exhausted
iteration budget, which `Checkpointer._is_final_iteration` detects from the
loop bound `iterk_loop` computed (`_stop_iteration`, the earlier of this run's
`PHIterLimit` and the study's `stop_at_iteration_number`). Resuming for more
iterations is an explicitly supported workflow (both bounds are
non-structural, so either may change at a resume), and that final iterate is
coherent and already in memory; discarding it to save a single write would be
a real loss for the most ordinary way a study gets extended. The limit is the
only stop knowable at `enditer` — convergence, the user converger and
`--time-limit` are all decided in the *next* iteration's top half, and the
cylinder-convergence test fires after the hook.

The original rationale for the other two triggers is preserved below for the
record.

**Other options**

- **`--checkpoint-dir <dir>`** — where per-rank files and the manifest are written
  (§10); its presence is what enables checkpointing.
- **`--checkpoint-backend {dill-reload, leaf}`** — how scenario-model state is
  restored (§2.2). `dill-reload` is the default and the only backend implemented in
  the planned phases (captures the warm start + cuts, dodges an expensive
  `scenario_creator` re-run). `leaf` — the **low-cost** option (tiny, fast,
  version-robust checkpoints, for small/cheap-creator runs or frequent kill-safety
  writes) — is designed but **not currently planned** (§11 Phase 6); until that
  phase lands, `dill-reload` is the only valid value.
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

### 8.2 ADMM (deterministic and stochastic)

`--admm` / `--stoch-admm` runs (`generic/admm.py`, `utils/admmWrapper.py`,
`utils/stoch_admmWrapper.py`) are plain PH hubs over *wrapped* scenarios, so
the checkpoint machinery applies in principle — but the wrapper path breaks
several assumptions made elsewhere in this design. Each of the following must
be honored or checkpointing will not work for ADMM:

1. **Scenario naming and file discovery.** The existing pickle paths that §4
   builds on are *hard-refused* for ADMM (`_check_admm_compatibility`,
   `generic/admm.py`) because `scenario_io.py` derives file names from
   `module.scenario_names_creator` and `sputils.extract_num` — wrapped names
   (`ADMM_STOCH__ADMM__<sub>__ADMM__<stoch>`) come from the wrapper, not the
   module, and `extract_num` scrapes trailing digits, colliding across ADMM
   subproblems that share a stochastic scenario. The checkpoint code must
   therefore (a) enumerate `opt.local_scenarios.keys()` — never a module name
   creator — for both write and restore, (b) never use `extract_num` in file
   names (§10), and (c) not be swept into the ADMM incompatibility checks the
   way the pickle flags were: checkpointing is *supposed* to work here, and a
   test should pin that it does.
2. **The creator-cost saving does not apply, and a naive resume doubles model
   memory.** §2.2 counts "avoids re-running an expensive `scenario_creator`"
   as a dill-reload benefit. Not for ADMM: `Stoch_AdmmWrapper.__init__` runs
   the user's `scenario_creator` for every local wrapped scenario (plus probe
   scenarios) during normal startup — it needs the built models to assemble
   consensus lists, `varprob_dict`, node names, and objective scaling — so an
   ADMM resume pays the full creator cost regardless. Worse, after the reload
   branch swaps the dilled models into `local_scenarios`, the fresh models
   remain referenced by `wrapper.local_admm_stoch_subproblem_scenarios` and by
   the `cfg._admm_variable_probability` bound method — a *persistent* 2×
   per-rank model footprint for large MIPs. The reload branch must release or
   replace the wrapper-held fresh models (§9, item 2).
3. **`variable_probability` is object-identity-keyed.** The wrapper's
   `varprob_dict` maps scenario *object* → `(id(var), prob)` pairs
   (`stoch_admmWrapper.py`; `AdmmBundler._bundle_varprob` likewise). This is
   safe today only because `_use_variable_probability_setter` runs exactly
   once, in `SPBase.__init__`, against the wrapper's own model objects, and
   its results land on the model itself (`s._mpisppy_data.prob_coeff` /
   `prob0_mask`) — which the dilled model carries back. The reload branch
   depends on that invariant: **variable probabilities are consumed only at
   construction; after the swap, the reloaded model's `_mpisppy_data` masks
   and fixed-at-0 dummy vars are authoritative, and `var_prob_list` must never
   be called with a reloaded model** (it would `KeyError` — or silently
   mismatch if the dict were rebuilt with new ids). mpi-sppy masks `W` (not
   prox) for zero-probability nonants and assumes each surrogate/dummy var is
   fixed at 0; dill-reload preserves the mask and the fixedness together, and
   the ADMM resume test must assert both survive. (A leaf-rebuild ADMM resume
   would have to re-apply the mask and re-fix the dummies explicitly — one
   more reason that backend is deferred, §11 Phase 6.)
4. **The dill round-trip is unvalidated for a wrapper-mutated model, and the
   intended vehicle cannot currently test it.** `stoch_distr` scenario models
   do not dill *at all* — not because of the wrapper, but because the model
   defines a Pyomo rule closing over `cfg` (§2.2, issue #828). A bare
   `scenario_creator` result fails identically, so the wrapper is exonerated
   and simultaneously untested. Validating this item needs either a fix to
   `stoch_distr` or a different stoch-ADMM model, and that is now a
   prerequisite of Phase 2 rather than a step within it. The rest of this item
   describes what still has to be proven once a vehicle exists. The MIP
   PoC (§6) dilled a plain `sizes` model. A stoch-ADMM scenario is stranger:
   inline dummy `pyo.Var()`s added post-construction with bracket-mangled
   names, rewritten `ScenarioNode`s carrying *unattached*
   `pyo.Expression(expr=0)` cost expressions and `surrogate_vardatas` sets of
   vardata references, a rescaled objective, an appended ADMM stage
   (a multistage tree even for a 2-stage-origin problem), and
   probability-mask arrays on `_mpisppy_data`. dill should handle the cycles,
   but this is a load-bearing assumption of the same kind §6 insisted on
   PoC-ing — validate a stoch-ADMM mid-run round-trip early
   (`mpisppy/tests/examples/stoch_distr` is the vehicle,
   `test_stoch_admmWrapper.py` the harness; §11 Phases 2 and 4).
5. **Bundled stoch-ADMM.** `--stoch-admm --scenarios-per-bundle`
   (`AdmmBundler`) creates bundles on the fly as EFs; they are first-class
   subproblems and should dill like other proper bundles (§8.1). Its
   `var_prob_list` has the same identity keying as item 3.
6. **The spoke set differs.** For stoch-ADMM cylinders: FWPH is refused,
   `xhatshuffle` requires `--stage2-ef-solver-name`, and `xhatxbar` is the
   variable-probability-native inner bounder. The Phase 4 test matrix must
   include a stoch-ADMM configuration (§11).

---

## 9. Core changes required

Touch-points an implementation needs beyond the PoC's extension/subclass hacks:

1. **Global iteration counter / resume offset.** `iterk_loop` hardcodes
   `for _PHIter in range(1, max+1)` (in `iterk_loop`, `phbase.py`), so a resumed run renumbers
   from 1 and its checkpoints collide with the pre-crash ones. Add a resume offset
   so checkpoint numbering is the global iteration. Termination is then counted
   two ways: `max_iterations` bounds the run being started, and the study bound
   `stop_at_iteration_number` (default unset) bounds the study across every run
   linked by checkpoints, whichever arrives first.
2. **A reload-model resume branch, in `Iter0`, replacing the iter-0 solve.**
   The branch lives where `iter0_from_pickle` already branches (§5.1), instead
   of the iter-0 `solve_loop` — so a resume never pays a throwaway `W = 0` solve
   of the fresh models — and the **model swap itself happens before
   `_create_solvers()`**, so the existing solver creation attaches a
   `_solver_plugin` (and calls `set_instance` once) to the reloaded model rather
   than to a model about to be discarded (§5.1). Set `solution_available` for
   the warm start (§5.2).

   Two attaches must be handled, and they are not symmetric. The deferred
   objective attach (`_attach_PH_to_objective_after_iter0`, driven by
   `_deferred_ph_attach`) runs at the **end** of `Iter0`, downstream of the
   branch, and on a reloaded model would duplicate the prox components and
   double the W terms — the branch **clears the flag** rather than relying on
   control flow to miss it. `attach_Ws_and_prox`, by contrast, runs in
   `PH_Prep`, *upstream* of `Iter0` entirely, so it cannot be skipped from this
   branch; it is also harmless, since it decorates the fresh models that the
   swap discards.

   Details the PoC and the ADMM analysis surfaced: **refresh
   `saved_objectives[sname]`** for each reloaded model — `Eobjective` reads
   those objective handles (populated by `_save_active_objectives` in
   `SPOpt.__init__`) and they otherwise dangle to the discarded fresh model;
   swap the reloaded model into `local_scenarios` (which `SPOpt.solve_loop`
   iterates); where the `local_subproblems` alias exists, refresh it too — a
   plain PH keeps no `local_subproblems`, but the **generic file-based path**
   the dill-reload backend builds on (`scenario_io.py` sets
   `sp.local_subproblems = sp.local_scenarios`) maintains it, and
   `CGBase.solve_loop` iterates it; and on ADMM runs, **release or replace the
   fresh models held by the wrapper** (`local_admm_stoch_subproblem_scenarios`
   and the `cfg._admm_variable_probability` closure), or the run keeps two
   copies of every local scenario alive for its whole life (§8.2, item 2). See
   also item 11 for the one piece of *opt-object* state that is keyed by
   variable identity and so cannot survive the swap untouched. This is a
   distinct branch from the leaf-rebuild "build fresh, overlay values" path;
   the `Checkpointer` picks the branch from `--checkpoint-backend`.
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
   only if improving (`update_best_solution_if_improving` in `spbase.py`). This
   also avoids a hub-triggered snapshot barrier and its stall/deadlock risk.
7. **Atomic writes with a single published generation.** Each rank writes only its
   local state (dilled models + leaf non-model data) to rank-tagged temp files and
   renames them into place; the set of per-rank files is then published as one
   checkpoint by atomically rewriting `manifest.json` (itself temp-then-rename) to
   point at the new complete generation (§10). That flip is the single commit
   point, so **one committed generation is enough**: a kill before it keeps the
   previous checkpoint, a kill after it keeps the new one. The prior generation is
   deleted once the manifest is in place. Retaining more than one checkpoint is
   **not supported**: exactly one committed generation exists at any time (plus
   the in-progress one transiently during a publish).
8. **A `Checkpointer` extension** that writes on its active triggers; restore
   itself is the in-core resume branch (item 2), with extension
   `restore_state` hooks (item 3) fired from it before `iterk_loop`:
   - *periodic* (`--checkpoint-every-iterations` / `--checkpoint-every-seconds`) —
     at `enditer`. The seconds trigger tests
     `allreduce_or(now − last_checkpoint ≥ S)` so all ranks decide together
     (mirroring the `time_limit` check in `phbase.py`), avoiding a rank-skew
     deadlock at the write barrier.
   - *anticipated one-shot* (`--checkpoint-before-seconds`) — also at `enditer`,
     testing `allreduce_or(elapsed + last_iteration_seconds ≥ S)` with the same
     collective pattern, then latching so it fires at most once (§8). It needs the
     most-recent iteration duration (item 9); everything else it shares with the
     periodic path.
   - *at each completed iteration* — in `enditer`, which fires after the
     subproblem solve and is therefore the only point in the loop where the
     dual weights and the nonants describe the same iteration (§8). There is
     no terminal trigger and no `--checkpoint-at-termination` flag: the flag
     was removed when the write moved, having briefly survived as a registered,
     documented option that nothing read.

   For spokes, the xhatter `main()` loop calls no per-iteration extension hook —
   add a single `self.opt.extobject.enditer()` (or a dedicated checkpoint hook)
   inside it so **one `Checkpointer` serves hub and xhatter uniformly** (restore
   already has a home: `pre_iter0`/`post_iter0` fire once in `xhat_prep` in
   `xhatbase.py`).
9. **Most-recent iteration duration kept on `self`.** `iterk_loop` (`phbase.py`)
   times each iteration into a *local* `iteration_start_time`, used only by the
   `display_progress` print. `--checkpoint-before-seconds` needs that duration at
   `enditer`, so record it on the object (e.g. `self._last_iteration_seconds`) as
   each iteration completes — and record iteration 0's duration the same way, since
   it is the seed the first time the trigger is tested (§8). Nothing else in PH
   changes: no new hook, no change to the loop's control flow.
10. **`toc` on both ends of every checkpoint write.** The `Checkpointer` emits a
    `global_toc` when a write begins and another when it completes — on every
    trigger, hub and spokes alike, gated on `cylinder_rank == 0` so a multi-rank
    cylinder prints one pair rather than one per rank. Because `tt_timer.toc`
    stamps absolute elapsed time, the pair *is* the measured write duration, which
    is what a user needs to choose `S` for `--checkpoint-before-seconds` — mpi-sppy
    deliberately does not estimate that cost for them (§8). It also makes an
    otherwise invisible multi-minute stall in a long run legible in the log.
11. **Restore the initially-fixed-nonant baseline, by name.** `SPOpt.__init__`
    builds `_initial_fixed_varibles`, a `ComponentSet` of the nonant *vardata
    objects* that were already fixed when the run started (`spopt.py`), and
    `_can_update_best_bound` refuses to update `best_bound_obj_val` whenever a
    nonant is fixed that is not in that set — because fixing a nonant mid-run
    invalidates the outer bound. This is **opt-object state keyed by variable
    identity**, so the model swap (item 2) breaks it in both directions and
    neither is acceptable:

    - *Left alone*, the set holds vardata from the discarded fresh models. Every
      reloaded nonant is a different object, so a fixed nonant reads as
      unrecognized and the gate refuses to update the bound.
    - *Naively rebuilt after the swap*, it absorbs whatever `fixer`/`slammer`
      pinned before the stop, so those mid-run fixings look original and the
      gate admits a bound the uninterrupted run would have refused.

    **How much this bites depends on the hub, and for a plain PH hub it is
    currently masked.** `PHBase._can_update_best_bound` first returns `False`
    whenever the proximal term is enabled, so PH consults the fixedness check
    only with prox off — which happens at exactly one place, the iteration-0
    trivial bound, and that is the path the resume branch replaces anyway. The
    hubs that consult the fixedness gate on their own terms are `Subgradient`
    (which calls the `SPOpt` version directly, every iteration, bypassing the
    prox short-circuit) and `FWPH` (its own override over the same set). Those
    are where a stale baseline would actually change results.

    The checkpoint therefore records the originally-fixed nonants **by variable
    name** (the same by-name discipline §5.4 uses for the incumbent cache), and
    the resume branch rebuilds `_initial_fixed_varibles` from those names against
    the reloaded models. This is the same identity-keying hazard §8.2 item 3
    catches for ADMM's `varprob_dict`. It belongs in the first phase not because
    plain PH is currently broken by it, but because it is the correct restore of
    opt-object state, it costs nothing, and it is load-bearing the moment resume
    covers a hub that consults the gate per iteration — leaving a knowingly stale
    ComponentSet behind for a later phase to trip over is the worse trade.

---

## 10. File layout (proposed)

```
<ckpt_dir>/
  manifest.json                       # cfg hash, n_proc, backend, cylinder map, latest complete hub generation
  hub/
    gen_<NNNN>/                        # NNNN = global PH iteration at the checkpoint
      hub_rank_<RRRR>.pkl             # non-model leaf state: iter counter, bounds, extension-object state
      hub_rank_<RRRR>_scen_<S>.dill   # dilled scenario model(s) for this rank (dill-reload backend)
  spokes/
    spoke_<name>_rank_<RRRR>.pkl      # each spoke's latest incumbent (best xhat, by name) + bound,
                                      #   overwritten asynchronously on improvement (§9, item 6)
```

The hub writes each checkpoint as an iteration-tagged generation under `hub/`,
deleting the prior one after the manifest flip (§9, item 7); each spoke keeps a
single latest-wins file under `spokes/` that it overwrites atomically on
improvement — the two are deliberately *not* aligned (§9, item 6).
`manifest.json` is the single commit point: it names the latest *complete* hub
generation and records the backend so resume loads the right way. Under the `leaf`
backend the `.dill` model files are replaced by numeric arrays inside the
`hub_rank_*.pkl`. Use plain `pickle` for the numeric/leaf state; `dill` for the
scenario models.

`<S>` is the scenario's full name sanitized for the filesystem (or its index in
the rank's local scenario list) — **never** `sputils.extract_num`, which is not
unique for ADMM wrapped names (§8.2, item 1). More generally, both write and
restore enumerate `opt.local_scenarios.keys()`, not a module name creator.

**Disk footprint.** Dilled large MIP models × scenarios/rank can be large. The
single-generation policy (§9, item 7) keeps exactly one checkpoint live, but
the peak is **two generations transiently during a publish** (the new one is
fully written before the manifest flip deletes the old one) — state the peak in
user docs so disk quotas are sized for it.

---

## 11. Phased rollout

Each phase is a review-sized PR that is green on its own and adds user-visible
value. New tests are wired into `run_coverage.bash` **and**
`test_pr_and_main.yml` in the same commit.

### 11.1 The A/B resume harness (every phase's acceptance test)

The core CI test shape, reused by every phase, is an **A/B comparison**:

- **Run A (reference):** an uninterrupted run of `N` iterations on a small
  instance.
- **Run B (checkpointed):** the same instance stopped at iteration `k < N`
  with a checkpoint written, then resumed in a **fresh process** and run to
  `N`.
- **Compare A and B** under the §7 determinism contract:
  - **Deterministic LP instances** (farmer; farmer + CVaR): `W`, nonants,
    `rho`, `xbar` at each common iteration and the final objective must be
    **bit-identical** (`max|diff| == 0.0`), and final bounds equal.
  - **MIP instances** (`sizes`): with single-thread deterministic solver
    settings (`Threads=1`, fixed seed, `MIPGap=0` — the §7 validation crutch)
    the same bit-identity check applies; under default settings assert instead
    that the run **continues** (global iteration numbering, no re-attach /
    duplicate-component errors), the **incumbent never regresses**, bounds
    stay valid, and the final objective agrees within a stated tolerance.
- Also assert the negatives: run B performs **no iter-0 subproblem solve** on
  resume (§5.1), and a geometry/cfg mismatch is refused with a clear error
  (§5.7).

Instances — all small enough for the pip-installed, size-limited CPLEX/Xpress
CI solvers:

- **farmer** — deterministic-LP baseline, serial and cylinders.
- **farmer + `--cvar`** (`utils/cvar.py`) — a mutate-after-creation transform:
  the deactivated risk-neutral objective, the active `WITH_CVAR` objective,
  and the eta var appended to the root nonants must all survive the dill
  round-trip, and the resume branch's `saved_objectives` refresh (§9, item 2)
  must resolve to `WITH_CVAR`, not the deactivated original.
- **stoch-distr (`--stoch-admm`)** — intended to exercise everything in §8.2:
  wrapped names in file discovery, variable-probability masks and fixed-at-0
  dummy vars, the wrapper-mutated model dill round-trip, and release of the
  wrapper-held fresh models. **Unblocked**: `stoch_distr`'s rules previously
  closed over `cfg`, which made its models unserializable; fixed and merged
  (#830), and a structural guard in `test_stoch_admmWrapper.py` keeps the
  pattern from returning.
- **`sizes`** — the MIP target: warm start taken on resume, incumbent carried.

The phase bullets below say where each instance enters (Phase 1a: serial
farmer; Phase 1b: farmer+CVaR and `sizes`; Phase 2: multi-rank, bundles,
stoch-ADMM; Phase 4: cylinders, including a stoch-ADMM configuration).

Phase 1 is split into two review-sized PRs. Phase 1a is the whole serial
stop-and-resume story with the single trigger the primary use case actually
needs; Phase 1b adds the optional triggers and the harder test instances on top.
Each is green on its own, and 1a is independently useful — a run that stops at
`--time-limit` and resumes the next morning needs nothing from 1b.

**The numbering is not a dependency chain, and phase 4 in particular can be
pulled forward.** The phases are ordered by how much machinery each adds, not
by what each requires, and cylinders is the phase that decides whether anyone
can use this at all: mpi-sppy is run as hub-and-spoke, so a serial-hub-only
feature has few real users. Phase 4 depends on neither phase 2 nor phase 3.
Not phase 2, because `n_proc` is `self.mpicomm.Get_size()` on the *cylinder's*
comm rather than `COMM_WORLD` (`global_rank` is tracked separately), so a
hub+lagrangian+xhatshuffle run with one rank per cylinder already has
`n_proc == 1` everywhere and passes the phase-1a multi-rank guard untouched —
phase 2 is multi-rank *within* a cylinder, phase 4 is *multiple* cylinders, and
the two are orthogonal. Not phase 3, because each spoke checkpoints its own
best xhat asynchronously on improvement with no hub↔spoke coordination (§9,
item 6), which is spoke-specific machinery rather than the general extension
`checkpoint_state`/`restore_state` contract. So phase 4 may follow 1a directly,
as a branch stacked on the 1a PR.

- **Phase 1a — Serial hub checkpoint/resume, writing at completed iterations.**
  (Titled "terminal trigger only" while that was the plan; the terminal
  trigger was removed once the write moved to iteration boundaries, and
  `--checkpoint-every-iterations` shipped here instead of in 1b — see §8.) The
  framework: `Checkpointer` extension; global iteration counter / resume offset
  (§9 item 1); reload-model resume branch **in `Iter0`, replacing the iter-0
  solve**, with the swap ahead of `_create_solvers()`, the deferred objective
  attach disarmed, `saved_objectives` refreshed, and the warm start set (§5.1, §9
  item 2); restore of the initially-fixed-nonant baseline by name (§9 item 11);
  geometry+cfg fingerprint (§5.7); atomic per-rank writes + manifest publish (§9
  item 7); the write at each completed iteration (§8), paced by
  `--checkpoint-every-iterations`; `toc` on
  both ends of every write (§9 item 10); and setup-time refusal of every
  configuration not supported — a non-PH hub (APH inherits this wiring through
  `aph_hub`), more than one rank, an unimplemented backend, an unwritable
  directory, and any run where the extension would not actually be attached.
  CLI flags `--checkpoint-dir`,
  `--checkpoint-backend`, `--resume-from`/`--resume`, with a clear error when
  `dill` is not installed (it is an optional `extras` dependency). Tests (the
  §11.1 A/B harness, serial): **farmer** bit-identical A vs B; no iter-0
  subproblem solve occurs on resume; geometry/cfg mismatch refused.
- **Phase 1b — The harder instances (the triggers are gone).**
  `--checkpoint-every-iterations` shipped in 1a, and `--checkpoint-every-seconds`
  and the anticipated one-shot `--checkpoint-before-seconds` are **not
  implemented and not planned**: §8 records why writing at every completed
  iteration subsumes them. What is left of this phase is the harder test
  instances, which the phase that needs them should absorb rather than wait
  for. Tests: **farmer + `--cvar`** bit-identical A vs B — the
  mutate-after-creation case, where the resume branch's `saved_objectives`
  refresh must resolve to `WITH_CVAR` and not the deactivated original;
  **`sizes`** (MIP) — bit-identical under deterministic solver settings, and
  under default settings run continues correctly, incumbent preserved, warm start
  taken (mid-run model dill round-trip proven, §6).
- **Phase 2 — Multi-rank + bundles + stoch-ADMM.** Barriers, rank-tagged files,
  single-generation atomic publish. Validate with **proper bundles** (§8.1) and
  refresh the stale `_restore_nonants` comment. Validate **stoch-ADMM** (§8.2):
  the wrapper-mutated model dill round-trip (item 4), file naming with wrapped
  scenario names (item 1), wrapper-held fresh models released on resume (item
  2), and the probability mask + dummy-var fixedness surviving restore (item
  3) — `mpisppy/tests/examples/stoch_distr` is the vehicle. Tests (the §11.1
  A/B harness under `mpiexec`): MIP stop+resume compared on every rank, incl.
  uneven distribution and `--scenarios-per-bundle`; a stoch-distr
  (`--stoch-admm`) A/B stop+resume; mismatch refusal.
- **Phase 3 — Extension-object state contract.** `checkpoint_state`/`restore_state`
  on `Extension`; implement for rho updaters, `fixer`, `slammer`, convergers.
  (Model-attached `fixer` counter and nonant fixedness ride in the dill.) Test: PH
  + norm-rho-updater, PH + `fixer`, PH + `slammer` each resume with state intact
  and consistent with variable fixedness.
- **Phase 4 — Cylinders / spokes.** One-line xhatter write hook; unified
  `Checkpointer` on spoke opts; each spoke checkpoints its own **best xhat** (by
  name) asynchronously on improvement — no hub↔spoke coordination (§9, item 6).
  **Also move the hub write onto that same dedicated hook.** Phase 1a writes
  from the `Checkpointer`'s `enditer`, and `MultiExtension` dispatches
  `enditer` in attach order with the `Checkpointer` first (`add_checkpointing`
  runs at the end of `ph_hub`, before `configure_extensions` appends the rest),
  so a *user* extension whose `enditer` mutates models does so after that
  iteration's checkpoint was written, and a resume never re-applies it. No
  shipped extension is affected — every `enditer` in the tree is a no-op or
  read-only — so phase 1a documents the constraint instead of reordering. A
  dedicated call in `iterk_loop` removes the dispatch-order dependency for
  every driver at once, not just the `do_decomp` path, and this phase is
  already adding the equivalent hook to the xhatter loop, so the two land
  together. Note `enditer_after_sync` is *not* a substitute: it fires after the
  `spcomm.is_converged()` break, so a run ending on cylinder convergence would
  write nothing.
  Tests (the §11.1 A/B harness on cylinders): farmer/`sizes`
  (hub+lagrangian+xhatshuffle) stop+resume — hub primal trajectory compared A
  vs B (bit-identical for farmer, per the §6 PoC), best xhat preserved, bounds
  valid best-so-far — **plus a stoch-ADMM cylinders configuration** (§8.2,
  item 6: no FWPH; `xhatshuffle` with `--stage2-ef-solver-name`, or
  `xhatxbar`).
- **Phase 5 — Exact spoke continuity (optional).** Hoist `ScenarioCycler`/`xh_iter`
  onto `self`; checkpoint the cursor (+ RNG getstate if a stream becomes stateful).
- **Phase 6 — Leaf-rebuild backend + broader coverage (not currently planned).**
  A possible future phase, deferred: the primary use case is fully served by the
  dill-reload backend (Phases 1–4), so this is recorded for when a lighter,
  version-robust checkpoint is actually needed rather than scheduled now. It would
  add the `--checkpoint-backend leaf` path (rebuild via `scenario_creator`, overlay
  W/rho/nonants/fixedness, replay prox `cut_values`, optional all-var warm start —
  what the PoC prototyped) plus lagranger, FWPH, and subgradient spoke coverage.
  The design deliberately keeps this backend's hooks and the shared
  framework/manifest (§2.2) so it can be added later without disturbing the shipped
  dill-reload path.

---

## 12. Design decisions (resolved) and deferrals

Resolved (given the §1 use case):

- **Backend choice.** dill the scenario models (§2.2): overhead is negligible at a
  few checkpoints, version robustness is unneeded (same-environment resume next
  day), and it captures the warm start + prox cuts + model-attached state for free
  while avoiding an expensive `scenario_creator` re-run (except on ADMM paths —
  §8.2, item 2).
- **Warm start.** Worthwhile for MIPs (branch-and-bound benefits), free via the
  dilled model, fed through the existing `warmstart_subproblems` /
  `solution_available` path.
- **Restore point.** An in-core resume branch in `Iter0` replacing the iter-0
  solve (§5.1; §9, item 2) — no throwaway `W = 0` solve on resume, and
  consequently no special iteration-0 checkpoint (§8). The PoCs' extension-hook
  restore was a validation crutch, not the design.
- **Checkpoint retention** (§9, item 7): exactly one manifest-published
  generation is kept; retaining older generations is not supported. The disk
  peak is two generations transiently during a publish (§10) — a documented
  cost, not an open question.
- **Spoke snapshot coordination** (§9, item 6): resolved by *not* coordinating.
- **variable_probability / surrogate vars (incl. ADMM).** Resolved by the §8.2
  contract: probabilities are consumed only at `SPBase` construction; after the
  reload swap, the reloaded model's `_mpisppy_data` masks and fixed-at-0
  dummy/surrogate vars are authoritative, and `var_prob_list` is never called
  with a reloaded model. Validation is scheduled (Phase 2).
- **Mid-run MIP model dill round-trip** — was the load-bearing unvalidated
  assumption; **validated by the MIP dill-reload PoC** (§6), including the
  linearized-prox cuts, in-process and cross-process, with serial stop→reload→
  continue bit-identical under a deterministic solver. The stoch-ADMM
  wrapper-mutated variant of the same assumption is not yet validated — that is
  a scheduled validation item (§6; §8.2, item 4; §11 Phase 2), not an open
  design question.

Deferred:

- **Cross-geometry resume** (different rank count or scenario-to-rank
  distribution) — a §1 non-goal; revisit if HPC users need to resume on a
  different node count.
- **Leaf-rebuild backend** — designed (§2.2) but not scheduled (§11, Phase 6).
