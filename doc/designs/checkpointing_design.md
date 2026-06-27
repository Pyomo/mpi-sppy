# Checkpoint / Resume for mpi-sppy — Design

Status: **draft** (PoC validated; not yet implemented in the library).
Scope: checkpoint a running mpi-sppy job so a killed/interrupted run can resume
where it left off. Must work on multiple MPI ranks and for cylinder
(hub-and-spoke) runs.

---

## 1. Goals and non-goals

**Goals**

- Resume a Progressive Hedging (PH) run — serial, multi-rank, or full cylinders
  (hub + spokes) — after a crash or a deliberate stop.
- Preserve the **best feasible solution found so far (the best xhat), not just
  the best bound.**
- Survive a hard kill (`kill -9`, node failure, walltime), not only a clean
  shutdown.
- Add no measurable overhead when checkpointing is off, and modest, tunable
  overhead when on.

**Non-goals (initially)**

- Resuming across a *different* rank count or scenario-to-rank distribution
  (cross-geometry remap). The first cut requires identical geometry and refuses
  a mismatch with a clear error.
- Bit-identical reproduction of *bounds* (see §7 — bounds are async and not
  reproducible; only the primal trajectory is).
- APH. A C++ APH is expected to replace the Python `opt/aph.py`; PH-family only.

---

## 2. Why "pickle every object with dill" does not work

The obvious idea — `dill.dump()` the whole opt/hub object graph and reload — is
not viable, because the core objects are built around **live, non-serializable
OS/MPI/solver handles**, not data:

- **MPI communicators** — `SPBase.mpicomm`, `SPBase.comms` (`spbase.py`), and the
  `fullcomm`/`strata_comm`/`cylinder_comm` on the spcomm
  (`cylinders/spcommunicator.py`). Handles into the running MPI runtime;
  meaningless once the process exits.
- **MPI RMA windows and `MPI.Alloc_mem` buffers** — `SPWindow`
  (`cylinders/spwindow.py`) and the `FieldArray` send/receive buffers. Kernel
  shared-memory regions.
- **Persistent solver handles** — `s._solver_plugin` for
  gurobi/cplex/xpress persistent interfaces (`spopt.py`). A C handle + license
  session.

The moment you write a custom `__getstate__`/`__reduce__` to drop and rebuild
those, you have conceded the "just pickle it" dream and are doing
reconstruct-scaffolding + restore-state anyway — only with the checkpoint format
welded to the live object graph (fragile across versions). Better to do that
split explicitly.

dill *is* useful for the leaf data (Pyomo scenario models, closures), and the
repo already uses it that way — see §4.

---

## 3. Approach: reconstruct the scaffolding, restore the state

A checkpoint is **not** a snapshot of the object graph. It is the **algorithmic
state** needed to continue. On resume:

1. **Reconstruct the scaffolding** via the normal startup path — comms, RMA
   windows, persistent solvers, and the Pyomo scenario models (from
   `scenario_creator`, or from the existing scenario-pickle path). This is
   exactly what a fresh run already does.
2. **Restore the state** into that fresh scaffolding: iteration counter, dual
   weights, nonant values, rho, the incumbent (best xhat), aggregated bounds,
   and any stateful-extension internals.

This is the design the library *already leans toward* (it pickles scenario
models alone, never the comms/solvers) — we are extending it from "warm-start
iteration 0" to "resume at iteration k".

---

## 4. Building blocks already in the repo

- **Scenario-model pickling (dill):** `utils/pickle_bundle.py`
  (`dill_pickle`/`dill_unpickle`) and `generic/scenario_io.py` pickle each
  scenario Pyomo model *alone*. Iter-0 solution + bounds can be baked into
  `pickle_metadata` and warm-started via `--iter0-from-pickle`. Precedent for
  persisting algorithmic state, and a ready path for reconstructing expensive /
  nondeterministic scenario models.
- **W / xbar persistence:** `utils/w_utils/wxbarwriter.py` (writes in
  `post_everything`) / `wxbarreader.py` (reads in `pre_iter0`) round-trip the
  dual weights `W` and the consensus `xbar` as CSV, as extensions. Covers part
  of the restore set; does **not** cover rho, the iteration counter, bounds, the
  incumbent, or RNG.
- **Incumbent-to-disk (spokes):** `cylinders/spoke.py`
  `_maybe_write_incumbent_on_improvement` (`--incumbent-on-improvement-filename-prefix`)
  already writes the first-stage solution on each improvement — a reference for
  serializing the incumbent.

---

## 5. State inventory

The key question for each piece: **reconstructed** (rebuilt by startup, no save
needed) vs **restored** (must be checkpointed), and if restored, is it
**bit-reproducible** on resume or only **carried forward** as a valid
best-so-far value?

### 5.1 Hub PH primal state — *restored, bit-reproducible*

The hub's primal trajectory is pure synchronous PH and is **independent of the
spokes** (lagrangian only contributes an outer bound; xhatshuffle only an
incumbent; neither perturbs the iterate). Minimum set, per local scenario:

| State | Where it lives | Note |
|---|---|---|
| `_PHIter` | `phbase` | iteration counter |
| `W[ndn_i]` (accumulated duals) | `s._mpisppy_model.W` | **must** restore — `Update_W` accumulates, so it is not recomputable |
| nonant values | `s._mpisppy_data.nonant_indices` vardata `_value` | restoring these lets `Compute_Xbar` reproduce `xbar`; `xbar` itself need not be saved |
| nonant **fixedness** | same vardata `.fixed` (+ the fixed value) | **must** restore — variable-fixing/forcing extensions (`fixer`, `slammer`) leave nonants fixed; a rebuilt model has them free. See §5.4 |
| `rho[ndn_i]` | `s._mpisppy_model.rho` | restore (rho-updaters mutate it) |
| smoothing `z/p/beta` | `s._mpisppy_model` | only if `--smoothing` |

Helpers already exist: `_populate_W_cache` / `W_from_flat_list` (`phbase.py`),
`_save_nonants` / `_restore_nonants` (`spopt.py`). Note `_save_nonants` already
captures fixedness (`fixedness_cache`) — the checkpoint must gather it too (the
PoC captured only `_value`).

**Restore point:** between `Iter0()` and `iterk_loop()`. In serial this can be a
driver call; in cylinders the hub runs `ph_main` internally, so restore happens
in the **`post_iter0_after_sync`** extension hook (fires at the end of `Iter0`,
`phbase.py:1079`). Iteration 1 of the resumed loop then reproduces what would
have been iteration k+1 — verified bit-identical (§6).

### 5.2 Hub bounds + incumbent objective — *restored, carried forward*

- `spcomm.BestInnerBound`, `spcomm.BestOuterBound` — aggregated bounds.
- `opt.best_bound_obj_val`, `opt.best_solution_obj_val`.

These are products of the **async** spoke interaction; their *timing* is not
reproducible, so they are carried forward (restored as best-so-far). They remain
valid: a restored looser bound is simply improved again by the fresh spokes; a
restored incumbent objective is never regressed because
`update_best_solution_if_improving` (`spbase.py:578`) only accepts improvements.

Note: in a cylinders run the hub's `opt.best_solution_obj_val` is often `None` —
the inner bound arrives as a scalar via `receive_innerbounds`
(`spcommunicator.py:1019`) and lands in `spcomm.BestInnerBound`. So the
incumbent **objective** is on the hub, but the incumbent **solution** is not (see
5.3).

### 5.3 Spoke incumbent — *the best xhat solution* — *restored, carried forward*

**The best xhat solution VALUES live on the xhat spoke, not the hub.** The
xhatshuffle spoke holds them in `spoke.opt.best_solution_cache` (a `ComponentMap`
over all vars) and `spoke.best_inner_bound`; `InnerBoundSpoke.finalize()`
(`spoke.py:293`) loads them back. The hub keeps only the bound scalar.

Therefore **"keep the best xhat" requires checkpointing the spoke incumbent.**
Checkpointing the hub alone preserves the incumbent *number* but can lose the
*solution* if the crash is late and a short resume does not re-find it.

Serialize the `ComponentMap` by **variable name** (`{var.name: value}`) — the
`Var` objects are tied to one process's model and cannot cross a restart; rebuild
the map by name lookup on the reconstructed model.

### 5.4 Stateful extensions — *restored, bit-reproducible (if saved)*

Several extensions hold trajectory-driving state and **must** be checkpointed or
resume diverges:

- rho updaters (`mult_rho_updater`, `norm_rho_updater`, `grad_rho`) — multipliers
  / previous-gradient history.
- convergers — convergence history.
- **variable-fixing/forcing extensions** — `fixer.py` and `slammer.py` both
  decide to pin nonants and then **skip what they have already pinned**, so their
  tracking *is* the trajectory. They illustrate two complications:
  - **Where the tracking lives differs.** `slammer` keeps `self._slammed`
    (`(ndn,i) → value`, sticky) on the **extension object** — covered directly by
    the contract below. `fixer` keeps its per-variable convergence counters on the
    **scenario model** (`s._mpisppy_data.conv_iter_count`), *not* on `self` — so a
    contract that only serializes `self.*` would miss them. The contract must let
    an extension serialize its model-attached state too.
  - **The tracker and the actual variable state must be restored together and
    agree.** Both extensions call `.fix()` / set a value on real nonant vardata.
    On resume the rebuilt model has those vars free. If the tracker is restored
    ("already fixed/slammed X") but X's actual `.fixed`/value is not (§5.1), they
    **disagree**: the extension skips X as done while the solver re-optimizes X
    freely — drift, and X is never re-pinned. This is *worse* than not tracking.
    So variable fixedness (§5.1) and the extension's records are one unit:
    checkpoint and restore them together.

The `Extension` base class has **no serialization hook today**. We add a
`checkpoint_state()` / `restore_state()` contract (§9, item 3) that can serialize
both `self.*` state and the extension's model-attached state
(`s._mpisppy_data.*`). The same contract serves the hub's extensions and an
xhatter's extensions, because both run through the same `extobject` machinery
(`MultiExtension`).

**Answer to "do fixing extensions keep tracking across resume?"** Yes — that is
the intent: their records are restored, so they continue to keep track (and
`slammer`'s slams stay sticky, `fixer` does not re-count from zero). But only if
both points above hold; the variable fixedness (§5.1) and model-attached counters
must be restored alongside the extension object, consistently.

### 5.5 RNG and spoke cursor — *partially restored*

- xhatshuffle seeds its stream to a fixed `42` and samples **once**
  (`xhatshufflelooper_bounder.py:88,94`) — the shuffle is deterministic, so no
  RNG state needs saving.
- The `ScenarioCycler` cursor and `xh_iter` are **local variables inside
  `main()`** — unreachable from outside. Exact spoke-cursor resume needs those
  hoisted onto `self` (Phase 5). Without it, the spoke restarts its cursor; for
  correctness this only changes *which* scenario it tries next, not the
  preserved best (which is restored from 5.3).
- lagrangian / lagranger spokes use **no RNG**; their bound is deterministic
  given the hub's W. State to carry: `_PHIter`, `trivial_bound`, last `bound`,
  received `localWs`.

### 5.6 Geometry / cfg fingerprint — *checkpoint metadata*

Each per-rank file records `{n_proc, rank, local scenario list}` (and, for the
header, a cfg hash). Resume verifies the current layout matches and **refuses a
mismatch with a clear error** (validated — see §6).

---

## 6. PoC evidence (what is already validated)

A throwaway PoC (serial + multi-rank + cylinders, farmer, gurobi_persistent)
confirmed the design:

- **Serial:** resume-from-iter-6 reproduced a full 12-iteration run with
  `max|diff| = 0.000e+00` for W, nonants, and rho. Persistent solver survives the
  rebuild (Iter0 re-creates + `set_instance`).
- **Multi-rank:** `-np 3` (1 scenario/rank) and uneven `-np 2` (2+1) both resume
  bit-identical on every rank. Per-rank rank-tagged files, barrier + atomic
  temp-then-rename write. Geometry mismatch (resume `-np 3` ckpt under `-np 2`)
  fails with a clear error.
- **Cylinders (PH hub + lagrangian + xhatshuffle):** hub primal resumes
  **bit-identical** inside `WheelSpinner` (windows, spokes, async). The best xhat
  *solution* (carried on the spoke) is preserved exactly across a crash;
  `BestInnerBound` carried exactly; `BestOuterBound` **differed** run-to-run
  (async timing) but stayed valid.

These also surfaced the findings folded into §5 and §9.

---

## 7. Determinism contract (what resume guarantees)

- **Primal trajectory (W, nonants, rho, xbar): bit-identical.** This is the
  correctness guarantee — the resumed run produces the same iterates as an
  uninterrupted run.
- **Bounds and incumbent: valid and best-so-far, but NOT bit-reproducible.** They
  come from async, timing-dependent spoke interaction. Resume never reports a
  *worse* best-so-far than the checkpoint, but it may report a different
  (equally valid) bound than the original run would have at the same iteration.

This distinction must be stated in user docs so a tightened/loosened bound after
resume is not mistaken for a bug.

---

## 8. Configuration and semantics

Checkpointing is **opt-in** and adds nothing when off.

- **`--checkpoint-every k`** — write a checkpoint every `k` PH iterations.
  - `k` unset or `0` ⇒ checkpointing is **disabled**: the `Checkpointer`
    extension is not attached, so there is zero overhead and no files.
  - `k ≥ 1` ⇒ checkpoint every `k` iterations **and always at the end of
    iteration 0**, regardless of `k`. Iteration 0 establishes the resume
    baseline — solvers are created, the trivial bound is computed, the initial
    (`W = 0`) solve is done — so a crash anywhere in the `iterk` loop has a valid
    resume point even before the first periodic checkpoint, and the iter-0 state
    composes with the existing `--iter0-from-pickle` warm-start.
- **`--checkpoint-dir <dir>`** — where per-rank checkpoint files and the
  manifest are written (see §10).
- **`--resume-from <dir>`** (or `--resume`, auto-selecting the latest *complete*
  checkpoint from the manifest) — rebuild the wheel and restore that checkpoint.
  Resume requires identical geometry (§5.6); a mismatch is refused with a clear
  error.

Hard-kill coverage is the user's tradeoff: small `k` is safest, larger `k` cuts
I/O. Combined with last-*k* retention (§9, item 7), a kill *during* a write still
leaves a usable earlier checkpoint.

### 8.1 Bundles

mpi-sppy has **only proper bundles** now — loose bundling (`bundles_per_rank`)
was removed in 2026 (`spbase.py`; see `doc/src/properbundles.rst`). A proper
bundle consumes whole second-stage tree nodes and is a **first-class
subproblem**: it appears in `local_scenarios` with its own `nonant_indices`, and
PH has no separate `local_subproblems` for it.

Consequently checkpointing **applies to bundled runs directly and uniformly** —
the same code that gathers/restores `W`, nonants, and `rho` by iterating
`local_scenarios` + `nonant_indices` covers a bundle exactly as it covers a plain
scenario. This holds whether bundles are built in memory
(`--scenarios-per-bundle`) or pickled and re-read
(`--pickle-bundles-dir` / `--unpickle-bundles-dir`); both produce proper bundles
and checkpoint identically. Pickling is an independent optimization (skip the
bundle rebuild on resume), not a requirement for checkpointing.

One cleanup: `_restore_nonants` still carries a 2019 comment that it "will not
work on bundles" (`spopt.py`). That predates proper bundles and refers to the
removed loose mechanism; it should be re-verified and refreshed when bundle
checkpointing is validated (Phase 2).

---

## 9. Core changes required

These are the touch-points an implementation needs beyond the PoC's
extension/subclass hacks:

1. **Global iteration counter / resume offset.** `iterk_loop` hardcodes
   `for _PHIter in range(1, max+1)` (`phbase.py:1156`), so a resumed run renumbers
   from 1 and its checkpoints collide with the pre-crash ones. Add a resume
   offset so checkpoint numbering is the global iteration, and so termination
   honors the original `max_iterations`.
2. **One-line xhatter write hook.** The xhatter `main()` loop calls no
   per-iteration extension hook. Add a single `self.opt.extobject.enditer()` (or a
   dedicated checkpoint hook) inside the loop. Then **one `Checkpointer` extension
   serves hub and xhatter uniformly** — restore at `post_iter0` /
   `post_iter0_after_sync`, write at `enditer` — instead of bespoke spoke
   subclasses. (Spoke restore already has a home: `pre_iter0`/`post_iter0` fire
   once in `xhat_prep`, `xhatbase.py:36,49`.)
3. **Extension `checkpoint_state` / `restore_state` contract** on the `Extension`
   base (no-ops by default; implemented by rho updaters, `fixer`, `slammer`,
   convergers). Must serialize both `self.*` state and the extension's
   model-attached state (`s._mpisppy_data.*`, e.g. `fixer`'s `conv_iter_count`),
   and be restored consistently with the variable fixedness in §5.1 (see §5.4).
   The `Checkpointer` aggregates the other extensions' dicts into the per-rank
   file.
4. **Full-var snapshot around any xhat/incumbent evaluation.** Evaluating an xhat
   fixes the first stage and re-solves; `_restore_nonants` restores *only*
   nonants, leaving second-stage recourse vars in the eval state. The checkpointed
   primal state (W/nonants/rho) is unaffected (it is gathered from nonants/params),
   but any *full-objective* read taken right after an eval is wrong. Snapshot and
   restore all vars around an eval, or evaluate on a copy.
5. **Geometry / cfg fingerprint** (§5.6) with a clear refusal on mismatch.
6. **Hub↔spoke snapshot coordination.** A spoke writes its *latest* incumbent and
   does not know the hub's iteration number, so a naive set of files is not a
   globally-consistent snapshot at a known iteration. The hub's checkpoint barrier
   should trigger spoke snapshots (e.g. a checkpoint Field broadcast), so a
   resume restores a coherent cross-cylinder state.
7. **Atomic, per-rank, barriered writes** (already in the PoC): each rank writes
   only its local state to a rank-tagged file via temp-then-rename, inside a
   barrier, so a hard kill never yields a half-written or partial-across-ranks
   checkpoint. Retain the last *k* checkpoints (configurable) so a kill *during*
   a checkpoint still leaves a usable earlier one.

---

## 10. File layout (proposed)

```
<ckpt_dir>/
  manifest.json                 # cfg hash, n_proc, cylinder map, latest complete iter
  iter_<NNNN>/
    hub_rank_<RRRR>.pkl         # hub primal + extension state (+ bounds/incumbent obj)
    spoke_<name>_rank_<RRRR>.pkl# spoke incumbent / bound state
```

`manifest.json` names the latest *complete* checkpoint (all ranks + cylinders
flushed); resume reads that. Use plain `pickle` for the numeric state; `dill`
only where models/closures must be serialized.

---

## 11. Phased rollout

Each phase is a review-sized PR that is green on its own and adds user-visible
value. New tests are wired into `run_coverage.bash` **and**
`test_pr_and_main.yml` in the same commit.

- **Phase 1 — Serial hub checkpoint/resume.** `Checkpointer` extension
  (write at `enditer`, restore at `post_iter0_after_sync`); global iteration
  counter / resume offset; geometry+cfg fingerprint; atomic per-rank writes;
  CLI flags `--checkpoint-dir`, `--checkpoint-every k` (0/unset ⇒ off; else every
  `k` iters **and always after iter 0** — §8), `--resume-from`/`--resume`. Restore
  W/nonants/rho. Test: serial farmer kill+resume bit-identical.
- **Phase 2 — Multi-rank + bundles.** Barriers, rank-tagged files, last-*k*
  retention. Validate checkpointing with **proper bundles** (bundle = first-class
  subproblem, §8.1) and refresh the stale `_restore_nonants` bundle comment.
  Test: `mpiexec` farmer kill+resume bit-identical on every rank, incl. uneven
  distribution and a `--scenarios-per-bundle` run; mismatch refusal.
- **Phase 3 — Extension state contract.** `checkpoint_state`/`restore_state` on
  `Extension` (covering `self.*` and model-attached state); restore nonant
  fixedness in the primal set (§5.1); implement the contract for rho updaters,
  `fixer`, and `slammer`. Test: PH + norm-rho-updater resumes bit-identical; PH +
  `fixer` resumes with the *same* variables fixed (and the same in-progress
  counters); PH + `slammer` resumes with slams intact.
- **Phase 4 — Cylinders / spokes.** One-line xhatter write hook; unified
  `Checkpointer` on spoke opts; checkpoint the **spoke incumbent (best xhat
  solution)**; hub↔spoke snapshot coordination. Test: farmer cylinders
  (hub+lagrangian+xhatshuffle) crash+resume — hub primal bit-identical, best xhat
  preserved.
- **Phase 5 — Exact spoke continuity (optional).** Hoist `ScenarioCycler`/`xh_iter`
  onto `self`; checkpoint cursor (+ RNG getstate if a stream becomes stateful).
- **Phase 6 — Broader coverage.** lagranger, FWPH, subgradient spokes;
  scenario-pickle reconstruction path for expensive `scenario_creator`s.

---

## 12. Open questions / risks

Resolved (see §8): the `--checkpoint-every k` semantics (0/unset ⇒ off; otherwise
every `k` iters and always after iter 0) and bundles (only proper bundles exist;
each is a first-class subproblem, so checkpointing applies uniformly — no special
handling needed).

Still open:

- **Coordinated spoke snapshots** (§9, item 6) add a sync point; need to confirm
  it does not stall fast spokes or deadlock with `got_kill_signal`.
- **variable_probability / surrogate vars.** The masked-W / prob-0 design must be
  re-checked under restore.
- **Cross-geometry resume** is explicitly deferred; revisit if HPC users need to
  resume on a different node count.
```
