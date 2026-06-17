# Slamming — design

**Status:** Draft — up for discussion. Nothing is implemented yet.
Intended branch: `slamming` (off Pyomo/mpi-sppy `main`).
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-06-17

Related: PySP's Watson–Woodruff (WW) PH extension, which let the user
specify slamming preferences in a configuration file. This design brings
a similar capability to mpi-sppy, modernized to the library's current
idioms (by-name nonant files, the `Config` system, the `Extension`
family of in-PH fixers).

---

## 0. Goals

1. Add **slamming** to mpi-sppy: forcing (fixing) selected nonanticipative
   variables to a chosen value *while a decomposition hub is running*, to
   drive toward a feasible / converged incumbent when ordinary convergence
   is slow. Slamming is an `Extension`, not a PH feature: it works with any
   hub that runs the extension hooks and maintains x-bar — the `PHBase`
   family (PH, APH, Subgradient, FWPH). See §4 for the applicability
   conditions. (L-shaped derives from `SPBase`, runs no extensions, and is
   out of scope.)

2. Let the user specify, per variable, **whether** it may be slammed,
   **which direction(s)** it may be slammed in, and in **what order**
   (priority) — supplied through a file (`--slamming-directives-file`),
   the way PySP did.
3. Support wildcards in the file so a single rule can cover a family of
   indexed variables.
4. Decouple the slamming **mechanism** (this design) from slam
   **triggering**, so that:
   - Phase 1 ships a simple iteration-count trigger (PySP's
     "slam-after-iter" plus an iters-between cadence), and
   - a later phase can drop a genuine **stall detector** into the same
     seam without touching the mechanism.
5. **Total backward compatibility.** A run that uses none of the new
   slamming options behaves byte-for-byte as it does today.
6. Multistage-capable from the start (the existing slam *spokes* are
   two-stage only — see §1).

### Non-goals

- **Stall detection.** Phase 1's trigger is iteration counting. Real
  stall/cycle detection is a named follow-up (§11, Phase 2). The point
  of this design is to build the slamming machinery *first*, with a
  pluggable trigger, so stall detection slots in later.
- **Automatic un-slamming / backtracking.** Phase 1 slams are sticky.
  We reserve an `unslam` hook (§7.5) for a later phase but do not
  implement release logic now.
- **Replacing the existing fixers.** `fixer.py`, `reduced_costs_fixer.py`,
  `relaxed_ph_fixer.py`, and `integer_relax_then_enforce.py` are
  untouched; the slammer coexists with them (§9).
- **Replacing the `SlamMin` / `SlamMax` spokes.** Those remain (§1); this
  is a different, complementary mechanism.
- **PySP config-file syntax compatibility.** We use a modern by-name file
  (§5). A WW-syntax importer is an optional later phase if there is
  demand to bring old files verbatim.

---

## 1. Current state in mpi-sppy — two senses of "slam"

The word "slam" is already used in the tree, so we are careful to keep
the two senses distinct.

**(a) The `SlamMin` / `SlamMax` *spokes*** (`mpisppy/cylinders/slam_heuristic.py`).
These are `InnerBoundNonantSpoke`s — *non-destructive incumbent finders*.
Each builds a candidate equal to the per-variable **min** (or **max**) of
the current nonant values across all scenarios, fixes *all* nonants to
that candidate, and evaluates the objective to report an inner bound.
They never perturb the hub. Limitations: globally all-to-min or
all-to-max (no per-variable control), **two-stage only** (explicit
`RuntimeError` on multistage), `rounding_bias` for integers. Wired via
`cfg_vanilla.slammin_spoke` / `slammax_spoke` and `config.slammin_args` /
`slammax_args`.

**(b) In-PH heuristic *fixing* extensions.** mpi-sppy already fixes
variables inside the PH loop:

| Extension | Fires on | Direction from | Per-var user spec? |
|---|---|---|---|
| `fixer.py` | converged within `th` for N iters | nearest / lb / ub (tuple) | yes — `id_fix_list_fct` callback |
| `reduced_costs_fixer.py` | \|rc\| ≥ quantile target, at bound | sign of reduced cost + sense | no (automatic) |
| `relaxed_ph_fixer.py` | relaxed-PH soln agrees with xbar at a bound | the relaxed value | no (automatic) |
| `integer_relax_then_enforce.py` | time / iters / near-convergence | n/a (re-enforces integrality) | no |

**Why none of these is slamming.** They all fix on *agreement /
convergence* — a variable is pinned because the evidence says it is
already settling there — and so they can derive direction
**automatically**. Slamming is the opposite regime: it forces
*non-converged* variables precisely *because* PH is not settling, and so
there is no signal to read direction from. That is exactly why slamming
needs **user-supplied** direction preferences. This design adds the
missing member of the in-PH-fixing family: a *stall-regime, user-directed*
fixer.

---

## 2. How PySP did it

PySP's WW PH extension read a configuration file with global directives
and per-variable override blocks. The slamming-relevant directives were,
in spirit:

- `CanSlamToLB`, `CanSlamToUB`, `CanSlamToAnywhere` — which directions a
  variable may be slammed in (`Anywhere` = round x-bar to the nearest
  integer).
- A slam **priority** per variable.
- `PH_Iters_Between_Slamming` and a "slam after iteration K" start point —
  the iteration-count trigger this design adopts for Phase 1.

When triggered (PySP triggered on detected cycling / stall), PySP slammed
variables according to these preferences to break the cycle.

We keep the *concepts* (per-variable can-slam, directions, priority; a
file; an iteration trigger) and modernize the *encoding* (§5) and the
*architecture* (§4).

---

## 3. The slam action — semantics

To **slam** a nonant `(ndn, i)` is to `fix()` its `Var` to a chosen value
**in every local scenario, on every rank**, and push the change to the
persistent solver. The chosen value depends on the variable's allowed
**direction**:

| Direction | Value the variable is fixed to |
|---|---|
| `lb` | `xvar.lb` |
| `ub` | `xvar.ub` |
| `nearest` | `lb` or `ub`, whichever the current x-bar is closer to |
| `anywhere` | x-bar, rounded to the nearest integer if the var is integer/binary (reusing the existing `rounding_bias`); plain x-bar otherwise |
| `min` | the **minimum** of the variable's value across all scenarios at this node |
| `max` | the **maximum** of the variable's value across all scenarios at this node |

`lb`, `ub`, `nearest`, `anywhere` use only data that is already identical
on every rank (the variable's own bounds; the globally-reduced x-bar), so
they are **coherent for free** — no communication. `min` and `max` need a
cross-scenario reduction (§7.4).

A direction is **applicable** only if it yields a finite value (e.g. `lb`
is skipped if the variable has no lower bound). The file may list an
ordered set of directions; the slammer uses the first applicable one
(§5).

---

## 4. Architecture — three decoupled layers

A single new extension, `mpisppy/extensions/slammer.py`, class `Slammer`,
in the in-PH-fixing family. It fires in `miditer` (like `fixer.py`). It is
organized as three independent layers so each can evolve separately:

```
   Trigger          Selection + preferences          Action
  (WHEN)      →        (WHO / WHERE)           →      (WHAT)
  should_slam()      directive map from file        slam(ndn_i, dir)
```

**Trigger (`when`)** — a predicate `should_slam(phiter) -> bool`. Phase 1:
iteration counting (§6). Phase 2: a stall detector implementing the same
predicate. The rest of the extension never changes.

**Preferences (`who`/`where`)** — a directive map built once from the file
(§5): for each nonant, `{can_slam, directions, priority}`. Only nonants
**present** in the file (matched by a `can_slam` rule) are eligible;
everything else is never slammed (the safe default that gives backward
compatibility).

**Action (`what`)** — when the trigger fires, choose the eligible nonant
of highest priority and slam it per §3 (§7).

**Applicable hubs.** The extension hooks (`pre_iter0`, `miditer`, …) are
driven from `PHBase`, so the Slammer runs under any `PHBase`-derived hub:
**PH, APH, Subgradient, FWPH**. The action layer additionally reads
`_mpisppy_model.xbars` (for `nearest` / `anywhere`) and the per-node
communicators `comms[ndn]` (for `min` / `max`), both of which the whole
`PHBase` family maintains. **L-shaped** (`SPBase`, no extension hooks) is
out. One caveat: the Phase-1 trigger counts iterations, and APH's
asynchronous notion of an "iteration" makes that cadence less meaningful
there — APH is a more natural fit for the Phase-2 stall trigger.

---

## 5. The directives file

`--slamming-directives-file PATH`. **CSV**, keyed **by nonant name**, with
wildcards. Chosen over JSON to match mpi-sppy's existing by-name nonant
CSVs (e.g. the multistage xhat file); JSON is trivial to add later if
wanted.

Columns:

| Column | Meaning |
|---|---|
| `name` | nonant name pattern matched against `xvar.name` (e.g. `DoBuild[*]`, `Reservoir*.spill[*]`). Shell-style `*`/`?` wildcards; `[` and `]` are **literal**, not character classes (so `DoBuild[*]` matches every `DoBuild[...]`). This is deliberately *not* `fnmatch`, whose bracket semantics would make `DoBuild[*]` fail to match an indexed Pyomo name. |
| `can_slam` | `1`/`0`. Optional; defaults to `1` when the row is present. `0` carves out an exception. |
| `directions` | `\|`-separated ordered list from `{lb, ub, nearest, anywhere, min, max}`. First *applicable* direction wins. |
| `priority` | float; the eligible nonant with the **largest** priority is slammed first. Ties broken by name (deterministic across ranks). |

Example:

```csv
name,can_slam,directions,priority
DoBuild[*],1,ub|lb,100
NumUnits[*],1,nearest,50
NumUnits[7],0,,0
Reservoir*.spill[*],1,min,10
```

A worked example translated from PySP's historical `wwph.suffixes` ships at
`examples/sizes/config/slamming_directives.csv`.

**Comments and quoting.** Blank lines and whole-line `#` comments are ignored
anywhere, including before the header. A multi-index name contains a comma
(e.g. `NumUnitsCutFirstStage[*,*]`), so it must be **quoted**
(`"NumUnitsCutFirstStage[*,*]"`) to survive the CSV split — standard CSV
quoting.

**Matching against scenario variables.** Each scenario is its own Pyomo
model, but the *same* nonant has the *same* `xvar.name` in every scenario
(e.g. `DoBuild[Seattle]`), so patterns are matched against `xvar.name` and
are scenario-independent and multistage-clean (one rule can span nodes).

**Precedence:** **last matching row wins** (gitignore-style). Write broad
defaults first and exceptions after. In the example, `NumUnits[7]` is
excluded even though `NumUnits[*]` would slam it.

**Coverage:** *not all nonants must appear.* A nonant matched by **no**
row — or whose last matching row has `can_slam=0` — is **never slammed**.
This is the requirement that only explicitly-specified variables can be
slammed, and it is what makes the feature backward compatible.

The file is parsed once and validated at startup: unknown direction
tokens, unparseable priorities, and patterns that match **zero** nonants
are reported (the last as a warning, gated on rank 0).

---

## 6. Trigger layer — Phase 1

Two options drive the iteration-count trigger:

- `--slam-start-iter K` — do not slam before hub iteration `K` (PySP's
  "slam after iter").
- `--iters-between-slams M` — once started, slam at most once every `M`
  iterations.

```python
def should_slam(self, phiter):
    if phiter < self.slam_start_iter:
        return False
    return (phiter - self.slam_start_iter) % self.iters_between_slams == 0
```

Purely a function of the iteration counter, so it is identical on every
rank with no communication.

**Phase 2 seam.** The stall detector will implement the same
`should_slam(phiter)` signature (reading the convergence history the hub
already tracks). Swapping it in is a one-line change in `Slammer`; nothing
else moves.

---

## 7. Action layer

### 7.1 Eligibility

A nonant is eligible to be slammed this event iff:
1. it is **present** in the directive map with `can_slam` true,
2. it is **not already fixed** (by the modeler, a fixer, or a prior slam),
   and
3. it is **not a surrogate nonant** (`all_surrogate_nonants`) — consistent
   with every other fixer.

There is deliberately **no convergence gate**: an eligible nonant is
slammed strictly by priority whether or not the scenarios already agree on
it. The user has already expressed intent by listing the variable with a
priority; second-guessing that with a dynamic "skip already-converged"
test adds a subtle behavior whose natural home is the Phase-2 stall
detector (which is, by definition, the layer that knows what is stuck).
Slamming an already-converged variable is harmless — it pins the variable
where it was settling and reduces the live problem, which is exactly what
`fixer.py` would have done. If a fixer is also active, no work is
duplicated: a slammed variable is `is_fixed()`, so the fixers skip it, and
a fixer-fixed variable fails eligibility test 2 here.

### 7.2 Selection

Among eligible nonants, pick the one with the **largest** `priority`
(ties → name order). Phase 1 slams **one nonant per event** — conservative,
PySP-faithful, and the trigger cadence (`--iters-between-slams`) controls
aggressiveness. A batched / fraction-based variant is a later option.

### 7.3 Determinism across ranks

Selection must use only globally-consistent inputs so that every rank
picks the *same* nonant with no communication:
- `priority` / `can_slam` come from the file (rank-identical), and
- the fixed mask is coherent because slamming/fixing is always applied to
  all scenarios on all ranks.

If any future input to selection could differ across ranks, the selection
result is reduced to agreement before acting; in Phase 1 none can.

### 7.4 `min` / `max` direction — on-demand reduction

`min` / `max` need the extremum of the selected variable's value across
all scenarios at its node. Because the selected `(ndn, i)` is identical on
every rank (§7.3), all ranks symmetrically:
1. take the local min/max of `xvar.value` over local scenarios at node
   `ndn`, then
2. `self.opt.comms[ndn].Allreduce(..., op=MPI.MIN / MPI.MAX)`

reusing the same per-node communicators x-bar uses. The cost (one small
collective) is paid **only** when a `min`/`max`-directed slam actually
fires — never on other iterations, never for other directions.

### 7.5 Applying and recording the slam

For the chosen `(ndn, i)` and value `v`, in every local scenario:
`xvar.fix(v)`; if the solver is persistent, `solver.update_var(xvar)`.
Record `self._slammed[(ndn, i)] = v`.

`_slammed` exists for reporting and for the **`unslam` hook** reserved for
Phase 2 (releasing a bad slam). Phase 1 never calls it; the seam is left
so the stall/backtracking phase need not refactor the action layer.

### 7.6 Reporting

On rank 0 (gated), log each slam (`name → value`, direction, priority) and
a running count, mirroring the fixers' progress output.

---

## 8. Config / CLI surface and backward compatibility

New options (added in `config.py`, exposed in `generic_cylinders.py`):

| Option | Meaning |
|---|---|
| `--slamming-directives-file PATH` | the file (§5); **its presence activates the Slammer** |
| `--slam-start-iter K` | first iteration at which slamming may occur |
| `--iters-between-slams M` | cadence once started |

**Activation & compatibility contract:**
- The `Slammer` extension is added **iff `--slamming-directives-file` is
  given.**
- Supplying any *other* new slam option **without** the file is a **hard
  error** ("slamming options require `--slamming-directives-file`").
- With no slamming options, the extension is never constructed and
  behavior is identical to today.
- The existing `--slammin` / `--slammax` spoke flags are **untouched** and
  independent; the new options are named so they cannot collide.

---

## 9. Plumbing

- **`config.py`:** a `slamming_args()` method adding the options above
  (following the pattern of the other `*_args` methods); reuse the
  existing `rounding_bias` option for `anywhere` integer rounding.
- **`cfg_vanilla.py`:** in the hub builder, when
  `cfg.slamming_directives_file` is set, append `Slammer` to the hub's
  extension list (via the existing `MultiExtension` mechanism, alongside
  any other extensions).
- **`generic_cylinders.py`:** register `slamming_args`; wire activation.
- A small farmer/sizes example directives file under `examples/` plus a
  doc page so the feature is discoverable.

The directional-fix-and-update inner loop overlaps with
`reduced_costs_fixer` / `relaxed_ph_fixer`; if it is clean to do so, factor
a tiny shared helper (`fix nonant to value across all scenarios + persistent
update`). Not required for correctness.

---

## 10. Interaction with existing features

- **`fixer.py` and the other fixers:** coexist. A slammed variable is
  `is_fixed()`, so every fixer's `is_fixed()` guard skips it; conversely a
  fixer-fixed variable fails the slammer's eligibility test (§7.1). No
  double-fixing.
- **Modeler-fixed and surrogate nonants:** never slammed (§7.1), matching
  the variable-probability surrogate-nonant contract.
- **`SlamMin`/`SlamMax` spokes:** orthogonal. The spokes propose
  incumbents; the slammer alters the hub search. A run may use both.
- **Bundling:** the slammer operates on nonant indices, which bundling
  preserves; reporting counts may need the same per-bundle care the
  fixers take.

---

## 11. Phasing (each its own review-sized PR)

**Phase 1 (this design):** the slammer mechanism — directives file
(by-name + wildcards), the five+`nearest` directions, one-slam-per-event
selection, iteration-count trigger, sticky slams, multistage, full
backward compatibility. Plus example + docs + tests.

**Phase 2:** a real **stall/cycle detector** implementing
`should_slam(phiter)`, dropped into the trigger seam; and the **`unslam`**
release logic the Phase 1 hook anticipates.

**Phase 3 (optional):** batched / fraction-based slamming; PySP WW-syntax
file importer; JSON file format.

---

## 12. Testing

- **Parser/unit:** wildcard matching, last-match-wins precedence,
  `can_slam=0` exceptions, unmatched ⇒ not slammable, bad-token /
  zero-match validation.
- **Action/unit:** each direction (`lb`/`ub`/`nearest`/`anywhere`/`min`/
  `max`) fixes to the expected value; `anywhere` rounding honors
  `rounding_bias`; infinite-bound directions are skipped.
- **Determinism:** same nonant selected on every rank (run under
  `mpiexec`).
- **Backward compatibility:** an existing test invocation with no slamming
  options produces identical results; supplying a slam option without the
  file errors.
- **Integration:** a small MIP (sizes / a netdes-like model) where
  slamming demonstrably forces convergence the plain run does not reach in
  the iteration budget.
- Wire the new `mpisppy/tests/test_slammer.py` into `run_coverage.bash`
  **and** `test_pr_and_main.yml` in the same commit.

---

## 13. Resolved decisions

Settled in review (DLW, 2026-06-17); recorded here for provenance.

1. **`priority` direction** — larger priority is slammed first. *Confirmed.*
2. **`nearest` token** — kept as a convenience for "either bound, pick the
   closer." *Confirmed.*
3. **Convergence gate** — the proposed `--slam-only-unconverged` option is
   **dropped**. Phase 1 slams strictly by priority with no convergence
   test (§7.1); convergence-aware targeting, if wanted, belongs in the
   Phase-2 stall detector.
4. **One slam per event** in Phase 1; batched / priority-band slamming is a
   later option (§7.2).
