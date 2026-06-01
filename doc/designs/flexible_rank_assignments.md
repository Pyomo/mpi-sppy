# Flexible Rank Assignments for Cylinders

Status: Design Document (Draft — second pass)
Date: 2026-05-24

**Key decision up front.**  Flexible ranks are added as a
*gated-additive* feature, not a rework of the path every run exercises.
At equal rank ratios (today's only case) the existing `strata_comm`
windows and single-source reader run unchanged; the new `fullcomm` +
overlap-map machinery is reached only when a rank ratio differs from
1.0.  The rationale, trade-offs, and rollout workflow are in
§Development and rollout strategy.

### Terminology

- **Window:** An MPI one-sided ("RMA") addressing scope, created
  collectively on a communicator via `MPI_Win_Allocate`.  The window
  is a handle, not a contiguous block of memory — it makes each
  participant's local buffer mutually addressable via `MPI_Get` /
  `MPI_Put`.
- **Buffer:** A rank's local memory region exposed through a window.
  Holds that rank's published fields (`NONANTS_VALS`, `DUALS`,
  `BEST_XHAT`, ...) packed according to the `buffer_layout` in
  `spwindow.py`.
- **Local-sized field:** A field whose buffer length is proportional
  to the rank's *local* scenario count (`NONANTS_VALS`, `DUALS`,
  `BEST_XHAT`, `RECENT_XHATS`, `CROSS_SCENARIO_COST`, ...).  These are
  the fields affected by unequal rank counts.
- **Global-sized field:** A field whose buffer length is proportional
  to the *total* nonant count and is identical on every rank
  (`NONANT_LOWER_BOUNDS`, `NONANT_UPPER_BOUNDS`,
  `EXPECTED_REDUCED_COST`).  Unaffected by rank asymmetry.

### Motivation

Currently, `WheelSpinner` enforces that every cylinder (hub and all
spokes) receives the same number of MPI ranks and the same scenario
distribution.  This is wasteful when different cylinders have different
computational requirements.  For example:

- The PH hub solves every subproblem at every iteration and benefits
  from many ranks.
- A Lagrangian spoke also solves subproblems but may converge with
  fewer iterations, so it could use fewer ranks (each handling more
  scenarios).
- An xhat shuffle spoke is lightweight and may need only one or two
  ranks.

Allowing different rank counts per cylinder would let users allocate
MPI resources more efficiently, potentially reducing total wall-clock
time for the same number of processors.

### User Interface

The user would specify a **target ratio** for each spoke relative to the
hub.  The hub always serves as the reference (ratio 1.0).  Example:
```
mpiexec -np 14 python -m mpi4py mpisppy/generic_cylinders.py \
    --module-name farmer --num-scens 100 \
    --solver-name gurobi --default-rho 1 --max-iterations 50 \
    --lagrangian --lagrangian-rank-ratio 0.5 \
    --xhatshuffle --xhatshuffle-rank-ratio 0.25
```

With 14 total ranks and ratios hub:1.0, lagrangian:0.5, xhat:0.25,
the system would allocate ranks proportionally: 8 for the hub, 4 for the
Lagrangian spoke, 2 for the xhat spoke.

Interface decisions:

- **Where the ratio lives:** per-spoke-type in `Config` (e.g.,
  `--lagrangian-rank-ratio`, `--xhatshuffle-rank-ratio`).  Matches
  mpi-sppy's modern Config style and gets CLI parsing for free.
- **Ratios only, no explicit counts:** ratios are portable across
  total rank counts; defer any `--spoke-ranks` count override
  until a concrete need surfaces.
- **Uneven division:** apportion ranks by the largest-remainder
  (Hare quota) method, then enforce a minimum of 1 rank per
  cylinder.  This guarantees the per-cylinder counts sum to `np`
  exactly and that every cylinder runs.  Do not warn; just
  `global_toc` the final per-cylinder rank allocation on rank 0
  so the actual numbers used appear in the run log.

  *Algorithm.* Let `C` be the number of cylinders, `np` the
  total rank count, and `r_i` the configured ratio for cylinder
  `i` (hub has `r_hub = 1.0`).

  1. Compute each cylinder's real-valued share:
     `alloc_i = (r_i / sum_j r_j) * np`.
  2. Floor each: `integer_i = floor(alloc_i)`.
  3. Distribute the leftover `np - sum_i integer_i` ranks one at
     a time to the cylinders with the largest fractional remainder
     `(alloc_i - integer_i)`, breaking ties by cylinder
     declaration order.  After this step
     `sum_i integer_i == np` exactly.
  4. Min-of-1 pass: while any cylinder has `integer_i == 0`,
     decrement the cylinder with the largest current allocation
     by 1 and increment the zero-cylinder by 1.  Repeat until
     every cylinder has at least 1 rank.
  5. If `C > np`, min-of-1 is infeasible — error out at startup
     with a message listing the requested cylinders and `np`.

  This preserves the requested ratio as closely as integer
  arithmetic allows while guaranteeing every cylinder gets at
  least one rank.


### Field Taxonomy

This is the section that the first pass got wrong, so it is grounded
in the actual write and consume sites.  Every local-sized field falls
into one of these categories; the category determines how it is
handled under unequal rank counts.

#### Category 1 — genuinely per-scenario *distinct* data

Buffers in this category hold one independent value (or block) per
local scenario, and consumers rely on the per-scenario differences.
Multi-source assembly across ranks is the correct primitive.

- **`NONANTS_VALS`** (hub → spokes).  `Hub.send_nonants`
  (`hub.py`) loops over `local_scenarios` and writes `xvar._value`
  for each scenario's `nonant_indices` — each scenario's *own*
  subproblem solution.  Consumers depend on the differences:
    - `slam_heuristic.py` reshapes the buffer to
      `(num_scen, num_vars)` and takes min/max **across scenarios**;
    - `cross_scen_spoke.py` sums values **across scenarios**;
    - the xhat loopers (`xhatlooper`, `xhatshufflelooper`,
      `xhatspecific`) try **each scenario's** first-stage solution as
      a candidate xhat.

- **`RELAXED_NONANTS_VALS`** (relaxed-PH spoke → `relaxed_ph_fixer`).
  `RelaxedPHSpoke` inherits the base `send_nonants`, so this is the
  same per-scenario layout as `NONANTS_VALS`, carrying the relaxed
  (LP) iterates.

- **`DUALS`** (hub → spokes).  Each scenario carries its own
  multiplier `W_s`.  Genuinely per-scenario; multi-source assembly is
  correct.  Unlike the nonant fields, `DUALS` requires **strict
  coherence** — see §Coherence.

- **`CROSS_SCENARIO_COST`** (cross-scenario cut source → cut spoke).
  Per-scenario cost data.

#### Category 2 — per-scenario *layout*, NAC-redundant first-stage

Buffers in this category are laid out per scenario, but the nonant
portion is an incumbent first-stage decision that is **identical across all
scenarios sharing a node** (non-anticipativity holds by construction
because the first stage is *fixed* across scenarios when the candidate
is evaluated).  A genuinely per-scenario scalar cost rides alongside.

- **`BEST_XHAT`** / **`RECENT_XHATS`** (inner-bound spoke →
  hub / FWPH).  `send_best_xhat` (`spoke.py`) writes, per local
  scenario, `[best_solution_cache nonant values, inner_bound]`.
  Because the candidate fixes the first stage across scenarios, the
  nonant values are NAC-redundant (one value per node, copied into
  each scenario's slot), while `inner_bound` is genuinely
  per-scenario.  `RECENT_XHATS` is a circular buffer of such entries.

The split matters for asymmetric ranks: the **first-stage portion
must stay NAC-consistent**, so it cannot be assembled from ranks at
different iterations; the **cost portion** is ordinary per-scenario
data assembled like Category 1.  See §Coherence for how this is
handled without canonicalization.

#### Category 3 — global-sized fields

`NONANT_LOWER_BOUNDS`, `NONANT_UPPER_BOUNDS`, `EXPECTED_REDUCED_COST`.
One canonical vector per sender, identical on every rank, sized by the
*total* nonant count.  No partitioning, nothing to assemble — a
no-op under rank asymmetry.  Their receiver-side application is
monotonic/idempotent (a bound is applied only when it tightens), so
staleness is safe.

#### Category 4 — scalars

`OBJECTIVE_INNER_BOUND`, `OBJECTIVE_OUTER_BOUND`,
`BEST_OBJECTIVE_BOUNDS`, `SHUTDOWN`.  One value (or a tiny fixed-size
record) per cylinder, written on rank 0.  Unaffected by rank
asymmetry; staleness yields a slightly later termination check, never
an incorrect one.


### Current Architecture

This section describes the pieces that would need to change.

#### Rank Partitioning (`spin_the_wheel.py`)

`WheelSpinner` requires `n_proc % n_spcomms == 0` and creates two
communicators via `MPI_Comm_split`:
```
strata_comm  = fullcomm.Split(color=global_rank // n_spcomms)
cylinder_comm = fullcomm.Split(color=global_rank % n_spcomms)
```

`strata_comm` groups rank *i* from each cylinder together (the
inter-cylinder communication channel).  `cylinder_comm` groups all
ranks within one cylinder (the intra-cylinder channel).

With equal rank counts, there is a clean 1-to-1 correspondence:
`strata_comm` rank 0 is always the hub, rank 1 is always spoke 1, etc.

#### Scenario Distribution (`spbase.py`)

`_calculate_scenario_ranks()` distributes scenarios across
`self.n_proc` ranks (the cylinder's rank count) using contiguous
blocks.  All cylinders currently have the same `n_proc`, so rank *i*
in every cylinder gets the same scenarios.  The function already works
with any `n_proc`; each cylinder just calls it with its own rank
count.

#### Buffer System (`spwindow.py`, `spcommunicator.py`)

Each rank creates an MPI RMA window with per-field buffers.  Buffer
sizes for local-sized fields are proportional to the rank's local
scenario count; global-sized fields are the same on every rank.

On receive, `_validate_recv_field()` checks that the remote buffer
size matches the local expectation.  This check assumes both sides
have the same local scenario count and must be relaxed (§Impact).

#### Communication Patterns

All inter-cylinder communication uses one-sided MPI (RMA) through
`SPWindow`.  The local-sized fields (Categories 1 and 2 above) are the
ones affected by unequal rank counts; the global-sized and scalar
fields are not.


### Design: Multi-Rank Mapping

The core idea: when a rank in one cylinder needs a local-sized field
from another cylinder with a different rank count, it reads from
multiple remote ranks and assembles the result (or reads from one
remote rank and extracts a subset).  **The same mechanism applies to
every local-sized field** — there is no special per-field layout.
What differs per field is only the *coherence policy* applied to the
reads (§Coherence).

#### Overlap Maps

At startup, each rank computes a static mapping for each peer cylinder:
which remote ranks have scenarios that overlap with its own local
scenarios, and at what offsets within those remote buffers.

The existing `scen_names_to_ranks(n_proc)` function already computes
scenario-to-rank mappings.  We would call it once per cylinder's rank
count to get each cylinder's distribution, then compute pairwise
overlaps.

For example, with a 4-rank hub and a 2-rank spoke, both handling 10
scenarios (the per-rank split below is illustrative — the real split is
whatever `scen_names_to_ranks` produces — and is chosen so that a hub
rank straddles the spoke's boundary):

========  ==================  ==================
          Hub (4 ranks)       Spoke (2 ranks)
========  ==================  ==================
Rank 0    scen0, scen1        scen0--scen4
Rank 1    scen2, scen3        scen5--scen9
Rank 2    scen4, scen5
Rank 3    scen6--scen9
========  ==================  ==================

Hub rank 0 reads from spoke rank 0 (which holds scen0--scen4),
extracting only the portion for scen0--scen1.  Hub rank 2 holds
scen4--scen5, which *straddles* the spoke's split: scen4 lives in spoke
rank 0 (at offset 4) and scen5 in spoke rank 1 (at offset 0), so hub
rank 2 reads one segment from each.  Hub rank 3's scen6--scen9 sit at
offsets 1--4 within spoke rank 1's buffer (which starts at scen5).

Taking one nonant per scenario for simplicity, the overlap map (in
nonant units) for hub rank 0 reading from the spoke would be:
```
[(spoke_rank=0, remote_offset=0, local_offset=0, count=2)]
```

For hub rank 2 (the straddle — two segments):
```
[(spoke_rank=0, remote_offset=4, local_offset=0, count=1),
 (spoke_rank=1, remote_offset=0, local_offset=1, count=1)]
```

For hub rank 3:
```
[(spoke_rank=1, remote_offset=1, local_offset=0, count=4)]
```

These maps are computed once at startup and reused for every
communication call.  The data structure could be:
```
@dataclass
class OverlapSegment:
    remote_rank: int       # rank in peer cylinder
    remote_offset: int     # nonant offset within remote buffer
    local_offset: int      # nonant offset within local buffer
    count: int             # number of nonants in this segment

    # Per (peer_cylinder, field): list of OverlapSegments
    overlap_map: dict[int, list[OverlapSegment]]
```

Note: offsets are in nonant units (not scenario units) because
different scenarios could have different numbers of nonants in
multi-stage problems (though for two-stage problems they are uniform).


#### Window Topology Options

Of the options below we settle on Option D, though only as the topology
for the unequal-rank path; it is similar to Option A.  See §Development
and rollout strategy for why it is added alongside the existing
`strata_comm` rather than replacing it.

**Option A: Single global window on `fullcomm`**

Every rank publishes its buffers in one shared window.  Readers address
remote ranks by their global rank.  The `strata_buffer_layouts`
(currently exchanged via `strata_comm.allgather`) would instead be
exchanged on `fullcomm` or a dedicated intercommunicator.

Pros: simple conceptually; any rank can read from any other rank; one
`MPI_Win_create`.  Cons: large window; all ranks participate even if
they never communicate.

**Option B: Per-cylinder-pair intercommunicators with separate windows**

For each communicating pair, create an `MPI_Intercomm` and a window on
it.  Pros: smaller windows, less contention.  Cons: number of windows
grows with communicating pairs (O(n^2) worst case with spoke-to-spoke);
more setup complexity; intercomm semantics are tricky.

**Option C: Keep strata_comm but make it asymmetric**

Group ranks from different-sized cylinders together, with a smaller
cylinder's rank appearing in multiple strata groups.  This does not
work with `MPI_Comm_split` (a rank appears once per split) and would
require `MPI_Comm_create_group` or manual addressing.  Rejected.

**Option D: Replace strata_comm with direct RMA addressing**

Abandon `strata_comm` entirely.  Use `fullcomm` for the window, and
have each rank directly address remote ranks by their global rank.  The
overlap maps provide the addressing information.  This is Option A with
the explicit framing that `strata_comm` is removed rather than
modified.

Pros: clean break from the 1-to-1 assumption; most flexible; single
window creation.  Cons:

- Requires rewriting `SPCommunicator` and `SPWindow` to work without
  `strata_comm`.
- The buffer layout exchange (currently `strata_comm.allgather`) needs
  a replacement.  The cheapest option is to compute layouts locally on
  every rank: per-cylinder rank count, the output of
  `_calculate_scenario_ranks`, and field-registration order are all
  static and deterministic at startup, so no communication is
  required.  If a runtime exchange is genuinely needed (e.g., dynamic
  field registration), prefer a two-level scheme —
  `cylinder_comm.allgather` followed by a small cross-cylinder gather
  over one anchor rank per cylinder — rather than a single
  `fullcomm.allgather`, which scales worse at N=thousands.

  *What the communication-layer cut actually ships, and the release
  gate.*  The first cut uses a single `fullcomm.allgather` for the
  unequal-rank layout exchange.  This is deliberate but interim: it is
  effectively zero new code (the existing `SPWindow` exchange run on
  `fullcomm` instead of `strata_comm`), it is a one-time *startup* cost
  on a cold path — not the RMA hot path — and at development/test scale
  (a handful of ranks) it is free.  It is **not** the end state.
  Because total rank counts in the thousands are a real operating
  regime here, the O(N) startup allgather and its O(N)-per-rank layout
  storage must be replaced by the two-level (or local-compute) scheme
  **before flexible ranks is documented or recommended for production
  use** (it can land on `main` before then, since it is inert until a
  non-default ratio).  That replacement is its own focused change — it
  touches only how
  `strata_buffer_layouts` is populated at startup, not the multi-source
  reader — so it is tracked as a release-gate item rather than folded
  into the feature phases, letting it be reviewed and scale-tested on
  its own.  See §Gate reliance on the feature with an MPI CI matrix.

  *Lock granularity.*  `MPI_Win_lock(rank=target, ...)` is per-
  target-rank in the MPI spec, not per-window — a writer's exclusive
  lock on its own buffer does not block readers fetching from a
  different rank's buffer.  Cross-cylinder serialization is a non-issue
  at the spec level.  If a workload ever showed measurable contention,
  splitting hot cylinder pairs into their own window (Option B for
  those pairs) is an incremental fix that does not require abandoning
  Option D.

**Recommendation:** Option D, *as the topology for the unequal-rank
path only*.  A given run uses exactly one topology, chosen at startup
from the rank ratios:

- *Equal-rank run* (all ratios 1.0): create `strata_comm` and
  `cylinder_comm`, and put the window on `strata_comm` -- exactly as
  today.
- *Unequal-rank run* (any ratio differs from 1.0): **`strata_comm` is
  not created at all.**  The strata grouping (`color =
  global_rank // n_spcomms`, "rank *i* of every cylinder") is undefined
  when cylinders have different rank counts -- that is the whole reason
  for Option D.  Create `cylinder_comm` only (each cylinder is a
  contiguous block of apportioned ranks, so the intra-cylinder grouping
  is still well-defined), put the window on `fullcomm`, and have each
  rank address peers by global rank via the overlap maps.  Each rank
  takes its cylinder index from the apportionment rather than from
  `strata_rank`.

So `strata_comm` is *retained in the codebase* -- the equal-rank path
still uses it unchanged -- but it is *not created in an unequal-rank
run*.  This is additive, not a replacement: the textbook "abandon
`strata_comm`" framing in the Option D description above would remove it
for *all* runs, which this design does not do.  See §Development and
rollout strategy.


#### Multi-Source Read Assembly

The current `get_receive_buffer()` reads one contiguous buffer from
one remote rank.  With multi-rank mapping, it would:

1. For each segment in the overlap map, issue an `MPI_Get` for that
   segment from the appropriate remote rank.
2. Place each segment at the correct offset in the local receive
   buffer.
3. Return the assembled buffer.

Schematically:
```
def get_receive_buffer_multi(self, field, peer_cylinder):
    local_buf = np.empty(self.local_field_length(field))
    for seg in self.overlap_map[peer_cylinder]:
        remote_buf = self.window.get(
            rank=seg.remote_rank,
            field=field,
            offset=seg.remote_offset,
            count=seg.count,
        )
        local_buf[seg.local_offset : seg.local_offset + seg.count] = remote_buf
    return local_buf
```

This requires `SPWindow.get()` to support partial reads (offset +
count within a field), which is straightforward with `MPI_Get`
displacement parameters.

For the common case where both cylinders have the same rank count, the
overlap map has exactly one segment per peer rank and the offsets are
identity — so the behavior degenerates to the current code.


### Coherence

The current system appends a `write_id` integer to each buffer; a
sender increments it on write, and a receiver checks whether it changed
to decide if data is "new."  With multi-source reads, a receiver reads
from multiple remote ranks that may not have updated at the same time
(the system is asynchronous).  The question — must all contributing
ranks share a `write_id` before the data is accepted? — is answered
**per field**, by category.

#### Strict coherence

Read all `write_id` values for the contributing segments first (cheap:
one int per segment); accept the data only if they match; otherwise
retry later.

- **`DUALS`.**  The dual normalization `sum_s p_s W_s = 0` holds only
  within a single iteration.  Assembling `W` from mixed iterations
  gives `sum_s p_s W_s != 0`, so the Lagrangian value is no longer a
  valid bound.  Strict coherence is required for bound validity.  This
  is cheap in practice: the hub is synchronous PH, so all hub ranks
  finish an iteration before writing `DUALS`, and their `write_id`
  values naturally agree.

- **`BEST_XHAT` / `RECENT_XHATS` (first-stage portion).**  The
  first-stage values must be NAC-consistent.  Two ways to guarantee
  that without canonicalization, both exploiting that the first-stage
  portion is *identical across scenarios*:

  - **Two-stage:** every scenario carries the same first-stage vector,
    so the receiver can read the first-stage portion from a **single**
    remote rank (no cross-rank assembly, hence no coherence question).
    Only the per-scenario `inner_bound` cost is assembled per-scenario
    (Category-1 style).

  - **Multistage:** a node's first-stage values are shared only by the
    scenarios passing through that node, which may be spread across
    ranks.  Source each node's first-stage portion from a single rank
    that holds a scenario through that node (deterministic from the
    scenario-to-rank map and the tree).  This is per-node single-source
    sourcing, computed at startup alongside the overlap maps — not a
    layout change and not multi-source assembly of the first stage.

  The sender side is synchronous in practice (a spoke writes
  `BEST_XHAT` only after its `cylinder_comm.Allreduce` feasibility
  verdict), so `write_id` values agree and the single-source reads see
  a coherent snapshot.

#### Relaxed coherence

Accept data from each source rank independently; track `write_id` per
source rank.

- **`NONANTS_VALS` / `RELAXED_NONANTS_VALS`.**  These are per-scenario
  iterates with no cross-scenario invariant on the wire.  The xhat
  loopers and `slam_heuristic` re-evaluate any candidate fresh
  (fixing the first stage and solving), so a stale or mixed-iteration
  per-scenario value just means evaluating a slightly older candidate
  — always honest, never invalid.

- **Global-sized bounds and scalars** (Categories 3 and 4).  Monotone,
  idempotent application; staleness is safe.

- **`CROSS_SCENARIO_COST`** (and `NONANTS_VALS` *as read by*
  `CrossScenarioCutSpoke`).  These build cross-scenario (Benders /
  L-shaped) cuts, and the relevant property is that **a Benders cut
  built at any first-stage point `x̂` is a valid global underestimator
  of the recourse value `Q(x)` for all `x`** — `x̂` only sets where the
  cut is tight, never whether it is valid.  In the code, cut validity
  comes entirely from `CrossScenarioCutSpoke.make_cut()` *re-solving
  the real subproblems* inside `opt.root.bender.generate_cut()` (the
  generator was handed `create_subproblem` callbacks); the received
  `NONANTS_VALS` / `CROSS_SCENARIO_COST` only steer (1) *which* point
  the cut is generated at (the "farthest `xhat`" heuristic and the
  `global_xbar` average) and (2) *trigger conditions* (the `eta_lb`
  violation test, and whether a generated cut is worth adding).  The
  `eta_lb` cuts (`eta >= LB`) and feasibility cuts are valid by
  construction too.  So assembling these fields from ranks at mixed
  `write_id`s can only cause a cut to be generated at a stale or
  blended candidate point — still a valid cut.  The cost is slower
  convergence (less useful cut placement), never an invalid cut or a
  wrong bound.  Hence: **relaxed coherence.**  (`CrossScenarioExtension`
  is two-stage only, so there is no multistage cut interaction to
  worry about.)

#### Recommendation

Per-field strict-vs-relaxed policy (the table above), implemented as
two code paths in the multi-source reader.  No cylinder-wide iteration
counter (it would add synchronization the async design avoids and is
unnecessary given the per-field analysis).


### Impact on Existing Components

#### `spin_the_wheel.py`

- Keep the `n_proc % n_spcomms == 0` check on the equal-rank path; the
  unequal-rank path bypasses it, since apportionment (§User Interface)
  removes the requirement that `n_proc` divide evenly by cylinder count.
- On the unequal-rank path, replace the uniform `MPI_Comm_split` with
  the apportionment algorithm (§User Interface) that respects
  per-cylinder rank counts.  The equal-rank path keeps the uniform
  split unchanged.
- `cylinder_comm` creation still uses `MPI_Comm_split` (all ranks in
  the same cylinder get the same color).
- `strata_comm` is **retained in the codebase** for the equal-rank
  path: when ranks are equal across cylinders, `strata_comm` and its
  per-strata windows are created and used exactly as today.  In an
  unequal-rank run `strata_comm` is **not created** -- the strata
  grouping is undefined for cylinders of different sizes -- and the
  window is built on `fullcomm` instead (Option D's addressing).  The
  choice is made once at startup; a run creates one or the other, never
  both.  `cylinder_comm` is created in both cases.  See §Development and
  rollout strategy.
- In the unequal-rank path, indexing that assumes a fixed `strata_rank`
  meaning does not apply; each rank addresses peers by global rank via
  the overlap maps, taking its cylinder index from the apportionment.
  The equal-rank path's `strata_rank` indexing is unchanged.

#### `spbase.py`

- `_calculate_scenario_ranks()` already works with any `n_proc`.
  No change needed.

#### `spwindow.py`

- `SPWindow.get()` must support partial reads (offset + count) -- an
  additive capability the equal-rank path does not exercise.
- For the unequal-rank path, the buffer layout must be exchanged over a
  communicator that spans all cylinders, or (preferred) computed
  locally from the static rank-count/scenario-map data.  The
  equal-rank path keeps its existing `strata_comm` allgather.
- Buffer validation must account for asymmetric sizes on the
  multi-source path: a receiver's expected size may differ from the
  sender's, and that is fine as long as each requested segment fits
  within the sender's buffer.
- No field-length or layout changes: every field keeps its current
  per-scenario / global / scalar sizing.  (Canonicalization is *not*
  part of this design.)

#### `spcommunicator.py`

- `register_receive_fields()` builds overlap maps for the unequal-rank
  path; the equal-rank path retains the 1-to-1 rank correspondence.
- `get_receive_buffer()` gains a multi-source assembly path (with the
  per-field coherence policy applied), selected when ratios differ; the
  single-source path is unchanged at equal ranks.
- `put_send_buffer()` is unchanged (each rank writes its own data).
- `_validate_recv_field()` gains per-segment validation for the
  multi-source path; equal-rank validation is unchanged.
- The `synchronize` parameter's `cylinder_comm.Barrier()` / `Allreduce`
  still work within a cylinder.

#### `hub.py` and `spoke.py`

- Packing (send side) is unchanged: each rank packs its own local
  scenarios linearly.
- Unpacking (receive side) uses the overlap map to place multi-source
  data at the right local offsets.
- For `BEST_XHAT` / `RECENT_XHATS`, the receiver reads the first-stage
  portion via single-source (per-node) sourcing and the cost portion
  via per-scenario assembly.
- Bound and scalar communication is unaffected.

#### `cfg_vanilla.py` and `config.py`

- Add per-spoke rank-ratio configuration.
- `shared_options()` may need to carry the rank-ratio information so
  `SPCommunicator` can compute overlap maps at init.

#### `generic_cylinders.py`

- After computing rank ratios, pass them to `WheelSpinner` via the
  hub/spoke dicts.  Default ratio is 1.0 for all cylinders (backward
  compatible).


### Phased Implementation Plan

There is **no canonicalization phase** in this design.  The rename that
the first pass bundled into "Phase 0" has already landed separately.

**Phase 1: Infrastructure**

- Implement overlap-map computation (per-cylinder rank counts +
  scenario lists → `OverlapSegment` lists).
- Add partial-read support to `SPWindow.get()`.
- Add rank-ratio fields to spoke dicts and `WheelSpinner`, plus the
  apportionment algorithm.
- Modify rank partitioning in `WheelSpinner` for unequal cylinder
  sizes.
- Unit tests for the apportionment algorithm and for overlap maps with
  various rank-count combinations (including the equal-rank identity
  case).

**Phase 2: Communication layer** (additive; reached only when a ratio
differs from 1.0)

- Add the `fullcomm.allgather` layout exchange for the unequal-rank
  path (Option D's addressing), *alongside* the existing
  `strata_comm`-based exchange, which the equal-rank path keeps using.
  This is the interim exchange; the scalable replacement is a release
  gate, not a feature phase (see the Option D layout-exchange note and
  §Gate reliance on the feature with an MPI CI matrix).
- Implement multi-source `get_receive_buffer()` using overlap maps, as
  a path taken only under non-default ratios; the single-source reader
  is unchanged for the equal-rank case.
- Add per-segment validation for the multi-source path; leave
  `_validate_recv_field()` as-is on the equal-rank path.
- Test the relaxed-coherence per-scenario fields first (`NONANTS_VALS`)
  with a simple case: hub with 4 ranks, Lagrangian spoke with 2 ranks.

**Phase 3: Coherence policy and integration**

- Implement the per-field strict-vs-relaxed reader paths.
- Add the strict `write_id` check for `DUALS`.
- Add single-source first-stage sourcing for `BEST_XHAT` /
  `RECENT_XHATS` in the **two-stage** case; cost portion per-scenario.
- Wire `cross_scen` with relaxed coherence (resolved — see §Coherence)
  and make `CrossScenarioCutSpoke` accept a multi-source `NONANTS_VALS`
  / `CROSS_SCENARIO_COST` (it currently asserts a single source rank).
- Wire up the rank-ratio CLI options end to end.
- Test all spoke types with various ratios at two stages.
- Performance check: does 8+4+2 beat 5+5+5 for a representative
  problem?

**Phase 4: Multistage and FWPH**

- Per-node single-source first-stage sourcing for `BEST_XHAT` /
  `RECENT_XHATS` in the **multistage** case.
- FWPH reading `RECENT_XHATS` from an InnerBoundSpoke with a different
  rank count.  This is the most complex consumer of the xhat fields and
  warrants targeted testing.

**Phase 5: APH support**

- Verify APH (asynchronous PH) works with the relaxed-coherence model.
- The strict-coherence fields already retry on `write_id` mismatch, so
  they should compose with async senders; verify and add tests.


### Development and rollout strategy

These changes touch a low-level, implementation-sensitive corner of the
stack -- MPI one-sided (RMA) communication through mpi4py -- where
behavior varies across MPI implementations (OpenMPI, MPICH, vendor
builds) and versions, and where passive-target RMA (`Lock` / `Get` /
`Put`) is the flakiest part of the spec.  Reviews of this material are
necessarily slow, and a regression in the communicator layer would
affect *every* run, not just flexible-rank runs.  The guiding principle
is therefore to **keep the equal-rank path off the table entirely**: the
common case that every current user exercises must execute the same code
after this feature lands as before it.

**Gated-additive, not a live-path rework.**  An earlier draft proposed
replacing `strata_comm` with a single `fullcomm` window (Option D) for
*all* runs, introduced via branch-by-abstraction behind a runtime flag
that defaulted to the old path and was to be removed once the new path
proved out.  We reject that here.  A replacement -- even one staged
behind a flag -- rewrites the path every run hits, and the temporary
"two implementations behind a flag, delete the old one later" state is
exactly the kind of clutter that becomes permanent: removing the proven
path is always scarier than keeping it, the two paths drift because tests
rarely exercise both equally, and the toggle outlives its purpose.

Instead, the new `fullcomm` + overlap-map machinery is **purely
additive and gated by the feature itself**.  The rank-ratio
configuration *is* the gate -- no separate runtime flag:

- When all ratios are 1.0 (today's only possibility), the system builds
  the existing per-`strata_comm` windows and uses the existing
  single-source reader, unchanged.  Current users execute the same bytes
  they execute now.
- When any ratio differs from 1.0, `make_windows()` builds the
  `fullcomm` window (Option D's topology) and the multi-source,
  overlap-map reader is used.  A run uses exactly one topology
  throughout, chosen once at startup; the two never coexist within a
  single run.

This makes *every* phase inert with respect to the equal-rank path --
there is no live-path phase to sequence around.  Each phase adds gated
code that is unreachable until a user opts in with a non-default ratio,
so each can land on `main` incrementally and safely.

**The production fallback comes for free.**  The branch-by-abstraction
flag was justified as a runtime escape hatch for clusters with buggy
RMA.  Gated-additive preserves that property without a second
implementation of the equal-rank path: if the `fullcomm` path
misbehaves on some MPI build, the user sets the ratios back to 1.0 and is
on the proven path.  Disabling the feature *is* the escape hatch.

**Cost we accept.**  Two window topologies live in the tree
permanently -- per-`strata_comm` windows for the equal-rank case and the
`fullcomm` window for the unequal case -- and we do *not* plan a later
unification onto a single communicator.  This is benign clutter: the two
are not competing implementations of the same behavior (which would
drift), but the right tool for two different cases, partitioned by a
condition decided once at startup.  The thing given up is the aesthetic
of a single communicator; if a future benchmark ever shows `fullcomm` is
strictly better even at equal ranks, the cutover remains available as
separate work, but it is not undertaken speculatively for an unstable
dependency we do not control.

**Workflow: stacked PRs.**  Because reviews of this material are slow,
phases are developed as a *stack* of pull requests rather than waiting
for each to merge before the next begins.  Each phase lives on its own
short-lived branch whose PR targets the branch *below* it (phase N+1 onto
phase N, phase 1 onto `main`); the stack is reviewed and merged
bottom-up, each PR being retargeted to `main` as the one beneath it
lands.  This keeps work moving while earlier PRs sit in review, yet --
unlike a long-lived feature branch or a development fork -- every change
stays small, individually reviewable, and revertible/bisectable, and
`main` stays continuously shippable.  (A big-bang integration merge of an
entire multi-phase feature is the opposite: a large, hard-to-review,
hard-to-bisect surface, which is *more* dangerous for code whose failures
are subtle.  If isolation is ever genuinely wanted, prefer a branch in
the upstream repository over a separate fork -- same isolation, far less
CI and merge friction.)

**Prerequisites before the feature is recommended for production use.**
There is no default to flip, but before the `fullcomm` path is
documented or recommended for production use, exercise it on at least
two MPI implementations (e.g. OpenMPI and MPICH) and more than one
mpi4py / MPI version, since that path is where the RMA-portability risk
lives.

The same "finish before recommending it" list carries the **scalable
layout exchange**: the interim `fullcomm.allgather` (see the Option D
layout-exchange note) must be replaced by the two-level or local-compute
scheme before the feature is documented or recommended for production
use, because total rank counts in the thousands are a real operating
regime here.  Both are prerequisites for recommending the feature, not
for landing the intervening phases on `main`.


### Possible future optimization (out of scope)

The first-stage portion of `BEST_XHAT` / `RECENT_XHATS` is genuinely
NAC-redundant (the same node value copied into every scenario's slot).
A future optimization could store it once per non-leaf node instead of
once per scenario, saving storage and bandwidth on those fields.  This
is *only* valid for the Category-2 fields — never for the Category-1
per-scenario fields — and is explicitly **not** required for flexible
ranks.  It is recorded here so the idea is not lost, not as planned
work.


### Backward Compatibility

When all rank ratios are 1.0 (the default), the system behaves
identically to the current implementation — not via a degenerate case of
the new machinery, but because the new machinery is **not reached at
all**.  At equal ratios the existing per-`strata_comm` windows,
single-source reader, and `strata_rank` addressing run verbatim; the
overlap maps, `fullcomm` window, and multi-source reader are gated behind
a non-default ratio (see §Development and rollout strategy).  Because
this design also makes **no field-layout changes**, the on-the-wire
format is unchanged at equal ranks — there is no analogue of the first
pass's "Phase 0 changes the wire format even at equal ranks" caveat.
Verify by running the full existing test suite with the new code and
default ratios.


---

### Correction note (why this is a second pass)

The first pass of this design was built on a wrong premise.  It
asserted that the hub-published nonant field carries `xbar` (the
consensus value), with one `xbar` value redundantly copied into every
scenario's slot, and proposed collapsing it to one canonical vector
per non-leaf node.

Reading the code disproved this.  The field carries **per-scenario
nonant variable values** — each scenario's own current iterate —
and several consumers *depend* on those values being per-scenario
distinct (see §Field Taxonomy).  `xbar` is a separate quantity,
stored in the `xbars` Pyomo Param (`phbase.py`), equal to the
transmitted values only at convergence.

Consequences for this design:

- The field-canonicalization refactor ("BN-2 / BX-2") is **gone**.
  The hub nonant field cannot and should not be collapsed to one
  value per node.
- The misnamed field has already been renamed in tree:
  `Field.NONANT` → (briefly `XBAR`) → `Field.NONANTS_VALS`, and
  `Field.RELAXED_NONANT` → `Field.RELAXED_NONANTS_VALS`.  This
  document uses the final names throughout.
- The remaining design is **simpler**: one uniform mechanism
  (overlap maps + multi-source assembly) for every local-sized
  field, plus a per-field coherence policy.
