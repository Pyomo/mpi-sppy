# Flexible Rank Assignments for Cylinders

Status: Design Document (Draft)
Date: 2026-03-16
Branch: `flexible_rank_assignments`

### Terminology

- **Window:** An MPI one-sided ("RMA") addressing scope, created
  collectively on a communicator via `MPI_Win_Allocate`.  The window
  is a handle, not a contiguous block of memory — it makes each
  participant's local buffer mutually addressable via `MPI_Get` /
  `MPI_Put`.
- **Buffer:** A rank's local memory region exposed through a window.
  Holds that rank's published fields (`NONANT`, `DUALS`,
  `BEST_XHAT`, ...) packed according to the `buffer_layout` in
  `spwindow.py`.

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
hub.  The hub always serves as the reference (ratio 1.0).  Example
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

Open questions on the interface:

- Should the ratio be per-spoke-type in `Config`, or specified in the
  spoke dict as a numeric field?
- Should there be a `--spoke-ranks` option that takes explicit counts
  instead of ratios?  Ratios are more portable across different total
  rank counts, but explicit counts give precise control.
- What happens when the ratios don't divide evenly into the total rank
  count?  Options: (a) round and warn, (b) error, (c) adjust ratios
  to fit.


### Current Architecture

This section describes the pieces that would need to change.

#### Rank Partitioning (`spin_the_wheel.py`)

`WheelSpinner` requires `n_proc % n_spcomms == 0` and creates two
communicators via `MPI_Comm_split`
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
in every cylinder gets the same scenarios.

#### Buffer System (`spwindow.py`, `spcommunicator.py`)

Each rank creates an MPI RMA window with per-field buffers.  Buffer
sizes for "local" fields (`NONANT`, `DUALS`, `BEST_XHAT`,
`RECENT_XHATS`, `CROSS_SCENARIO_COST`) are proportional to the
rank's local scenario count.  Buffer sizes for "global" fields
(`NONANT_LOWER_BOUNDS`, `NONANT_UPPER_BOUNDS`) are proportional to
the total nonant count and are the same on every rank.

On receive, `_validate_recv_field()` checks that the remote buffer
size matches the local expectation.  This check assumes both sides have
the same local scenario count.

Note: the per-scenario classification of `BEST_XHAT` /
`RECENT_XHATS` is itself revisited below (§Coherence options) —
under Option BX-1 they remain local-sized with added strict
coherence; under Option BX-2 they are reframed as global-sized
per-node vectors.

#### Communication Patterns

All inter-cylinder communication uses one-sided MPI (RMA) through
`SPWindow`.  The communication graph is:

**Hub sends to spokes:**
  - `NONANT` (local-sized): current nonant values for PH iterates
  - `DUALS` (local-sized): W values (dual weights)
  - `SHUTDOWN`, `BEST_OBJECTIVE_BOUNDS` (scalars on rank 0 only)

**Spokes send to hub:**
  - `OBJECTIVE_INNER_BOUND`, `OBJECTIVE_OUTER_BOUND` (scalars)
  - `BEST_XHAT` (local-sized): best feasible solution found
  - `RECENT_XHATS` (local-sized, circular buffer)

**Spoke-to-spoke (via RMA windows, not point-to-point):**
  - FWPH spoke reads `BEST_XHAT` and `RECENT_XHATS` from an
    InnerBoundSpoke
  - ReducedCostsSpoke sends `NONANT_LOWER_BOUNDS` and
    `NONANT_UPPER_BOUNDS` (global-sized, already works)
  - CrossScenarioCutSpoke reads `NONANT` and
    `CROSS_SCENARIO_COST` from a source

The local-sized fields are the ones affected by unequal rank
counts.  Of those, `NONANT` and `DUALS` get the multi-source
assembly treatment described below; `BEST_XHAT` and `RECENT_XHATS`
need different handling (see §Coherence options) because
per-scenario assembly across iterations violates NAC.


### Design: Multi-Rank Mapping

This section describes the design for fields where multi-source
assembly across scenarios is mathematically valid (`NONANT`,
`DUALS`).  Fields where per-scenario assembly would violate
non-anticipativity (`BEST_XHAT`, `RECENT_XHATS`) are handled
separately — see §Coherence options.

The core idea (for the applicable fields): when a rank in one
cylinder needs data from another cylinder with a different rank
count, it reads from multiple remote ranks and assembles the
result (or reads from one remote rank and extracts a subset).

#### Overlap Maps

At startup, each rank computes a static mapping for each peer cylinder:
which remote ranks have scenarios that overlap with its own local
scenarios, and at what offsets within those remote buffers.

The existing `scen_names_to_ranks(n_proc)` function already computes
scenario-to-rank mappings.  We would call it once per cylinder's rank
count to get each cylinder's distribution, then compute pairwise
overlaps.

For example, with a 4-rank hub and a 2-rank spoke, both handling 10
scenarios:

========  ==================  ==================
          Hub (4 ranks)       Spoke (2 ranks)
========  ==================  ==================
Rank 0    scen0, scen1        scen0--scen4
Rank 1    scen2, scen3        scen5--scen9
Rank 2    scen4, scen5
Rank 3    scen6--scen9
========  ==================  ==================

Hub rank 0 needs to read from spoke rank 0 (which has scen0--scen4),
extracting only the portion for scen0--scen1.  Hub rank 2 also reads
from spoke rank 0, extracting scen4--scen5.

The overlap map for hub rank 0 reading from the spoke would be
```
[(spoke_rank=0, remote_offset=0, local_offset=0, count=2)]
```

For hub rank 2
```
[(spoke_rank=0, remote_offset=4, local_offset=0, count=2)]
```

For hub rank 3
```
[(spoke_rank=1, remote_offset=0, local_offset=0, count=4)]
```

These maps are computed once at startup and reused for every
communication call.  The data structure could be
```
@dataclass
class OverlapSegment:
    remote_rank: int       # rank in peer cylinder
    remote_offset: int     # nonant offset within remote buffer
    local_offset: int      # nonant offset within local buffer
    count: int             # number of nonants in this segment
```

    # Per (peer_cylinder, field): list of OverlapSegments
    overlap_map: dict[int, list[OverlapSegment]]

Note: offsets are in nonant units (not scenario units) because
different scenarios could have different numbers of nonants in
multi-stage problems (though for two-stage problems they are uniform).


#### Window Topology Options

Spoiler alert: we are going to recommend option D, which is similar to A.

**Option A: Single global window on MPI_COMM_WORLD**

Every rank publishes its buffers in one shared window.  Readers address
remote ranks by their global rank.  The `strata_buffer_layouts`
(currently exchanged via `strata_comm.allgather`) would instead be
exchanged on `MPI_COMM_WORLD` or a dedicated intercommunicator.

Pros:

- Simple conceptually: any rank can read from any other rank.
- No need for multiple communicator splits.
- A single `MPI_Win_create` call.

Cons:

- The window is large (sum of all ranks' buffers across all cylinders).
- `MPI_Win_lock` granularity is per-window; concurrent accesses to
  different cylinders' data may contend.
- All ranks must participate in window creation even if they never
  communicate with most other ranks.

**Option B: Per-cylinder-pair intercommunicators with separate windows**

For each pair of cylinders that need to communicate, create an
`MPI_Intercomm` and a window on it.  For example, hub-lagrangian and
hub-xhat would each get their own window.

Pros:

- Smaller windows, less contention.
- Clean separation of concerns.

Cons:

- Number of windows grows with the number of communicating pairs.
  With spoke-to-spoke communication, this could be O(n^2) in the
  worst case (though in practice most spokes only talk to the hub
  plus at most one other spoke).
- More complex setup code.
- MPI intercomm semantics can be tricky.

**Option C: Keep strata_comm but make it asymmetric**

Create `strata_comm` groupings where ranks from different-sized
cylinders are grouped together.  Ranks without a counterpart in a
smaller cylinder would be grouped with the "nearest" rank in that
cylinder.

For example, with 4 hub ranks and 2 spoke ranks, strata groups would
be
```
strata 0: hub_rank_0, spoke_rank_0
strata 1: hub_rank_1, spoke_rank_0  (spoke_rank_0 appears twice)
strata 2: hub_rank_2, spoke_rank_1
strata 3: hub_rank_3, spoke_rank_1  (spoke_rank_1 appears twice)
```

Pros:

- Closest to the current architecture; least code change.
- Each strata_comm still has one rank per cylinder.

Cons:

- A spoke rank appears in multiple strata comms, which may cause
  issues with MPI (a rank can only be in one communicator of a given
  color).
- Actually, this doesn't work with `MPI_Comm_split` because a rank
  can only appear once per split.  Would need to use
  `MPI_Comm_create_group` or manual point-to-point addressing
  instead.

**Option D: Replace strata_comm with direct RMA addressing**

Abandon `strata_comm` entirely.  Use `MPI_COMM_WORLD` for the
window, and have each rank directly address remote ranks by their
global rank.  The overlap maps provide the addressing information.

This is essentially Option A but with the explicit framing that
`strata_comm` is removed rather than modified.

Pros:

- Clean break from the current 1-to-1 assumption.
- Most flexible: any rank can read from any other rank.
- Single window creation.

Cons:

- Requires rewriting `SPCommunicator` and `SPWindow` to work
  without `strata_comm`.
- The buffer layout exchange (currently `strata_comm.allgather`)
  needs a replacement. The cheapest option is to compute layouts
  locally on every rank: per-cylinder rank count, the output of
  `_calculate_scenario_ranks`, and field-registration order are all
  static and deterministic at startup, so no communication is
  required. If a runtime exchange is genuinely needed (e.g., dynamic
  field registration), prefer a two-level scheme —
  `cylinder_comm.allgather` followed by a small cross-cylinder gather
  over one anchor rank per cylinder — rather than a single
  `fullcomm.allgather`, which scales worse at N=thousands.
- Single-window structural cost relative to Option B.  Two
  separable concerns; neither is a deal-breaker.

  *Lock granularity.*  `MPI_Win_lock(rank=target, ...)` is per-
  target-rank in the MPI spec, not per-window — a writer's
  exclusive lock on its own buffer does not block readers fetching
  from a different rank's buffer, regardless of cylinder.  Cross-
  cylinder serialization is therefore a non-issue at the spec
  level.  Implementation-dependent progress-engine state per
  window object can cause minor contention in practice, but
  nothing like "the whole window stalls."  If a workload ever did
  show measurable contention, splitting hot cylinder pairs into
  their own window (Option B for those pairs) is an incremental
  fix that doesn't require abandoning Option D.

  *Bookkeeping.*  MPI-side, the window object on `fullcomm` must
  track all participating ranks' base addresses, displacement
  units, and lock state — modest (KB-to-low-MB per rank at
  N=thousands) but unavoidable.  mpi-sppy-side, a naive
  `fullcomm.allgather` of buffer layouts would scale poorly,
  which is why the preceding bullet recommends computing layouts
  locally from the static rank-count/scenario-map data instead.

**Recommendation:** Option D is the cleanest long-term solution.
Option A is equivalent but Option D better describes the intent.  The
window is on `fullcomm` (or a subset), and each rank knows which
global ranks to read from via the overlap maps.


#### Multi-Source Read Assembly

The current `get_receive_buffer()` reads one contiguous buffer from
one remote rank.  With multi-rank mapping, it would need to:

1. For each segment in the overlap map, issue an `MPI_Get` for that
   segment from the appropriate remote rank.
2. Place each segment at the correct offset in the local receive
   buffer.
3. Return the assembled buffer.

Schematically
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

For the common case where both cylinders have the same rank count,
the overlap map has exactly one segment per peer rank and the offsets
are identity — so the behavior degenerates to the current code.


#### Write-ID Coherence

The current system uses a `write_id` integer appended to each buffer.
When a sender writes new data, it increments `write_id`.  The
receiver checks whether the `write_id` has changed since the last
read to determine if the data is "new."

With multi-source reads, a receiver reads from multiple remote ranks
in the same cylinder.  These ranks may not have updated their buffers
at exactly the same time (the system is asynchronous).  This raises
the question: should we require all ranks in a cylinder to have the
same `write_id` before accepting the data?

The answer depends on the field.  Some fields carry coupled data that
must come from the same iteration to be mathematically valid, while
others are independent candidates that tolerate staleness.

##### Per-field coherence requirements

**Fields requiring strict coherence:**

- `NONANT` (xbar values from the hub) and `DUALS` (W values from
  the hub) are coupled: they define the Lagrangian subproblem at a
  specific PH iteration.  If a spoke assembles xbar from iteration k
  for some scenarios and iteration k+1 for others, the dual function
  is evaluated at an inconsistent dual point and the resulting bound
  is invalid.

  In practice this is manageable because the hub is synchronous PH:
  all hub ranks complete the same iteration before writing their
  buffers, so their `write_id` values naturally agree after each
  PH iteration.  The receiver just needs to verify they match before
  accepting the assembled data.

**Fields where multi-source assembly breaks non-anticipativity:**

- `BEST_XHAT` is per-scenario local-sized, but every scenario's
  first-stage slot must hold the same first-stage values (NAC).
  Assembling segments from iteration *k* for some scenarios and
  iteration *k+1* for others yields a vector where different
  scenarios' first-stage slots disagree — i.e., not a feasible
  first-stage point at all.  "Evaluate fresh" does not save this:
  the receiver fixes first-stage variables across all scenarios
  from the assembled buffer and would see scenario-dependent
  first-stage values, violating NAC.  Multistage is worse: stage-2+
  decisions are conditional on stage-1, so mixing iterations also
  mixes branches with different parent decisions.
- `RECENT_XHATS` is the same: a circular buffer of multiple xhat
  solutions, each entry per-scenario local-sized.  Each entry is a
  candidate first-stage point and inherits the same NAC problem
  when assembled from mixed iterations.

There are two ways to address this; both are on the table (but DLW
favors BX-2, which has a very long description here):

**Option BX-1: Require strict coherence on `BEST_XHAT` /
`RECENT_XHATS` as well.**  Treat them the same as `NONANT` /
`DUALS`: read `write_id` values first, verify they match across the
remote ranks contributing segments, then read data.  The sender
side is already synchronous in practice (an xhat spoke writes
`BEST_XHAT` only after its local `_try_one` solve finishes on all
its ranks), so `write_id` values naturally agree after each xhat
iteration and the check is almost free.  Minimal change; keeps the
current per-scenario field layout.

**Option BX-2: Reframe `BEST_XHAT` and `RECENT_XHATS` so each
non-anticipative decision lives in exactly one place.**  Replace
the per-scenario layout with one canonical vector per non-leaf
node.

*What the current layout looks like.*  `BEST_XHAT` on a given
rank is a concatenation, one entry per local scenario, of that
scenario's `nonant_indices` values (plus a per-scenario scalar
total cost stored alongside).  Since `nonant_indices` contains
exactly the NAC-bound variables — those at non-leaf nodes —
every scenario passing through a given node carries its own copy
of that node's values.  Two scenarios sharing the ROOT node, for
example, each store the same stage-1 vector in their own slot;
nothing in the layout enforces that the two copies agree, even
though NAC requires they must.

*What BX-2 changes it to.*  The layout becomes a concatenation,
one entry per non-leaf node touched by the local scenarios, of
that node's canonical nonant vector (size = the node's nonant
count).  Scenarios no longer have private xhat slots; they share
the per-node entries.  Per-scenario total cost, which is
genuinely scenario-specific, remains per-scenario in a parallel
small array.

*Why non-leaf nodes specifically.*  NAC binds at non-leaf nodes:
all scenarios sharing a non-leaf node must agree on that node's
decisions.  Leaf nodes correspond one-to-one with scenarios, so
their decisions are scenario-specific by construction and are
not included in `nonant_indices` anyway — there is nothing to
canonicalize at leaves.

*Concrete example.*  A two-stage problem with 100 scenarios and
20 stage-1 (root) variables.  The current per-scenario layout
stores 100 × 20 = 2000 doubles for the nonant portion of
`BEST_XHAT` (across all ranks combined), even though by NAC all
100 copies must agree.  BX-2 stores 1 × 20 = 20 doubles for the
ROOT node — a 100× reduction in this dimension.  Multistage gives
the same kind of reduction per node, scaled by how many scenarios
share that node.

*Writer rule (uniform across stages and cylinders):* the writer for
a non-leaf node is the lowest-rank holder, within the publishing
cylinder, of any scenario passing through that node.  For two-stage
this collapses to a single writer per cylinder — rank 0, since rank
0 holds the first scenario block and ROOT is the only non-leaf
node.  Hub-side rank-0-special-cases like `SHUTDOWN` and
`BEST_OBJECTIVE_BOUNDS` are untouched; this rule applies only to
`BEST_XHAT` / `RECENT_XHATS`.

*Certification and publish.*  A natural question is how the
elected writer knows that all other ranks in the cylinder have
certified the candidate xhat as feasible before it publishes the
canonical vector.  The answer is that certification is not done
via the SPWindow; it goes through the existing cylinder-internal
`cylinder_comm.Allreduce` that already runs in every xhat spoke
today.  Sequence:

1. The hub broadcasts a candidate xhat to the spoke via `NONANT`.
2. Each spoke rank fixes the first stage to that candidate and
   solves its locally-held scenarios' subproblems.
3. The spoke does a `cylinder_comm.Allreduce` combining per-rank
   feasibility flags and per-scenario costs into a single global
   verdict plus total cost.  After this Allreduce returns, every
   rank in the cylinder — including the elected writer — knows
   whether the candidate is certified.
4. If certified, the elected writer transcribes the canonical
   node vector(s) from its in-memory model into its buffer and bumps
   `write_id`.  Other ranks do not write `BEST_XHAT` at all.
5. Readers (hub, other spokes) Get from the writer's buffer.

So the writer never has to learn the certification result
out-of-band: the Allreduce *is* the certification, and it is
globally synchronous within the cylinder.  Step 4 is just the
elected scribe transcribing the already-agreed answer.

*Why the writer has the value to transcribe.*  By the writer
rule, the elected writer for a node holds at least one scenario
passing through that node, so the node's nonant values are in
its local Pyomo model.  NAC guarantees that all scenarios through
that node agree on those values, so it does not matter which
local scenario's slot the writer reads from when transcribing.

*Read side:*

- **Two-stage:** one Get from the publishing cylinder's writer.  No
  assembly on either side.
- **Multistage:** one Get per non-leaf node from that node's
  writer, then stitch those node vectors into the reader's local
  view.  Not "no assembly," but structurally different and smaller
  than the per-scenario layout: O(non-leaf nodes) Gets instead of
  O(local-scenarios × node-depth) source segments, and no
  cross-source merging *within* a node.

The chosen-writer table is deterministic from the scenario-to-rank
map and the scenario tree; compute it at startup alongside the
overlap maps.

*Benefits independent of asymmetric ranks:* storage drops from
O(scenarios × all-nodes-len) to O(non-leaf-nodes × node-len), and
the layout encodes the actual NAC invariant rather than implying
scenarios can disagree.

*Costs:* field-layout change that touches FWPH (consumes
`RECENT_XHATS`), any receiver that reads these fields, and the
writer-assignment plumbing.  Probably the right long-term shape;
arguably should be done first as a prerequisite refactor.

**Fields tolerating staleness (relaxed coherence):**

- `NONANT_LOWER_BOUNDS`, `NONANT_UPPER_BOUNDS`.  Already
  global-sized: one canonical bound vector per sender, no
  partitioning to assemble.  Staleness is safe because the
  receiver-side application is monotonic and idempotent —
  `recv_buf[ci]` is applied only when it tightens the existing
  local bound (`spcommunicator.py:517` and the upper-bound mirror),
  so an older snapshot is never "wrong," it just fails to tighten
  as aggressively as a fresh one would.  Asymmetric ranks are a
  no-op here.
- Scalar bounds (`OBJECTIVE_INNER_BOUND`, `OBJECTIVE_OUTER_BOUND`).
  One float per cylinder.  Nothing to partition, and the receiver's
  use of these (termination comparison) is monotonically improving
  on the sender's side.  Staleness yields a slightly later
  termination check, never an incorrect one.

##### Coherence options

**Option 1: Per-field strict check**

For fields that require coherence (`NONANT`, `DUALS`, and
`BEST_XHAT` / `RECENT_XHATS` if Option BX-1 is chosen): read all
`write_id` values first (cheap: one int per segment), check they
match, then read the data.  If they don't match, retry later.

For fields that tolerate staleness (the global-sized bound fields
and the scalar objective bounds): accept data from each source
rank independently.  Track `write_id` per source rank.

Pros: correct semantics for each field type.  No unnecessary stalls
for fields that don't need coherence.

Cons: two code paths for multi-source reads (strict and relaxed).

**Option 2: Cylinder-wide iteration counter (synchronized)**

Each cylinder maintains a shared iteration counter (via
`cylinder_comm.Allreduce` after each update).  Receivers check
this counter for fields requiring coherence.

Pros: clean coherence semantics.
Cons: adds synchronization within the cylinder, which is exactly
what the async design tries to avoid.  Only acceptable for
synchronous algorithms (PH, not APH).

**Option 3: Accept staleness everywhere (fully relaxed)**

Accept data from each source rank independently for all fields.

Pros: simplest implementation, no stalls.
Cons: Lagrangian bounds may be invalid when assembled from mixed
iterations.

**Recommendation:** Option 1 (per-field strict check) for the
multi-source-read mechanics, paired with either Option BX-1 or
Option BX-2 to fix the NAC problem with `BEST_XHAT` / `RECENT_XHATS`.
Option 1's check is almost free under synchronous PH: `write_id`
values naturally agree after each iteration, so receivers verify
and rarely retry.  Whether to land BX-1 (strict coherence, minimal field-layout
change) or BX-2 (reframe as global-sized, prerequisite refactor)
is still open.


### Impact on Existing Components

#### `spin_the_wheel.py`

- Remove the `n_proc % n_spcomms == 0` check.
- Replace the uniform `MPI_Comm_split` with a rank assignment
  algorithm that respects per-cylinder rank counts.
- The `cylinder_comm` creation still uses `MPI_Comm_split` (all
  ranks in the same cylinder get the same color).
- The `strata_comm` is either removed (Option D) or replaced with
  a global window communicator.
- The `communicator_list[strata_rank]` indexing must change because
  `strata_rank` no longer has a fixed meaning.  Each rank needs to
  know its cylinder index directly.

#### `spbase.py`

- `_calculate_scenario_ranks()` already works with any `n_proc`.
  No change needed; each cylinder just calls it with its own rank
  count.

#### `spwindow.py`

- `FieldLengths` stays the same (it's per-rank, based on local
  scenarios).
- `SPWindow` must support partial `get()` calls (offset + count).
- The buffer layout exchange must use a communicator that spans all
  cylinders (not just `strata_comm`).
- Buffer validation must account for asymmetric sizes: a receiver's
  expected size may differ from the sender's buffer size, and that's
  OK as long as the requested segment fits within the sender's buffer.

#### `spcommunicator.py`

- `register_receive_fields()` must build overlap maps instead of
  assuming 1-to-1 rank correspondence.
- `get_receive_buffer()` must support multi-source assembly.
- `put_send_buffer()` is unchanged (each rank writes its own data).
- `_validate_recv_field()` must be relaxed or replaced with
  segment-level validation (check that each requested segment fits
  within the remote buffer, not that the total sizes match).
- The `synchronize` parameter in `get_receive_buffer()` uses
  `cylinder_comm.Barrier()` and `Allreduce` — this still works
  within a cylinder but the cross-cylinder sync semantics change.

#### `hub.py` and `spoke.py`

- `send_nonants()`, `update_nonants()`, and similar methods
  currently iterate over local scenarios and pack/unpack linearly.
  The packing is unchanged (each rank packs its own data).  The
  unpacking on the receiver side must use the overlap map to
  correctly place data from multi-source reads.
- Bound communication (scalars) is unaffected.

#### `cfg_vanilla.py` and `config.py`

- Add per-spoke rank ratio configuration.
- `shared_options()` may need to carry the rank ratio information
  so that `SPCommunicator` can compute overlap maps at init time.

#### `generic_cylinders.py`

- After computing rank ratios, pass them to `WheelSpinner` via the
  hub/spoke dicts.
- Default ratio is 1.0 for all cylinders (backward compatible).


### Phased Implementation Plan

**Phase 1: Infrastructure**

- Implement the overlap map computation (given per-cylinder rank
  counts and scenario lists, produce `OverlapSegment` lists).
- Add partial-read support to `SPWindow.get()`.
- Add rank ratio fields to spoke dicts and `WheelSpinner`.
- Modify rank partitioning in `WheelSpinner` to support unequal
  cylinder sizes.
- Unit tests for overlap maps with various rank count combinations.

**Phase 2: Communication layer**

- Replace `strata_comm`-based buffer layout exchange with
  `fullcomm`-based exchange.
- Implement multi-source `get_receive_buffer()` using overlap maps.
- Relax `_validate_recv_field()` to check per-segment instead of
  total.
- Test with simple cases: hub with 4 ranks, Lagrangian spoke with 2
  ranks.

**Phase 3: Integration and testing**

- Wire up the rank ratio CLI options.
- Auto-disable for cylinders that cannot support asymmetric ranks
  (if any).
- Test all spoke types with various ratios.
- Performance benchmarks: does 8+4+2 outperform 5+5+5 for a given
  problem?

**Phase 4: Spoke-to-spoke communication**

- Extend multi-source reads for spoke-to-spoke fields (FWPH reading
  `BEST_XHAT` from an InnerBoundSpoke with different rank count).
- Test FWPH + xhatshuffle with unequal ranks.

**Phase 5: APH support**

- Verify that APH (asynchronous PH) works correctly with the relaxed
  write-ID coherence model.
- Consider adding optional strict coherence (Option 1) as a flag.


### Backward Compatibility

When all rank ratios are 1.0 (the default), the system must behave
identically to the current implementation.  The overlap maps degenerate
to single-segment identity mappings, and multi-source reads become
single-source reads.  This should be verified by running the full
existing test suite with the new code and default ratios.


### Open Questions

1. For the FWPH spoke-to-spoke case, the FWPH spoke reads
   `RECENT_XHATS` which is a circular buffer of multiple xhat
   solutions.  Each entry is local-sized.  The multi-source read
   would need to be applied to each entry in the circular buffer.
   Is this feasible, or should FWPH be restricted to equal rank
   counts?





