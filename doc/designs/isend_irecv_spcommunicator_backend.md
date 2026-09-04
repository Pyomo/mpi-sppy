# Isend/Irecv Backend for SPCommunicator

Status: proposed implementation plan

Date: 2026-08-31

## Decision summary

Add a two-sided, nonblocking transport beneath `SPCommunicator`. The new
backend will publish immutable field snapshots with typed `MPI_Isend` calls
and keep typed `MPI_Irecv` operations posted at consumers. It will preserve
the existing `put_send_buffer` / `get_receive_buffer` API, write-ID semantics,
equal-rank behavior, and unequal-rank overlap assembly.

The first implementation will be opt-in so it can be compared directly with
RMA. After correctness, performance, and production soak validation, the
two-sided backend is intended to replace RMA rather than remain one of two
permanent implementations. RMA will remain selectable only during migration
and will then be deprecated and removed.

No background progress thread is part of this design. The backend progresses
requests cooperatively whenever mpi-sppy publishes, polls, synchronizes, or
finalizes communication.

## Motivation

mpi-sppy currently uses passive-target MPI RMA as its inter-cylinder transport:

- a publisher updates its own `SPWindow` allocation with
  `Lock` / `Put` / `Unlock`;
- a consumer snapshots a remote field with `Lock` / `Get` / `Unlock`;
- a trailing write ID distinguishes new values from old values;
- cylinder collectives prevent ranks that consume the data collectively from
  making inconsistent control-flow decisions.

A standalone two-rank C reproducer and an equivalent mpi4py reproducer show an
intermittent cross-node hang with MVAPICH 2.3.7. The reproducer uses one window,
one double, and a standards-compliant loop consisting of a remote
`Lock` / `Get` / `Unlock` followed by `MPI_Barrier`. This rules out solvers,
Python object serialization, mpi4py, multiple windows, and complicated
mpi-sppy control flow.

One-sided communication has also proven brittle across more than one MPI
implementation. The transport should therefore be replaceable rather than
embedded in `SPCommunicator`'s field and coherence logic.

## Goals

1. Preserve the public behavior of `SPCommunicator`:
   `register_send_field`, `register_recv_field`, `put_send_buffer`, and
   `get_receive_buffer` remain the interface used by hubs, spokes, and
   extensions.
2. Preserve latest-value semantics. Consumers need the newest complete
   snapshot available; they need not observe every intermediate publication.
3. Preserve the trailing write ID and all existing strict/relaxed coherence
   policies.
4. Support both equal-rank strata and unequal-rank overlap maps.
5. Bound memory usage even when a publisher runs faster than a consumer.
6. Avoid object messages on hot paths. Data movement uses NumPy buffers and
   typed MPI operations.
7. Require no MPI progress thread and no `MPI_THREAD_MULTIPLE`.
8. Keep the RMA backend working behind the same transport interface long
   enough for differential testing and a controlled migration.
9. Shut down without leaked requests, unmatched messages, or ranks stranded in
   transport cleanup.

## Non-goals

- Replacing `cylinder_comm` collectives used by algorithms and write-ID
  agreement.
- Guaranteeing delivery of every intermediate publication. The RMA backend
  already permits readers to skip versions.
- Solving general asynchronous messaging for arbitrary user-defined message
  types. The backend transports the existing fixed `Field` snapshots.
- Adding a Python or MPI progress thread.
- Optimizing unequal-rank traffic in the first correctness implementation.

## Semantics that must not change

### Complete snapshots

A publication consists of field data and its trailing write ID. A consumer
must never accept data from one publication paired with the ID from another.
An MPI message naturally provides this atomic unit: the sender copies the
entire padded field after `_next_write_id`, and the receiver installs it only
after the receive request completes.

### Latest-value behavior

`put_send_buffer` may overwrite publications that no consumer has observed.
This is intentional. Bounds, shutdown, nonants, duals, and xhat fields are
polled state, not an event log. `RECENT_XHATS` already carries its own circular
history inside one field snapshot.

### Coherence

Message atomicity eliminates torn data/ID reads from one source, but does not
eliminate differences between sources or between ranks in a receiving
cylinder. The following logic remains above the transport:

- `_write_ids_agree` protects collective control flow across a receiving
  cylinder;
- strict multi-source fields require every contributing source to have the
  same write ID;
- relaxed multi-source fields may assemble snapshots from different IDs and
  use their minimum ID as the accepted version.

No backend may weaken these rules merely because each individual message is
atomic.

### Nonblocking reads

`get_receive_buffer` must continue to return `False` when no acceptable new
snapshot is available. It must not wait for a publisher that is solving or has
stopped publishing.

## Proposed architecture

### Separate metadata from transport

Today `SPWindow` performs two jobs:

1. allocates the RMA storage;
2. gathers every rank's field layout into `strata_buffer_layouts`.

The layout exchange must move out of `SPWindow`. Introduce a transport-neutral
layout catalog built before any backend resource is created:

```python
local_layout = build_layout(send_buffers)
layout_catalog = fullcomm.allgather(rank_metadata_and_layout)
```

The catalog records global rank, cylinder index, cylinder rank, fields,
logical lengths, and padded lengths. Equal-rank field mappings and
unequal-rank overlap maps consume this catalog rather than
`self.window.strata_buffer_layouts`.

This change also makes it possible to inspect actual source/target edges before
creating an RMA window, improving the MVAPICH safety guard.

### Transport interface

Add `mpisppy/cylinders/sptransport.py` with a narrow internal interface:

```python
class SPTransport(abc.ABC):
    @abc.abstractmethod
    def publish(self, field, snapshot): ...

    @abc.abstractmethod
    def poll(self, source_global_rank, field): ...
    # Return the newest complete cached snapshot, or None when none exists.

    @abc.abstractmethod
    def progress(self): ...

    @abc.abstractmethod
    def close(self): ...
```

`RmaTransport` adapts the existing `SPWindow`. `MessageTransport` implements
the Isend/Irecv protocol. During migration, `SPCommunicator.window` may remain
as a compatibility alias for the RMA transport, but new code must use
`self.transport`.

Field registration, write-ID decisions, overlap assembly, and bound/xhat
interpretation remain in `SPCommunicator`; the transport only moves complete
snapshots.

### Setup phases

Replace the implicit setup inside `make_windows` with these explicit phases:

1. **Register producers.** Call all cylinder and extension
   `register_send_fields` hooks.
2. **Exchange layouts.** Allgather transport-neutral layout metadata.
3. **Build field mappings.** Populate `fields_to_ranks` and
   `ranks_to_fields` from the layout catalog.
4. **Register consumers.** Call all cylinder and extension
   `register_receive_fields` hooks and build unequal-rank overlap maps.
5. **Build subscriptions.** Each consumer describes every
   `(source global rank, destination global rank, field)` stream it needs.
6. **Exchange subscriptions.** Allgather the descriptors so publishers know
   their destinations.
7. **Create the backend.** Duplicate a communicator for message traffic and
   post initial receives, or allocate the RMA window.
8. **Freeze registration.** A send/receive registration attempted after this
   point is a startup error. The current extension hooks already run from the
   hub/spoke registration methods during `make_windows`; tests will pin this
   ordering before the restriction is enforced.

Keep `make_windows` and `free_windows` as compatibility wrappers initially;
rename them only after external users have a deprecation period.

## Message protocol

### Communicator and tags

Duplicate `fullcomm` once for transport traffic. Using a dedicated context
prevents collisions with solver, cylinder, and user messages.

Assign one compact deterministic tag per `Field`, based on the sorted field
enumeration rather than its raw integer value. Source rank plus tag uniquely
identifies a stream. Check the largest assigned tag against `MPI_TAG_UB` at
startup.

There are no Python-object headers on the data path. Source, destination,
field tag, expected count, and unpack mapping are fixed by subscriptions.

### Receive state

For every incoming stream, keep:

- one fixed-size NumPy receive slot;
- one posted `Irecv` request;
- a bounded history of completed immutable snapshots indexed by write ID;
- the newest write ID seen on that stream.

When a request completes:

1. copy or rotate the completed slot into snapshot history;
2. validate that the write ID is finite, integral, and nondecreasing;
3. repost `Irecv` immediately with the alternate slot;
4. discard history older than the configured bound once it cannot be selected
   by a synchronized reader.

A two-slot receive arrangement avoids copying a buffer that MPI may already be
writing into. Start with two slots plus a small snapshot history; optimize only
after profiling.

### Send state and backpressure

For every outgoing stream, keep at most:

- one immutable buffer owned by an in-flight `Isend` request;
- one pending buffer containing the latest publication.

On publish:

1. call `progress` on existing requests;
2. copy the complete field snapshot, including its write ID;
3. if no send is active, start `Isend` immediately;
4. otherwise replace the pending snapshot with the newer one.

When the active request completes, launch the pending snapshot, if any. This
bounds sender memory at two snapshots per edge and preserves RMA's ability to
skip intermediate versions.

MPI send buffers remain alive and immutable until their requests complete.
Application `SendArray` objects are never passed directly to `Isend` and may be
modified immediately after `publish` returns.

### Equal-rank routes

For equal cylinders, a consumer's `origin` identifies a peer cylinder. The
layout catalog maps `(origin cylinder, local cylinder rank)` to one source
global rank. Each subscription therefore replaces one current strata-window
Get with one source/destination message stream.

### Unequal-rank routes

For unequal cylinders:

- global/scalar fields subscribe only to the peer cylinder's base rank, as the
  current single-source path does;
- local-sized fields subscribe to every source global rank present in the
  existing overlap map.

The first implementation sends each source's complete padded field snapshot.
The receiver slices and assembles it with the existing `OverlapSegment`
metadata. This is intentionally simple and directly comparable with the
current whole-field RMA reads.

A later optimization may pack only subscribed segments. That requires a
route-specific layout and must be justified by measurements; it is not part of
the initial correctness implementation.

## Synchronized-reader coherence

Independent message streams may deliver different write IDs to ranks in the
same receiving cylinder. Correctness requires a common snapshot ID before
those ranks enter downstream collectives.

The message backend therefore separates **received** snapshots from
**committed** snapshots:

1. Progress all streams needed by the requested field.
2. Construct the locally available candidate ID set from bounded history.
3. Exchange candidate sets across `cylinder_comm` for synchronized reads.
4. Select the greatest write ID available to every reader rank and, for strict
   multi-source fields, to every source used by that rank.
5. Commit only that common ID. If the intersection is empty, return `False`
   without modifying the public `RecvArray`.

For relaxed multi-source fields, retain the current policy after selecting a
cross-reader-safe candidate: source snapshots may have different IDs and the
minimum source ID becomes the committed ID.

The prototype may begin with the existing MIN/MAX agreement gate, but it must
not become the default backend until stress tests demonstrate liveness under
skipped publications. Candidate-history intersection is the planned robust
implementation.

Snapshot history is bounded. If no common ID remains because different streams
coalesced past one another, readers reject the update and wait for publisher
quiescence, at which point every stream must converge on the final published
ID. Instrument this condition; persistent failure to converge is a transport
bug, not a reason to accept inconsistent data.

## Progress model

No background thread is required. `MessageTransport.progress` is called:

- at entry and exit of `put_send_buffer`;
- at entry to `get_receive_buffer`;
- at established hub/spoke synchronization points;
- in polling loops that wait for an initial value or shutdown;
- repeatedly during transport close.

MPI does not guarantee completion while a sender performs no MPI calls. That
may delay a snapshot while a cylinder is inside a long solver call, but it must
not make `get_receive_buffer` block. The consumer continues using its last
committed snapshot until the publisher next enters mpi-sppy communication.

The implementation must record request age and high-water marks for pending
sends. These diagnostics will show whether cooperative progress is adequate on
real workloads before the backend becomes the general default.

## Shutdown and cleanup

Cleanup must be collective and explicit:

1. Freeze new publications.
2. Repeatedly progress receives and outgoing requests.
3. Use a `fullcomm` reduction to determine whether any outgoing request or
   pending snapshot remains.
4. Continue until the global outstanding count reaches zero.
5. Process completed final snapshots.
6. Cancel unmatched posted receives and `Wait` for their cancellation.
7. Assert that every request is null/completed and release staging buffers.
8. Free the duplicated transport communicator.

The existing finalization barriers in `spin_the_wheel.py` provide an orderly
entry to this protocol, but the transport must not rely on process exit to
complete requests.

The final `SHUTDOWN` publication may not be discarded. Coalescing may replace
an older pending shutdown-field snapshot with the terminating snapshot, but
close must flush it before canceling receives.

## Configuration and rollout

Add one user-facing choice:

```text
spcomm_transport = auto | rma | p2p
```

Roll out in stages:

1. **`p2p` experimental.** Implement equal-rank communication and require an
   explicit selection.
2. **Unequal-rank parity.** Add subscription routes from overlap maps and run
   all flexible-rank tests under both backends.
3. **`auto` transition.** Select `p2p` for known-unsafe MVAPICH 2.3.7
   cross-node routes while retaining RMA elsewhere for direct comparison.
   Explicit `rma` remains subject to the existing safety guard unless the
   unsafe override is set.
4. **`p2p` default.** After performance and production soak testing, select
   the two-sided backend unconditionally. Keep explicit `rma` temporarily for
   regression diagnosis.
5. **Remove RMA.** Deprecate and delete the RMA backend, its safety guard, and
   the transport selector after the two-sided backend has completed the agreed
   compatibility window.

Log the selected backend once on global rank zero, including the reason when
`auto` chooses it.

## Implementation phases

### Phase 0: characterize and pin current behavior

- Add backend-independent tests for write-ID advancement, stale reads,
  circular buffers, strict/relaxed coherence, and shutdown publication.
- Record representative equal-rank and unequal-rank field routes.
- Preserve the minimal C and mpi4py RMA reproducers as external-MPI diagnostics.

### Phase 1: extract metadata and introduce the transport interface

- Move layout creation/allgather out of `SPWindow`.
- Make field mapping and overlap-map code consume the layout catalog.
- Wrap current RMA behavior in `RmaTransport`.
- Keep RMA as the only selectable implementation in this phase and require all
  existing tests to remain unchanged.

### Phase 2: equal-rank message backend

- Build equal-rank subscriptions.
- Implement deterministic tags, posted receives, immutable Isend staging,
  bounded coalescing, polling, and cleanup.
- Route the existing equal-rank `put_send_buffer` and `get_receive_buffer`
  paths through `MessageTransport`.
- Add delay/skew tests that make publishers and consumers run at very
  different rates.

### Phase 3: coherence hardening

- Add bounded receive history and common-ID selection.
- Test mismatched arrival order across every rank in a receiving cylinder.
- Verify that no rank enters a downstream collective unless all ranks commit
  the same ID.
- Add diagnostics for coalesced sends, history misses, rejected candidates,
  and maximum request age.

### Phase 4: unequal-rank support

- Convert overlap maps into subscriptions to global source ranks.
- Cache complete source snapshots and assemble accepted data with existing
  `OverlapSegment` logic.
- Preserve strict policies for `DUALS`, `BEST_XHAT`, and `RECENT_XHATS`.
- Validate global/scalar single-source routes separately from local-sized
  multi-source routes.

### Phase 5: production integration

- Add `auto/rma/p2p` configuration and rank-zero startup reporting.
- Integrate transport progress at audited hub/spoke loop boundaries.
- Add collective close to `free_windows` (or its eventual replacement).
- Run cross-MPI and production-scale soak tests before changing defaults.

## Test plan

### Unit tests

- Deterministic layout catalog and field-tag assignment.
- Equal and unequal subscription construction.
- One active plus one pending send per edge; newest pending snapshot wins.
- Send buffers remain immutable until request completion.
- Receive completion copies/rotates before reposting.
- Stale, new, skipped, and regressing write IDs.
- Common-ID selection with delayed and out-of-order streams.
- Strict and relaxed multi-source assembly.
- Circular-buffer snapshots with skipped publications.
- Cleanup of active sends, pending sends, and unmatched receives.
- Late field registration fails with a useful startup error.

Use fake request objects for deterministic state-machine tests; do not make
unit tests depend on MPI timing.

### MPI integration tests

Run each backend against the same assertions:

- two and four ranks with equal cylinders;
- unequal ratios such as 2:1 and 4:2:1;
- one-node and cross-node placement;
- messages below and above MPI eager thresholds;
- delayed publisher, delayed consumer, and deliberately skewed receiver ranks;
- repeated shutdown and final-value delivery;
- every currently supported field, including `RECENT_XHATS` and flexible-rank
  multi-source fields.

Tests must impose timeouts so a progress regression fails rather than hanging
the suite indefinitely.

### Application parity

- Run existing cylinder tests once per backend.
- Compare final bounds, incumbent values, iteration counts where deterministic,
  and write-ID/coherence diagnostics.
- Run the 100-scenario PH/Lagrangian/RelaxedPH/XhatShuffle configuration that
  motivated this work.
- Soak on Open MPI, MPICH, and MVAPICH 2.3.7. The p2p backend must pass repeated
  cross-node runs on the system where the RMA reproducer fails.

### Performance gates

Measure:

- bytes sent per field and per iteration;
- publications coalesced;
- peak staging/history memory;
- oldest outstanding request age;
- solver wall time and communication fraction;
- equal- and unequal-rank scaling.

Correctness is the first gate. Optimize segment packing, history depth, and
progress frequency only after parity is established.

## Risks and explicit decisions

1. **MPI progress is still implementation-dependent.** Cooperative progress
   may delay messages during long solves. The API remains nonblocking, and
   request-age instrumentation plus stress tests are required.
2. **Per-destination coalescing can produce different observed ID sequences.**
   Never relax correctness to hide this. Use bounded history and common-ID
   selection; treat persistent non-convergence as a defect.
3. **Full snapshots may amplify unequal-rank traffic.** Accept this in the
   first implementation. Segment packing is a later optimization.
4. **Shutdown is the highest-risk lifecycle point.** No backend is complete
   until all requests are accounted for under normal termination and raised
   exceptions.
5. **Do not add a progress thread as a shortcut.** Requiring
   `MPI_THREAD_MULTIPLE` would exchange one portability problem for another.
6. **RMA is transitional, not a permanent second backend.** Keep it only long
   enough for differential testing, performance comparison, and one controlled
   compatibility window; maintaining two transports indefinitely would add
   complexity without enough expected performance benefit.

## Expected file changes

- `mpisppy/cylinders/sptransport.py`: abstract interface, route descriptors,
  and shared request state.
- `mpisppy/cylinders/spwindow.py`: retained RMA implementation, adapted behind
  `RmaTransport`.
- `mpisppy/cylinders/spcommunicator.py`: transport-neutral layout, routing,
  publish/poll, and coherence orchestration.
- `mpisppy/spin_the_wheel.py`: backend selection and lifecycle naming.
- `mpisppy/utils/config.py` and `mpisppy/utils/cfg_vanilla.py`: transport
  option plumbing.
- `mpisppy/tests/`: deterministic transport state-machine tests and MPI parity
  tests.
- `doc/designs/flexible_rank_assignments.md`: update references from a
  full-world window to transport-neutral global-rank routes.

## Completion criteria

The backend is ready to replace RMA when:

1. every existing cylinder and flexible-rank test passes with both transports;
2. synchronized readers cannot commit different write IDs;
3. shutdown leaves no active MPI requests;
4. bounded-memory behavior is demonstrated under a deliberately stalled
   consumer;
5. the MVAPICH 2.3.7 cross-node production configuration completes repeated
   runs with the p2p backend while the minimal RMA reproducer continues to
   demonstrate the vendor defect;
6. startup and runtime diagnostics make backend choice, coalescing, and stalled
   progress observable.
