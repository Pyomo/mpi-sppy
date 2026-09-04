###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2025, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# Sep 2026: windows are byte-addressed records. Each double payload retains
# its 512-bit padding and is surrounded by transmitted canaries and local-only
# red zones; the record also reserves metadata for a later publication protocol.

from mpisppy import MPI

import numpy as np
import numpy.typing as nptyping

import enum
import struct
from typing import NamedTuple

import pyomo.environ as pyo

class Field(enum.IntEnum):
    SHUTDOWN=-1000
    NONANTS_VALS=1
    DUALS=2
    RELAXED_NONANTS_VALS=3
    BEST_OBJECTIVE_BOUNDS=100 # Both inner and outer bounds from the hub. Layout: [OUTER INNER ID]
    OBJECTIVE_INNER_BOUND=101
    OBJECTIVE_OUTER_BOUND=102
    EXPECTED_REDUCED_COST=200
    SCENARIO_REDUCED_COST=201
    CROSS_SCENARIO_CUT=300
    CROSS_SCENARIO_COST=400
    NONANT_LOWER_BOUNDS=500
    NONANT_UPPER_BOUNDS=501
    BEST_XHAT=600 # buffer having the best xhat and its total cost per scenario
    RECENT_XHATS=601 # buffer having some recent xhats and their total cost per scenario
    XFEAS=602 # buffer having a feasible x and its total cost per scenario
    XHAT_FEASIBILITY_CUT=700  # feasibility cuts emitted by xhat spokes
    WHOLE=1_000_000


field_length_components = pyo.ConcreteModel()
field_length_components._local_nonant_length = pyo.Param(mutable=True)
field_length_components._local_scenario_length = pyo.Param(mutable=True)
field_length_components._total_number_nonants = pyo.Param(mutable=True)
field_length_components._total_number_scenarios = pyo.Param(mutable=True)

# these could be modified by the user...
field_length_components.total_number_recent_xhats = pyo.Param(mutable=True, initialize=10, within=pyo.NonNegativeIntegers)
# max cuts an xhat spoke may emit per iteration (set by the xhatter
# spoke from cfg.xhat_feasibility_cuts_count before field registration)
field_length_components.xhat_feasibility_cuts_per_iter = pyo.Param(mutable=True, initialize=0, within=pyo.NonNegativeIntegers)

_field_lengths = {
        Field.SHUTDOWN : 1,
        Field.NONANTS_VALS : field_length_components._local_nonant_length,
        Field.DUALS : field_length_components._local_nonant_length,
        Field.RELAXED_NONANTS_VALS : field_length_components._local_nonant_length,
        Field.BEST_OBJECTIVE_BOUNDS : 2,
        Field.OBJECTIVE_INNER_BOUND : 1,
        Field.OBJECTIVE_OUTER_BOUND : 1,
        Field.EXPECTED_REDUCED_COST : field_length_components._total_number_nonants,
        Field.SCENARIO_REDUCED_COST : field_length_components._local_nonant_length,
        # CROSS_SCENARIO_CUT is global-sized: every rank holds one cut row (eta
        # coef + nonant coefs + constant) for every scenario, replicated across a
        # cylinder's ranks. CROSS_SCENARIO_COST is local-sized: one recourse-cost
        # (eta) value per global scenario, for each *local* scenario. Both must use
        # the global scenario count (_total_number_scenarios), not the local one.
        Field.CROSS_SCENARIO_CUT : field_length_components._total_number_scenarios * (field_length_components._total_number_nonants + 1 + 1),
        Field.CROSS_SCENARIO_COST : field_length_components._total_number_scenarios * field_length_components._local_scenario_length,
        Field.NONANT_LOWER_BOUNDS : field_length_components._total_number_nonants,
        Field.NONANT_UPPER_BOUNDS : field_length_components._total_number_nonants,
        Field.BEST_XHAT : field_length_components._local_nonant_length + field_length_components._local_scenario_length,
        Field.RECENT_XHATS : field_length_components.total_number_recent_xhats * (field_length_components._local_nonant_length + field_length_components._local_scenario_length),
        Field.XFEAS: field_length_components._local_nonant_length + field_length_components._local_scenario_length,
        # rows: [constant, nonant_coef_1, ..., nonant_coef_N]; trailing slot holds the
        # actual number of cuts written this batch (0..per_iter).
        Field.XHAT_FEASIBILITY_CUT : field_length_components.xhat_feasibility_cuts_per_iter * (field_length_components._total_number_nonants + 1) + 1,
        }


class FieldLengths:
    def __init__(self, opt):
        number_nonants = (
            sum(
                len(s._mpisppy_data.nonant_indices)
                for s in opt.local_scenarios.values()
               )
        )

        field_length_components._local_nonant_length.value = number_nonants
        field_length_components._local_scenario_length.value = len(opt.local_scenarios)
        field_length_components._total_number_nonants.value = opt.nonant_length
        field_length_components._total_number_scenarios.value = len(opt.all_scenario_names)
        # user-tunable cap on feasibility cuts per iteration (0 = off).
        # getattr guard: FieldLengths only requires the nonant/scenario
        # attributes above, so tolerate an opt without .options (feature
        # off in that case).
        field_length_components.xhat_feasibility_cuts_per_iter.value = int(
            getattr(opt, "options", {}).get("xhat_feasibility_cuts_count", 0)
        )

        self._field_lengths = {k : pyo.value(v) for k, v in _field_lengths.items()}

        # reset the field_length_components
        for p in field_length_components.component_data_objects():
            # leave user-set parameter alone, just clear the
            # "private" parameters
            if p.name.startswith("_"):
                p.clear()

    def __getitem__(self, field: Field):
        return self._field_lengths[field]


PAD_N_DOUBLES = 8  # padding granularity in doubles (PAD_N_DOUBLES*8 bytes)
METADATA_NBYTES = 16
CANARY_NBYTES = 16
RED_ZONE_NBYTES = 16
RED_ZONE_CANARY = bytes.fromhex("d37c" * (RED_ZONE_NBYTES // 2))
PUBLICATION_BUSY_OFFSET = 0
PUBLICATION_GENERATION_OFFSET = 8
PUBLICATION_IDLE = 0
PUBLICATION_BUSY = 1
PUBLICATION_MAX_GENERATION = (1 << 64) - 1
_PUBLICATION_METADATA = struct.Struct("<B7xQ")


def transmitted_canary(window_rank: int, field: Field, side: str,
                       padded_len: int) -> bytes:
    """Return a canary identifying one boundary of a published field.

    The fixed-width representation makes a successful guard check evidence
    that a Get reached the requested window rank, field, side, and padded
    extent--not merely some other guarded record.
    """
    if side == "left":
        magic = b"SPL1"
    elif side == "right":
        magic = b"SPR1"
    else:
        raise ValueError(f"Unknown canary side {side!r}")
    return struct.pack("<4sIiI", magic, window_rank, int(field), padded_len)


def padded_len_n_doubles(logical_len: int) -> int:
    """Round up length (in doubles) to a multiple of PAD_N_DOUBLES doubles (PAD_N_DOUBLES*8 bytes)."""
    if PAD_N_DOUBLES < 1:
        raise ValueError(f"PAD_N_DOUBLES must be >= 1, got {PAD_N_DOUBLES}")
    return ((logical_len + PAD_N_DOUBLES - 1) // PAD_N_DOUBLES) * PAD_N_DOUBLES


class FieldLayout(NamedTuple):
    """Location and extent of one field in a byte-addressed MPI window."""

    offset_bytes: int
    logical_len: int
    padded_len: int

    @property
    def metadata_offset_bytes(self) -> int:
        return self.offset_bytes - CANARY_NBYTES - RED_ZONE_NBYTES - METADATA_NBYTES

    @property
    def left_red_zone_offset_bytes(self) -> int:
        return self.offset_bytes - CANARY_NBYTES - RED_ZONE_NBYTES

    @property
    def left_canary_offset_bytes(self) -> int:
        return self.offset_bytes - CANARY_NBYTES

    @property
    def right_canary_offset_bytes(self) -> int:
        return self.offset_bytes + self.padded_nbytes

    @property
    def right_red_zone_offset_bytes(self) -> int:
        return self.right_canary_offset_bytes + CANARY_NBYTES

    @property
    def padded_nbytes(self) -> int:
        return self.padded_len * MPI.DOUBLE.size

    @property
    def transfer_offset_bytes(self) -> int:
        return self.left_canary_offset_bytes

    @property
    def transfer_nbytes(self) -> int:
        return CANARY_NBYTES + self.padded_nbytes + CANARY_NBYTES

    @property
    def record_nbytes(self) -> int:
        return (METADATA_NBYTES + RED_ZONE_NBYTES + self.transfer_nbytes
                + RED_ZONE_NBYTES)


class SPWindow:

    def __init__(self, my_fields: dict, strata_comm: MPI.Comm, field_order=None):
        """Allocate guarded, byte-addressed records for the local fields.

        A field record contains publication metadata, a left red zone, a left
        transmitted canary, the padded double payload, a right transmitted
        canary, and a right red zone. Put/Get transfer both canaries and the
        complete padded payload; the red zones remain local to each buffer.

        ``Field.WHOLE`` is retained in ``buffer_layout`` as a description of
        the complete allocation for compatibility. It is not an addressable
        record and therefore cannot be passed to :meth:`put` or :meth:`get`.
        """
        self.strata_comm = strata_comm
        self.strata_rank = strata_comm.Get_rank()

        # Sorted by the integer value of the enumeration value
        if field_order is None:
            self.field_order = sorted(f for f in my_fields.keys() if f != Field.WHOLE)
        else:
            self.field_order = [f for f in field_order if f != Field.WHOLE]

        record_offset_bytes = 0
        layout = {}

        for field in self.field_order:
            logical_len, padded_len = my_fields[field]

            if padded_len < logical_len:
                raise ValueError(f"{field=} has {padded_len=} < {logical_len=}")

            # padded_len must be a multiple of 8 doubles (64 bytes)
            expected_padded = padded_len_n_doubles(logical_len)
            if padded_len != expected_padded:
                raise ValueError(
                    f"{field=} has {logical_len=} but {padded_len=}; expected padded_len={expected_padded}"
                )

            layout[field] = FieldLayout(
                offset_bytes=(record_offset_bytes + METADATA_NBYTES
                              + RED_ZONE_NBYTES + CANARY_NBYTES),
                logical_len=logical_len,
                padded_len=padded_len,
            )
            record_offset_bytes += layout[field].record_nbytes

        # WHOLE covers the entire padded window extent
        if Field.WHOLE not in layout:
            total_logical = sum(layout[f][1] for f in layout.keys() if f != Field.WHOLE)
            total_padded = record_offset_bytes // MPI.DOUBLE.size
            layout[Field.WHOLE] = FieldLayout(0, total_logical, total_padded)

        self.buffer_layout = layout
        total_buffer_length = record_offset_bytes // MPI.DOUBLE.size
        window_size_bytes = record_offset_bytes

        self.buffer_length = total_buffer_length
        self.window = MPI.Win.Allocate(window_size_bytes, 1, comm=strata_comm)

        # Bind a byte view to the heterogeneous window memory. Each field's
        # payload remains a naturally-aligned array of doubles.
        self.buff = np.ndarray(
            dtype=np.uint8,
            shape=(window_size_bytes,),
            buffer=self.window.tomemory(),
        )
        self.buff[:] = 0
        self._field_views = {}

        # Initialize guard regions, publication metadata, and payloads.
        self._publication_generations = {}
        for field, field_layout in self.buffer_layout.items():
            if field == Field.WHOLE:
                continue
            metadata = _PUBLICATION_METADATA.pack(PUBLICATION_IDLE, 0)
            metadata_start = field_layout.metadata_offset_bytes
            self.buff[metadata_start:metadata_start + METADATA_NBYTES] = \
                np.frombuffer(metadata, dtype=np.uint8)
            self._publication_generations[field] = 0
            self._set_guard_bytes(field, field_layout)
            payload = np.ndarray(
                dtype="d",
                shape=(field_layout.padded_len,),
                buffer=self.window.tomemory(),
                offset=field_layout.offset_bytes,
            )
            payload[:] = np.nan
            payload[field_layout.logical_len - 1] = 0.0
            self._field_views[field] = payload

        # Keep one passive-target access epoch open for the lifetime of the
        # window. Individual operations complete at their target with Flush;
        # they do not repeatedly acquire and release target locks.
        self.window.Lock_all()
        self._epoch_open = True
        # Publish the direct initialization above on separate-memory-model
        # implementations. This is normally a no-op for unified windows.
        self.window.Sync()

        # Besides exchanging layouts, this ensures every target has completed
        # initialization and Sync before any constructor returns to its caller.
        self.strata_buffer_layouts = strata_comm.allgather(self.buffer_layout)

    def free(self):
        if self.window is not None:
            guard_error = None
            try:
                for field in self.field_order:
                    self._verify_window_guards(field, "free", "before")
            except RuntimeError as error:
                guard_error = error
            if self._epoch_open:
                self.window.Unlock_all()
                self._epoch_open = False
            self.window.Free()
            self.buff = None
            self.buffer_layout = None
            self.buffer_length = 0
            self.window = None
            self.strata_buffer_layouts = None
            self._field_views = None
            self._publication_generations = None
            if guard_error is not None:
                raise guard_error
        return

    def _set_guard_bytes(self, field: Field,
                         field_layout: FieldLayout) -> None:
        left_red = field_layout.left_red_zone_offset_bytes
        left_canary = field_layout.left_canary_offset_bytes
        right_canary = field_layout.right_canary_offset_bytes
        right_red = field_layout.right_red_zone_offset_bytes
        self.buff[left_red:left_red + RED_ZONE_NBYTES] = \
            np.frombuffer(RED_ZONE_CANARY, dtype=np.uint8)
        self.buff[left_canary:left_canary + CANARY_NBYTES] = \
            np.frombuffer(transmitted_canary(
                self.strata_rank, field, "left", field_layout.padded_len),
                dtype=np.uint8,
            )
        self.buff[right_canary:right_canary + CANARY_NBYTES] = \
            np.frombuffer(transmitted_canary(
                self.strata_rank, field, "right", field_layout.padded_len),
                dtype=np.uint8,
            )
        self.buff[right_red:right_red + RED_ZONE_NBYTES] = \
            np.frombuffer(RED_ZONE_CANARY, dtype=np.uint8)

    @staticmethod
    def _make_guarded_transfer_buffer(field_layout: FieldLayout, field: Field,
                                      target_rank: int):
        storage = np.empty(
            RED_ZONE_NBYTES + field_layout.transfer_nbytes + RED_ZONE_NBYTES,
            dtype=np.uint8,
        )
        storage[:RED_ZONE_NBYTES] = np.frombuffer(
            RED_ZONE_CANARY, dtype=np.uint8)
        storage[-RED_ZONE_NBYTES:] = np.frombuffer(
            RED_ZONE_CANARY, dtype=np.uint8)
        transfer = storage[RED_ZONE_NBYTES:-RED_ZONE_NBYTES]
        transfer[:CANARY_NBYTES] = np.frombuffer(
            transmitted_canary(
                target_rank, field, "left", field_layout.padded_len),
            dtype=np.uint8,
        )
        transfer[-CANARY_NBYTES:] = np.frombuffer(
            transmitted_canary(
                target_rank, field, "right", field_layout.padded_len),
            dtype=np.uint8,
        )
        return storage, transfer

    @classmethod
    def _make_transfer_buffer(cls, values, field_layout: FieldLayout,
                              field: Field, target_rank: int):
        storage, transfer = cls._make_guarded_transfer_buffer(
            field_layout, field, target_rank)
        payload = transfer[
            CANARY_NBYTES:CANARY_NBYTES + field_layout.padded_nbytes
        ].view("d")
        payload[:] = values
        return storage, transfer

    def _raise_guard_error(self, field, operation, phase, region,
                           actual, expected, target_rank=None):
        mismatch = np.flatnonzero(actual != expected)
        first = int(mismatch[0])
        target = "" if target_rank is None else f" target_rank={target_rank}"
        raise RuntimeError(
            f"SPWindow guard corruption: operation={operation} phase={phase} "
            f"local_rank={self.strata_rank}{target} field={field.name} "
            f"region={region} byte_offset={first} "
            f"expected=0x{int(expected[first]):02x} "
            f"actual=0x{int(actual[first]):02x}"
        )

    def _verify_pattern(self, actual, pattern, field, operation, phase,
                        region, target_rank=None):
        expected = np.frombuffer(pattern, dtype=np.uint8)
        if not np.array_equal(actual, expected):
            self._raise_guard_error(
                field, operation, phase, region, actual, expected,
                target_rank=target_rank,
            )

    def _verify_window_guards(self, field, operation, phase):
        field_layout = self.buffer_layout[field]
        regions = (
            ("left-red-zone", field_layout.left_red_zone_offset_bytes,
             RED_ZONE_NBYTES, RED_ZONE_CANARY),
            ("left-transmitted-canary", field_layout.left_canary_offset_bytes,
             CANARY_NBYTES, transmitted_canary(
                 self.strata_rank, field, "left", field_layout.padded_len)),
            ("right-transmitted-canary", field_layout.right_canary_offset_bytes,
             CANARY_NBYTES, transmitted_canary(
                 self.strata_rank, field, "right", field_layout.padded_len)),
            ("right-red-zone", field_layout.right_red_zone_offset_bytes,
             RED_ZONE_NBYTES, RED_ZONE_CANARY),
        )
        for region, offset, size, pattern in regions:
            self._verify_pattern(
                self.buff[offset:offset + size], pattern, field,
                operation, phase, region,
            )

    def _verify_transfer_guards(self, storage, field_layout, field,
                                operation, phase, target_rank=None):
        if target_rank is None:
            raise ValueError("target_rank is required for transmitted canaries")
        transfer_start = RED_ZONE_NBYTES
        right_canary = transfer_start + field_layout.transfer_nbytes \
            - CANARY_NBYTES
        right_red = transfer_start + field_layout.transfer_nbytes
        regions = (
            ("left-red-zone", storage[:RED_ZONE_NBYTES], RED_ZONE_CANARY),
            ("left-transmitted-canary",
             storage[transfer_start:transfer_start + CANARY_NBYTES],
             transmitted_canary(
                 target_rank, field, "left", field_layout.padded_len)),
            ("right-transmitted-canary",
             storage[right_canary:right_canary + CANARY_NBYTES],
             transmitted_canary(
                 target_rank, field, "right", field_layout.padded_len)),
            ("right-red-zone",
             storage[right_red:right_red + RED_ZONE_NBYTES], RED_ZONE_CANARY),
        )
        for region, actual, pattern in regions:
            self._verify_pattern(
                actual, pattern, field, operation, phase, region,
                target_rank=target_rank,
            )

    @staticmethod
    def _make_guarded_bytes(nbytes: int, contents=None):
        storage = np.empty(RED_ZONE_NBYTES + nbytes + RED_ZONE_NBYTES,
                           dtype=np.uint8)
        expected = np.frombuffer(RED_ZONE_CANARY, dtype=np.uint8)
        storage[:RED_ZONE_NBYTES] = expected
        storage[-RED_ZONE_NBYTES:] = expected
        body = storage[RED_ZONE_NBYTES:-RED_ZONE_NBYTES]
        if contents is not None:
            body[:] = np.frombuffer(contents, dtype=np.uint8)
        return storage, body

    def _verify_byte_buffer_red_zones(self, storage, field, operation, phase,
                                      target_rank):
        self._verify_pattern(
            storage[:RED_ZONE_NBYTES], RED_ZONE_CANARY, field,
            operation, phase, "metadata-left-red-zone",
            target_rank=target_rank,
        )
        self._verify_pattern(
            storage[-RED_ZONE_NBYTES:], RED_ZONE_CANARY, field,
            operation, phase, "metadata-right-red-zone",
            target_rank=target_rank,
        )

    def _put_metadata_bytes(self, contents: bytes, target_offset: int,
                            field: Field) -> None:
        storage, body = self._make_guarded_bytes(len(contents), contents)
        self._verify_byte_buffer_red_zones(
            storage, field, "put-metadata", "before", self.strata_rank)
        self.window.Put(
            (body, len(contents), MPI.BYTE),
            self.strata_rank,
            (target_offset, len(contents), MPI.BYTE),
        )
        self.window.Flush(self.strata_rank)
        self._verify_byte_buffer_red_zones(
            storage, field, "put-metadata", "after", self.strata_rank)

    def _get_publication_metadata(self, strata_rank: int, field: Field,
                                  field_layout: FieldLayout):
        storage, body = self._make_guarded_bytes(METADATA_NBYTES)
        self._verify_byte_buffer_red_zones(
            storage, field, "get-metadata", "before", strata_rank)
        self.window.Get(
            (body, METADATA_NBYTES, MPI.BYTE),
            strata_rank,
            (field_layout.metadata_offset_bytes, METADATA_NBYTES, MPI.BYTE),
        )
        self.window.Flush(strata_rank)
        self._verify_byte_buffer_red_zones(
            storage, field, "get-metadata", "after", strata_rank)
        busy, generation = _PUBLICATION_METADATA.unpack(bytes(body))
        if busy not in (PUBLICATION_IDLE, PUBLICATION_BUSY):
            raise RuntimeError(
                "SPWindow publication metadata corruption: "
                f"local_rank={self.strata_rank} target_rank={strata_rank} "
                f"field={field.name} busy={busy}"
            )
        return busy, generation

    #### Functions ####
    def get(self, dest: nptyping.ArrayLike, strata_rank: int, field: Field,
            item_offset: int = 0, item_count: int = None):
        """Read a remote rank's buffer for ``field`` into ``dest``.

        By default the whole padded field is transferred (``dest`` must be
        ``padded_len`` long), preserving the original behavior.  When
        ``item_count`` is given, only that many doubles are read, starting
        ``item_offset`` doubles into the field -- a partial read used for
        multi-source assembly across cylinders with different rank counts
        (see ``overlap_map.py``).  ``dest`` must then be ``item_count`` long.

        Publication metadata is sampled before and after the guarded payload.
        A read overlapping a Put is discarded and retried until both samples
        identify the same idle generation.
        """
        if field == Field.WHOLE:
            raise ValueError("Field.WHOLE is not an addressable guarded record")
        assert (0 <= strata_rank < len(self.strata_buffer_layouts))

        that_layout = self.strata_buffer_layouts[strata_rank]
        assert field in that_layout

        field_layout = that_layout[field]
        padded_len = field_layout.padded_len

        if item_count is None:
            count = padded_len
            item_offset = 0
        else:
            assert item_offset >= 0 and item_count >= 0
            assert item_offset + item_count <= padded_len, \
                f"{field=} partial get {item_offset=}+{item_count=} exceeds {padded_len=}"
            count = item_count
        assert np.size(dest) == count

        window = self.window
        storage, transfer = self._make_guarded_transfer_buffer(
            field_layout, field, strata_rank)
        while True:
            metadata_before = self._get_publication_metadata(
                strata_rank, field, field_layout)
            if metadata_before[0] == PUBLICATION_BUSY:
                continue

            self._verify_transfer_guards(
                storage, field_layout, field, "get", "before",
                target_rank=strata_rank,
            )
            window.Get(
                (transfer, field_layout.transfer_nbytes, MPI.BYTE),
                strata_rank,
                (field_layout.transfer_offset_bytes,
                 field_layout.transfer_nbytes,
                 MPI.BYTE),
            )
            # Flush provides local completion, so the destination can be
            # inspected once the second metadata snapshot accepts it.
            window.Flush(strata_rank)
            metadata_after = self._get_publication_metadata(
                strata_rank, field, field_layout)
            if metadata_before == metadata_after \
                    and metadata_after[0] == PUBLICATION_IDLE:
                break

        self._verify_transfer_guards(
            storage, field_layout, field, "get", "after",
            target_rank=strata_rank,
        )
        payload = transfer[
            CANARY_NBYTES:CANARY_NBYTES + field_layout.padded_nbytes
        ].view("d")
        dest[:] = payload[item_offset:item_offset + count]
        return

    def put(self, values: nptyping.ArrayLike, field: Field):
        if field == Field.WHOLE:
            raise ValueError("Field.WHOLE is not an addressable guarded record")
        field_layout = self.buffer_layout[field]
        padded_len = field_layout.padded_len
        assert np.size(values) == padded_len

        storage, transfer = self._make_transfer_buffer(
            values, field_layout, field, self.strata_rank)
        self._verify_transfer_guards(
            storage, field_layout, field, "put", "before",
            target_rank=self.strata_rank,
        )
        self._verify_window_guards(field, "put", "before")
        window = self.window
        metadata_offset = field_layout.metadata_offset_bytes
        generation = self._publication_generations[field]
        if generation >= PUBLICATION_MAX_GENERATION:
            raise OverflowError(
                "SPWindow publication generation exhausted: "
                f"local_rank={self.strata_rank} field={field.name} "
                f"generation={generation}"
            )

        # Mark the record busy and complete that update before modifying the
        # guarded payload. Readers accept a snapshot only when clean metadata
        # with one generation brackets the payload Get.
        self._put_metadata_bytes(
            bytes((PUBLICATION_BUSY,)),
            metadata_offset + PUBLICATION_BUSY_OFFSET,
            field,
        )
        window.Put(
            (transfer, field_layout.transfer_nbytes, MPI.BYTE),
            self.strata_rank,
            (field_layout.transfer_offset_bytes,
             field_layout.transfer_nbytes,
             MPI.BYTE),
        )
        window.Flush(self.strata_rank)
        self._verify_transfer_guards(
            storage, field_layout, field, "put", "after",
            target_rank=self.strata_rank,
        )
        self._verify_window_guards(field, "put", "after")

        generation += 1
        self._put_metadata_bytes(
            struct.pack("<Q", generation),
            metadata_offset + PUBLICATION_GENERATION_OFFSET,
            field,
        )
        self._put_metadata_bytes(
            bytes((PUBLICATION_IDLE,)),
            metadata_offset + PUBLICATION_BUSY_OFFSET,
            field,
        )
        self._publication_generations[field] = generation
        return

## End SPWindow
