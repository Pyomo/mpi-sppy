###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
""" Conventional wisdom seems to be that we should use Put calls locally (i.e.
    a process should Put() into its own buffer), and Get calls for
    communication (i.e. call Get on a remote target, rather than your local
    buffer). The following implementation uses this paradigm.

    The communication in this paradigm is a star graph, with the hub at the
    center and the spokes on the outside. Each spoke is concerned only
    with the hub, but the hub must track information about all of the
    spokes.

    Separate hub and spoke classes for memory/window management?
"""

import abc
import time
import logging
import numpy as np
from math import inf

from mpisppy import MPI, global_toc
from mpisppy.cylinders.spwindow import Field, FieldLengths, SPWindow, padded_len_n_doubles
from mpisppy.cylinders.overlap_map import OverlapSegment, compute_overlap_segments
from mpisppy.utils.rank_apportionment import (
    apportion_ranks,
    cylinder_bases,
    rank_to_cylinder,
)

logger = logging.getLogger(__name__)

# Fields whose buffer length is the same on every rank (global-sized) or a
# fixed-size record (scalar): Categories 3 and 4 in the design doc. Under
# unequal rank counts these are read single-source from the publishing
# cylinder's base rank -- there is nothing to assemble. Every other field is
# local-sized / per-scenario (Categories 1 and 2) and is read via the
# overlap-map multi-source path.
_GLOBAL_OR_SCALAR_FIELDS = frozenset((
    Field.SHUTDOWN,
    Field.OBJECTIVE_INNER_BOUND,
    Field.OBJECTIVE_OUTER_BOUND,
    Field.BEST_OBJECTIVE_BOUNDS,
    Field.NONANT_LOWER_BOUNDS,
    Field.NONANT_UPPER_BOUNDS,
    Field.EXPECTED_REDUCED_COST,
))

# Per-scenario fields that require *strict* coherence when assembled across
# ranks: every contributing source must be at the same write_id, or the read
# is rejected and retried. DUALS carries the PH multipliers W_s, whose
# normalization sum_s p_s W_s = 0 holds only within a single iteration --
# stitching W from mixed iterations yields an invalid Lagrangian bound. Every
# other per-scenario field uses *relaxed* coherence (the assembled value may
# mix iterations; its consumers re-evaluate, so it is always honest).
_STRICT_COHERENCE_FIELDS = frozenset((
    Field.DUALS,
))

# Category-2 xhat fields: per-scenario layout [first-stage nonants, cost] whose
# nonant portion is an incumbent first-stage decision identical across all
# scenarios by non-anticipativity (the candidate fixes the first stage when it
# is evaluated). Multi-source assembly fills each scenario's block from the rank
# that holds it; if those ranks sit at different write_ids the first-stage
# values could end up inconsistent across scenarios. After assembly we restore
# NAC by overwriting every scenario's first-stage portion with a single coherent
# reference (see _enforce_first_stage_nac). The per-scenario cost slot is left
# alone -- it is genuinely per-scenario (Category 1). XFEAS is *not* here: it
# carries per-scenario *distinct* iterates (its consumer tries each scenario's x
# as its own candidate), so it is assembled per-scenario with no NAC fix-up.
_FIRST_STAGE_NAC_FIELDS = frozenset((
    Field.BEST_XHAT,
    Field.RECENT_XHATS,
))


def reduce_source_write_ids(source_ids, strict: bool) -> int:
    """Reduce the per-source write_ids of a multi-source read to the single id
    this rank will compare against its last-accepted id.

    Args:
        source_ids: write_ids read from each contributing remote rank.
        strict: if True, the sources must agree on one id (strict coherence);
            a disagreement returns the sentinel ``-1`` so the read is rejected
            (it is < any real id, and it breaks the cross-reader agreement
            check, so every reader rejects together). If False (relaxed), the
            floor over sources is taken -- a mixed-iteration assembly is
            accepted, which is fine for fields whose consumers re-evaluate.

    Returns:
        int: the accepted write_id, or ``-1`` to reject (strict mismatch),
        or ``0`` when there are no sources.
    """
    if not source_ids:
        return 0
    if strict:
        return source_ids[0] if len(set(source_ids)) == 1 else -1
    return min(source_ids)

def communicator_array(data_length: int):
    """
    Allocate an MPI memory region with a padded length (multiple of 8 doubles = 64B),
    but expose a logical view of length (data_length + 1) where the last element is
    the read/write id.

    Returns:
        full_arr:  padded array (used for SPWindow put/get)
        logical_arr: logical view (data + id), last element is id
        data_length: number of data entries (excluding id)
        logical_len: data_length + 1
        padded_len: multiple-of-8 length used in the MPI window
    """
    logical_len = data_length + 1
    padded_len = padded_len_n_doubles(logical_len)

    itemsize = np.dtype("d").itemsize
    mem = MPI.Alloc_mem(padded_len * itemsize)

    full_arr = np.frombuffer(mem, dtype="d", count=padded_len)
    full_arr[:] = np.nan

    logical_arr = full_arr[:logical_len]
    logical_arr[-1] = 0.0

    return full_arr, logical_arr, data_length, logical_len, padded_len


class FieldArray:
    """
    The intention here is that these are passive data holding classes. That is, other classes are
    expected to update the internal fields. The lone exception to this is the read/write id field.
    See the `SendArray` and `RecvArray` classes for how that field is updated.
    Wrapper around an MPI-allocated numpy buffer with:
      - a padded "window" array used for MPI RMA
      - a logical view used by mpi-sppy code (data + id)
    """

    def __init__(self, length: int):
        # length is the data length (excluding the id)
        (self._full_array,
         self._array,
         self._data_length,
         self._logical_len,
         self._padded_len) = communicator_array(length)
        self._id = 0

    def window_array(self) -> np.typing.NDArray:
        """Full padded array (used for SPWindow get/put)."""
        return self._full_array

    def array(self) -> np.typing.NDArray:
        """Logical array (data + id)."""
        return self._array

    def value_array(self) -> np.typing.NDArray:
        """Data only (excludes id)."""
        return self._array[:self._data_length]

    def padded_len(self) -> int:
        return self._padded_len

    def logical_len(self) -> int:
        return self._logical_len

    def data_len(self) -> int:
        return self._data_length

    def __getitem__(self, key):
        # Preserve old behavior: indexing into the logical view.
        return self._array[key]

    def id(self) -> int:
        return self._id


class SendArray(FieldArray):

    def __init__(self, length: int):
        super().__init__(length)
        return

    def __setitem__(self, key, value):
        # Use value_array function to hide the read/write id field so it is
        # not accidentally overwritten
        np_array = self.value_array()
        np_array[key] = value
        return

    def _next_write_id(self) -> int:
        """
        Updates the internal id field to the next write id, sets that id in the
        field data array, and returns that id
        """
        self._id += 1
        self._array[-1] = self._id
        return self._id


class RecvArray(FieldArray):

    def __init__(self, length: int):
        super().__init__(length)
        self._is_new = False
        return

    def is_new(self) -> bool:
        return self._is_new

    def _pull_id(self) -> int:
        """
        Updates the internal id field to the write id currently held in the numpy buffer
        and returns that id
        """
        self._id = int(self._array[-1])
        return self._id


class _CircularBuffer:
    """
    The circular buffer is meant for holding several versions of a Field
    (defined by the `buffer_size`). The `data` object is an instance of
    `FieldArray`.

    To know where in the buffer we are, we use the FieldArray._id. The layout
    looks like this for a `buffer_size` of 4:

    |--0--|--1--|--2--|--3--|id|

    The id % buffer_size tells us which data point is the most recent, such
    that individual ids are not needed for each instance.
    """

    def __init__(self, data: FieldArray, field_length: int, buffer_size: int):
        # last byte is the "write pointer"
        assert len(data.value_array()) == field_length * buffer_size
        self.data = data
        self._field_length = field_length
        self._buffer_size = buffer_size

    def _get_value_array(self, read_write_index):
        position = read_write_index % self._buffer_size
        arr = self.data.array()  # logical view
        return arr[(position*self._field_length):((position+1)*self._field_length)]


class SendCircularBuffer(_CircularBuffer):

    def next_value_array_reference(self):
        # NOTE: The id gets incremented in the call
        #       to `put_send_buffer`, which is necessarily
        #       called *after* this method. Therefore
        #       we start at 0 and go up, and when sent
        #       will be the id of the next *open* position
        return self._get_value_array(self.data.id())


class RecvCircularBuffer(_CircularBuffer):
    # The _read_id tells us where we last read from, and the
    # (data.id % buffer_size) - 1
    # has the last place written to. Therefore, we know which
    # items in the buffer are new based on their difference.

    def __init__(self, data: RecvArray, field_length: int, buffer_size: int):
        super().__init__(data, field_length, buffer_size)
        self._read_id = 0

    def most_recent_value_arrays(self):
        # if the writes have already "wrapped around" the buffer,
        # we need to fast-forward the read index so we don't read
        # the same data multiple times
        while self.data.id() > self._read_id + self._buffer_size:
            self._read_id += 1
        while self._read_id < self.data.id():
            yield self._get_value_array(self._read_id)
            self._read_id += 1


class SPCommunicator:
    """ Base class for communicator objects. Each communicator object should register
        as a class attribute what Field attributes it provides in its buffer
        or expects to receive from another SPCommunicator object.
    """
    send_fields = ()
    receive_fields = (Field.OBJECTIVE_INNER_BOUND, Field.OBJECTIVE_OUTER_BOUND,
                      Field.NONANT_LOWER_BOUNDS, Field.NONANT_UPPER_BOUNDS,)

    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, communicators, options=None):
        self.fullcomm = fullcomm
        self.strata_comm = strata_comm
        self.cylinder_comm = cylinder_comm
        self.communicators = communicators
        self.global_rank = fullcomm.Get_rank()
        self.cylinder_rank = cylinder_comm.Get_rank()

        # Flexible rank assignments (Option D in the design doc): when
        # strata_comm is None the cylinders have unequal rank counts, so the
        # strata grouping is undefined and the window lives on fullcomm with
        # peers addressed by global rank via overlap maps. The cylinder index
        # plays the role strata_rank does on the equal path -- it indexes
        # `communicators` and marks the hub as 0 -- so we set strata_rank to it
        # for the code paths shared with the equal-rank case.
        self._flex_ranks = strata_comm is None
        if self._flex_ranks:
            rank_ratios = [d.get("rank_ratio", 1.0) for d in communicators]
            self.rank_counts = apportion_ranks(rank_ratios, fullcomm.Get_size())
            self._cylinder_bases = cylinder_bases(self.rank_counts)
            self.cylinder_index, _ = rank_to_cylinder(self.global_rank, self.rank_counts)
            self.strata_rank = self.cylinder_index
            self.n_spokes = len(communicators) - 1
        else:
            assert len(communicators) == strata_comm.Get_size()
            self.strata_rank = strata_comm.Get_rank()
            self.cylinder_index = self.strata_rank
            self.n_spokes = strata_comm.Get_size() - 1

        self.opt = spbase_object
        self.inst_time = time.time() # For diagnostics
        if options is None:
            self.options = dict()
        else:
            self.options = options

        # Common fields for spokes and hubs
        self.receive_buffers = {}
        self.send_buffers = {}
        # key: Field, value: list of (strata_rank, SPComm, buffer) with that Field
        self.receive_field_spcomms = {}

        # Overlap-map state for the unequal-rank path; empty on the equal path.
        # Keyed by (field, peer_cylinder_index).
        self.overlap_maps = {}            # -> list[OverlapSegment] (global ranks)
        self._overlap_source_ranks = {}   # -> sorted distinct source global ranks

        # setup FieldLengths which calculates
        # the length of each buffer type based
        # on the problem data
        self._field_lengths = FieldLengths(self.opt)

        self.window = None

        # attach the SPCommunicator to
        # the SPBase object
        self.opt.spcomm = self

        # for communicating with bounders
        self.latest_ib_char = None
        self.latest_ob_char = None
        self.last_ib_idx = None
        self.last_ob_idx = None
        self.initialize_bound_values()

        return

    def _make_key(self, field: Field, origin: int):
        """
        Given a field and an origin (i.e. a strata_rank), generate a key for indexing
        into the self.receive_buffers dictionary and getting the corresponding RecvArray.

        Undone by `_split_key`. Currently, the key is simply a Tuple[field, origin].
        """
        return (field, origin)

    def _split_key(self, key) -> tuple[Field, int]:
        """
        Take the given key and return a tuple (field, origin) where origin in the strata_rank
        from which the field comes.

        Undoes `_make_key`.  Currently, this is a no-op.
        """
        return key

    def _build_window_spec(self) -> dict[Field, tuple[int, int]]:
        """ Build dict with fields and padded lengths needed for local MPI window
        """
        window_spec = dict()
        for (field, buf) in self.send_buffers.items():
            window_spec[field] = (buf.logical_len(), buf.padded_len())
        return window_spec

    def _create_field_rank_mappings(self) -> None:
        self.fields_to_ranks = {}
        self.ranks_to_fields = {}

        if self._flex_ranks:
            # On the unequal-rank path the window spans fullcomm, so
            # strata_buffer_layouts is indexed by global rank. Map fields to
            # peer *cylinder* indices instead (so downstream `origin` values
            # stay cylinder indices, as on the equal path): every rank in a
            # cylinder registers the same send fields, so the base rank's
            # layout names exactly the fields that cylinder publishes.
            for peer in range(len(self.rank_counts)):
                if peer == self.cylinder_index:
                    continue
                base_layout = self.window.strata_buffer_layouts[self._cylinder_bases[peer]]
                self.ranks_to_fields[peer] = []
                for field in base_layout.keys():
                    if field == Field.WHOLE:
                        continue
                    self.fields_to_ranks.setdefault(field, []).append(peer)
                    self.ranks_to_fields[peer].append(field)
            return

        for rank, buffer_layout in enumerate(self.window.strata_buffer_layouts):
            if rank == self.strata_rank:
                continue
            self.ranks_to_fields[rank] = []
            for field in buffer_layout.keys():
                if field == Field.WHOLE:
                    continue
                if field not in self.fields_to_ranks:
                    self.fields_to_ranks[field] = []
                self.fields_to_ranks[field].append(rank)
                self.ranks_to_fields[rank].append(field)

        # print(f"{self.__class__.__name__}: {self.fields_to_ranks=}, {self.ranks_to_fields=}")

    def _validate_recv_field(self, field: Field, origin: int, length: int):
        remote_buffer_layout = self.window.strata_buffer_layouts[origin]
        if field not in remote_buffer_layout:
            raise RuntimeError(
                f"{self.__class__.__name__} on local {self.strata_rank=} "
                f"could not find {field=} on remote rank {origin} with "
                f"class {self.communicators[origin]['spcomm_class']}."
            )

        _, remote_logical_len, remote_padded_len = remote_buffer_layout[field]
        expected_logical_len = length + 1
        expected_padded_len = padded_len_n_doubles(expected_logical_len)

        if remote_logical_len != expected_logical_len or remote_padded_len != expected_padded_len:
            raise RuntimeError(
                f"{self.__class__.__name__} on local {self.strata_rank=} "
                f"{field=} expects (logical={expected_logical_len}, padded={expected_padded_len}) "
                f"but remote rank {origin} advertises (logical={remote_logical_len}, padded={remote_padded_len}) "
                f"with class {self.communicators[origin]['spcomm_class']}."
            )

    def register_recv_field(self, field: Field, origin: int, length: int = -1) -> RecvArray:
        # print(f"{self.__class__.__name__}.register_recv_field, {field=}, {origin=}")
        key = self._make_key(field, origin)
        if length == -1:
            length = self._field_lengths[field]
        if key in self.receive_buffers:
            my_fa = self.receive_buffers[key]
            expected_logical_len = length + 1
            expected_padded_len = padded_len_n_doubles(expected_logical_len)
            assert expected_logical_len == my_fa.logical_len()
            assert expected_padded_len == my_fa.padded_len()
        else:
            # On the equal-rank path the remote buffer for `origin` has the
            # same local sizing as ours, so we can validate it directly. On the
            # unequal-rank path `origin` is a peer cylinder spanning several
            # global ranks of differing sizes, so validation is per-segment
            # (in _build_overlap_map) instead.
            if not self._flex_ranks:
                self._validate_recv_field(field, origin, length)
            my_fa = RecvArray(length)
            self.receive_buffers[key] = my_fa
        ## End if
        return my_fa

    def register_send_field(self, field: Field, length: int = -1) -> SendArray:
        assert field not in self.send_buffers, "Field {} is already registered".format(field)
        if length == -1:
            length = self._field_lengths[field]
        my_fa = SendArray(length)
        self.send_buffers[field] = my_fa
        return my_fa

    @abc.abstractmethod
    def main(self):
        """ Every hub/spoke must have a main function
        """
        pass

    def sync(self):
        """ Every hub/spoke may have a sync function
        """
        pass

    def is_converged(self):
        """ Every hub/spoke may have a is_converged function
        """
        return False

    def finalize(self):
        """ Every hub/spoke may have a finalize function,
            which does some final calculations/flushing to
            disk after convergence
        """
        pass

    def hub_finalize(self):
        """ Every hub may have another finalize function,
            which collects any results from finalize
        """
        pass

    def allreduce_or(self, val):
        return self.opt.allreduce_or(val)

    def make_windows(self) -> None:
        """ Make MPI windows: blocking call for all ranks in `strata_comm`.
        """

        if self.window is not None:
            return

        self.register_send_fields()

        window_spec = self._build_window_spec()
        # Equal-rank: window on strata_comm (rank i of every cylinder),
        # addressed by strata_rank. Unequal-rank: window on fullcomm,
        # addressed by global rank via overlap maps (strata_comm is None).
        window_comm = self.fullcomm if self._flex_ranks else self.strata_comm
        self.window = SPWindow(window_spec, window_comm)

        self._create_field_rank_mappings()
        self.register_receive_fields()

        return

    def free_windows(self) -> None:
        """ Free MPI windows: blocking call for all ranks in `strata_comm`.
        """

        if self.window is None:
            return

        self.receive_buffers = {}
        self.send_buffers = {}
        self.receive_field_spcomms = {}
        self.overlap_maps = {}
        self._overlap_source_ranks = {}

        self.window.free()

        self.window = None

    def is_send_field_registered(self, field: Field) -> bool:
        return field in self.send_buffers

    def register_send_fields(self) -> None:
        for field in self.send_fields:
            self.register_send_field(field)

    def register_receive_fields(self) -> None:
        # print(f"{self.__class__.__name__}: {self.receive_fields=}")
        if self._flex_ranks:
            # local global scenario indices held by this rank (its slice in
            # its own cylinder's partition); used to build overlap maps
            self._local_scen_idxs = list(self.opt._rank_slices[self.cylinder_rank])
        for field in self.receive_fields:
            # NOTE: If this list is empty after this method, it is up
            #       to the caller to raise an error. Sometimes optional
            #       receive fields are perfectly sensible, and sometimes
            #       they are nonsensical.
            self.receive_field_spcomms[field] = []
            for strata_rank, comm in enumerate(self.communicators):
                if strata_rank == self.strata_rank:
                    continue
                cls = comm["spcomm_class"]
                if field != Field.WHOLE and field in self.ranks_to_fields[strata_rank]:
                    buff = self.register_recv_field(field, strata_rank)
                    self.receive_field_spcomms[field].append((strata_rank, cls, buff))
                    # On the unequal-rank path, a per-scenario (local-sized)
                    # field is assembled from several remote buffers; precompute
                    # its overlap map now. Global/scalar fields are read
                    # single-source and need no map.
                    if self._flex_ranks and field not in _GLOBAL_OR_SCALAR_FIELDS:
                        self._build_overlap_map(field, strata_rank)

    def put_send_buffer(self, buf: SendArray, field: Field):
        """ Put the specified values into the specified locally-owned buffer
            for the another cylinder to pick up.

            Notes:
                This automatically updates handles the write id.
        """
        buf._next_write_id()
        self.window.put(buf.window_array(), field)
        return

    def _write_ids_agree(self, new_id: int, synchronize: bool) -> bool:
        """Cross-rank write_id agreement on ``cylinder_comm`` (shared by every
        reader: the equal-rank path and both unequal-rank helpers).

        Consumers that run collectives on freshly-received data (Eobjective
        Allreduce, ROOT bcast, ...) must enter them in lockstep, so every reader
        rank has to agree on whether a read is "new". A synchronous writer
        stamps all of its ranks at one write_id, so when ids agree every rank
        computes the same answer; a transient mixed-id read (the writer Put
        between two readers' Gets) fails here and is rejected, to be retried,
        rather than accepted out of lockstep.

        Returns True if not synchronizing, or if all ranks read the same id.
        """
        if not synchronize:
            return True
        local_val = np.array((new_id,), 'i')
        sum_ids = np.zeros(1, 'i')
        self.cylinder_comm.Allreduce((local_val, MPI.INT),
                                     (sum_ids, MPI.INT),
                                     op=MPI.SUM)
        return new_id * self.cylinder_comm.size == sum_ids[0]

    def _mark_new(self, buf: RecvArray, new_id: int) -> bool:
        """Commit an accepted read: stamp ``new_id`` into the buffer's id slot
        and mark it new. The data itself must already be in ``buf`` (placed by
        the caller's single Get, or by overlap-map assembly for a multi-source
        read). Writing the id slot before ``_pull_id`` lets the multi-source
        path -- whose assembly fills only the data region -- reuse the same
        commit as the single-source/equal-rank paths.
        """
        buf._is_new = True
        buf._array[-1] = new_id
        buf._pull_id()
        return True

    def get_receive_buffer(self,
                           buf: RecvArray,
                           field: Field,
                           origin: int,
                           synchronize: bool = True,
                          ):
        """ Gets the specified values from another cylinder and copies them into
        the specified locally-owned buffer. Updates the write_id in the locally-
        owned buffer, if appropriate.

        Args:
            buf (RecvArray) : Buffer to put the data in
            field (Field) : The source field
            origin (int) : The rank on strata_comm to get the data.
            synchronize (:obj:`bool`, optional) : If True, will only report
                updated data if the write_ids are the same across the cylinder_comm
                are identical. Default: True.

        Returns:
            is_new (bool): Indicates whether the "gotten" values are new,
                based on the write_id.
        """
        if self._flex_ranks:
            # Unequal-rank path: `origin` is a peer cylinder index. A
            # per-scenario field is assembled from several of that cylinder's
            # global ranks via its overlap map; a global/scalar field is read
            # single-source from the cylinder's base rank.
            if (field, origin) in self.overlap_maps:
                return self._flex_get_multi_source(buf, field, origin, synchronize)
            return self._flex_get_single_source(buf, field, origin, synchronize)

        if synchronize:
            self.cylinder_comm.Barrier()

        last_id = buf.id()

        self.window.get(buf.window_array(), origin, field)  # padded view

        new_id = int(buf.array()[-1])  # logical view

        if not self._write_ids_agree(new_id, synchronize):
            buf._is_new = False
            return False

        if new_id > last_id:
            # data is already in buf from the Get above
            return self._mark_new(buf, new_id)
        buf._is_new = False
        return False

    # ------------------------------------------------------------------
    # Unequal-rank (Option D) helpers. These are reached only when
    # self._flex_ranks is True; the equal-rank path above never calls them.
    # ------------------------------------------------------------------

    def _items_per_scen_for_field(self, field: Field):
        """Number of field items each scenario contributes, indexed by global
        scenario index, for a per-scenario field (used to build overlap maps).

        For the nonant-valued fields this is the per-scenario nonant count,
        which is uniform across scenarios in a two-stage problem
        (``opt.nonant_length``). For the xhat fields each scenario contributes a
        ``[nonants, cost]`` block, so the count is ``nonant_length + 1``; the
        circular ``RECENT_XHATS`` returns the same single-version block size and
        is expanded across versions in ``_build_overlap_map``.
        """
        if field in (Field.NONANTS_VALS, Field.RELAXED_NONANTS_VALS, Field.DUALS):
            k = self.opt.nonant_length
            return [k] * len(self.opt.all_scenario_names)
        if field in (Field.BEST_XHAT, Field.RECENT_XHATS, Field.XFEAS):
            # Each scenario's block is [first-stage nonants, per-scenario cost].
            # Two-stage only: under multistage the NAC-redundant portion is
            # per-tree-node (not one global first stage), so the cost-vs-nonant
            # split and the NAC fix-up need per-node sourcing -- a later phase.
            if self.opt.multistage:
                raise NotImplementedError(
                    f"Flexible (unequal) rank assignments: {self.__class__.__name__} "
                    f"reads {field.name} on a multistage problem, whose per-node "
                    f"first-stage sourcing is not implemented yet (two-stage is). "
                    f"Run the cylinders that exchange {field.name} at equal rank "
                    f"counts (rank_ratio == 1.0), or wait for the multistage phase."
                )
            k = self.opt.nonant_length
            return [k + 1] * len(self.opt.all_scenario_names)
        # Reached at startup (window creation), not mid-solve: this cylinder is
        # in a flexible-rank run and reads a per-scenario field whose
        # multi-source assembly across unequal rank counts is not implemented
        # yet. Fail here with an actionable message rather than mis-assembling.
        raise NotImplementedError(
            f"Flexible (unequal) rank assignments: {self.__class__.__name__} "
            f"reads {field.name}, whose multi-source assembly across cylinders "
            f"with different rank counts is not supported yet. Run the cylinders "
            f"that exchange {field.name} at equal rank counts (rank_ratio == 1.0), "
            f"or wait for the phase that adds {field.name} support."
        )

    def _validate_segment(self, field: Field, remote_global_rank: int, seg) -> None:
        """Check that one overlap segment fits within the remote buffer it
        will be read from (the per-segment analogue of _validate_recv_field)."""
        layout = self.window.strata_buffer_layouts[remote_global_rank]
        if field not in layout:
            raise RuntimeError(
                f"{self.__class__.__name__} on cylinder {self.cylinder_index} "
                f"expected {field} on global rank {remote_global_rank} but it "
                f"is not published there."
            )
        _, logical_len, _ = layout[field]
        remote_data_len = logical_len - 1  # exclude the trailing write_id slot
        # The segment occupies the half-open remote range
        # [remote_offset, remote_offset + count); the guard is an overrun
        # check on its exclusive end, so '>' (ending exactly at remote_data_len
        # is legal) and remote_offset belongs in the LHS (a segment may start
        # mid-buffer). A reader needing only a subset of a source's scenarios
        # legitimately ends before remote_data_len, so this is not '!='.
        if seg.remote_offset + seg.count > remote_data_len:
            raise RuntimeError(
                f"{self.__class__.__name__}: overlap segment for {field} reads "
                f"items [{seg.remote_offset}, {seg.remote_offset + seg.count}) "
                f"from global rank {remote_global_rank} whose {field} data "
                f"length is only {remote_data_len}."
            )

    def _build_overlap_map(self, field: Field, peer_cylinder: int) -> None:
        """Precompute, for a per-scenario field and a peer cylinder, the list
        of remote segments (in global-rank terms) that assemble this rank's
        local buffer, validating each against the remote layout."""
        base = self._cylinder_bases[peer_cylinder]
        rc_peer = self.rank_counts[peer_cylinder]
        # peer cylinder's scenario partition for its own rank count
        _, peer_slices, _ = self.opt._scenario_tree.scen_names_to_ranks(rc_peer)
        items_per_scen = self._items_per_scen_for_field(field)
        segments = compute_overlap_segments(
            self._local_scen_idxs, peer_slices, items_per_scen
        )
        for seg in segments:
            seg.remote_rank = base + seg.remote_rank  # peer-local -> global
        if field == Field.RECENT_XHATS:
            # The single-version segments computed above describe one xhat
            # version's per-scenario layout; replicate them across all versions
            # of the circular buffer (see _expand_to_circular_versions).
            segments = self._expand_to_circular_versions(
                segments, peer_slices, base, items_per_scen
            )
        for seg in segments:
            self._validate_segment(field, seg.remote_rank, seg)
        self.overlap_maps[(field, peer_cylinder)] = segments
        self._overlap_source_ranks[(field, peer_cylinder)] = \
            sorted({seg.remote_rank for seg in segments})

    def _expand_to_circular_versions(self, base_segments, peer_slices, base,
                                     items_per_scen):
        """Replicate single-version overlap segments across the versions of the
        ``RECENT_XHATS`` circular buffer.

        ``RECENT_XHATS`` holds ``V`` versions, each a ``BEST_XHAT``-sized block
        laid out per scenario. Within one version the per-scenario blocks are
        contiguous, but version ``v``'s block for a given rank sits at
        ``v * (that rank's version size)`` -- and the version size differs per
        rank because it scales with the rank's local scenario count. So a
        single-version segment is emitted once per version, shifting its remote
        offset by the *source rank's* version size and its local offset by this
        rank's version size. Reading every physical slot faithfully (just
        re-partitioned across scenarios) lets the receiver's ``RecvCircularBuffer``
        interpret the write_id exactly as on the equal-rank path.
        """
        nversions = (self._field_lengths[Field.RECENT_XHATS]
                     // self._field_lengths[Field.BEST_XHAT])
        local_version_size = self._field_lengths[Field.BEST_XHAT]
        # global source rank -> its single-version (BEST_XHAT) block size, in items
        remote_version_size = {
            base + r: sum(items_per_scen[s] for s in scens)
            for r, scens in enumerate(peer_slices)
        }
        expanded = []
        for v in range(nversions):
            for seg in base_segments:
                expanded.append(OverlapSegment(
                    remote_rank=seg.remote_rank,
                    remote_offset=v * remote_version_size[seg.remote_rank] + seg.remote_offset,
                    local_offset=v * local_version_size + seg.local_offset,
                    count=seg.count,
                ))
        return expanded

    def _enforce_first_stage_nac(self, data_view, field):
        """Restore non-anticipativity of the first-stage portion of a Category-2
        xhat field after multi-source assembly (two-stage case).

        Each per-scenario block is ``[first-stage nonants, cost]``. The nonant
        portion is identical across scenarios by construction, but the blocks
        were assembled from possibly different ranks/write_ids, so overwrite
        every scenario's nonants with a single coherent reference -- the first
        scenario's, which came whole from one rank. The cost slots are left
        untouched (genuinely per-scenario). For ``RECENT_XHATS`` this is applied
        independently within each version block. Two-stage only; multistage
        per-node sourcing is guarded in _items_per_scen_for_field.
        """
        k = self.opt.nonant_length            # first-stage nonants per scenario
        block = k + 1                          # [nonants, cost]
        nscen = len(self._local_scen_idxs)
        version_size = nscen * block
        nversions = len(data_view) // version_size  # 1 for BEST_XHAT, V for RECENT_XHATS
        for v in range(nversions):
            v0 = v * version_size
            ref = data_view[v0:v0 + k].copy()  # this version's scenario-0 first stage
            for i in range(1, nscen):
                off = v0 + i * block
                data_view[off:off + k] = ref

    def _flex_get_single_source(self, buf, field, peer_cylinder, synchronize):
        """Read a global/scalar field single-source from a peer cylinder's base
        rank. Every rank of the cylinder holds the identical value, so reading
        the base rank is sufficient and keeps the cross-cylinder write_id
        agreement check (every local rank reads the same remote rank)."""
        window_rank = self._cylinder_bases[peer_cylinder]
        if synchronize:
            self.cylinder_comm.Barrier()
        last_id = buf.id()
        self.window.get(buf.window_array(), window_rank, field)
        new_id = int(buf.array()[-1])
        if not self._write_ids_agree(new_id, synchronize):
            buf._is_new = False
            return False
        if new_id > last_id:
            # data is already in buf from the Get above
            return self._mark_new(buf, new_id)
        buf._is_new = False
        return False

    def _flex_get_multi_source(self, buf, field, peer_cylinder, synchronize):
        """Assemble a per-scenario field from several of a peer cylinder's
        global ranks using the precomputed overlap map.

        Coherence note. The per-source write_ids are reduced to this rank's
        `new_id` by a per-field coherence policy (reduce_source_write_ids):

          * relaxed (default): `new_id` is the floor (min) over sources -- the
            assembled value may mix iterations, which is fine for fields whose
            consumers re-evaluate (NONANTS_VALS, ...);
          * strict (`_STRICT_COHERENCE_FIELDS`, e.g. DUALS): all sources must
            share one id, else `new_id` is a sentinel (-1) that rejects the
            read -- a mixed-iteration assembly would be invalid.

        `new_id` then feeds the shared _write_ids_agree check, so the strict
        rejection is collective-safe: the sentinel (below any real id) simply
        breaks agreement and every reader rejects together (no early return, no
        deadlock). With a synchronous hub all sources share an id, so strict
        reads pass normally and a transient mixed-id read is retried.

        For the Category-2 xhat fields (`_FIRST_STAGE_NAC_FIELDS`) the assembled
        per-scenario blocks then pass through _enforce_first_stage_nac, which
        restores the non-anticipativity of their first-stage portion (the cost
        portion stays per-scenario).
        """
        if synchronize:
            self.cylinder_comm.Barrier()
        last_id = buf.id()

        # Read each source's whole field once -- data and trailing write_id
        # from a single atomic snapshot, exactly as the equal-rank reader does.
        # Reading the data segment and the id in separate Gets would race the
        # writer: the hub could Put a source between the two reads, yielding
        # pre-write NaN data paired with a fresh id, which the gate below would
        # then accept. One whole-field Get per source closes that window. We
        # also defer writing into buf until the read is accepted, so a rejected
        # (mixed-id) read never leaves stale data behind.
        source_snapshots = {}
        source_ids = []
        for r in self._overlap_source_ranks[(field, peer_cylinder)]:
            _, logical_len, padded_len = self.window.strata_buffer_layouts[r][field]
            snapshot = np.empty(padded_len, dtype="d")
            self.window.get(snapshot, r, field)
            source_snapshots[r] = snapshot
            source_ids.append(int(snapshot[logical_len - 1]))

        new_id = reduce_source_write_ids(
            source_ids, strict=field in _STRICT_COHERENCE_FIELDS
        )

        if not self._write_ids_agree(new_id, synchronize):
            buf._is_new = False
            return False

        if new_id > last_id:
            # assemble the accepted data into buf, then commit via the shared
            # _mark_new (which stamps the id slot the assembly does not touch)
            data_view = buf.value_array()
            for seg in self.overlap_maps[(field, peer_cylinder)]:
                snapshot = source_snapshots[seg.remote_rank]
                data_view[seg.local_offset : seg.local_offset + seg.count] = \
                    snapshot[seg.remote_offset : seg.remote_offset + seg.count]
            if field in _FIRST_STAGE_NAC_FIELDS:
                self._enforce_first_stage_nac(data_view, field)
            return self._mark_new(buf, new_id)
        buf._is_new = False
        return False

    def receive_nonant_bounds(self):
        """ receive the bounds on the nonanticipative variables based on
        Field.NONANT_LOWER_BOUNDS and Field.NONANT_UPPER_BOUNDS. Updates the
        NONANT_LOWER_BOUNDS and NONANT_UPPER_BOUNDS buffers, and if new,
        updates the corresponding Pyomo nonant variables
        """
        bounds_modified = 0
        for idx, _, recv_buf in self.receive_field_spcomms[Field.NONANT_LOWER_BOUNDS]:
            is_new = self.get_receive_buffer(recv_buf, Field.NONANT_LOWER_BOUNDS, idx)
            if not is_new:
                continue
            for s in self.opt.local_scenarios.values():
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    xvarlb = xvar.lb
                    if xvarlb is None:
                        xvarlb = -inf
                    if recv_buf[ci] > xvarlb:
                        # global_toc(f"{self.__class__.__name__}: tightened {xvar.name} lower bound from {xvar.lb} to {recv_buf[ci]}, value: {xvar.value}", self.cylinder_rank == 0)
                        xvar.lb = recv_buf[ci]
                        bounds_modified += 1
        for idx, _, recv_buf in self.receive_field_spcomms[Field.NONANT_UPPER_BOUNDS]:
            is_new = self.get_receive_buffer(recv_buf, Field.NONANT_UPPER_BOUNDS, idx)
            if not is_new:
                continue
            for s in self.opt.local_scenarios.values():
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    xvarub = xvar.ub
                    if xvarub is None:
                        xvarub = inf 
                    if recv_buf[ci] < xvarub:
                        # global_toc(f"{self.__class__.__name__}: tightened {xvar.name} upper bound from {xvar.ub} to {recv_buf[ci]}, value: {xvar.value}", self.cylinder_rank == 0)
                        xvar.ub = recv_buf[ci]
                        bounds_modified += 1

        bounds_modified /= len(self.opt.local_scenarios)

        if bounds_modified > 0:
            global_toc(f"{self.__class__.__name__}: tightened {int(bounds_modified)} variable bounds", self.cylinder_rank == 0)

    def receive_best_xhat(self):
        """
        Receive the BEST_XHAT buffer from spokes.
        If new, process the xhat values (e.g., store or use them).
        """
        is_new = False
        for idx, _, recv_buf in self.receive_field_spcomms.get(Field.BEST_XHAT, []):
            is_new = self.get_receive_buffer(recv_buf, Field.BEST_XHAT, idx)
            if is_new:
                self.best_xhat = recv_buf.value_array().copy()
        return is_new


    def receive_xfeas(self):
        """
        Receive the  XFEAS buffer from spokes.
        If new, process the x values (e.g., store or use them).
        """
        is_new = False
        for idx, _, recv_buf in self.receive_field_spcomms.get(Field.XFEAS, []):
            is_new = self.get_receive_buffer(recv_buf, Field.XFEAS, idx)
            if is_new:
                self.xfeas = recv_buf.value_array().copy()

        return is_new

    def receive_latest_xhat(self):
        """
        Receive the RECENT_XHATS circular buffer from spokes.
        If new, extract all recent xhats via the circular buffer wrapper.
        """
        is_new_any = False
        all_received_xhats = []

        for idx, cls, recv_buf_circular in self._recent_xhat_recv_circular_buffers:
            is_new = self.get_receive_buffer(recv_buf_circular.data, Field.RECENT_XHATS, idx)
            if not is_new:
                continue

            is_new_any = True

            for value_array in recv_buf_circular.most_recent_value_arrays():
                all_received_xhats.append(value_array.copy())
        
        
        if not is_new_any or not all_received_xhats:
            self.recent_xhats_list = []
            return False

        self.recent_xhats_list = all_received_xhats
        self.latest_xhat = all_received_xhats[-1].copy()

        return True


    def receive_innerbounds(self):
        """ Get inner bounds from inner bound providers
        """
        logger.debug(f"{self.__class__.__name__} is trying to receive from InnerBounds")
        for idx, cls, recv_buf in self.receive_field_spcomms[Field.OBJECTIVE_INNER_BOUND]:
            is_new = self.get_receive_buffer(recv_buf, Field.OBJECTIVE_INNER_BOUND, idx)
            if is_new:
                bound = recv_buf[0]
                logger.debug(f"new InnerBound from {cls.__name__} in {self.__class__.__name__}, {bound=}")
                self.BestInnerBound = self.InnerBoundUpdate(bound, cls, idx)
        logger.debug(f"{self.__class__.__name__} back from InnerBounds")

    def receive_outerbounds(self):
        """ Get outer bounds from outer bound providers
        """
        logger.debug(f"{self.__class__.__name__} is trying to receive from OuterBounds")
        for idx, cls, recv_buf in self.receive_field_spcomms[Field.OBJECTIVE_OUTER_BOUND]:
            is_new = self.get_receive_buffer(recv_buf, Field.OBJECTIVE_OUTER_BOUND, idx) 
            if is_new:
                bound = recv_buf[0]
                logger.debug(f"new OuterBound from {cls.__name__} in {self.__class__.__name__}, {bound=}")
                self.BestOuterBound = self.OuterBoundUpdate(bound, cls, idx)
        logger.debug(f"{self.__class__.__name__} back from OuterBounds")

    def OuterBoundUpdate(self, new_bound, cls=None, idx=None, char='*'):
        current_bound = self.BestOuterBound
        if self._outer_bound_update(new_bound, current_bound):
            if cls is None:
                self.latest_ob_char = char
                self.last_ob_idx = self.strata_rank
            else:
                self.latest_ob_char = cls.converger_spoke_char
                self.last_ob_idx = idx
            return new_bound
        else:
            return current_bound

    def InnerBoundUpdate(self, new_bound, cls=None, idx=None, char='*'):
        current_bound = self.BestInnerBound
        if self._inner_bound_update(new_bound, current_bound):
            if cls is None:
                self.latest_ib_char = char
                self.last_ib_idx = self.strata_rank
            else:
                self.latest_ib_char = cls.converger_spoke_char
                self.last_ib_idx = idx
            return new_bound
        else:
            return current_bound

    def initialize_bound_values(self):
        if self.opt.is_minimizing:
            self.BestInnerBound = inf
            self.BestOuterBound = -inf
            self._inner_bound_update = lambda new, old : (new < old)
            self._outer_bound_update = lambda new, old : (new > old)
        else:
            self.BestInnerBound = -inf
            self.BestOuterBound = inf
            self._inner_bound_update = lambda new, old : (new > old)
            self._outer_bound_update = lambda new, old : (new < old)

    def compute_gaps(self):
        """ Compute the current absolute and relative gaps,
            using the current self.BestInnerBound and self.BestOuterBound
        """
        if self.opt.is_minimizing:
            abs_gap = self.BestInnerBound - self.BestOuterBound
        else:
            abs_gap = self.BestOuterBound - self.BestInnerBound

        if abs_gap != inf:
            rel_gap = ( abs_gap /
                        max(1e-10,
                            abs(self.BestOuterBound),
                            abs(self.BestInnerBound),
                           )
                      )
        else:
            rel_gap = inf
        return abs_gap, rel_gap
