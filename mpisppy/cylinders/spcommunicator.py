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
from mpisppy.cylinders.spwindow import Field, FieldLengths, SPWindow

logger = logging.getLogger(__name__)

def communicator_array(size):
    arr = np.empty(size+1, dtype='d')
    arr[:] = np.nan
    arr[-1] = 0
    return arr


class FieldArray:
    """
    Notes: Buffer that tracks new/old state as well. Light-weight wrapper around a numpy array.

    The intention here is that these are passive data holding classes. That is, other classes are
    expected to update the internal fields. The lone exception to this is the read/write id field.
    See the `SendArray` and `RecvArray` classes for how that field is updated.
    """

    def __init__(self, length: int):
        self._array = communicator_array(length)
        self._id = 0
        return

    def __getitem__(self, key):
        # TODO: Should probably be hiding the read/write id field but there are many functions
        # that expect it to be there and being able to read it is not really a problem.
        np_array = self.array()
        return np_array[key]

    def array(self) -> np.typing.NDArray:
        """
        Returns the numpy array for the field data including the read id
        """
        return self._array

    def value_array(self) -> np.typing.NDArray:
        """
        Returns the numpy array for the field data without the read id
        """
        return self._array[:-1]

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
        assert len(communicators) == strata_comm.Get_size()
        self.global_rank = fullcomm.Get_rank()
        self.strata_rank = strata_comm.Get_rank()
        self.cylinder_rank = cylinder_comm.Get_rank()
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

    def _build_window_spec(self) -> dict[Field, int]:
        """ Build dict with fields and lengths needed for local MPI window
        """
        window_spec = dict()
        for (field,buf) in self.send_buffers.items():
            window_spec[field] = np.size(buf.array())
        ## End for
        return window_spec

    def _create_field_rank_mappings(self) -> None:
        self.fields_to_ranks = {}
        self.ranks_to_fields = {}

        for rank, buffer_layout in enumerate(self.window.strata_buffer_layouts):
            if rank == self.strata_rank:
                continue
            self.ranks_to_fields[rank] = []
            for field in buffer_layout:
                if field not in self.fields_to_ranks:
                    self.fields_to_ranks[field] = []
                self.fields_to_ranks[field].append(rank)
                self.ranks_to_fields[rank].append(field)

        # print(f"{self.__class__.__name__}: {self.fields_to_ranks=}, {self.ranks_to_fields=}")

    def _validate_recv_field(self, field: Field, origin: int, length: int):
        remote_buffer_layout = self.window.strata_buffer_layouts[origin]
        if field not in remote_buffer_layout:
            raise RuntimeError(f"{self.__class__.__name__} on local {self.strata_rank=} "
                               f"could not find {field=} on remote rank {origin} with "
                               f"class {self.communicators[origin]['spcomm_class']}."
                              )
        _, remote_length = remote_buffer_layout[field]
        if (length + 1) != remote_length:
            raise RuntimeError(f"{self.__class__.__name__} on local {self.strata_rank=} "
                               f"{field=} has length {length} on local "
                               f"{self.strata_rank=} and length {remote_length} "
                               f"on remote rank {origin} with class "
                               f"{self.communicators[origin]['spcomm_class']}."
                              )

    def register_recv_field(self, field: Field, origin: int, length: int = -1) -> RecvArray:
        # print(f"{self.__class__.__name__}.register_recv_field, {field=}, {origin=}")
        key = self._make_key(field, origin)
        if length == -1:
            length = self._field_lengths[field]
        if key in self.receive_buffers:
            my_fa = self.receive_buffers[key]
            assert(length + 1 == np.size(my_fa.array()))
        else:
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
        self.window = SPWindow(window_spec, self.strata_comm)

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

        self.window.free()

        self.window = None

    def is_send_field_registered(self, field: Field) -> bool:
        return field in self.send_buffers

    def register_send_fields(self) -> None:
        for field in self.send_fields:
            self.register_send_field(field)

    def register_receive_fields(self) -> None:
        # print(f"{self.__class__.__name__}: {self.receive_fields=}")
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
                if field in self.ranks_to_fields[strata_rank]:
                    buff = self.register_recv_field(field, strata_rank)
                    self.receive_field_spcomms[field].append((strata_rank, cls, buff))

    def put_send_buffer(self, buf: SendArray, field: Field):
        """ Put the specified values into the specified locally-owned buffer
            for the another cylinder to pick up.

            Notes:
                This automatically updates handles the write id.
        """
        buf._next_write_id()
        self.window.put(buf.array(), field)
        return

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
        if synchronize:
            self.cylinder_comm.Barrier()

        last_id = buf.id()

        self.window.get(buf.array(), origin, field)

        new_id = int(buf.array()[-1])

        if synchronize:
            local_val = np.array((new_id,), 'i')
            sum_ids = np.zeros(1, 'i')
            self.cylinder_comm.Allreduce((local_val, MPI.INT),
                                         (sum_ids, MPI.INT),
                                         op=MPI.SUM)
            if new_id * self.cylinder_comm.size != sum_ids[0]:
                buf._is_new = False
                return False

        if new_id > last_id:
            buf._is_new = True
            buf._pull_id()
            return True
        else:
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
