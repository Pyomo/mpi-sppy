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

import numpy as np
import abc
import time

from mpisppy.cylinders.spwindow import Field, FieldLengths, SPWindow

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
        self._length = length
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
        Updates the internal id field to the next write id and returns that id
        """
        self._id += 1
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
    receive_fields = ()

    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, communicators, options=None):
        # flag for if the windows have been constructed
        self._windows_constructed = False
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
        # key: Field, value: list of (strata_rank, SPComm) with that Field
        self.receive_field_spcomms = {}

        # setup FieldLengths which calculates
        # the length of each buffer type based
        # on the problem data
        self._field_lengths = FieldLengths(self.opt)

        # attach the SPCommunicator to
        # the SPBase object
        self.opt.spcomm = self

        self.register_send_fields()

        self._exchange_send_fields()
        # TODO: here we can have a dynamic exchange of the send fields
        #       so we can do error checking (all-to-all in send fields)
        self.register_receive_fields()

        # TODO: check that we have something in receive_field_spcomms??

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

    def _exchange_send_fields(self) -> None:
        """ Do an all-to-all so we know what the other communicators are sending """
        send_buffers = tuple((k, buff._length) for k, buff in self.send_buffers.items())
        self.send_fields_lengths_by_rank = self.strata_comm.allgather(send_buffers)

        self.send_fields_by_rank = {}

        self.available_receive_fields = {}
        for rank, fields_lengths in enumerate(self.send_fields_lengths_by_rank):
            if rank == self.strata_rank:
                continue
            self.send_fields_by_rank[rank] = []
            for f, length in fields_lengths:
                if f not in self.available_receive_fields:
                    self.available_receive_fields[f] = []
                self.available_receive_fields[f].append(rank)
                self.send_fields_by_rank[rank].append(f)

        # print(f"{self.__class__.__name__}: {self.available_receive_fields=}")

    def register_recv_field(self, field: Field, origin: int, length: int = -1) -> RecvArray:
        # print(f"{self.__class__.__name__}.register_recv_field, {field=}, {origin=}")
        key = self._make_key(field, origin)
        if length == -1:
            length = self._field_lengths[field]
        if key in self.receive_buffers:
            my_fa = self.receive_buffers[key]
            assert(length + 1 == np.size(my_fa.array()))
        else:
            available_fields_from_origin = self.send_fields_lengths_by_rank[origin]
            for _field, _length in available_fields_from_origin:
                if field == _field:
                    assert length == _length
                    break
            else: # couldn't find field!
                raise RuntimeError(f"Couldn't find {field=} from {origin=}")
            my_fa = RecvArray(length)
            self.receive_buffers[key] = my_fa
        ## End if
        return my_fa

    def register_send_field(self, field: Field, length: int = -1) -> SendArray:
        assert field not in self.send_buffers, "Field {} is already registered".format(field)
        if length == -1:
            length = self._field_lengths[field]
        # if field in self.send_buffers:
        #     my_fa = self.send_buffers[field]
        #     assert(length + 1 == np.size(my_fa.array()))
        # else:
        #     my_fa = SendArray(length)
        #     self.send_buffers[field] = my_fa
        # ## End if else
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

    def free_windows(self):
        """
        """
        if self._windows_constructed:
            self.window.free()
        self._windows_constructed = False

    def make_windows(self) -> None:
        if self._windows_constructed:
            return

        window_spec = self._build_window_spec()
        self.window = SPWindow(window_spec, self.strata_comm)
        self._windows_constructed = True

        return

    def register_send_fields(self) -> None:
        for field in self.send_fields:
            self.register_send_field(field)

    def register_receive_fields(self) -> None:
        # print(f"{self.__class__.__name__}: {self.receive_fields=}")
        for field in self.receive_fields:
            self.receive_field_spcomms[field] = []
            for strata_rank, comm in enumerate(self.communicators):
                if strata_rank == self.strata_rank:
                    continue
                cls = comm["spcomm_class"]
                if field in self.send_fields_by_rank[strata_rank]:
                    buff = self.register_recv_field(field, strata_rank)
                    self.receive_field_spcomms[field].append((strata_rank, cls, buff))
