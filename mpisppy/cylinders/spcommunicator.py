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
from mpisppy import MPI

from mpisppy.cylinders.spwindow import Field, SPWindow


def communicator_array(size):
    arr = np.empty(size+1)
    arr[:] = np.nan
    arr[-1] = 0
    return arr


class FieldArray:
    """ Notes: Buffer that tracks new/old state as well
    """

    def __init__(self, length: int):
        self._array = communicator_array(length)
        self._is_new = False
        self._id = 0
        return

    def array(self):
        # return self._array[:-1]
        return self._array

    def is_new(self):
        return self._is_new

    def id(self):
        return self._id

    # def raw_array(self):
    #     return self._array

class SendArray(FieldArray):

    def __init__(self, length: int):
        super().__init__(length)
        return

    def next_write_id(self):
        self._id += 1
        return self._id


class RecvArray(FieldArray):

    def __init__(self, length: int):
        super().__init__(length)
        return

    def pull_id(self):
        self._id = int(self._array[-1])
        return self._id


class SPCommunicator:
    """ Notes: TODO
    """

    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, options=None):
        # flag for if the windows have been constructed
        self._windows_constructed = False
        self.fullcomm = fullcomm
        self.strata_comm = strata_comm
        self.cylinder_comm = cylinder_comm
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
        self._locals = dict()
        self._sends = dict()

        # attach the SPCommunicator to
        # the SPBase object
        self.opt.spcomm = self

    def _make_key(self, field: Field, origin: int):
        """Given a field and an origin (i.e. a strata_rank), generate a key for indexing
           into the self._locals dictionary and getting the corresponding RecvArray.
        """
        return (field, origin)

    def _split_key(self, key):
        """Take the given key and return a tuple (field, origin) where origin in the strata_rank
           from which the field comes.

           Undoes `_make_key`.  Currently, this is a no-op.
        """
        return key

    def register_recv_field(self, field: Field, origin: int, length: int) -> np.typing.NDArray:
        key = self._make_key(field, origin)
        my_fa = RecvArray(length)
        self._locals[key] = my_fa
        return my_fa.array()

    def register_send_field(self, field: Field, length: int) -> np.typing.NDArray:
        self._sends[field] = SendArray(length)
        return self._sends[field].array()

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
            # for i in range(self.n_spokes):
            #     self.windows[i].Free()
            # del self.buffers
            self.window.free()
        self._windows_constructed = False

    # def _make_window(self, length, comm=None):
    #     """ Create a local window object and its corresponding
    #         memory buffer using MPI.Win.Allocate()

    #         Args:
    #             length (int): length of the buffer to create
    #             comm (MPI Communicator, optional): MPI communicator object to
    #                 create the window over. Default is self.strata_comm.

    #         Returns:
    #             window (MPI.Win object): The created window
    #             buff (ndarray): Pointer to corresponding memory

    #         Notes:
    #             The created buffer will actually be +1 longer than length.
    #             The last entry is a write number to keep track of new info.

    #             This function assumes that the user has provided the correct
    #             window size for the local buffer based on whether this process
    #             is a hub or spoke, etc.
    #     """
    #     if comm is None:
    #         comm = self.strata_comm
    #     size = MPI.DOUBLE.size * (length + 1)
    #     window = MPI.Win.Allocate(size, MPI.DOUBLE.size, comm=comm)
    #     buff = np.ndarray(dtype="d", shape=(length + 1,), buffer=window.tomemory())
    #     buff[:] = np.nan
    #     buff[-1] = 0. # Initialize the write number to zero
    #     return window, buff

    # def _make_window(self, comm):
    #     buffer_spec = self.build_buffer_spec()
    #     self.window = SPWindow(buffer_spec, comm)
    #     return

    def make_windows(self) -> None:
        if self._windows_constructed:
            return

        window_spec = self.build_window_spec()
        self.window = SPWindow(window_spec, self.strata_comm)
        self._windows_constructed = True

        return

    @abc.abstractmethod
    def build_window_spec(self) -> dict[Field, int]:
        """ Build dict with fields and lengths needed for local MPI window
        """
        return {}
