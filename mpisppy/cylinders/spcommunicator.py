# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
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


class SPCommunicator:
    """ Notes: TODO
    """

    # magic constant for spoke_sleep_time calculation below
    _SLEEP_TIME_MUTLIPLIER = 1e-5

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

        self.spoke_sleep_time = self.options.get('spoke_sleep_time')
        # the user could set None
        if self.spoke_sleep_time is None:
                self.spoke_sleep_time = self._SLEEP_TIME_MUTLIPLIER * spbase_object.nonant_length

        # attach the SPCommunicator to
        # the SPBase object
        self.opt.spcomm = self

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

            Spokes use the implementation below, which just
            puts a small sleep in so windows are not freed
            too soon.
        """
        ## give the hub the chance to catch new values
        time.sleep(self.spoke_sleep_time)

    def allreduce_or(self, val):
        local_val = np.array([val], dtype='int8')
        global_val = np.zeros(1, dtype='int8')
        self.cylinder_comm.Allreduce(local_val, global_val, op=MPI.LOR)
        if global_val[0] > 0:
            return True
        else:
            return False

    def free_windows(self):
        """
        """
        if self._windows_constructed:
            for i in range(self.n_spokes):
                self.windows[i].Free()
            del self.buffers
        self._windows_constructed = False

    def _make_window(self, length, comm=None):
        """ Create a local window object and its corresponding 
            memory buffer using MPI.Win.Allocate()

            Args: 
                length (int): length of the buffer to create
                comm (MPI Communicator, optional): MPI communicator object to
                    create the window over. Default is self.strata_comm.

            Returns:
                window (MPI.Win object): The created window
                buff (ndarray): Pointer to corresponding memory

            Notes:
                The created buffer will actually be +1 longer than length.
                The last entry is a write number to keep track of new info.

                This function assumes that the user has provided the correct
                window size for the local buffer based on whether this process
                is a hub or spoke, etc.
        """
        if comm is None:
            comm = self.strata_comm
        size = MPI.DOUBLE.size * (length + 1)
        window = MPI.Win.Allocate(size, MPI.DOUBLE.size, comm=comm)
        buff = np.ndarray(dtype="d", shape=(length + 1,), buffer=window.tomemory())
        buff[-1] = 0. # Initialize the write number to zero
        return window, buff
