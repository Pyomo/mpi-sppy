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
from mpi4py import MPI

# for SLEEP_TIME
import mpisppy.cylinders as cylinders

class SPCommunicator:
    """ Notes: TODO
    """

    def __init__(self, spbase_object, fullcomm, intercomm, intracomm, options=None):
        # flag for if the windows have been constructed
        self._windows_constructed = False
        self.fullcomm = fullcomm
        self.intercomm = intercomm
        self.intracomm = intracomm
        self.rank_global = fullcomm.Get_rank()
        self.rank_inter = intercomm.Get_rank()
        self.rank_intra = intracomm.Get_rank()
        self.n_spokes = intercomm.Get_size() - 1
        self.opt = spbase_object
        self.inst_time = time.time() # For diagnostics
        self.options = options

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
        time.sleep(cylinders.SPOKE_SLEEP_TIME)

    def allreduce_or(self, val):
        local_val = np.array([val], dtype='int8')
        global_val = np.zeros(1, dtype='int8')
        self.intracomm.Allreduce(local_val, global_val, op=MPI.LOR)
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
                    create the window over. Default is self.intercomm.

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
            comm = self.intercomm
        size = MPI.DOUBLE.size * (length + 1)
        window = MPI.Win.Allocate(size, MPI.DOUBLE.size, comm=comm)
        buff = np.ndarray(dtype="d", shape=(length + 1,), buffer=window.tomemory())
        buff[-1] = 0. # Initialize the write number to zero
        return window, buff
