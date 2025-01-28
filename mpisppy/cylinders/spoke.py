###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import numpy as np
import abc
import enum
import time
import os
import math

from mpisppy import MPI
from mpisppy.cylinders.spcommunicator import SPCommunicator, communicator_array


class ConvergerSpokeType(enum.Enum):
    OUTER_BOUND = 1
    INNER_BOUND = 2
    W_GETTER = 3
    NONANT_GETTER = 4

class Spoke(SPCommunicator):
    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, options=None):
        super().__init__(spbase_object, fullcomm, strata_comm, cylinder_comm, options)
        self.local_write_id = 0
        self.remote_write_id = 0
        self.local_length = 0  # Does NOT include the + 1
        self.remote_length = 0  # Length on hub; does NOT include + 1

        self.last_call_to_got_kill_signal = time.time()

    def _make_windows(self, local_length, remote_length):
        # Spokes notify the hub of the buffer sizes
        pair_of_lengths = np.array([local_length, remote_length], dtype="i")
        self.strata_comm.Send((pair_of_lengths, MPI.INT), dest=0, tag=self.strata_rank)
        self.local_length = local_length
        self.remote_length = remote_length

        # Make the windows of the appropriate buffer sizes
        # To do?: Spoke should not need to know how many other spokes there are.
        # Just call a single _make_window()? Do you need to create empty
        # windows?
        # ANSWER (dlw July 2020): Since the windows have zero length and since
        # the number of spokes is not expected to be large, it is probably OK.
        # The (minor) benefit is that free_windows does not need to know if it
        # was called by a hub or a spoke. If we ever move to dynamic spoke
        # creation, then this needs to be reimagined.
        self.windows = [None for _ in range(self.n_spokes)]
        self.buffers = [None for _ in range(self.n_spokes)]
        for i in range(self.n_spokes):
            length = self.local_length if self.strata_rank == i + 1 else 0
            win, buff = self._make_window(length)
            self.windows[i] = win
            self.buffers[i] = buff

        self._windows_constructed = True

    def spoke_to_hub(self, values):
        """ Put the specified values into the locally-owned buffer for the hub
            to pick up.

            Notes:
                This automatically does the -1 indexing

                This assumes that values contains a slot at the end for the
                write_id
        """
        expected_length = self.local_length + 1
        if len(values) != expected_length:
            raise RuntimeError(
                f"Attempting to put array of length {len(values)} "
                f"into local buffer of length {expected_length}"
            )
        self.cylinder_comm.Barrier()
        self.local_write_id += 1
        values[-1] = self.local_write_id
        window = self.windows[self.strata_rank - 1]
        window.Lock(self.strata_rank)
        window.Put((values, len(values), MPI.DOUBLE), self.strata_rank)
        window.Unlock(self.strata_rank)

    def spoke_from_hub(self, values):
        """
        """
        expected_length = self.remote_length + 1
        if len(values) != expected_length:
            raise RuntimeError(
                f"Spoke trying to get buffer of length {expected_length} "
                f"from hub, but provided buffer has length {len(values)}."
            )
        self.cylinder_comm.Barrier()
        window = self.windows[self.strata_rank - 1]
        window.Lock(0)
        window.Get((values, len(values), MPI.DOUBLE), 0)
        window.Unlock(0)

        # On rare occasions a NaN is seen...
        new_id = int(values[-1]) if not math.isnan(values[-1]) else 0
        local_val = np.array((new_id,-new_id), 'i')
        max_min_ids = np.zeros(2, 'i')
        self.cylinder_comm.Allreduce((local_val, MPI.INT),
                                     (max_min_ids, MPI.INT),
                                     op=MPI.MAX)

        max_id = max_min_ids[0]
        min_id = -max_min_ids[1]
        # NOTE: we only proceed if all the ranks agree
        #       on the ID
        if max_id != min_id:
            return False

        assert max_id == min_id == new_id

        if (new_id > self.remote_write_id) or (new_id < 0):
            self.remote_write_id = new_id
            return True
        return False

    def got_kill_signal(self):
        """ Spoke should call this method at least every iteration
            to see if the Hub terminated
        """
        return self._got_kill_signal()

    @abc.abstractmethod
    def main(self):
        """
        The main call for the Spoke. Derived classe
        should call the got_kill_signal method
        regularly to ensure all ranks terminate
        with the Hub.
        """
        pass

    def get_serial_number(self):
        return self.remote_write_id

    @abc.abstractmethod
    def _got_kill_signal(self):
        """ Every spoke needs a way to get the signal to terminate
            from the hub
        """
        pass


class _BoundSpoke(Spoke):
    """ A base class for bound spokes
    """
    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, options=None):
        super().__init__(spbase_object, fullcomm, strata_comm, cylinder_comm, options)
        if self.cylinder_rank == 0 and \
                'trace_prefix' in spbase_object.options and \
                spbase_object.options['trace_prefix'] is not None:
            trace_prefix = spbase_object.options['trace_prefix']

            filen = trace_prefix+self.__class__.__name__+'.csv'
            if os.path.exists(filen):
                raise RuntimeError(f"Spoke trace file {filen} already exists!")
            with open(filen, 'w') as f:
                f.write("time,bound\n")
            self.trace_filen = filen
            self.start_time = spbase_object.start_time
        else:
            self.trace_filen = None

        self._new_locals = False
        self._bound = None
        self._locals = None

    def make_windows(self):
        """ Makes the bound window and a remote window to
            look for a kill signal
        """
        self._make_windows(1, 2) # kill signals are accounted for in _make_window
        self._bound = communicator_array(1) # spoke bound + kill signal
        self._locals = communicator_array(2) # hub outer/inner bounds and kill signal

    @property
    def bound(self):
        return self._bound[0]

    @bound.setter
    def bound(self, value):
        self._append_trace(value)
        self._bound[0] = value
        self.spoke_to_hub(self._bound)

    @property
    def hub_inner_bound(self):
        """Returns the local copy of the inner bound from the hub"""
        return self._locals[-2]

    @property
    def hub_outer_bound(self):
        """Returns the local copy of the outer bound from the hub"""
        return self._locals[-3]

    def _got_kill_signal(self):
        """Looks for the kill signal and returns True if sent"""
        self._new_locals = self.spoke_from_hub(self._locals)
        return self.remote_write_id == -1

    def _append_trace(self, value):
        if self.cylinder_rank != 0 or self.trace_filen is None:
            return
        with open(self.trace_filen, 'a') as f:
            f.write(f"{time.perf_counter()-self.start_time},{value}\n")


class _BoundNonantLenSpoke(_BoundSpoke):
    """ A base class for bound spokes which also
        want something of len nonants from OPT
    """

    def make_windows(self):
        """ Makes the bound window and with a remote buffer long enough
            to hold an array as long as the nonants.

            Input:
                opt (SPBase): Must have local_scenarios attached already!

        """
        if not hasattr(self.opt, "local_scenarios"):
            raise RuntimeError("Provided SPBase object does not have local_scenarios attribute")

        if len(self.opt.local_scenarios) == 0:
            raise RuntimeError("Rank has zero local_scenarios")

        vbuflen = 2
        for s in self.opt.local_scenarios.values():
            vbuflen += len(s._mpisppy_data.nonant_indices)

        self._make_windows(1, vbuflen)
        self._bound = communicator_array(1)
        self._locals = communicator_array(vbuflen)

class InnerBoundSpoke(_BoundSpoke):
    """ For Spokes that provide an inner bound through self.bound to the
        Hub, and do not need information from the main PH OPT hub.
    """
    converger_spoke_types = (ConvergerSpokeType.INNER_BOUND,)
    converger_spoke_char = 'I'


class OuterBoundSpoke(_BoundSpoke):
    """ For Spokes that provide an outer bound through self.bound to the
        Hub, and do not need information from the main PH OPT hub.
    """
    converger_spoke_types = (ConvergerSpokeType.OUTER_BOUND,)
    converger_spoke_char = 'O'


class _BoundWSpoke(_BoundNonantLenSpoke):
    """ A base class for bound spokes which also want the W's from the OPT
        threads
    """

    @property
    def localWs(self):
        """Returns the local copy of the weights"""
        return self._locals[:-3] # -3 for the bounds and kill signal

    @property
    def new_Ws(self):
        """ Returns True if the local copy of
            the weights has been updated since
            the last call to got_kill_signal
        """
        return self._new_locals


class OuterBoundWSpoke(_BoundWSpoke):
    """
    For Spokes that provide an outer bound
    through self.bound to the Hub,
    and receive the Ws (or weights) from
    the main PH OPT hub.
    """

    converger_spoke_types = (
        ConvergerSpokeType.OUTER_BOUND,
        ConvergerSpokeType.W_GETTER,
    )
    converger_spoke_char = 'O'


class _BoundNonantSpoke(_BoundNonantLenSpoke):
    """ A base class for bound spokes which also
        want the xhat's from the OPT threads
    """

    @property
    def localnonants(self):
        """Returns the local copy of the nonants"""
        return self._locals[:-3]

    @property
    def new_nonants(self):
        """Returns True if the local copy of
           the nonants has been updated since
           the last call to got_kill_signal"""
        return self._new_locals


class InnerBoundNonantSpoke(_BoundNonantSpoke):
    """ For Spokes that provide an inner (incumbent)
        bound through self.bound to the Hub,
        and receive the nonants from
        the main SPOpt hub.

        Includes some helpful methods for saving
        and restoring results
    """
    converger_spoke_types = (
        ConvergerSpokeType.INNER_BOUND,
        ConvergerSpokeType.NONANT_GETTER,
    )
    converger_spoke_char = 'I'

    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, options=None):
        super().__init__(spbase_object, fullcomm, strata_comm, cylinder_comm, options)
        self.is_minimizing = self.opt.is_minimizing
        self.best_inner_bound = math.inf if self.is_minimizing else -math.inf
        self.solver_options = None # can be overwritten by derived classes

    def update_if_improving(self, candidate_inner_bound, update_best_solution_cache=True):
        if update_best_solution_cache:
            update = self.opt.update_best_solution_if_improving(candidate_inner_bound)
        else:
            update = ( (candidate_inner_bound < self.best_inner_bound)
                if self.is_minimizing else
                (self.best_inner_bound < candidate_inner_bound)
                )
        if update:
            self.best_inner_bound = candidate_inner_bound
            # send to hub
            self.bound = candidate_inner_bound
            return True
        return False

    def finalize(self):
        if self.opt.load_best_solution():
            self.final_bound = self.bound
            return self.final_bound
        return None


class OuterBoundNonantSpoke(_BoundNonantSpoke):
    """ For Spokes that provide an outer
        bound through self.bound to the Hub,
        and receive the nonants from
        the main OPT hub.
    """
    converger_spoke_types = (
        ConvergerSpokeType.OUTER_BOUND,
        ConvergerSpokeType.NONANT_GETTER,
    )
    converger_spoke_char = 'A'  # probably Lagrangian
