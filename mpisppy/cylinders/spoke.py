# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import numpy as np
import abc
import enum
import logging
import time
import os

# for SLEEP_TIME
import mpisppy.cylinders as cylinders

from mpi4py import MPI
from mpisppy.cylinders.spcommunicator import SPCommunicator


class ConvergerSpokeType(enum.Enum):
    OUTER_BOUND = 1
    INNER_BOUND = 2
    W_GETTER = 3
    NONANT_GETTER = 4

class Spoke(SPCommunicator):
    def __init__(self, spbase_object, fullcomm, intercomm, intracomm):
        super().__init__(spbase_object, fullcomm, intercomm, intracomm)
        self.local_write_id = 0
        self.remote_write_id = 0
        self.local_length = 0  # Does NOT include the + 1
        self.remote_length = 0  # Length on hub; does NOT include + 1

        self.last_call_to_got_kill_signal = time.time()

    def _make_windows(self, local_length, remote_length):
        # Spokes notify the hub of the buffer sizes
        pair_of_lengths = np.array([local_length, remote_length], dtype="i")
        self.intercomm.Send((pair_of_lengths, MPI.INT), dest=0, tag=self.rank_inter)
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
            length = self.local_length if self.rank_inter == i + 1 else 0
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
        self.local_write_id += 1
        values[-1] = self.local_write_id
        window = self.windows[self.rank_inter - 1]
        window.Lock(self.rank_inter)
        window.Put((values, len(values), MPI.DOUBLE), self.rank_inter)
        window.Unlock(self.rank_inter)

    def spoke_from_hub(self, values):
        """
        """
        expected_length = self.remote_length + 1
        if len(values) != expected_length:
            raise RuntimeError(
                f"Spoke trying to get buffer of length {expected_length} "
                f"from hub, but provided buffer has length {len(values)}."
            )
        window = self.windows[self.rank_inter - 1]
        window.Lock(0)
        window.Get((values, len(values), MPI.DOUBLE), 0)
        window.Unlock(0)

        if values[-1] > self.remote_write_id:
            self.remote_write_id = values[-1]
            return True
        return False

    def got_kill_signal(self):
        """ Spoke should call this method at least every iteration
            to see if the Hub terminated
        """
        # Spokes can sometimes call this frequently in a tight loop,
        # causing the Allreduces to become out of sync
        diff = time.time() - self.last_call_to_got_kill_signal
        if diff < cylinders.SPOKE_SLEEP_TIME:
            time.sleep(cylinders.SPOKE_SLEEP_TIME - diff)
        self.last_call_to_got_kill_signal = time.time()
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

    @abc.abstractmethod
    def get_serial_number(self):
        pass

    @abc.abstractmethod
    def _got_kill_signal(self):
        """ Every spoke needs a way to get the signal to terminate
            from the hub
        """
        pass


class _BoundSpoke(Spoke):
    """ A base class for bound spokes
    """
    def __init__(self, spbase_object, fullcomm, intercomm, intracomm):
        super().__init__(spbase_object, fullcomm, intercomm, intracomm)
        if self.rank_intra == 0 and \
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

    def make_windows(self):
        """ Makes the bound window and a remote window to
            look for a kill signal
        """

        ## need a remote_length for the kill signal
        self._make_windows(1, 0)
        self._kill_sig = np.zeros(0 + 1)
        self._bound = np.zeros(1 + 1)

    @property
    def bound(self):
        return self._bound[0]

    @bound.setter
    def bound(self, value):
        self._append_trace(value)
        self._bound[0] = value
        self.spoke_to_hub(self._bound)

    def get_serial_number(self):
        return int(self._kill_sig[-1])

    def _got_kill_signal(self):
        """Looks for the kill signal and returns True if sent"""
        self.spoke_from_hub(self._kill_sig)
        kill = self._kill_sig[-1] == -1
        return kill

    def _append_trace(self, value):
        if self.rank_intra != 0 or self.trace_filen is None:
            return
        with open(self.trace_filen, 'a') as f:
            f.write(f"{time.time()-self.start_time},{value}\n")


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
            raise RuntimeError(f"Rank has zero local_scenarios")

        vbuflen = 0
        for s in self.opt.local_scenarios.values():
            vbuflen += len(s._nonant_indexes)

        self._make_windows(1, vbuflen)
        self._locals = np.zeros(vbuflen + 1) # Also has kill signal
        self._bound = np.zeros(1 + 1)
        self._new_locals = False

    def get_serial_number(self):
        return int(self._locals[-1])

    def _got_kill_signal(self):
        """ returns True if a kill signal was received, 
            and refreshes the array and _locals"""
        self._new_locals = self.spoke_from_hub(self._locals)
        kill = self._locals[-1] == -1
        return kill


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
        return self._locals[:-1]

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
        return self._locals[:-1]

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
        the main PH OPT hub.
    """
    converger_spoke_types = (
        ConvergerSpokeType.INNER_BOUND,
        ConvergerSpokeType.NONANT_GETTER,
    )
    converger_spoke_char = 'I'


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
    
