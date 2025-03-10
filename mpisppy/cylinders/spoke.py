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
import time
import os
import math

from mpisppy import MPI
from mpisppy.cylinders.spcommunicator import RecvArray, SendArray, SPCommunicator
from mpisppy.cylinders.spwindow import Field


class Spoke(SPCommunicator):

    send_fields = (*SPCommunicator.send_fields, )
    receive_fields = (*SPCommunicator.receive_fields, Field.SHUTDOWN, )

    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, communicators, options=None):

        super().__init__(spbase_object, fullcomm, strata_comm, cylinder_comm, communicators, options)

        self.last_call_to_got_kill_signal = time.time()

        return

    def spoke_from_hub(self,
                       buf: RecvArray,
                       field: Field,
                       ):
        buf._is_new = self._spoke_from_hub(buf.array(), field, buf.id())
        if buf.is_new():
            buf._pull_id()
        return buf.is_new()

    def _spoke_from_hub(self,
                        values: np.typing.NDArray,
                        field: Field,
                        last_write_id: int
                        ):
        """
        """

        self.cylinder_comm.Barrier()
        self.window.get(values, 0, field)

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

        if new_id > last_write_id or new_id < 0:
            return True

        return False

    def _got_kill_signal(self):
        shutdown_buf = self.receive_buffers[self._make_key(Field.SHUTDOWN, 0)]
        if shutdown_buf.is_new():
            shutdown = (shutdown_buf[0] == 1.0)
        else:
            shutdown = False
        ## End if
        return shutdown

    def got_kill_signal(self):
        """ Spoke should call this method at least every iteration
            to see if the Hub terminated
        """
        self.update_receive_buffers()
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

    def update_receive_buffers(self):
        for (key, recv_buf) in self.receive_buffers.items():
            field, rank = self._split_key(key)
            # The below code will need to be updated for spoke to spoke communication
            assert(rank == 0)
            self.spoke_from_hub(recv_buf, field)
        ## End for
        return


class _BoundSpoke(Spoke):
    """ A base class for bound spokes
    """

    send_fields = (*Spoke.send_fields, )
    receive_fields = (*Spoke.receive_fields, Field.BEST_OBJECTIVE_BOUNDS)

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

        return


    def register_send_fields(self) -> None:
        super().register_send_fields()
        self._bound = self.send_buffers[self.bound_type()]
        return

    def register_receive_fields(self) -> None:
        super().register_receive_fields()
        self._hub_bounds = self.register_recv_field(Field.BEST_OBJECTIVE_BOUNDS, 0, 2)

    @abc.abstractmethod
    def bound_type(self) -> Field:
        pass

    @property
    def bound(self):
        return self._bound[0]

    @bound.setter
    def bound(self, value):
        self._append_trace(value)
        self._bound[0] = value
        self.put_send_buffer(self._bound, self.bound_type())
        return

    @property
    def hub_inner_bound(self):
        """Returns the local copy of the inner bound from the hub"""
        return self._hub_bounds[1]

    @property
    def hub_outer_bound(self):
        """Returns the local copy of the outer bound from the hub"""
        return self._hub_bounds[0]

    def _append_trace(self, value):
        if self.cylinder_rank != 0 or self.trace_filen is None:
            return
        with open(self.trace_filen, 'a') as f:
            f.write(f"{time.perf_counter()-self.start_time},{value}\n")


class _BoundNonantLenSpoke(_BoundSpoke):
    """ A base class for bound spokes which also
        want something of len nonants from OPT
    """

    @abc.abstractmethod
    def nonant_len_type(self) -> Field:
        # TODO: Make this a static method?
        pass


class InnerBoundSpoke(_BoundSpoke):
    """ For Spokes that provide an inner bound through self.bound to the
        Hub, and do not need information from the main PH OPT hub.
    """

    send_fields = (*_BoundSpoke.send_fields, Field.OBJECTIVE_INNER_BOUND, )
    receive_fields = (*_BoundSpoke.receive_fields, )

    converger_spoke_char = 'I'

    def bound_type(self) -> Field:
        return Field.OBJECTIVE_INNER_BOUND


class OuterBoundSpoke(_BoundSpoke):
    """ For Spokes that provide an outer bound through self.bound to the
        Hub, and do not need information from the main PH OPT hub.
    """

    send_fields = (*_BoundSpoke.send_fields, Field.OBJECTIVE_OUTER_BOUND, )
    receive_fields = (*_BoundSpoke.receive_fields, )

    converger_spoke_char = 'O'

    def bound_type(self) -> Field:
        return Field.OBJECTIVE_OUTER_BOUND


class _BoundWSpoke(_BoundNonantLenSpoke):
    """ A base class for bound spokes which also want the W's from the OPT
        threads
    """

    def nonant_len_type(self) -> Field:
        return Field.DUALS

    @property
    def localWs(self):
        """Returns the local copy of the weights"""
        key = self._make_key(Field.DUALS, 0)
        return self.receive_buffers[key].value_array()

    @property
    def new_Ws(self):
        """ Returns True if the local copy of
            the weights has been updated since
            the last call to got_kill_signal
        """
        key = self._make_key(Field.DUALS, 0)
        return self.receive_buffers[key].is_new()


class OuterBoundWSpoke(_BoundWSpoke):
    """
    For Spokes that provide an outer bound
    through self.bound to the Hub,
    and receive the Ws (or weights) from
    the main PH OPT hub.
    """

    send_fields = (*_BoundWSpoke.send_fields, Field.OBJECTIVE_OUTER_BOUND, )
    receive_fields = (*_BoundWSpoke.receive_fields, Field.DUALS)

    converger_spoke_char = 'O'

    def bound_type(self) -> Field:
        return Field.OBJECTIVE_OUTER_BOUND


class _BoundNonantSpoke(_BoundNonantLenSpoke):
    """ A base class for bound spokes which also
        want the xhat's from the OPT threads
    """

    def nonant_len_type(self) -> Field:
        return Field.NONANT

    @property
    def localnonants(self):
        """Returns the local copy of the nonants"""
        key = self._make_key(Field.NONANT, 0)
        return self.receive_buffers[key].value_array()

    @property
    def new_nonants(self):
        """Returns True if the local copy of
           the nonants has been updated since
           the last call to got_kill_signal"""
        key = self._make_key(Field.NONANT, 0)
        return self.receive_buffers[key].is_new()


class InnerBoundNonantSpoke(_BoundNonantSpoke):
    """ For Spokes that provide an inner (incumbent)
        bound through self.bound to the Hub,
        and receive the nonants from
        the main SPOpt hub.

        Includes some helpful methods for saving
        and restoring results
    """

    send_fields = (*_BoundNonantSpoke.send_fields, Field.OBJECTIVE_INNER_BOUND, )
    receive_fields = (*_BoundNonantSpoke.receive_fields, Field.NONANT)

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

    def bound_type(self) -> Field:
        return Field.OBJECTIVE_INNER_BOUND



class OuterBoundNonantSpoke(_BoundNonantSpoke):
    """ For Spokes that provide an outer
        bound through self.bound to the Hub,
        and receive the nonants from
        the main OPT hub.
    """

    send_fields = (*_BoundNonantSpoke.send_fields, Field.OBJECTIVE_OUTER_BOUND, )
    receive_fields = (*_BoundNonantSpoke.receive_fields, Field.NONANT)

    converger_spoke_char = 'A'  # probably Lagrangian

    def bound_type(self) -> Field:
        return Field.OBJECTIVE_OUTER_BOUND
