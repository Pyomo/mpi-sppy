###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import abc
import time
import os
import math

from mpisppy.cylinders.spcommunicator import SPCommunicator, SendCircularBuffer
from mpisppy.cylinders.spwindow import Field


class Spoke(SPCommunicator):

    send_fields = (*SPCommunicator.send_fields, )
    receive_fields = (*SPCommunicator.receive_fields, Field.SHUTDOWN, Field.BEST_OBJECTIVE_BOUNDS, )

    def got_kill_signal(self):
        """ Spoke should call this method at least every iteration
            to see if the Hub terminated
        """
        shutdown_buf = self.receive_buffers[self._make_key(Field.SHUTDOWN, 0)]
        self.get_receive_buffer(shutdown_buf, Field.SHUTDOWN, 0, synchronize=False)
        return self.allreduce_or(shutdown_buf[0] == 1.0)

    def is_converged(self, screen_trace=False):
        """ Alias for got_kill_signal; useful for algorithms working as both
            hub and spoke
        """
        return self.got_kill_signal()

    @abc.abstractmethod
    def main(self):
        """
        The main call for the Spoke. Derived classe
        should call the got_kill_signal method
        regularly to ensure all ranks terminate
        with the Hub.
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

    def send_bound(self, value):
        if self.bound_type() == Field.OBJECTIVE_INNER_BOUND:
            self.BestInnerBound = self.InnerBoundUpdate(value)
        elif self.bound_type() == Field.OBJECTIVE_OUTER_BOUND:
            self.BestOuterBound = self.OuterBoundUpdate(value)
        else:
            raise RuntimeError(f"Unexpected bound_type {self.bound_type()}")
        self._append_trace(value)
        self._bound[0] = value
        self.put_send_buffer(self._bound, self.bound_type())
        return

    def update_hub_bounds(self) -> None:
        """ get new hub inner / outer bounds from the hub """
        return self.get_receive_buffer(self._hub_bounds, Field.BEST_OBJECTIVE_BOUNDS, 0)

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

    def register_receive_fields(self) -> None:
        super().register_receive_fields()
        nonant_len_ranks = self.fields_to_ranks[self.nonant_len_type()]
        if len(nonant_len_ranks) > 1:
            raise RuntimeError(
                f"More than one cylinder to select from for {self.nonant_len_type()}!"
            )
        key = self._make_key(self.nonant_len_type(), nonant_len_ranks[0])
        self._nonant_len_receive_buffer = self.receive_buffers[key]
        self._nonant_len_receive_rank = nonant_len_ranks[0]

    def _update_nonant_len_buffer(self) -> bool:
        """ get new data from the hub """
        return self.get_receive_buffer(self._nonant_len_receive_buffer, self.nonant_len_type(), self._nonant_len_receive_rank)


class InnerBoundSpoke(_BoundSpoke):
    """ For Spokes that provide an inner bound through self.send_bound to the
        Hub, and do not need information from the main PH OPT hub.
    """

    send_fields = (*_BoundSpoke.send_fields, Field.OBJECTIVE_INNER_BOUND, Field.BEST_XHAT, Field.RECENT_XHATS, )
    receive_fields = (*_BoundSpoke.receive_fields, )

    converger_spoke_char = 'I'

    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, options=None):
        super().__init__(spbase_object, fullcomm, strata_comm, cylinder_comm, options)
        self.is_minimizing = self.opt.is_minimizing
        self.best_inner_bound = math.inf if self.is_minimizing else -math.inf
        self.solver_options = None # can be overwritten by derived classes

    def register_send_fields(self):
        super().register_send_fields()
        self._recent_xhat_send_circular_buffer = SendCircularBuffer(
            self.send_buffers[Field.RECENT_XHATS],
            self._field_lengths[Field.BEST_XHAT],
            self._field_lengths[Field.RECENT_XHATS] // self._field_lengths[Field.BEST_XHAT],
        )

    def update_if_improving(self, candidate_inner_bound, update_best_solution_cache=True):
        if candidate_inner_bound is None:
            return False
        if update_best_solution_cache:
            update = self.opt.update_best_solution_if_improving(candidate_inner_bound)
        else:
            update = ( (candidate_inner_bound < self.best_inner_bound)
                if self.is_minimizing else
                (self.best_inner_bound < candidate_inner_bound)
                )
        self.send_latest_xhat()
        if update:
            self.best_inner_bound = candidate_inner_bound
            # send to hub
            self.send_bound(candidate_inner_bound)
            self.send_best_xhat()
            return True
        return False

    def send_best_xhat(self):
        best_xhat_buf = self.send_buffers[Field.BEST_XHAT]
        ci = 0
        for s in self.opt.local_scenarios.values():
            solution_cache = s._mpisppy_data.best_solution_cache._dict
            for ndn_varid in s._mpisppy_data.varid_to_nonant_index:
                best_xhat_buf[ci] = solution_cache[ndn_varid][1]
                ci += 1
            best_xhat_buf[ci] = s._mpisppy_data.inner_bound
            ci += 1
        # print(f"{self.cylinder_rank=} sending {best_xhat_buf.value_array()=}")
        self.put_send_buffer(best_xhat_buf, Field.BEST_XHAT)

    def send_latest_xhat(self):
        recent_xhat_buf = self._recent_xhat_send_circular_buffer.next_value_array_reference()
        ci = 0
        for s in self.opt.local_scenarios.values():
            solution_cache = s._mpisppy_data.latest_nonant_solution_cache
            len_nonants = len(s._mpisppy_data.nonant_indices)
            recent_xhat_buf[ci:ci+len_nonants] = solution_cache[:]
            ci += len_nonants
            recent_xhat_buf[ci] = s._mpisppy_data.inner_bound
            ci += 1
        # print(f"{self.cylinder_rank=} sending {recent_xhat_buf=}")
        self.put_send_buffer(self._recent_xhat_send_circular_buffer.data, Field.RECENT_XHATS)

    def finalize(self):
        if self.opt.load_best_solution():
            self.final_bound = self.bound
            return self.final_bound
        return None

    def bound_type(self) -> Field:
        return Field.OBJECTIVE_INNER_BOUND


class OuterBoundSpoke(_BoundSpoke):
    """ For Spokes that provide an outer bound through self.send_bound to the
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
        return self._nonant_len_receive_buffer.value_array()

    def update_Ws(self) -> bool:
        """ Check for new Ws from the source.
        Returns True if the Ws are new. False otherwise.
        Puts the result in `localWs`.
        """
        return self._update_nonant_len_buffer()


class OuterBoundWSpoke(_BoundWSpoke):
    """
    For Spokes that provide an outer bound
    through self.send_bound to the Hub,
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
        return self._nonant_len_receive_buffer.value_array()

    def update_nonants(self) -> bool:
        """ Check for new nonants from the source.
        Returns True if the nonants are new. False otherwise.
        Puts the result in `localnonants`.
        """
        return self._update_nonant_len_buffer()


class InnerBoundNonantSpoke(_BoundNonantSpoke, InnerBoundSpoke):
    """ For Spokes that provide an inner (incumbent)
        bound through self.send_bound to the Hub,
        and receive the nonants from
        the main SPOpt hub.

        Includes some helpful methods for saving
        and restoring results
    """

    send_fields = (*InnerBoundSpoke.send_fields, )
    receive_fields = (*InnerBoundSpoke.receive_fields, Field.NONANT)

    converger_spoke_char = 'I'


class OuterBoundNonantSpoke(_BoundNonantSpoke):
    """ For Spokes that provide an outer
        bound through self.send_bound to the Hub,
        and receive the nonants from
        the main OPT hub.
    """

    send_fields = (*_BoundNonantSpoke.send_fields, Field.OBJECTIVE_OUTER_BOUND, )
    receive_fields = (*_BoundNonantSpoke.receive_fields, Field.NONANT)

    converger_spoke_char = 'A'  # probably Lagrangian

    def bound_type(self) -> Field:
        return Field.OBJECTIVE_OUTER_BOUND
