###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# TODO Remove after this program no longer support Python 3.8
from __future__ import annotations

import numpy as np
import abc
import logging
import mpisppy.log
from mpisppy.opt.aph import APH

from mpisppy import MPI
from mpisppy.cylinders.spcommunicator import RecvArray, SendArray, SPCommunicator
from math import inf
from mpisppy.cylinders.spoke import ConvergerSpokeType

from mpisppy import global_toc

from mpisppy.cylinders.spwindow import Field

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.Hub",
                         "hub.log",
                         level=logging.CRITICAL)
logger = logging.getLogger("mpisppy.cylinders.Hub")

class Hub(SPCommunicator):
    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, spokes, options=None):
        super().__init__(spbase_object, fullcomm, strata_comm, cylinder_comm, options=options)
        assert len(spokes) == self.n_spokes
        self.spokes = spokes  # List of dicts
        logger.debug(f"Built the hub object on global rank {fullcomm.Get_rank()}")
        # for logging
        self.print_init = True
        self.latest_ib_char = None
        self.latest_ob_char = None
        self.last_ib_idx = None
        self.last_ob_idx = None
        # for termination based on stalling out
        self.stalled_iter_cnt = 0
        self.last_gap = float('inf')  # abs_gap tracker

        # All hubs need to be able to tell spokes to terminate. Register that here.
        self.shutdown = self.register_send_field(Field.SHUTDOWN, 1)

        self.extension_recv = set()

        return

    @abc.abstractmethod
    def setup_hub(self):
        pass

    @abc.abstractmethod
    def sync(self):
        """ To be called within the whichever optimization algorithm
            is being run on the hub (e.g. PH)
        """
        pass

    @abc.abstractmethod
    def is_converged(self):
        """ The hub has the ability to halt the optimization algorithm on the
            hub before any local convergers.
        """
        pass

    @abc.abstractmethod
    def current_iteration(self):
        """ Returns the current iteration count - however the hub defines it.
        """
        pass

    @abc.abstractmethod
    def main(self):
        pass


    def register_extension_recv_field(self, field: Field, strata_rank: int, buf_len: int) -> RecvArray:
        """
        Register an extensions interest in the given field from the given spoke. The hub
        is then responsible for updating this field into a local buffer prior to the call
        to the extension sync_with_spokes function.
        """
        # TODO: What is the correct action when registering a field that is already registered?
        ra = self.register_recv_field(field, strata_rank, buf_len)
        key = self._make_key(field, strata_rank)
        self.extension_recv.add(key)
        return ra

    def register_extension_send_field(self, field: Field, buf_len: int) -> SendArray:
        """
        Register a field with the hub that an extension will be making available to spokes. Returns a
        buffer that is usable for sending the desired values. The extension is responsible for calling
        the hub publish_extension_field when ready to send the values. Returns a SendArray to use
        to publish values to spokes.
        """
        # TODO: What is the correct action when registering a field that is already registered?
        # TODO: This needs to be executed BEFORE the SPWindow is setup and somehow be included in
        # the buffer spec...
        sa = self.register_send_field(field, buf_len)
        return sa

    def get_extension_recv_field(self, strata_rank: int, field: Field) -> RecvArray:
        key = self._make_key(field, strata_rank)
        assert(key in self.extension_recv)
        return self._locals[key]

    def publish_extension_field(self, field: Field, buf: SendArray):
        # TODO: Implement this...
        return

    def sync_extension_fields(self):
        """
        Update all registered extension fields. Safe to call even when there are no extension fields.
        """
        for key in self.extension_recv:
            ext_buf = self._locals[key]
            (field, srank) = self._split_key(key)
            ext_buf._is_new = self.hub_from_spoke(ext_buf.array(), srank, field, ext_buf.id())
            if ext_buf.is_new():
                ext_buf.pull_id()
            ## End if
        ## End for
        return

    def clear_latest_chars(self):
        self.latest_ib_char = None
        self.latest_ob_char = None


    def compute_gaps(self):
        """ Compute the current absolute and relative gaps,
            using the current self.BestInnerBound and self.BestOuterBound
        """
        if self.opt.is_minimizing:
            abs_gap = self.BestInnerBound - self.BestOuterBound
        else:
            abs_gap = self.BestOuterBound - self.BestInnerBound

        ## define by the best solution, as is common
        nano = float("nan")  # typing aid
        if (
            abs_gap != nano
            and abs_gap != float("inf")
            and abs_gap != float("-inf")
            and self.BestOuterBound != nano
            and self.BestOuterBound != 0
        ):
            rel_gap = abs_gap / abs(self.BestOuterBound)
        else:
            rel_gap = float("inf")
        return abs_gap, rel_gap


    def get_update_string(self):
        if self.latest_ib_char is None and \
                self.latest_ob_char is None:
            return '   '
        if self.latest_ib_char is None:
            return self.latest_ob_char + '  '
        if self.latest_ob_char is None:
            return '  ' + self.latest_ib_char
        return self.latest_ob_char+' '+self.latest_ib_char

    def screen_trace(self):
        current_iteration = self.current_iteration()
        abs_gap, rel_gap = self.compute_gaps()
        best_solution = self.BestInnerBound
        best_bound = self.BestOuterBound
        update_source = self.get_update_string()
        if self.print_init:
            row = f'{"Iter.":>5s}  {"   "}  {"Best Bound":>14s}  {"Best Incumbent":>14s}  {"Rel. Gap":>12s}  {"Abs. Gap":>14s}'
            global_toc(row, True)
            self.print_init = False
        row = f"{current_iteration:5d}  {update_source}  {best_bound:14.4f}  {best_solution:14.4f}  {rel_gap*100:12.3f}%  {abs_gap:14.4f}"
        global_toc(row, True)
        self.clear_latest_chars()

    def determine_termination(self):
        # return True if termination is indicated, otherwise return False

        if not hasattr(self,"options") or self.options is None\
           or ("rel_gap" not in self.options and "abs_gap" not in self.options\
           and "max_stalled_iters" not in self.options):
            return False  # Nothing to see here folks...

        # If we are still here, there is some option for termination
        abs_gap, rel_gap = self.compute_gaps()

        abs_gap_satisfied = False
        rel_gap_satisfied = False
        max_stalled_satisfied = False

        if "rel_gap" in self.options and rel_gap <= self.options["rel_gap"]:
            rel_gap_satisfied = True
        if "abs_gap" in self.options and abs_gap <= self.options["abs_gap"]:
            abs_gap_satisfied = True

        if "max_stalled_iters" in self.options:
            if abs_gap < self.last_gap:  # liberal test (we could use an epsilon)
                self.last_gap = abs_gap
                self.stalled_iter_cnt = 0
            else:
                self.stalled_iter_cnt += 1
                if self.stalled_iter_cnt >= self.options["max_stalled_iters"]:
                    max_stalled_satisfied = True

        if abs_gap_satisfied:
            global_toc(f"Terminating based on inter-cylinder absolute gap {abs_gap:12.4f}")
        if rel_gap_satisfied:
            global_toc(f"Terminating based on inter-cylinder relative gap {rel_gap*100:12.3f}%")
        if max_stalled_satisfied:
            global_toc(f"Terminating based on max-stalled-iters {self.stalled_iter_cnt}")

        return abs_gap_satisfied or rel_gap_satisfied or max_stalled_satisfied

    def hub_finalize(self):
        if self.has_outerbound_spokes:
            self.receive_outerbounds()
        if self.has_innerbound_spokes:
            self.receive_innerbounds()

        if self.global_rank == 0:
            self.print_init = True
            global_toc("Statistics at termination", True)
            self.screen_trace()

    def receive_innerbounds(self):
        """ Get inner bounds from inner bound spokes
            NOTE: Does not check if there _are_ innerbound spokes
            (but should be harmless to call if there are none)
        """
        logging.debug("Hub is trying to receive from InnerBounds")
        for idx in self.innerbound_spoke_indices:
            key = self._make_key(Field.INNER_BOUND, idx)
            recv_buf = self._locals[key]
            is_new = self.hub_from_spoke(recv_buf.array(),
                                         idx,
                                         Field.INNER_BOUND,
                                         recv_buf.id())
            if is_new:
                recv_buf.pull_id()
                bound = recv_buf.array()[0]
                logging.debug("!! new InnerBound to opt {}".format(bound))
                self.BestInnerBound = self.InnerBoundUpdate(bound, idx)
        logging.debug("ph back from InnerBounds")

    def receive_outerbounds(self):
        """ Get outer bounds from outer bound spokes
            NOTE: Does not check if there _are_ outerbound spokes
            (but should be harmless to call if there are none)
        """
        logging.debug("Hub is trying to receive from OuterBounds")
        for idx in self.outerbound_spoke_indices:
            key = self._make_key(Field.OUTER_BOUND, idx)
            recv_buf = self._locals[key]
            is_new = self.hub_from_spoke(recv_buf.array(),
                                         idx,
                                         Field.OUTER_BOUND,
                                         recv_buf.id())
            if is_new:
                recv_buf.pull_id()
                bound = recv_buf.array()[0]
                logging.debug("!! new OuterBound to opt {}".format(bound))
                self.BestOuterBound = self.OuterBoundUpdate(bound, idx)
        logging.debug("ph back from OuterBounds")

    def OuterBoundUpdate(self, new_bound, idx=None, char='*'):
        current_bound = self.BestOuterBound
        if self._outer_bound_update(new_bound, current_bound):
            if idx is None:
                self.latest_ob_char = char
                self.last_ob_idx = 0
            else:
                self.latest_ob_char = self.outerbound_spoke_chars[idx]
                self.last_ob_idx = idx
            return new_bound
        else:
            return current_bound

    def InnerBoundUpdate(self, new_bound, idx=None, char='*'):
        current_bound = self.BestInnerBound
        if self._inner_bound_update(new_bound, current_bound):
            if idx is None:
                self.latest_ib_char = char
                self.last_ib_idx = 0
            else:
                self.latest_ib_char = self.innerbound_spoke_chars[idx]
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

    def initialize_outer_bound_buffers(self):
        """ Initialize outer bound receive buffers
        """
        self.outerbound_receive_buffers = dict()
        for idx in self.outerbound_spoke_indices:
            self.outerbound_receive_buffers[idx] = self.register_recv_field(
                Field.OUTER_BOUND, idx, 1,
            )
        ## End for
        return

    def initialize_inner_bound_buffers(self):
        """ Initialize inner bound receive buffers
        """
        self.innerbound_receive_buffers = dict()
        for idx in self.innerbound_spoke_indices:
            self.innerbound_receive_buffers[idx] = self.register_recv_field(
                Field.INNER_BOUND, idx, 1
            )
        ## End for
        return

    def initialize_nonants(self):
        """ Initialize the buffer for the hub to send nonants
            to the appropriate spokes
        """
        self.nonant_send_buffer = None
        if self.has_nonant_spokes:
            # self.nonant_send_buffer = self._sends[Field.NONANT].array()
            self.nonant_send_buffer = self._sends[Field.NONANT]
        ## End if
        return

    def initialize_boundsout(self):
        """ Initialize the buffer for the hub to send bounds
            to bounds only spokes
        """
        self.boundsout_send_buffer = None
        if self.has_bounds_only_spokes:
            # self.boundsout_send_buffer = self._sends[Field.BOUNDS].array()
            self.boundsout_send_buffer = self._sends[Field.BOUNDS]
        ## End if
        return

    def _populate_boundsout_cache(self, buf):
        """ Populate a given buffer with the current bounds
        """
        buf[-3] = self.BestOuterBound
        buf[-2] = self.BestInnerBound

    def send_boundsout(self):
        """ Send bounds to the appropriate spokes
        This is called only for spokes which are bounds only.
        w and nonant spokes are passed bounds through the w and nonant buffers
        """
        # NOTE: boundsout_send_buffer should be the same numpy array as my_bounds.array()
        # my_bounds = self._sends[Field.BOUNDS]
        my_bounds = self.boundsout_send_buffer
        self._populate_boundsout_cache(my_bounds.array())
        logging.debug("hub is sending bounds={}".format(my_bounds.array()))
        self.hub_to_spoke(my_bounds.array(), Field.BOUNDS, my_bounds.next_write_id())
        return

    def initialize_spoke_indices(self):
        """ Figure out what types of spokes we have,
        and sort them into the appropriate classes.

        Note:
            Some spokes may be multiple types (e.g. outerbound and nonant),
            though not all combinations are supported.
        """
        self.outerbound_spoke_indices = set()
        self.innerbound_spoke_indices = set()
        self.nonant_spoke_indices = set()
        self.w_spoke_indices = set()

        self.outerbound_spoke_chars = dict()
        self.innerbound_spoke_chars = dict()

        for (i, spoke) in enumerate(self.spokes):
            spoke_class = spoke["spoke_class"]
            if hasattr(spoke_class, "converger_spoke_types"):
                for cst in spoke_class.converger_spoke_types:
                    if cst == ConvergerSpokeType.OUTER_BOUND:
                        self.outerbound_spoke_indices.add(i + 1)
                        self.outerbound_spoke_chars[i+1] = spoke_class.converger_spoke_char
                    elif cst == ConvergerSpokeType.INNER_BOUND:
                        self.innerbound_spoke_indices.add(i + 1)
                        self.innerbound_spoke_chars[i+1] = spoke_class.converger_spoke_char
                    elif cst == ConvergerSpokeType.W_GETTER:
                        self.w_spoke_indices.add(i + 1)
                    elif cst == ConvergerSpokeType.NONANT_GETTER:
                        self.nonant_spoke_indices.add(i + 1)
                    else:
                        raise RuntimeError(f"Unrecognized converger_spoke_type {cst}")

            else:  ##this isn't necessarily wrong, i.e., cut generators
                logger.debug(f"Spoke class {spoke_class} not recognized by hub")

        # all _BoundSpoke spokes get hub bounds so we determine which spokes
        # are "bounds only"
        self.bounds_only_indices = \
            (self.outerbound_spoke_indices | self.innerbound_spoke_indices) - \
            (self.w_spoke_indices | self.nonant_spoke_indices)

        self.has_outerbound_spokes = len(self.outerbound_spoke_indices) > 0
        self.has_innerbound_spokes = len(self.innerbound_spoke_indices) > 0
        self.has_nonant_spokes = len(self.nonant_spoke_indices) > 0
        self.has_w_spokes = len(self.w_spoke_indices) > 0
        self.has_bounds_only_spokes = len(self.bounds_only_indices) > 0

        # Not all opt classes may have extensions
        if getattr(self.opt, "extensions", None) is not None:
            self.opt.extobject.initialize_spoke_indices()

        return


    def build_window_spec(self) -> dict[Field, int]:

        required_fields = set()
        for spoke in self.spokes:
            spoke_class = spoke["spoke_class"]
            if hasattr(spoke_class, "converger_spoke_types"):
                for cst in spoke_class.converger_spoke_types:
                    if cst == ConvergerSpokeType.W_GETTER:
                        required_fields.add(Field.DUALS)
                    elif cst == ConvergerSpokeType.NONANT_GETTER:
                        required_fields.add(Field.NONANT)
                    elif cst == ConvergerSpokeType.INNER_BOUND or cst == ConvergerSpokeType.OUTER_BOUND:
                        required_fields.add(Field.BOUNDS)
                    else:
                        pass # Intentional no-op
                    ## End if
                ## End for
            else:
                # TODO: Do non-converger spoke types use info from the hub? If so,
                # need to account for that here.
                pass
            ## End if
        ## End for

        window_spec = dict()
        window_spec[Field.SHUTDOWN] = 1

        n_nonants = 0
        for s in self.opt.local_scenarios.values():
            n_nonants += len(s._mpisppy_data.nonant_indices)
        ## End for

        if Field.DUALS in required_fields:
            window_spec[Field.DUALS] = n_nonants
            self.register_send_field(Field.DUALS, n_nonants)
        if Field.NONANT in required_fields:
            window_spec[Field.NONANT] = n_nonants
            self.register_send_field(Field.NONANT, n_nonants)
        if Field.BOUNDS in required_fields:
            window_spec[Field.BOUNDS] = 2
            self.register_send_field(Field.BOUNDS, 2)

        # TODO: We can build the window spec directly from the self._sends dictionary
        # after all the register_send_field calls are complete.

        return window_spec


    def hub_to_spoke(self, values: np.typing.NDArray, field: Field, write_id: int):
        """ Put the specified values into the specified locally-owned buffer
            for the spoke to pick up.

            Notes:
                This automatically does the -1 indexing

                This assumes that values contains a slot at the end for the
                write_id
        """

        if not isinstance(self.opt, APH):
            self.cylinder_comm.Barrier()
        ## End if

        values[-1] = write_id
        self.window.put(values, field)

        return


    def hub_from_spoke(self,
                       values: np.typing.NDArray,
                       spoke_num: int,
                       field: Field,
                       last_write_id: int,
                       ):
        """ spoke_num is the rank in the strata_comm, so it is 1-based not 0-based

            Returns:
                is_new (bool): Indicates whether the "gotten" values are new,
                    based on the write_id.
        """
        # so the window in each rank gets read at approximately the same time,
        # and so has the same write_id
        if not isinstance(self.opt, APH):
            self.cylinder_comm.Barrier()
        ## End if
        self.window.get(values, spoke_num, field)

        if isinstance(self.opt, APH):
            # # reverting part of changes from Ben getting rid of spoke sleep DLW jan 2023
            if values[-1] > last_write_id:
                return True
        else:
            new_id = int(values[-1])
            local_val = np.array((new_id,), 'i')
            sum_ids = np.zeros(1, 'i')
            self.cylinder_comm.Allreduce((local_val, MPI.INT),
                                         (sum_ids, MPI.INT),
                                         op=MPI.SUM)
            if new_id != sum_ids[0] / self.cylinder_comm.size:
                return False
            ## End if
            if new_id > last_write_id or new_id < 0:
                return True
            ## End if
        ## End if

        return False


    def send_terminate(self):
        """ Send an array of zeros with a -1 appended to the
            end to indicate termination. This function puts to the local
            buffer, so every spoke will see it simultaneously.
            processes (don't need to call them one at a time).
        """
        # term_buf = self._sends[Field.SHUTDOWN]
        shutdown = self.shutdown
        shutdown[0] = 1.0
        self.hub_to_spoke(shutdown.array(), Field.SHUTDOWN, shutdown.next_write_id())
        return


class PHHub(Hub):
    def setup_hub(self):
        """ Must be called after make_windows(), so that
            the hub knows the sizes of all the spokes windows
        """
        if not self._windows_constructed:
            raise RuntimeError(
                "Cannot call setup_hub before memory windows are constructed"
            )

        # attribute to set False if some extension
        # modified the iteration 0 subproblems such
        # that the trivial bound is no longer valid
        self.use_trivial_bound = True

        self.initialize_spoke_indices()
        self.initialize_bound_values()

        if self.has_outerbound_spokes:
            self.initialize_outer_bound_buffers()
        if self.has_innerbound_spokes:
            self.initialize_inner_bound_buffers()
        if self.has_w_spokes:
            self.initialize_ws()
        if self.has_nonant_spokes:
            self.initialize_nonants()
        if self.has_bounds_only_spokes:
            self.initialize_boundsout()  # bounds going out

        ## Do some checking for things we currently don't support
        if len(self.outerbound_spoke_indices & self.innerbound_spoke_indices) > 0:
            raise RuntimeError(
                "A Spoke providing both inner and outer "
                "bounds is currently unsupported"
            )
        if len(self.w_spoke_indices & self.nonant_spoke_indices) > 0:
            raise RuntimeError(
                "A Spoke needing both Ws and nonants is currently unsupported"
            )

        ## Generate some warnings if nothing is giving bounds
        if not self.has_outerbound_spokes:
            logger.warn(
                "No OuterBound Spokes defined, this converger "
                "will not cause the hub to terminate"
            )

        if not self.has_innerbound_spokes:
            logger.warn(
                "No InnerBound Spokes defined, this converger "
                "will not cause the hub to terminate"
            )
        if self.opt.extensions is not None:
            self.opt.extobject.setup_hub()

    def sync(self):
        """
            Manages communication with Spokes
        """
        if self.has_w_spokes:
            self.send_ws()
        if self.has_nonant_spokes:
            self.send_nonants()
        if self.has_bounds_only_spokes:
            self.send_boundsout()
        if self.has_outerbound_spokes:
            self.receive_outerbounds()
        if self.has_innerbound_spokes:
            self.receive_innerbounds()
        if self.opt.extensions is not None:
            self.sync_extension_fields()
            self.opt.extobject.sync_with_spokes()

    def sync_with_spokes(self):
        self.sync()

    def is_converged(self):
        ## might as well get a bound, in this case
        if self.opt._PHIter == 1 and self.use_trivial_bound:
            self.BestOuterBound = self.OuterBoundUpdate(self.opt.trivial_bound)

        if not self.has_innerbound_spokes:
            if self.opt._PHIter == 1:
                logger.warning(
                    "PHHub cannot compute convergence without "
                    "inner bound spokes."
                )

            ## you still want to output status, even without inner bounders configured
            if self.global_rank == 0:
                self.screen_trace()

            return False

        if not self.has_outerbound_spokes:
            if self.opt._PHIter == 1:
                global_toc(
                    "Without outer bound spokes, no progress "
                    "will be made on the Best Bound")

        ## log some output
        if self.global_rank == 0:
            self.screen_trace()

        return self.determine_termination()

    def current_iteration(self):
        """ Return the current PH iteration."""
        return self.opt._PHIter

    def main(self):
        """ SPComm gets attached in self.__init__ """
        self.opt.ph_main(finalize=False)

    def finalize(self):
        """ does PH.post_loops, returns Eobj """
        Eobj = self.opt.post_loops(self.opt.extensions)
        return Eobj

    def send_nonants(self):
        """ Gather nonants and send them to the appropriate spokes
            TODO: Will likely fail with bundling
        """
        self.opt._save_nonants()
        ci = 0  ## index to self.nonant_send_buffer
        # my_nonants = self._sends[Field.NONANT]
        nonant_send_buffer = self.nonant_send_buffer
        for k, s in self.opt.local_scenarios.items():
            for xvar in s._mpisppy_data.nonant_indices.values():
                nonant_send_buffer[ci] = xvar._value
                ci += 1
        logging.debug("hub is sending X nonants={}".format(nonant_send_buffer))

        self.hub_to_spoke(nonant_send_buffer.array(), Field.NONANT, nonant_send_buffer.next_write_id())

        return

    def initialize_ws(self):
        """ Initialize the buffer for the hub to send dual weights
            to the appropriate spokes
        """
        self.w_send_buffer = None
        if self.has_w_spokes:
            self.w_send_buffer = self._sends[Field.DUALS].array()
        ## End if
        return

    def send_ws(self):
        """ Send dual weights to the appropriate spokes
        """
        # NOTE: my_ws.array() and self.w_send_buffer should be the same array.
        my_ws = self._sends[Field.DUALS]
        self.opt._populate_W_cache(my_ws.array(), padding=1)
        logging.debug("hub is sending Ws={}".format(my_ws.array()))

        self.hub_to_spoke(my_ws.array(), Field.DUALS, my_ws.next_write_id())

        return

class LShapedHub(Hub):

    def setup_hub(self):
        """ Must be called after make_windows(), so that
            the hub knows the sizes of all the spokes windows
        """
        if not self._windows_constructed:
            raise RuntimeError(
                "Cannot call setup_hub before memory windows are constructed"
            )

        self.initialize_spoke_indices()
        self.initialize_bound_values()

        if self.has_outerbound_spokes:
            self.initialize_outer_bound_buffers()
        if self.has_innerbound_spokes:
            self.initialize_inner_bound_buffers()

        ## Do some checking for things we currently
        ## do not support
        if self.has_w_spokes:
            raise RuntimeError("LShaped hub does not compute dual weights (Ws)")
        if self.has_nonant_spokes:
            self.initialize_nonants()
        if len(self.outerbound_spoke_indices & self.innerbound_spoke_indices) > 0:
            raise RuntimeError(
                "A Spoke providing both inner and outer "
                "bounds is currently unsupported"
            )

        ## Generate some warnings if nothing is giving bounds
        if not self.has_innerbound_spokes:
            logger.warn(
                "No InnerBound Spokes defined, this converger "
                "will not cause the hub to terminate"
            )

    def sync(self, send_nonants=True):
        """
        Manages communication with Bound Spokes
        """
        if send_nonants and self.has_nonant_spokes:
            self.send_nonants()
        if self.has_outerbound_spokes:
            self.receive_outerbounds()
        if self.has_innerbound_spokes:
            self.receive_innerbounds()
        # in case LShaped ever gets extensions
        if getattr(self.opt, "extensions", None) is not None:
            self.sync_extension_fields()
            self.opt.extobject.sync_with_spokes()

    def is_converged(self):
        """ Returns a boolean. If True, then LShaped will terminate

        Side-effects:
            The L-shaped method produces outer bounds during execution,
            so we will check it as well.
        """
        bound = self.opt._LShaped_bound
        self.BestOuterBound = self.OuterBoundUpdate(bound)

        ## log some output
        if self.global_rank == 0:
            self.screen_trace()

        return self.determine_termination()

    def current_iteration(self):
        """ Return the current L-shaped iteration."""
        return self.opt.iter

    def main(self):
        """ SPComm gets attached in self.__init__ """
        self.opt.lshaped_algorithm()

    def send_nonants(self):
        """ Gather nonants and send them to the appropriate spokes
            TODO: Will likely fail with bundling
        """
        ci = 0  ## index to self.nonant_send_buffer
        nonant_send_buffer = self.nonant_send_buffer
        for k, s in self.opt.local_scenarios.items():
            nonant_to_root_var_map = s._mpisppy_model.subproblem_to_root_vars_map
            for xvar in s._mpisppy_data.nonant_indices.values():
                ## Grab the value from the associated root variable
                nonant_send_buffer[ci] = nonant_to_root_var_map[xvar]._value
                ci += 1
        logging.debug("hub is sending X nonants={}".format(nonant_send_buffer))

        my_nonants = self._sends[Field.NONANT]
        self.hub_to_spoke(nonant_send_buffer.array(), Field.NONANT, nonant_send_buffer.next_write_id())

        return

class APHHub(PHHub):

    def main(self):
        """ SPComm gets attached by self.__init___; holding APH harmless """
        logger.critical("aph debug main in hub.py")
        self.opt.APH_main(spcomm=self, finalize=False)

    def finalize(self):
        """ does PH.post_loops, returns Eobj """
        # NOTE: APH_main does NOT pass in extensions
        #       to APH.post_loops
        Eobj = self.opt.post_loops()
        return Eobj
