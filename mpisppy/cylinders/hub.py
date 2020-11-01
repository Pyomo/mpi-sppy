# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import numpy as np
import abc
import logging
import time
import mpisppy.log

from mpi4py import MPI
from mpisppy.cylinders.spcommunicator import SPCommunicator
from math import inf
from mpisppy.cylinders.spoke import ConvergerSpokeType

from mpisppy import tt_timer

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.Hub",
                         "hub.log",
                         level=logging.CRITICAL)
logger = logging.getLogger("mpisppy.cylinders.Hub")

class Hub(SPCommunicator):
    def __init__(self, spbase_object, fullcomm, intercomm, intracomm, spokes, options=None):
        super().__init__(spbase_object, fullcomm, intercomm, intracomm, options=options)
        assert len(spokes) == self.n_spokes
        self.local_write_ids = np.zeros(self.n_spokes, dtype=np.int64)
        self.remote_write_ids = np.zeros(self.n_spokes, dtype=np.int64)
        self.local_lengths = np.zeros(self.n_spokes, dtype=np.int64)
        self.remote_lengths = np.zeros(self.n_spokes, dtype=np.int64)
        self.spokes = spokes  # List of dicts
        logger.debug(f"Built the hub object on global rank {fullcomm.Get_rank()}")
        # ^^^ Does NOT include +1
        # for logging
        self.print_init = True
        self.latest_ib_char = None
        self.latest_ob_char = None
        self.last_ib_idx = None
        self.last_ob_idx = None

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

    def clear_latest_chars(self):
        self.latest_ib_char = None
        self.latest_ob_char = None

    def compute_gap(self, compute_relative=True):
        """ Compute the current absolute or relative gap, 
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
        if compute_relative:
            return rel_gap
        else:
            return abs_gap

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
        rel_gap = self.compute_gap(compute_relative=True)
        abs_gap = self.compute_gap(compute_relative=False)
        best_solution = self.BestInnerBound
        best_bound = self.BestOuterBound
        update_source = self.get_update_string()
        if self.print_init:
            row = f'{"Iter.":>5s}  {"   "}  {"Best Bound":>14s}  {"Best Incumbent":>14s}  {"Rel. Gap":>12s}  {"Abs. Gap":>14s}'
            tt_timer.toc(row, delta=False)
            self.print_init = False
        row = f"{current_iteration:5d}  {update_source}  {best_bound:14.4f}  {best_solution:14.4f}  {rel_gap*100:12.3f}%  {abs_gap:14.4f}"
        tt_timer.toc(row, delta=False)
        self.clear_latest_chars()

    def determine_termination(self):
        abs_gap_satisfied = False
        rel_gap_satisfied = False
        if hasattr(self,"options") and self.options is not None:
            if "rel_gap" in self.options:
                rel_gap = self.compute_gap(compute_relative=True)
                rel_gap_satisfied = rel_gap <= self.options["rel_gap"]
            if "abs_gap" in self.options:
                abs_gap = self.compute_gap(compute_relative=False)
                abs_gap_satisfied = abs_gap <= self.options["abs_gap"]
        if abs_gap_satisfied and self.rank_global == 0:
            tt_timer.toc(f"Terminating based on inter-cylinder absolute gap {abs_gap:12.4f}", delta=False)
        if rel_gap_satisfied and self.rank_global == 0:
            tt_timer.toc(f"Terminating based on inter-cylinder relative gap {rel_gap*100:12.3f}%", delta=False)
        return abs_gap_satisfied or rel_gap_satisfied

    def hub_finalize(self):
        if self.has_outerbound_spokes:
            self.receive_outerbounds()
        if self.has_innerbound_spokes:
            self.receive_innerbounds()

        if self.rank_global == 0:
            self.print_init = True
            tt_timer.toc(f" ", delta=False)
            tt_timer.toc(f"Statistics at termination", delta=False)
            self.screen_trace()

    def receive_innerbounds(self):
        """ Get inner bounds from inner bound spokes
            NOTE: Does not check if there _are_ innerbound spokes
            (but should be harmless to call if there are none)
        """
        logging.debug("Hub is trying to receive from InnerBounds")
        for idx in self.innerbound_spoke_indices:
            is_new = self.hub_from_spoke(self.innerbound_receive_buffers[idx], idx)
            if is_new:
                bound = self.innerbound_receive_buffers[idx][0]
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
            is_new = self.hub_from_spoke(self.outerbound_receive_buffers[idx], idx)
            if is_new:
                bound = self.outerbound_receive_buffers[idx][0]
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
        """ Initialize value of BestOuterBound, and outer bound receive buffers
        """
        self.outerbound_receive_buffers = dict()
        for idx in self.outerbound_spoke_indices:
            self.outerbound_receive_buffers[idx] = np.zeros(
                self.remote_lengths[idx - 1] + 1
            )

    def initialize_inner_bound_buffers(self):
        """ Initialize value of BestInnerBound, and inner bound receive buffers
        """
        self.innerbound_receive_buffers = dict()
        for idx in self.innerbound_spoke_indices:
            self.innerbound_receive_buffers[idx] = np.zeros(
                self.remote_lengths[idx - 1] + 1
            )

    def initialize_nonants(self):
        """ Initialize the buffer for the hub to send nonants
            to the appropriate spokes
        """
        self.nonant_send_buffer = None
        for idx in self.nonant_spoke_indices:
            if self.nonant_send_buffer is None:
                self.nonant_send_buffer = np.zeros(self.local_lengths[idx - 1] + 1)
            elif self.local_lengths[idx - 1] + 1 != len(self.nonant_send_buffer):
                raise RuntimeError("Nonant buffers disagree on size")

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

        self.has_outerbound_spokes = len(self.outerbound_spoke_indices) > 0
        self.has_innerbound_spokes = len(self.innerbound_spoke_indices) > 0
        self.has_nonant_spokes = len(self.nonant_spoke_indices) > 0
        self.has_w_spokes = len(self.w_spoke_indices) > 0

    def make_windows(self):
        if self._windows_constructed:
            # different parts of the hub may call make_windows,
            # we just care about the first call
            return

        # Spokes notify the hub of the buffer sizes
        for i in range(self.n_spokes):
            pair_of_sizes = np.zeros(2, dtype="i")
            self.intercomm.Recv((pair_of_sizes, MPI.INT), source=i + 1, tag=i + 1)
            self.remote_lengths[i] = pair_of_sizes[0]
            self.local_lengths[i] = pair_of_sizes[1]

        # Make the windows of the appropriate buffer sizes
        self.windows = [None for _ in range(self.n_spokes)]
        self.buffers = [None for _ in range(self.n_spokes)]
        for i in range(self.n_spokes):
            length = self.local_lengths[i]
            win, buff = self._make_window(length)
            self.windows[i] = win
            self.buffers[i] = buff

        # flag this for multiple calls from the hub
        self._windows_constructed = True

    def hub_to_spoke(self, values, spoke_rank_inter):
        """ Put the specified values into the specified locally-owned buffer
            for the spoke to pick up.

            Notes:
                This automatically does the -1 indexing

                This assumes that values contains a slot at the end for the
                write_id
        """
        expected_length = self.local_lengths[spoke_rank_inter - 1] + 1
        if len(values) != expected_length:
            raise RuntimeError(
                f"Attempting to put array of length {len(values)} "
                f"into local buffer of length {expected_length}"
            )
        self.local_write_ids[spoke_rank_inter - 1] += 1
        values[-1] = self.local_write_ids[spoke_rank_inter - 1]
        window = self.windows[spoke_rank_inter - 1]
        window.Lock(self.rank_inter)
        window.Put((values, len(values), MPI.DOUBLE), self.rank_inter)
        window.Unlock(self.rank_inter)

    def hub_from_spoke(self, values, spoke_num):
        """ spoke_num is the rank in the intercomm, so it is 1-based not 0-based
            
            Returns:
                is_new (bool): Indicates whether the "gotten" values are new,
                    based on the write_id.
        """
        expected_length = self.remote_lengths[spoke_num - 1] + 1
        if len(values) != expected_length:
            raise RuntimeError(
                f"Hub trying to get buffer of length {expected_length} "
                f"from spoke, but provided buffer has length {len(values)}."
            )
        window = self.windows[spoke_num - 1]
        window.Lock(spoke_num)
        window.Get((values, len(values), MPI.DOUBLE), spoke_num)
        window.Unlock(spoke_num)

        if values[-1] > self.remote_write_ids[spoke_num - 1]:
            self.remote_write_ids[spoke_num - 1] = values[-1]
            return True
        return False

    def send_terminate(self):
        """ Send an array of zeros with a -1 appended to the
            end to indicate termination. This function puts to the local
            buffer, so every spoke will see it simultaneously. 
            processes (don't need to call them one at a time).
        """
        for rank in range(1, self.n_spokes + 1):
            dummies = np.zeros(self.local_lengths[rank - 1] + 1)
            dummies[-1] = -1
            window = self.windows[rank - 1]
            window.Lock(0)
            window.Put((dummies, len(dummies), MPI.DOUBLE), 0)
            window.Unlock(0)


class PHHub(Hub):
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
        if self.has_w_spokes:
            self.initialize_ws()
        if self.has_nonant_spokes:
            self.initialize_nonants()

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

    def sync(self):
        """
            Manages communication with Spokes
        """
        if self.has_w_spokes:
            self.send_ws()
        if self.has_nonant_spokes:
            self.send_nonants()
        if self.has_outerbound_spokes:
            self.receive_outerbounds()
        if self.has_innerbound_spokes:
            self.receive_innerbounds()

    def sync_with_spokes(self):
        self.sync()

    def is_converged(self):
        ## might as well get a bound, in this case
        if self.opt._PHIter == 1:
            self.BestOuterBound = self.OuterBoundUpdate(self.opt.trivial_bound)

        if not self.has_innerbound_spokes:
            if self.opt._PHIter == 1:
                logger.warning(
                    "PHHub cannot compute convergence without "
                    "inner bound spokes."
                )

            ## you still want to output status, even without inner bounders configured
            if self.rank_global == 0:                
                self.screen_trace()
                
            return False

        if not self.has_outerbound_spokes:
            if self.opt._PHIter == 1 and self.rank_global == 0:
                tt_timer.toc(
                    "Without outer bound spokes, no progress "
                    "will be made on the Best Bound", delta=False)

        ## log some output
        if self.rank_global == 0:
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
        Eobj = self.opt.post_loops(self.opt.PH_extensions)
        return Eobj

    def send_nonants(self):
        """ Gather nonants and send them to the appropriate spokes
            TODO: Will likely fail with bundling
        """
        self.opt._save_nonants()
        ci = 0  ## index to self.nonant_send_buffer
        nonant_send_buffer = self.nonant_send_buffer
        for k, s in self.opt.local_scenarios.items():
            for xvar in s._nonant_indexes.values():
                nonant_send_buffer[ci] = xvar._value
                ci += 1
        logging.debug("hub is sending X nonants={}".format(nonant_send_buffer))
        for idx in self.nonant_spoke_indices:
            self.hub_to_spoke(nonant_send_buffer, idx)

    def initialize_ws(self):
        """ Initialize the buffer for the hub to send dual weights
            to the appropriate spokes
        """
        self.w_send_buffer = None
        for idx in self.w_spoke_indices:
            if self.w_send_buffer is None:
                self.w_send_buffer = np.zeros(self.local_lengths[idx - 1] + 1)
            elif self.local_lengths[idx - 1] + 1 != len(self.w_send_buffer):
                raise RuntimeError("W buffers disagree on size")

    def send_ws(self):
        """ Send dual weights to the appropriate spokes
        """
        self.opt._populate_W_cache(self.w_send_buffer)
        logging.debug("hub is sending Ws={}".format(self.w_send_buffer))
        for idx in self.w_spoke_indices:
            self.hub_to_spoke(self.w_send_buffer, idx)


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

    def is_converged(self):
        """ Returns a boolean. If True, then LShaped will terminate

        Side-effects:
            The L-shaped method produces outer bounds during execution,
            so we will check it as well.
        """
        bound = self.opt._LShaped_bound
        self.BestOuterBound = self.OuterBoundUpdate(bound)

        ## log some output
        if self.rank_global == 0:                
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
            nonant_to_master_var_map = s._subproblem_to_master_vars_map
            for xvar in s._nonant_indexes.values():
                ## Grab the value from the associated master variable
                nonant_send_buffer[ci] = nonant_to_master_var_map[xvar]._value
                ci += 1
        logging.debug("hub is sending X nonants={}".format(nonant_send_buffer))
        for idx in self.nonant_spoke_indices:
            self.hub_to_spoke(nonant_send_buffer, idx)


class APHHub(PHHub):
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
            raise RuntimeError("APH not ready for outer bound spokes yet")
            self.initialize_outer_bound_buffers()
        if self.has_innerbound_spokes:
            self.initialize_inner_bound_buffers()
        if self.has_w_spokes:
            raise RuntimeError("APH not ready for W spokes")
            self.initialize_ws()
        if self.has_nonant_spokes:
            self.initialize_nonants()

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


    def sync(self):
        """
            Manages communication with Spokes
        """
        if self.has_w_spokes:
            self.send_ws()
        if self.has_nonant_spokes:
            self.send_nonants()
        if self.has_outerbound_spokes:
            self.receive_outerbounds()
        if self.has_innerbound_spokes:
            self.receive_innerbounds()


    def sync_with_spokes(self):
        self.sync()

    def current_iteration(self):
        """ Return the current APH iteration."""
        return self.opt._PHIter

    def main(self):
        """ SPComm gets attached by self.__init___; holding APH harmless """
        logger.critical("aph debug main in hub.py")
        self.opt.APH_main(spcomm=self, finalize=False)

    def finalize(self):
        """ does PH.post_loops, returns Eobj """
        # NOTE: APH_main does NOT pass in PH_extensions
        #       to APH.post_loops
        Eobj = self.opt.post_loops()
        return Eobj
