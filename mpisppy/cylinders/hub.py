###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import abc
import logging
import mpisppy.log

from mpisppy.cylinders.spcommunicator import RecvArray, SPCommunicator

from mpisppy import global_toc

from mpisppy.cylinders.spwindow import Field
from mpisppy.cylinders.fwph_cylinder import FWPH_Cylinder

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger(__name__,
                         "hub.log",
                         level=logging.CRITICAL)
logger = logging.getLogger(__name__)

class Hub(SPCommunicator):

    send_fields = (*SPCommunicator.send_fields, Field.SHUTDOWN, Field.BEST_OBJECTIVE_BOUNDS,)
    receive_fields = (*SPCommunicator.receive_fields,)

    _hub_algo_best_bound_provider = False

    def __init__(self, spbase_object, fullcomm, strata_comm, cylinder_comm, communicators, options=None):
        super().__init__(spbase_object, fullcomm, strata_comm, cylinder_comm, communicators, options=options)

        logger.debug(f"Built the hub object on global rank {fullcomm.Get_rank()}")
        # for logging
        self.print_init = True
        # for termination based on stalling out
        self.stalled_iter_cnt = 0
        self.last_outer_bound = self.BestOuterBound
        self.last_inner_bound = self.BestInnerBound

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

    def clear_latest_chars(self):
        self.latest_ib_char = None
        self.latest_ob_char = None

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

    def determine_termination(self, screen_trace=True):
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
            if self.last_outer_bound != self.BestOuterBound or self.last_inner_bound != self.BestInnerBound:
                self.last_outer_bound = self.BestOuterBound
                self.last_inner_bound = self.BestInnerBound
                self.stalled_iter_cnt = 0
            else:
                self.stalled_iter_cnt += 1
                if self.stalled_iter_cnt >= self.options["max_stalled_iters"]:
                    max_stalled_satisfied = True

        if screen_trace:
            if abs_gap_satisfied:
                global_toc(f"Terminating based on inter-cylinder absolute gap {abs_gap:12.4f}")
            if rel_gap_satisfied:
                global_toc(f"Terminating based on inter-cylinder relative gap {rel_gap*100:12.3f}%")
            if max_stalled_satisfied:
                global_toc(f"Terminating based on max-stalled-iters {self.stalled_iter_cnt}")

        return abs_gap_satisfied or rel_gap_satisfied or max_stalled_satisfied

    def hub_finalize(self):
        self.receive_outerbounds()
        self.receive_innerbounds()

        if self.global_rank == 0:
            self.print_init = True
            global_toc("Statistics at termination", True)
            self.screen_trace()

    def _populate_boundsout_cache(self, buf):
        """ Populate a given buffer with the current bounds
        """
        buf[0] = self.BestOuterBound
        buf[1] = self.BestInnerBound

    def send_boundsout(self):
        """ Send bounds to the appropriate spokes
        """
        my_bounds = self.send_buffers[Field.BEST_OBJECTIVE_BOUNDS]
        self._populate_boundsout_cache(my_bounds.array())
        logging.debug("hub is sending bounds={}".format(my_bounds))
        self.put_send_buffer(my_bounds, Field.BEST_OBJECTIVE_BOUNDS)
        return

    def register_receive_fields(self):
        """ Figure out what types of spokes we have,
        and sort them into the appropriate classes.

        Note:
            Some spokes may be multiple types (e.g. outerbound and nonant),
            though not all combinations are supported.
        """
        super().register_receive_fields()

        # Not all opt classes may have extensions
        if getattr(self.opt, "extensions", None) is not None:
            self.opt.extobject.register_receive_fields()

        return

    def register_send_fields(self):
        super().register_send_fields()

        # Not all opt classes may have extensions
        if getattr(self.opt, "extensions", None) is not None:
            self.opt.extobject.register_send_fields()

        return

    def send_terminate(self):
        """ Send an array of zeros with a -1 appended to the
            end to indicate termination. This function puts to the local
            buffer, so every spoke will see it simultaneously.
            processes (don't need to call them one at a time).
        """
        self.send_buffers[Field.SHUTDOWN][0] = 1.0
        self.put_send_buffer(self.send_buffers[Field.SHUTDOWN], Field.SHUTDOWN)
        return

    def sync_bounds(self):
        self.receive_nonant_bounds()
        self.receive_outerbounds()
        self.receive_innerbounds()
        self.send_boundsout()


class PHPrimalHub(Hub):
    """
    Like PHHub, but only sends nonants and omits Ws. To be used
    when another cylinder is supplying Ws (like RelaxedPHSpoke).
    Could be removed when mpi-sppy supports pointing consuming
    spokes like Lagrangian to a specific dual (W) buffer.
    """

    send_fields = (*Hub.send_fields, Field.NONANT, )
    receive_fields = (*Hub.receive_fields,)

    @property
    def nonant_field(self):
        return Field.NONANT

    def setup_hub(self):
        ## Generate some warnings if nothing is giving bounds
        if not self.receive_field_spcomms[Field.OBJECTIVE_OUTER_BOUND]:
            logger.warn(
                "No OuterBound Spokes defined, this converger "
                "will not cause the hub to terminate"
            )

        if not self.receive_field_spcomms[Field.OBJECTIVE_INNER_BOUND]:
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
        self.sync_Ws()
        self.sync_nonants()
        self.sync_bounds()
        self.sync_extensions()

    def sync_with_spokes(self):
        self.sync()

    def sync_extensions(self):
        if self.opt.extensions is not None:
            self.opt.extobject.sync_with_spokes()

    def sync_nonants(self):
        self.send_nonants()

    def sync_Ws(self):
        self.send_ws()

    def is_converged(self, screen_trace=True):
        if self.opt.best_bound_obj_val is not None:
            self.BestOuterBound = self.OuterBoundUpdate(self.opt.best_bound_obj_val)
        if self.opt.best_solution_obj_val is not None:
            self.BestInnerBound = self.InnerBoundUpdate(self.opt.best_solution_obj_val)

        if not self.receive_field_spcomms[Field.OBJECTIVE_INNER_BOUND]:
            if self.opt._PHIter == 1:
                logger.warning(
                    "PHHub cannot compute convergence without "
                    "inner bound spokes."
                )

            ## you still want to output status, even without inner bounders configured
            if self.global_rank == 0 and screen_trace:
                self.screen_trace()

            return False

        if not self.receive_field_spcomms[Field.OBJECTIVE_OUTER_BOUND]:
            if self.opt._PHIter == 1 and not self._hub_algo_best_bound_provider:
                global_toc(
                    "Without outer bound spokes, no progress "
                    "will be made on the Best Bound")

        ## log some output
        if self.global_rank == 0 and screen_trace:
            self.screen_trace()

        return self.determine_termination(screen_trace)

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
        """
        ci = 0  ## index to self.nonant_send_buffer
        nonant_send_buffer = self.send_buffers[self.nonant_field]
        for k, s in self.opt.local_scenarios.items():
            for xvar in s._mpisppy_data.nonant_indices.values():
                nonant_send_buffer[ci] = xvar._value
                ci += 1
        logging.debug("hub is sending X nonants={}".format(nonant_send_buffer))

        self.put_send_buffer(nonant_send_buffer, self.nonant_field)

        return

    def send_ws(self):
        """ Nonant hub; do not send Ws
        """
        pass


class PHHub(PHPrimalHub):
    send_fields = (*PHPrimalHub.send_fields, Field.DUALS, )
    receive_fields = (*PHPrimalHub.receive_fields,)

    def send_ws(self):
        """ Send dual weights to the appropriate spokes
        """
        # NOTE: my_ws.array() and self.w_send_buffer should be the same array.
        my_ws = self.send_buffers[Field.DUALS]
        self.opt._populate_W_cache(my_ws.array(), padding=1)
        logging.debug("hub is sending Ws={}".format(my_ws.array()))

        self.put_send_buffer(my_ws, Field.DUALS)

        return


class LShapedHub(Hub):

    send_fields = (*Hub.send_fields, Field.NONANT,)
    receive_fields = (*Hub.receive_fields,)

    def setup_hub(self):
        ## Generate some warnings if nothing is giving bounds
        if not self.receive_field_spcomms[Field.OBJECTIVE_INNER_BOUND]:
            logger.warn(
                "No InnerBound Spokes defined, this converger "
                "will not cause the hub to terminate"
            )

    def sync(self, send_nonants=True):
        """
        Manages communication with Bound Spokes
        """
        if send_nonants:
            self.send_nonants()
        self.sync_bounds()
        # in case LShaped ever gets extensions
        if getattr(self.opt, "extensions", None) is not None:
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
        """
        ci = 0  ## index to self.nonant_send_buffer
        nonant_send_buffer = self.send_buffers[Field.NONANT]
        for k, s in self.opt.local_scenarios.items():
            nonant_to_root_var_map = s._mpisppy_model.subproblem_to_root_vars_map
            for xvar in s._mpisppy_data.nonant_indices.values():
                ## Grab the value from the associated root variable
                nonant_send_buffer[ci] = nonant_to_root_var_map[xvar]._value
                ci += 1
        logging.debug("hub is sending X nonants={}".format(nonant_send_buffer))

        self.put_send_buffer(nonant_send_buffer, Field.NONANT)

        return


class SubgradientHub(PHHub):

    # send / receive fields are same as PHHub

    _hub_algo_best_bound_provider = True

    def main(self):
        """ SPComm gets attached in self.__init__ """
        self.opt.subgradient_main(finalize=False)


class APHHub(PHHub):

    # send / receive fields are same as PHHub

    def main(self):
        """ SPComm gets attached by self.__init___; holding APH harmless """
        logger.critical("aph debug main in hub.py")
        self.opt.APH_main(spcomm=self, finalize=False)

    # overwrite the default behavior of this method for APH
    def get_receive_buffer(self,
                           buf: RecvArray,
                           field: Field,
                           origin: int,
                           synchronize: bool = False,
                          ):
        return super().get_receive_buffer(buf, field, origin, synchronize)

    def finalize(self):
        """ does PH.post_loops, returns Eobj """
        # NOTE: APH_main does NOT pass in extensions
        #       to APH.post_loops
        Eobj = self.opt.post_loops()
        return Eobj

class FWPHHub(PHHub, FWPH_Cylinder):

    send_fields = (*PHHub.send_fields, *FWPH_Cylinder.send_fields)
    receive_fields = (*PHHub.receive_fields, *FWPH_Cylinder.receive_fields)

    _hub_algo_best_bound_provider = True

    def main(self):
        self.opt.fwph_main(finalize=False)
