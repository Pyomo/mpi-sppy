###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# updated April 2020
from mpisppy.extensions.xhatlooper import XhatLooper
from mpisppy.cylinders.xhatbase import XhatInnerBoundBase
import logging
import mpisppy.log

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.xhatlooper_bounder",
                         "xhatlp.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.cylinders.xhatlooper_bounder")


class XhatLooperInnerBound(XhatInnerBoundBase):

    converger_spoke_char = 'X'

    def xhat_extension(self):
        return XhatLooper(self.opt)

    def main(self):
        logger.debug(f"Entering main on xhatlooper spoke rank {self.global_rank}")

        xhatter = self.xhat_prep()

        scen_limit = self.opt.options['xhat_looper_options']['scen_limit']

        xh_iter = 1
        while not self.got_kill_signal():
            if (xh_iter-1) % 10000 == 0:
                logger.debug(f'   Xhatlooper loop iter={xh_iter} on rank {self.global_rank}')
                logger.debug(f'   Xhatlooper got from opt on rank {self.global_rank}')

            if self.new_nonants:
                logger.debug(f'   *Xhatlooper loop iter={xh_iter}')
                logger.debug(f'   *got a new one! on rank {self.global_rank}')
                logger.debug(f'   *localnonants={str(self.localnonants)}')

                self.opt._put_nonant_cache(self.localnonants)
                # just for sending the values to other scenarios
                # so we don't need to tell persistent solvers
                self.opt._restore_nonants(update_persistent=False)
                upperbound, srcsname = xhatter.xhat_looper(scen_limit=scen_limit, restore_nonants=True)

                # send a bound to the opt companion
                # XhatBase._try_one updates the solution cache on the opt object for us
                self.update_if_improving(upperbound, update_best_solution_cache=False)
            xh_iter += 1
