###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# udpated April 20
# specific xhat supplied (copied from xhatlooper_bounder by DLW, Dec 2019)

from mpisppy.extensions.xhatspecific import XhatSpecific
from mpisppy.cylinders.xhatbase import XhatInnerBoundBase
from mpisppy.cylinders._jensens_mixin import _JensensMixin

import mpisppy.MPI as mpi
import logging

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()
fullcom_n_proc = fullcomm.Get_size()


############################################################################
class XhatSpecificInnerBound(_JensensMixin, XhatInnerBoundBase):

    converger_spoke_char = 'S'

    def xhat_extension(self):
        return XhatSpecific(self.opt)

    def main(self):
        """
        Entry point. Communicates with the optimization companion.

        """
        dtm = logging.getLogger(f'dtm{global_rank}')
        logging.debug("Enter xhatspecific main on rank {}".format(global_rank))

        # What to try does not change, but the data in the scenarios should
        xhat_scenario_dict = self.opt.options["xhat_specific_options"]\
                                             ["xhat_scenario_dict"]

        xhatter = self.xhat_prep()

        if self._jensens_enabled():
            ev_model = self._jensens_build_ev()
            _, nonant_values = self._jensens_solve(ev_model)
            cache = self._jensens_pack_nonant_cache(nonant_values)
            Eobj = self.opt.evaluate(cache)
            self.update_if_improving(Eobj)

        ib_iter = 1  # ib is for inner bound
        while (not self.got_kill_signal()):
            logging.debug('   IB loop iter={} on global rank {}'.\
                          format(ib_iter, global_rank))
            # _log_values(ib_iter, self._locals, dtm)

            logging.debug('   IB got from opt on global rank {}'.\
                          format(global_rank))
            if self.update_nonants():
                logging.debug('  and its new! on global rank {}'.\
                              format(global_rank))
                logging.debug('  localnonants={}'.format(str(self.localnonants)))

                self.opt._put_nonant_cache(self.localnonants)  # don't really need all caches
                # just for sending the values to other scenarios
                # so we don't need to tell persistent solvers
                self.opt._restore_nonants(update_persistent=False)

                innerbound = xhatter.xhat_tryit(xhat_scenario_dict, restore_nonants=False)

                self.update_if_improving(innerbound)

            ib_iter += 1

        dtm.debug(f'IB specific thread ran {ib_iter} iterations\n')
    
if __name__ == "__main__":
    print("no main.")
