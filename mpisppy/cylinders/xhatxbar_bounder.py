###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# udpated April 20
# xbar from xhat (copied from xhat specific, DLW Feb 2023)

import pyomo.environ as pyo
from mpisppy.extensions.xhatxbar import XhatXbar
from mpisppy.cylinders.xhatbase import XhatInnerBoundBase

import mpisppy.MPI as mpi
import logging

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()
fullcom_n_proc = fullcomm.Get_size()


def _attach_xbars(opt):
    # attach xbars to an Xhat_Eval object given as opt
    for scenario in opt.local_scenarios.values():
        scenario._mpisppy_model.xbars = pyo.Param(
            scenario._mpisppy_data.nonant_indices.keys(), initialize=0.0, mutable=True
        )
        scenario._mpisppy_model.xsqbars = pyo.Param(
            scenario._mpisppy_data.nonant_indices.keys(), initialize=0.0, mutable=True
        )


############################################################################
class XhatXbarInnerBound(XhatInnerBoundBase):

    converger_spoke_char = 'B'

    def xhat_extension(self):
        return XhatXbar(self.opt)

    def xhat_prep(self):
        """
        Set up the objects needed for bounding.

        Returns:
            xhatter (xhatxbar object): Constructed by a call to Prep
        """
        xhatter = super().xhat_prep()
        _attach_xbars(self.opt)
        return xhatter

    def main(self):
        """
        Entry point. Communicates with the optimization companion.

        """
        dtm = logging.getLogger(f'dtm{global_rank}')
        logging.debug("Enter xhatxbar main on rank {}".format(global_rank))

        xhatter = self.xhat_prep()

        ib_iter = 1  # ib is for inner bound
        while (not self.got_kill_signal()):
            logging.debug('   IB loop iter={} on global rank {}'.\
                          format(ib_iter, global_rank))
            # _log_values(ib_iter, self._locals, dtm)

            logging.debug('   IB got from opt on global rank {}'.\
                          format(global_rank))
            if (self.new_nonants):
                logging.debug('  and its new! on global rank {}'.\
                              format(global_rank))
                logging.debug('  localnonants={}'.format(str(self.localnonants)))
                self.opt._put_nonant_cache(self.localnonants)  # don't really need all caches
                # just for sending the values to other scenarios
                # so we don't need to tell persistent solvers
                self.opt._restore_nonants(update_persistent=False)
                innerbound = xhatter.xhat_tryit(restore_nonants=False)

                self.update_if_improving(innerbound)

            ib_iter += 1

        dtm.debug(f'IB xbar thread ran {ib_iter} iterations\n')
    
        # for debugging
        #print("output .txt files for debugging in xhatxbar_bounder.py")
        #for k,s in self.opt.local_scenarios.items():
        #    fname = f"xhatxbar_model_{k}.txt"
        #    with open(fname, "w") as f:
        #        s.pprint(f)

if __name__ == "__main__":
    print("no main.")
