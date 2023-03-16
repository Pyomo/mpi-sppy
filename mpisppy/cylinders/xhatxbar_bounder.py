# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# udpated April 20
# xbar from xhat (copied from xhat specific, DLW Feb 2023)

import pyomo.environ as pyo
import mpisppy.cylinders.spoke as spoke
from mpisppy.extensions.xhatxbar import XhatXbar
from mpisppy.utils.xhat_eval import Xhat_Eval

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
class XhatXbarInnerBound(spoke.InnerBoundNonantSpoke):

    converger_spoke_char = 'B'

    def ib_prep(self):
        """
        Set up the objects needed for bounding.

        Returns:
            xhatter (xhatxbar object): Constructed by a call to Prep
        """
        if "bundles_per_rank" in self.opt.options\
           and self.opt.options["bundles_per_rank"] != 0:
            raise RuntimeError("xhat spokes cannot have bundles (yet)")

        if not isinstance(self.opt, Xhat_Eval):
            raise RuntimeError("XhatXbarInnerBound must be used with Xhat_Eval.")

        verbose = self.opt.options['verbose']
        xhatter = XhatXbar(self.opt)
        # somehow deal with the prox option .... TBD .... important for aph APH

        # begin iter0 stuff
        xhatter.pre_iter0()
        self.opt._save_original_nonants()

        self.opt._lazy_create_solvers()  # no iter0 loop, but we need the solvers

        self.opt._update_E1()  
        if (abs(1 - self.opt.E1) > self.opt.E1_tolerance):
            if self.opt.cylinder_rank == 0:
                print("ERROR")
                print("Total probability of scenarios was ", self.opt.E1)
                print("E1_tolerance = ", self.opt.E1_tolerance)
            quit()

        ### end iter0 stuff

        xhatter.post_iter0()
        print("about to attach xbars")
        _attach_xbars(self.opt)
        self.opt._save_nonants()  # make the cache

        return xhatter

    def main(self):
        """
        Entry point. Communicates with the optimization companion.

        """
        dtm = logging.getLogger(f'dtm{global_rank}')
        verbose = self.opt.options["verbose"] # typing aid  
        logging.debug("Enter xhatxbar main on rank {}".format(global_rank))

        xhatter = self.ib_prep()

        ib_iter = 1  # ib is for inner bound
        got_kill_signal = False
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
                self.opt._restore_nonants()
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
