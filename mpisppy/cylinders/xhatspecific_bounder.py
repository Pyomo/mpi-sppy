# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# udpated April 20
# specific xhat supplied (copied from xhatlooper_bounder by DLW, Dec 2019)

import mpisppy.cylinders.spoke as spoke
from mpisppy.extensions.xhatspecific import XhatSpecific

import mpi4py.MPI as mpi
import logging

fullcomm = mpi.COMM_WORLD
rank_global = fullcomm.Get_rank()
fullcom_n_proc = fullcomm.Get_size()


############################################################################
class XhatSpecificInnerBound(spoke.InnerBoundNonantSpoke):

    converger_spoke_char = 'S'

    def ib_prep(self):
        """
        Set up the objects needed for bounding.

        Returns:
            xhatter (xhatspecific object): Constructed by a call to Prep
        """
        if "bundles_per_rank" in self.opt.options\
           and self.opt.options["bundles_per_rank"] != 0:
            raise RuntimeError("xhat spokes cannot have bundles (yet)")
        verbose = self.opt.options['verbose']
        xhatter = XhatSpecific(self.opt)
        # somehow deal with the prox option .... TBD .... important for aph APH
        self.opt.PH_Prep()  
        logging.debug("  ib back from Prep global rank {}".format(rank_global))

        self.opt.subproblem_creation(verbose)

        # begin iter0 stuff
        xhatter.pre_iter0()
        self.opt._save_original_nonants()
        self.opt._create_solvers()

        teeme = False
        if ("tee-rank0-solves" in self.opt.options):
            teeme = self.opt.options['tee-rank0-solves']

        self.opt.solve_loop(solver_options=self.opt.current_solver_options,
                            dtiming=False,
                            gripe=True,
                            tee=teeme,
                            verbose=verbose)
        self.opt._update_E1()  # Apologies for doing this after the iter0 solves...
        if (abs(1 - self.opt.E1) > self.opt.E1_tolerance):
            if opt.rank == opt.rank0:
                print("ERROR")
                print("Total probability of scenarios was ", self.opt.E1)
                print("E1_tolerance = ", self.opt.E1_tolerance)
            quit()
        infeasP = self.opt.infeas_prob()
        if infeasP != 0.:
            if self.opt.rank == self.opt.rank0:
                print("ERROR")
                print("Infeasibility detected; E_infeas, E1=", infeasP, self.opt.E1)
            quit()

        ### end iter0 stuff

        xhatter.post_iter0()
        self.opt._save_nonants()  # make the cache

        return xhatter

    def main(self):
        """
        Entry point. Communicates with the optimization companion.

        """
        dtm = logging.getLogger(f'dtm{rank_global}')
        verbose = self.opt.options["verbose"] # typing aid  
        logging.debug("Enter xhatspecific main on rank {}".format(rank_global))

        # What to try does not change, but the data in the scenarios should
        xhat_scenario_dict = self.opt.options["xhat_specific_options"]\
                                             ["xhat_scenario_dict"]

        xhatter = self.ib_prep()

        ib_iter = 1  # ib is for inner bound
        got_kill_signal = False
        while (not self.got_kill_signal()):
            logging.debug('   IB loop iter={} on global rank {}'.\
                          format(ib_iter, rank_global))
            # _log_values(ib_iter, self._locals, dtm)

            logging.debug('   IB got from opt on global rank {}'.\
                          format(rank_global))
            if (self.new_nonants):
                logging.debug('  and its new! on global rank {}'.\
                              format(rank_global))
                logging.debug('  localnonants={}'.format(str(self.localnonants)))

                self.opt._put_nonant_cache(self.localnonants)  # don't really need all caches
                self.opt._restore_nonants()
                innerbound = xhatter.xhat_tryit(xhat_scenario_dict)

                # send a bound to the opt companion
                if innerbound is not None:
                    self.bound = innerbound
                    logging.debug('send ib={}; global rank={} (specified dict)'\
                                  .format(innerbound, rank_global))
                    dtm.debug(f'Computed inner bound on rank {rank_global}: {self.bound:.4f}')
                    logging.debug('   bottom of ib loop on global rank {}'\
                                  .format(rank_global))

            ib_iter += 1

        dtm.debug(f'IB specific thread ran {ib_iter} iterations\n')
    
if __name__ == "__main__":
    print("no main.")
