###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import mpisppy.cylinders.spoke as spoke

from math import inf
from mpisppy.utils.xhat_eval import Xhat_Eval

class XhatLShapedInnerBound(spoke.InnerBoundNonantSpoke):

    converger_spoke_char = 'X'

    def xhatlshaped_prep(self):
        verbose = self.opt.options['verbose']

        if not isinstance(self.opt, Xhat_Eval):
            raise RuntimeError("XhatLShapedInnerBound must be used with Xhat_Eval.")

        teeme = False
        if "tee-rank0-solves" in self.opt.options:
            teeme = self.opt.options['tee-rank0-solves']

        self.opt.solve_loop(
            solver_options=self.opt.current_solver_options,
            dtiming=False,
            gripe=True,
            tee=teeme,
            verbose=verbose
        )
        self.opt._update_E1()  # Apologies for doing this after the solves...
        if abs(1 - self.opt.E1) > self.opt.E1_tolerance:
            if self.opt.cylinder_rank == 0:
                print("ERROR")
                print("Total probability of scenarios was ", self.opt.E1)
                print("E1_tolerance = ", self.opt.E1_tolerance)
            quit()
        infeasP = self.opt.infeas_prob()
        if infeasP != 0.:
            if self.opt.cylinder_rank == 0:
                print("ERROR")
                print("Infeasibility detected; E_infeas, E1=", infeasP, self.opt.E1)
            quit()

        self.opt._save_nonants() # make the cache

        self.opt.current_solver_options = self.opt.options["iterk_solver_options"]
        ### end iter0 stuff

    def main(self):

        self.xhatlshaped_prep()
        is_minimizing = self.opt.is_minimizing

        self.ib = inf if is_minimizing else -inf

        #xh_iter = 1
        while not self.got_kill_signal():

            if self.new_nonants:
                
                self.opt._put_nonant_cache(self.localnonants)
                self.opt._restore_nonants()
                obj = self.opt.calculate_incumbent(fix_nonants=True)

                if obj is None:
                    continue

                self.update_if_improving(obj)
