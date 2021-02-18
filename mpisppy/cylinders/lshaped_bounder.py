# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import logging
import time
import random
import mpisppy.log
import mpisppy.utils.sputils as sputils
import mpisppy.cylinders.spoke as spoke

from math import inf
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from mpisppy.phbase import PHBase
from mpisppy.utils.xhat_tryer import XhatTryer

class XhatLShapedInnerBound(spoke.InnerBoundNonantSpoke):

    converger_spoke_char = 'X'

    def xhatlshaped_prep(self):
        verbose = self.opt.options['verbose']

        if not isinstance(self.opt, XhatTryer):
            raise RuntimeError("XhatLShapedInnerBound must be used with XhatTryer.")

        self.opt.PH_Prep(attach_duals=False, attach_prox=False)  

        self.opt.subproblem_creation(verbose)

        self.opt._create_solvers()

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
            if self.rank_global == self.opt.rank0:
                print("ERROR")
                print("Total probability of scenarios was ", self.opt.E1)
                print("E1_tolerance = ", self.opt.E1_tolerance)
            quit()
        infeasP = self.opt.infeas_prob()
        if infeasP != 0.:
            if self.rank_global == self.opt.rank0:
                print("ERROR")
                print("Infeasibility detected; E_infeas, E1=", infeasP, self.opt.E1)
            quit()

        self.opt._save_nonants() # make the cache

        self.opt.current_solver_options = self.opt.PHoptions["iterk_solver_options"]
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

                update = (obj < self.ib) if is_minimizing else (self.ib < obj)
                if update:
                    self.bound = obj
                    self.ib = obj

        ## TODO: Save somewhere?
