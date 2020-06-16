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
from mpisppy.cylinders.xhatshufflelooper_bounder import XhatTryer

# custom PH-subclass for LShaped incuments
class LShapedXhatTryer(XhatTryer):
    """
    Basic incumbent solution tryer for LShaped
    """

    def _fix_nonants_at_value(self):
        """ Fix the Vars subject to non-anticipativity at their current values.
            Loop over the scenarios to restore, but loop over subproblems
            to alert persistent solvers.
        """
        for k,s in self.local_scenarios.items():

            persistent_solver = None
            if not self.bundling:
                if (sputils.is_persistent(s._solver_plugin)):
                    persistent_solver = s._solver_plugin

            for var in s._nonant_indexes.values():
                var.fix()
                if not self.bundling and persistent_solver is not None:
                    persistent_solver.update_var(var)

        if self.bundling:  # we might need to update persistent solvers
            rank_local = self.rank
            for k,s in self.local_subproblems.items():
                if (sputils.is_persistent(s._solver_plugin)):
                    persistent_solver = s._solver_plugin
                else:
                    break  # all solvers should be the same

                # the bundle number is the last number in the name
                bunnum = sputils.extract_num(k)
                # for the scenarios in this bundle, update Vars
                for sname, scen in self.local_scenarios.items():
                    if sname not in self.names_in_bundles[rank_local][bunnum]:
                        break
                    for var in scen._nonant_indexes.values():
                        persistent_solver.update_var(var)

    def calculate_incumbent(self, verbose=False):
        """
        Calculates the current incumbent

        Args:
            solver_options (dict): passed through to the solver
            verbose (boolean): controls debugging output
        Returns:
            xhatobjective (float or None): the objective function
                or None if one could not be obtained.
        """

        self._fix_nonants_at_value()

        self.solve_loop(solver_options=self.current_solver_options, 
                        verbose=verbose)

        feasP = self.feas_prob()
        if feasP != self.E1:
            return None
        else:
            if verbose and self.rank == self.rank0:
                print("  Feasible xhat found")
            return self.Eobjective(verbose=verbose)



class XhatLShapedInnerBound(spoke.InnerBoundNonantSpoke):

    def xhatlshaped_prep(self):
        verbose = self.opt.options['verbose']

        if not isinstance(self.opt, LShapedXhatTryer):
            raise RuntimeError("XhatLShapedInnerBound must be used with LShapedXhatTryer.")

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
        feasP = self.opt.feas_prob()
        if feasP != self.opt.E1:
            if self.rank_global == self.opt.rank0:
                print("ERROR")
                print("Infeasibility detected; E_feas, E1=", feasP, self.opt.E1)
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
                obj = self.opt.calculate_incumbent()

                if obj is None:
                    continue

                update = (obj < self.ib) if is_minimizing else (self.ib < obj)
                if update:
                    self.bound = obj
                    self.ib = obj

        ## TODO: Save somewhere?
