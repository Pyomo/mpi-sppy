# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import time
import mpisppy.utils.sputils as sputils
import mpisppy.log
import logging

from mpisppy.phbase import PHBase
from mpisppy.extensions.xhatbase import XhatBase
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.utils.xhat_tryer",
                         "xhattryer.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.utils.xhat_tryer")

# This custom PH class,
# which overwrites a lot of methods for
# and provides a few others for
# just computing incumbent solutions

class XhatTryer(PHBase):

    def attach_Ws_and_prox(self):
        pass

    def attach_PH_to_objective(self, add_duals=False , add_prox=False):
        if add_duals:
            raise RuntimeError("XhatTryer has no notion of duals")
        if add_prox:
            raise RuntimeError("XhatTryer has no notion of prox")
        pass

    def solve_loop(self, solver_options=None,
                   use_scenarios_not_subproblems=False,
                   dtiming=False,
                   dis_W=True,
                   dis_prox=True,
                   gripe=False,
                   disable_pyomo_signal_handling=False,
                   tee=False,
                   verbose=False):
        """ Loop over self.local_subproblems and solve them in a manner 
            dicated by the arguments. In addition to changing the Var
            values in the scenarios, update _PySP_feas_indictor for each.

        ASSUMES:
            Every scenario already has a _solver_plugin attached.

        Args:
            solver_options (dict or None): the scenario solver options
            use_scenarios_not_subproblems (boolean): for use by bounds
            dtiming (boolean): indicates that timing should be reported
            dis_W (boolean): indicates that W should be disabled (and re-enabled)
            dis_prox (boolean): disable (and re-enable) prox term
            gripe (boolean): output a message if a solve fails
            disable_pyomo_signal_handling (boolean): set to true for asynch, 
                                                     ignored for persistent solvers.
            tee (boolean): show solver output to screen if possible
            verbose (boolean): indicates verbose output

        NOTE: I am not sure what happens with solver_options None for
              a persistent solver. Do options persist?

        NOTE: set_objective takes care of W and prox changes.
        """
        if not dis_W:
            raise RuntimeError("XhatTryer has no notion of W")
        if not dis_prox:
            raise RuntimeError("XhatTryer has no notion of prox")
        def _vb(msg): 
            if verbose and self.rank == self.rank0:
                print ("(rank0) " + msg)
        logger.debug("  early solve_loop for rank={}".format(self.rank))

        # note that when there is no bundling, scenarios are subproblems
        if use_scenarios_not_subproblems:
            s_source = self.local_scenarios
        else:
            s_source = self.local_subproblems
        for k,s in s_source.items():
            logger.debug("  in loop solve_loop k={}, rank={}".format(k, self.rank))

            pyomo_solve_time = self.solve_one(solver_options, k, s,
                                              dtiming=dtiming,
                                              verbose=verbose,
                                              tee=tee,
                                              gripe=gripe,
                disable_pyomo_signal_handling=disable_pyomo_signal_handling
            )

        if dtiming:
            all_pyomo_solve_times = self.mpicomm.gather(pyomo_solve_time, root=0)
            if self.rank == self.rank0:
                print("Pyomo solve times (seconds):")
                print("\tmin=%4.2f mean=%4.2f max=%4.2f" %
                      (np.min(all_pyomo_solve_times),
                      np.mean(all_pyomo_solve_times),
                      np.max(all_pyomo_solve_times)))

    def Update_W(self, verbose):
        pass

    def _disable_prox(self):
        pass

    def _disable_W_and_prox(self):
        pass

    def _disable_W(self):
        pass

    def _reenable_prox(self):
        pass

    def _reenable_W_and_prox(self):
        pass

    def _reenable_W(self):
        pass

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

    def calculate_incumbent(self, fix_nonants=True, verbose=False):
        """
        Calculates the current incumbent

        Args:
            solver_options (dict): passed through to the solver
            verbose (boolean): controls debugging output
        Returns:
            xhatobjective (float or None): the objective function
                or None if one could not be obtained.
        """

        if fix_nonants:
            self._fix_nonants_at_value()

        self.solve_loop(solver_options=self.current_solver_options, 
                        verbose=verbose)

        infeasP = self.infeas_prob()
        if infeasP != 0.:
            return None
        else:
            if verbose and self.rank == self.rank0:
                print("  Feasible xhat found")
            return self.Eobjective(verbose=verbose)

