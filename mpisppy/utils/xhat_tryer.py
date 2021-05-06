# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import time
import mpisppy.utils.sputils as sputils
import mpisppy.log
import logging

from mpisppy.spbase import SPBase
from mpisppy.extensions.xhatbase import XhatBase
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.utils.xhat_tryer",
                         "xhattryer.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.utils.xhat_tryer")


class XhatTryer(SPBase):

    # for playing nice with PH Extensions
    def _disable_W_and_prox(self):
        pass

    def _reenable_W_and_prox(self):
        pass

    def _fix_nonants_at_value(self):
        """ Fix the Vars subject to non-anticipativity at their current values.
            Loop over the scenarios to restore, but loop over subproblems
            to alert persistent solvers.
        """
        for k,s in self.local_scenarios.items():

            persistent_solver = s._solver_plugin \
                    if (sputils.is_persistent(s._solver_plugin)) else \
                    None

            for var in s._mpisppy_data.nonant_indices.values():
                var.fix()
                if persistent_solver is not None:
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
            if verbose and self.cylinder_rank == 0:
                print("  Feasible xhat found")
            return self.Eobjective(verbose=verbose)

