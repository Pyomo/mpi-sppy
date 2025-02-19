###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import mpisppy.opt.ph
import mpisppy.MPI as _mpi

_global_rank = _mpi.COMM_WORLD.Get_rank()

class Subgradient(mpisppy.opt.ph.PH):
    """ Subgradient Algorithm """

    def subgradient_main(self, finalize=True):
        """ Execute the subgradient algorithm.

        Args:
            finalize (bool, optional, default=True):
                If True, call `PH.post_loops()`, if False, do not,
                and return None for Eobj

        Returns:
            tuple:
                Tuple containing

                conv (float):
                    The convergence value (not easily interpretable).
                Eobj (float or `None`):
                    If `finalize=True`, this is the expected, weighted
                    objective value. This value is not directly useful.
                    If `finalize=False`, this value is `None`.
                trivial_bound (float):
                    The "trivial bound", computed by solving the model with no
                    nonanticipativity constraints (immediately after iter 0).
        """
        # for use with the PH hub
        if self.options["smoothed"] != 0:
            raise RuntimeError("Cannnot use smoothing with Subgradient algorithm")
        return super().ph_main(finalize=finalize, attach_prox=False)

    def solve_loop(self,
                   solver_options=None,
                   use_scenarios_not_subproblems=False,
                   dtiming=False,
                   dis_W=False,
                   dis_prox=False,
                   gripe=False,
                   disable_pyomo_signal_handling=False,
                   tee=False,
                   verbose=False,
                   need_solution=True,
                   ):
        """ Loop over `local_subproblems` and solve them in a manner
        dicated by the arguments.

        In addition to changing the Var values in the scenarios, this function
        also updates the `_PySP_feas_indictor` to indicate which scenarios were
        feasible/infeasible.

        Args:
            solver_options (dict, optional):
                The scenario solver options.
            use_scenarios_not_subproblems (boolean, optional):
                If True, solves individual scenario problems, not subproblems.
                This distinction matters when using bundling. Default is False.
            dtiming (boolean, optional):
                If True, reports solve timing information. Default is False.
            dis_W (boolean, optional):
                If True, duals weights (Ws) are disabled before solve, then
                re-enabled after solve. Default is False.
            dis_prox (boolean, optional):
                If True, prox terms are disabled before solve, then
                re-enabled after solve. Default is False.
            gripe (boolean, optional):
                If True, output a message when a solve fails. Default is False.
            disable_pyomo_signal_handling (boolean, optional):
                True for asynchronous PH; ignored for persistent solvers.
                Default False.
            tee (boolean, optional):
                If True, displays solver output. Default False.
            verbose (boolean, optional):
                If True, displays verbose output. Default False.
            need_solution (boolean, optional):
                If True, raises an exception if a solution is not available.
                Default True
        """
        super().solve_loop(
            solver_options=solver_options,
            use_scenarios_not_subproblems=use_scenarios_not_subproblems,
            dtiming=dtiming,
            dis_W=dis_W,
            dis_prox=dis_prox,
            gripe=gripe,
            disable_pyomo_signal_handling=disable_pyomo_signal_handling,
            tee=tee,
            verbose=verbose,
            need_solution=need_solution,
        )

        # set self.best_bound_obj_val if we don't have any additional fixed variables
        if self._can_update_best_bound():
            self.best_bound_obj_val = self.Ebound(verbose)


