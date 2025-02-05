###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import mpisppy.phbase
import mpisppy.MPI as mpi

from pyomo.common.collections import ComponentSet

_global_rank = mpi.COMM_WORLD.Get_rank()

class Subgradient(mpisppy.phbase.PHBase):
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
        verbose = self.options['verbose']
        smoothed = self.options['smoothed']
        if smoothed != 0:
            raise RuntimeError("Cannnot use smoothing with Subgradient algorithm")
        self.create_fixed_nonant_cache()
        self.PH_Prep(attach_prox=False, attach_smooth=smoothed)

        if (verbose):
            print('Calling Subgradient Iter0 on global rank {}'.format(_global_rank))
        trivial_bound = self.Iter0()
        self.best_bound_obj_val = trivial_bound
        if (verbose):
            print('Completed Subgradient Iter0 on global rank {}'.format(_global_rank))

        self.iterk_loop()

        if finalize:
            Eobj = self.post_loops(self.extensions)
        else:
            Eobj = None

        return self.conv, Eobj, trivial_bound

    def ph_main(self, finalize=True):
        # for working with a PHHub
        return self.subgradient_main(finalize=finalize)

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
        if self.can_update_best_bound():
            self.best_bound_obj_val = self.Ebound(verbose)


    def create_fixed_nonant_cache(self):
        self._initial_fixed_varibles = ComponentSet()
        for s in self.local_scenarios.values():
            for v in s._mpisppy_data.nonant_indices.values():
                if v.fixed:
                    self._initial_fixed_varibles.add(v)

    def can_update_best_bound(self):
        for s in self.local_scenarios.values():
            for v in s._mpisppy_data.nonant_indices.values():
                if v.fixed:
                    if v not in self._initial_fixed_varibles:
                        return False
        return True
