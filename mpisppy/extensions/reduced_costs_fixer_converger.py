###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import numpy as np
import pandas as pd
import os

from mpisppy import MPI
from mpisppy.extensions.phtracker import TrackedData
from mpisppy.extensions.extension import Extension
from mpisppy.convergers.converger import Converger

from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke
from mpisppy.utils.sputils import is_persistent

from mpisppy.cylinders.spwindow import Field

class ReducedCostsFixerConverger(Extension, Converger):

    def __init__(self, spobj):
        super().__init__(spobj)

        rc_options = spobj.options['rc_fixer_converger_options']

        self.verbose = spobj.options['verbose'] or rc_options['verbose']
        self.debug = rc_options['debug']

        self.prev_xbars = None
        self._rank = spobj.cylinder_rank

        # reduced costs less than this in absolute value
        # will be considered 0
        self.zero_rc_tol = rc_options['zero_rc_tol']
        self.convergence_threshold = rc_options['rc_converger_tol']

        # fixing variables based on reduces costs, outer bound, and inner bound improvement
        # self._rc_fixer_require_improving_lagrangian = rc_options['rc_fixer_require_improving_lagrangian']
        self._rc_fixer_require_improving_outer_bound = rc_options['rc_fixer_require_improving_outer_bound']
        self._rc_fixer_require_improving_inner_bound = rc_options['rc_fixer_require_improving_inner_bound']

        # Percentage of variables which are at the bound we will target
        # to fix. We never fix varibles with reduced costs less than
        # the `zero_rc_tol` in absolute value
        self._fix_fraction_target_pre_iter0 = rc_options.get('fix_fraction_target_pre_iter0', 0)
        if self._fix_fraction_target_pre_iter0 < 0 or self._fix_fraction_target_pre_iter0 > 1:
            raise ValueError("fix_fraction_target_pre_iter0 must be between 0 and 1")
        self._fix_fraction_target_iter0 = rc_options['fix_fraction_target_iter0']
        if self._fix_fraction_target_iter0 < 0 or self._fix_fraction_target_iter0 > 1:
            raise ValueError("fix_fraction_target_iter0 must be between 0 and 1")
        self._fix_fraction_target_iterK = rc_options['fix_fraction_target_iterK']
        if self._fix_fraction_target_iterK < 0 or self._fix_fraction_target_iterK > 1:
            raise ValueError("fix_fraction_target_iterK must be between 0 and 1")
        self.fix_fraction_target = self._fix_fraction_target_pre_iter0

        self.bound_tol = rc_options['rc_bound_tol']
        self._fixed_vars = 0
        if spobj.is_minimizing:
            self._best_outer_bound = -float("inf")
            self._outer_bound_update = lambda new, old : (new > old)
            self._best_inner_bound = float("inf")
            self._inner_bound_update = lambda new, old : (new < old)
        else:
            self._best_outer_bound = float("inf")
            self._outer_bound_update = lambda new, old : (new < old)
            self._best_inner_bound = -float("inf")
            self._inner_bound_update = lambda new, old : (new > old)

        self._current_reduced_costs = None
        self._is_new_outer_bound = False
        self._is_new_inner_bound = False


    def _update_best_outer_bound(self, new_outer_bound):
        if self._outer_bound_update(new_outer_bound, self._best_outer_bound):
            self._best_outer_bound = new_outer_bound
            return True
        return False
    
    def _update_best_inner_bound(self, new_inner_bound):
        if self._inner_bound_update(new_inner_bound, self._best_inner_bound):
            self._best_inner_bound = new_inner_bound
            return True
        return False

    def pre_iter0(self):
        self._modeler_fixed_nonants = set()
        self._integer_nonants = set()
        self.nonant_length = self.opt.nonant_length
        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants.add(ndn_i)
                    continue
                if xvar.is_integer():
                    self._integer_nonants.add(ndn_i)

    def iter0_post_solver_creation(self):
        self.fix_fraction_target = self._fix_fraction_target_pre_iter0
        if self.fix_fraction_target > 0:
            # wait for the reduced costs
            if self.opt.cylinder_rank == 0 and self.verbose:
                print("Fixing based on reduced costs prior to iteration 0!")
            while self._current_reduced_costs is None:
                self.sync_with_spokes()
            self.reduced_costs_fixing_step(self._current_reduced_costs, fix_variables=True, relax_variables=False, pre_iter0 = True)
        self.fix_fraction_target = self._fix_fraction_target_iter0

    def post_iter0_after_sync(self):
        self.fix_fraction_target = self._fix_fraction_target_iterK

    def register_receive_fields(self):
        spcomm = self.opt.spcomm
        expected_reduced_cost_ranks = spcomm.fields_to_ranks[Field.EXPECTED_REDUCED_COST]
        assert len(expected_reduced_cost_ranks) == 1
        index_reduced_costs_spoke = expected_reduced_cost_ranks[0]
        self.reduced_costs_spoke_index = index_reduced_costs_spoke
        self.reduced_cost_buf = spcomm.register_recv_field(
            Field.EXPECTED_REDUCED_COST,
            self.reduced_costs_spoke_index,
        )
        return
    
    def sync_with_spokes(self):
        self.opt.spcomm.get_receive_buffer(
            self.reduced_cost_buf,
            Field.EXPECTED_REDUCED_COST,
            self.reduced_costs_spoke_index,
        )

        if self.reduced_cost_buf.is_new():
            reduced_costs = self.reduced_cost_buf.value_array()
            self._current_reduced_costs = np.array(reduced_costs[:])
            if self._rank == 0 and self.verbose:
                print("Received new reduced costs I'm in the loop")   
        else:
            if self.opt.cylinder_rank == 0 and self.verbose:
                print("No new reduced costs! I'm in the loop")
            ## End if
        ## End if

        this_best_outer_bound = self.opt.spcomm.BestOuterBound
        this_best_inner_bound = self.opt.spcomm.BestInnerBound

        self._is_new_inner_bound = self._update_best_inner_bound(this_best_inner_bound)
        self._is_new_outer_bound = self._update_best_outer_bound(this_best_outer_bound)
        
        return
        
    def enditer_after_sync(self):
        
        primal_gap = self._compute_primal_convergence()
        self.prev_xbars = self._get_xbars()
        ret_val = primal_gap <= self.convergence_threshold
        if self._rank == 0 and self.verbose:
            print(f"Primal residual = {round(primal_gap, 5)}")
        ## End if
        if ret_val: # this statment should be removed if is_converged is used
            # unfix variables if primal gap is less than convergence threshold
            self.reduced_costs_fixing_step(self._current_reduced_costs, fix_variables=False, relax_variables=True, pre_iter0 = False)
        else:
            # fixing/unfixing variables based on bound improvement or if the setting doesn't require improvement in bounds
            if (self._is_new_outer_bound and self._rc_fixer_require_improving_outer_bound) or (self._is_new_inner_bound and self._rc_fixer_require_improving_inner_bound) or (not self._rc_fixer_require_improving_outer_bound and not self._rc_fixer_require_improving_inner_bound):
                self.reduced_costs_fixing_step(self._current_reduced_costs, fix_variables=True, relax_variables=True, pre_iter0 = False)
            self._is_new_outer_bound = False
            self._is_new_inner_bound = False
            ## End if
        ## End if

    def reduced_costs_fixing_step(self, reduced_costs, fix_variables=True, relax_variables=True, pre_iter0 = False):

        if np.all(np.isnan(reduced_costs)):
            # Note: If all rc = nan at some later iteration,
            # this will skip unfixing
            if self.opt.cylinder_rank == 0 and self.verbose:
                print("All reduced costs are nan, heuristic fixing will not be applied")
            return
        
        # compute the quantile target
        abs_reduced_costs = np.abs(reduced_costs)
        fix_fraction_target = self.fix_fraction_target

        # excludes nan
        nonzero_rc = abs_reduced_costs[abs_reduced_costs > self.zero_rc_tol]
        if len(nonzero_rc) == 0:
            # still need to continue, for unfixing
            target = self.zero_rc_tol
        else:
            target = np.nanquantile(nonzero_rc, 1 - fix_fraction_target, method="median_unbiased")

        if target < self.zero_rc_tol:
            # shouldn't be reached
            target = self.zero_rc_tol

        if self.opt.cylinder_rank == 0 and self.verbose:
            print(f"Heuristic fixing reduced cost cutoff: {target}")

        raw_fixed_this_iter = 0

        for sub in self.opt.local_scenarios.values():
            persistent_solver = is_persistent(sub._solver_plugin)
            for ci, (ndn_i, xvar) in enumerate(sub._mpisppy_data.nonant_indices.items()):
                if ndn_i in self._modeler_fixed_nonants:
                    continue
                if xvar in sub._mpisppy_data.all_surrogate_nonants:
                    continue
                this_expected_rc = abs_reduced_costs[ci]
                update_var = False
                if np.isnan(this_expected_rc):
                    # is nan, variable is not converged in LP-LR
                    if xvar.fixed and relax_variables:
                        xvar.unfix()
                        update_var = True
                        raw_fixed_this_iter -= 1
                        if self.debug and self.opt.cylinder_rank == 0:
                            print(f"unfixing var {xvar.name}; not converged in LP-LR")
                else: # not nan, variable is converged in LP-LR
                    if xvar.fixed and relax_variables:
                        xb = sub._mpisppy_model.xbars[ndn_i].value
                        if (this_expected_rc < target):
                            xvar.unfix()
                            update_var = True
                            raw_fixed_this_iter -= 1
                            if self.debug and self.opt.cylinder_rank == 0:
                                print(f"unfixing var {xvar.name}; reduced cost is zero/below target in LP-LR")
                        # in case somebody else unfixs a variable in another rank...
                        if abs(xb - xvar.value) > self.bound_tol:
                            xvar.unfix()
                            update_var = True
                            raw_fixed_this_iter -= 1
                            if self.debug and self.opt.cylinder_rank == 0:
                                print(f"unfixing var {xvar.name}; xbar differs from the fixed value")
                    elif not xvar.fixed and fix_variables:
                        xb = sub._mpisppy_model.xbars[ndn_i].value
                        if (this_expected_rc >= target):
                            if self.opt.is_minimizing:
                                # TODO: First check can be simplified as abs(rc) is already checked above
                                if (reduced_costs[ci] > 0 + self.zero_rc_tol) and (pre_iter0 or (xb - xvar.lb <= self.bound_tol)):
                                    xvar.fix(xvar.lb)
                                    if self.debug and self.opt.cylinder_rank == 0:
                                        print(f"fixing var {xvar.name} to lb {xvar.lb}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                                elif (reduced_costs[ci] < 0 - self.zero_rc_tol) and (pre_iter0 or (xvar.ub - xb <= self.bound_tol)):
                                    xvar.fix(xvar.ub)
                                    if self.debug and self.opt.cylinder_rank == 0:
                                        print(f"fixing var {xvar.name} to ub {xvar.ub}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                                else:
                                    # rc is near 0 or
                                    # xbar from MIP might differ from rc from relaxation
                                    pass
                            else:
                                if (reduced_costs[ci] < 0 - self.zero_rc_tol) and (xb - xvar.lb <= self.bound_tol):
                                    xvar.fix(xvar.lb)
                                    if self.debug and self.opt.cylinder_rank == 0:
                                        print(f"fixing var {xvar.name} to lb {xvar.lb}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                                elif (reduced_costs[ci] > 0 + self.zero_rc_tol) and (xvar.ub - xb <= self.bound_tol):
                                    xvar.fix(xvar.ub)
                                    if self.debug and self.opt.cylinder_rank == 0:
                                        print(f"fixing var {xvar.name} to ub {xvar.ub}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                                else:
                                    # rc is near 0 or
                                    # xbar from MIP might differ from rc from relaxation
                                    pass

                if update_var and persistent_solver:
                    sub._solver_plugin.update_var(xvar)

        # Note: might count incorrectly with bundling?
        self._fixed_vars += raw_fixed_this_iter / len(self.opt.local_scenarios)
        if self.opt.cylinder_rank == 0 and self.verbose:
            print(f"Total unique vars fixed by heuristic: {int(round(self._fixed_vars))}/{self.nonant_length}")

    def _get_xbars(self):
        """
        Get the current xbar values from the local scenarios
        Returns:
            xbars (dict): dictionary of xbar values indexed by
                          (decision node name, index)
        """
        xbars = {}
        for s in self.opt.local_scenarios.values():
            for ndn_i, xbar in s._mpisppy_model.xbars.items():
                xbars[ndn_i] = xbar.value
            break
        return xbars

    def _compute_primal_convergence(self):
        """
        Compute the primal convergence metric
        Returns:
            global_sum_diff (float): primal convergence metric
        """
        local_sum_diff = np.zeros(1)
        global_sum_diff = np.zeros(1)
        for _, s in self.opt.local_scenarios.items():
            # we iterate over decision nodes instead of
            # s._mpisppy_data.nonant_indices to use numpy
            for node in s._mpisppy_node_list:
                ndn = node.name
                nlen = s._mpisppy_data.nlens[ndn]
                x_bars = np.fromiter((s._mpisppy_model.xbars[ndn,i]._value
                                      for i in range(nlen)), dtype='d')

                nonants_array = np.fromiter(
                    (v._value for v in node.nonant_vardata_list),
                    dtype='d', count=nlen)
                _l1 = np.abs(x_bars - nonants_array)

                # invariant to prob_coeff being a scalar or array
                prob = s._mpisppy_data.prob_coeff[ndn] * np.ones(nlen)
                local_sum_diff[0] += np.dot(prob, _l1)

        self.opt.comms["ROOT"].Allreduce(local_sum_diff, global_sum_diff, op=MPI.SUM)
        return global_sum_diff[0]
    
    # TODO: This code can be used if the extension can terminate the algorithm. is_converged function checks the convergence criterion and trigger termination. If is_converged used, remove the if ret_val check in enditer_after_sync and use the else statment
    # def is_converged(self):

    #     if not hasattr(self, "reduced_cost_buf"):
    #         if self._rank == 0 and self.verbose:
    #             print("Adding receive reduced cost buffer object for convergence check...")
    #         self.register_receive_fields()
    #         self.pre_iter0()
    #     terminate_check = False

    #     primal_gap = self._compute_primal_convergence()
    #     self.prev_xbars = self._get_xbars()
    #     ret_val = primal_gap <= self.convergence_threshold

    #     self.sync_with_spokes(pre_iter0 = False, converger_sync = False)
    #     if self.verbose and self._rank == 0:
    #         print(f"current reduced costs: {self._current_reduced_costs}")
    #         print(f"converger object name is {self.__class__.__name__} with id {id(self)}")
    #     if self.verbose and self._rank == 0:
    #         print(f"primal gap = {round(primal_gap, 5)}")

    #         if ret_val:
    #             print("Primal convergence check passed")
    #         else:
    #             print("Primal convergence check failed "
    #                   f"(requires primal gap) <= {self.convergence_threshold}")

    #     if self.tracking and self._rank == 0:
    #         self.tracker.add_row([self.opt._PHIter, primal_gap])
    #         self.tracker.write_out_data()

    #     if ret_val:
    #         # if not hasattr(self, "reduced_cost_buf"):
    #         #     if self._rank == 0 and self.verbose:
    #         #         print("Adding receive reduced cost buffer object for convergence check...")
    #         #     self.pre_iter0()
    #         #     self.register_receive_fields()
                

    #         # self.sync_with_spokes(pre_iter0 = False, converger_sync = True)
    #         previous_fixed_vars = 0
    #         print(f"Checking reduced costs for unfixing variables...")
    #         print(f"Value of reduced cost type of {type(self._current_reduced_costs)} is {self._current_reduced_costs}")
    #         self.reduced_costs_fixing(self._current_reduced_costs, fix_variables=False, relax_variables=True)
    #         current_fixed_vars = self._fixed_vars
    #         print(f"Number of variables un-fixed by reduced cost fixing: {int(round(previous_fixed_vars - current_fixed_vars))}")
    #         if self.verbose and self._rank == 0:
    #             print(f"Number of variables un-fixed by reduced cost fixing: {int(round(previous_fixed_vars - current_fixed_vars))}")
    #             # print(f"Total unique vars fixed by heuristic: {int(round(self._fixed_vars))}/{self.nonant_length}")   

    #             if current_fixed_vars < previous_fixed_vars:
    #                 print("Unfixing variables based on reduced costs, continuing iterations")
    #             else:
    #                 print("No variables were unfixed based on reduced costs, convergence check is valid, terminating")
    #                 terminate_check = True
    #     else:            
    #         if self.verbose and self._rank == 0:
    #             print("Not checking reduced costs for unfixing because primal convergence check failed")    
        
    #     return terminate_check