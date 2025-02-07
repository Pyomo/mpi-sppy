###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import numpy as np


from mpisppy.extensions.extension import Extension
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke 
from mpisppy.utils.sputils import is_persistent

class ReducedCostsFixer(Extension):

    def __init__(self, spobj):
        super().__init__(spobj)

        ph_options = spobj.options
        rc_options = ph_options['rc_options']
        self.verbose = ph_options['verbose'] or rc_options['verbose']
        self.debug = rc_options['debug']

        self._use_rc_bt = rc_options['use_rc_bt']
        # reduced costs less than this in absolute value
        # will be considered 0
        self.zero_rc_tol = rc_options['zero_rc_tol']
        self._use_rc_fixer = rc_options['use_rc_fixer']
        self._rc_fixer_require_improving_lagrangian = rc_options.get('rc_fixer_require_improving_lagrangian', True)
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

        # TODO: This should be same as in rc spoke?
        self.bound_tol = rc_options['rc_bound_tol']

        if not (self._use_rc_bt or self._use_rc_fixer) and \
            self.opt.cylinder_rank == 0:
            print("Warning: ReducedCostsFixer will be idle. Enable use_rc_bt or use_rc_fixer in options.")

        self._last_serial_number = -1
        self._heuristic_fixed_vars = 0
        if spobj.is_minimizing:
            self._best_outer_bound = -float("inf")
            self._outer_bound_update = lambda new, old : (new > old)
        else:
            self._best_outer_bound = float("inf")
            self._outer_bound_update = lambda new, old : (new < old)

    def _get_serial_number(self):
        return int(round(self.opt.spcomm.outerbound_receive_buffers[self.reduced_costs_spoke_index][-1]))

    def _update_best_outer_bound(self, new_outer_bound):
        if self._outer_bound_update(new_outer_bound, self._best_outer_bound):
            self._best_outer_bound = new_outer_bound
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
        if self._use_rc_fixer and self.fix_fraction_target > 0:
            # wait for the reduced costs
            if self.opt.cylinder_rank == 0 and self.verbose:
                print("Fixing based on reduced costs prior to iteration 0!")
            if self._get_serial_number() == 0:
                while not self.opt.spcomm.hub_from_spoke(self.opt.spcomm.outerbound_receive_buffers[self.reduced_costs_spoke_index], self.reduced_costs_spoke_index):
                    continue
            self.sync_with_spokes(pre_iter0 = True)
            self.opt.spcomm.use_trivial_bound = False
        self.fix_fraction_target = self._fix_fraction_target_iter0

    def post_iter0_after_sync(self):
        self.fix_fraction_target = self._fix_fraction_target_iterK

    def initialize_spoke_indices(self):
        for (i, spoke) in enumerate(self.opt.spcomm.spokes):
            if spoke["spoke_class"] == ReducedCostsSpoke:
                self.reduced_costs_spoke_index = i + 1

    def sync_with_spokes(self, pre_iter0 = False):
        serial_number = self._get_serial_number()
        if serial_number > self._last_serial_number:
            spcomm = self.opt.spcomm
            idx = self.reduced_costs_spoke_index
            self._last_serial_number = serial_number
            reduced_costs = spcomm.outerbound_receive_buffers[idx][1:1+self.nonant_length]
            this_outer_bound = spcomm.outerbound_receive_buffers[idx][0]
            new_outer_bound = self._update_best_outer_bound(this_outer_bound)
            if not pre_iter0 and self._use_rc_bt:
                self.reduced_costs_bounds_tightening(reduced_costs, this_outer_bound)
            if self._use_rc_fixer and self.fix_fraction_target > 0.0:
                if new_outer_bound or not self._rc_fixer_require_improving_lagrangian:
                    self.reduced_costs_fixing(reduced_costs)
        else:
            if self.opt.cylinder_rank == 0 and self.verbose:
                print("No new reduced costs!")


    def reduced_costs_bounds_tightening(self, reduced_costs, this_outer_bound):

        bounds_reduced_this_iter = 0
        inner_bound = self.opt.spcomm.BestInnerBound
        outer_bound = this_outer_bound
        is_minimizing = self.opt.is_minimizing
        if np.isinf(inner_bound) or np.isinf(outer_bound):
            if self.opt.cylinder_rank == 0 and self.verbose:
                print("Bounds tightened by reduced cost: 0 (inner or outer bound not available)")
            return

        for sub in self.opt.local_subproblems.values():
            persistent_solver = is_persistent(sub._solver_plugin)
            for sn in sub.scen_list:
                tightened_this_scenario = 0
                s = self.opt.local_scenarios[sn]
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    if ndn_i in self._modeler_fixed_nonants:
                        continue
                    this_expected_rc = reduced_costs[ci]
                    update_var = False
                    if np.isnan(this_expected_rc) or np.isinf(this_expected_rc):
                        continue

                    if np.isclose(xvar.lb, xvar.ub):
                        continue

                    # TODO: could simplify if/else blocks using sign variable - might reduce readability?
                    # alternatively, could move some blocks into functions
                    if is_minimizing:
                        # var at lb
                        if this_expected_rc > 0 + self.zero_rc_tol:
                            new_ub = xvar.lb + (inner_bound - outer_bound)/ this_expected_rc
                            old_ub = xvar.ub
                            if new_ub < old_ub:
                                if ndn_i in self._integer_nonants:
                                    new_ub = np.floor(new_ub)
                                    xvar.setub(new_ub)
                                else:
                                    xvar.setub(new_ub)
                                if self.debug and self.opt.cylinder_rank == 0:
                                    print(f"tightening ub of var {xvar.name} to {new_ub} from {old_ub}; reduced cost is {this_expected_rc}")
                                update_var = True
                                bounds_reduced_this_iter += 1
                                tightened_this_scenario += 1
                        # var at ub
                        elif this_expected_rc < 0 - self.zero_rc_tol:
                            new_lb = xvar.ub + (inner_bound - outer_bound)/ this_expected_rc
                            old_lb = xvar.lb
                            if new_lb > old_lb:
                                if ndn_i in self._integer_nonants:
                                    new_lb = np.ceil(new_lb)
                                    xvar.setlb(new_lb)
                                else:
                                    xvar.setlb(new_lb)
                                if self.debug and self.opt.cylinder_rank == 0:
                                    print(f"tightening lb of var {xvar.name} to {new_lb} from {old_lb}; reduced cost is {this_expected_rc}")
                                update_var = True
                                bounds_reduced_this_iter += 1
                                tightened_this_scenario += 1
                    # maximization
                    else:
                        # var at lb
                        if this_expected_rc < 0 - self.zero_rc_tol:
                            new_ub = xvar.lb - (outer_bound - inner_bound)/ this_expected_rc
                            old_ub = xvar.ub
                            if new_ub < old_ub:
                                if ndn_i in self._integer_nonants:
                                    new_ub = np.floor(new_ub)
                                    xvar.setub(new_ub)
                                else:
                                    xvar.setub(new_ub)
                                if self.debug and self.opt.cylinder_rank == 0:
                                    print(f"tightening ub of var {xvar.name} to {new_ub} from {old_ub}; reduced cost is {this_expected_rc}")
                                update_var = True
                                bounds_reduced_this_iter += 1
                        # var at ub
                        elif this_expected_rc > 0 + self.zero_rc_tol:
                            new_lb = xvar.ub - (outer_bound - inner_bound)/ this_expected_rc
                            old_lb = xvar.lb
                            if new_lb > old_lb:
                                if ndn_i in self._integer_nonants:
                                    new_lb = np.ceil(new_lb)
                                    xvar.setlb(new_lb)
                                else:
                                    xvar.setlb(new_lb)
                                if self.debug and self.opt.cylinder_rank == 0:
                                    print(f"tightening lb of var {xvar.name} to {new_lb} from {old_lb}; reduced cost is {this_expected_rc}")
                                update_var = True
                                bounds_reduced_this_iter += 1

                    if update_var and persistent_solver:
                        sub._solver_plugin.update_var(xvar)

        total_bounds_tightened = bounds_reduced_this_iter / len(self.opt.local_scenarios)
        if self.opt.cylinder_rank == 0 and self.verbose:
            print(f"Bounds tightened by reduced cost: {int(round(total_bounds_tightened))}/{self.nonant_length}")


    def reduced_costs_fixing(self, reduced_costs, pre_iter0 = False):

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
        
        for sub in self.opt.local_subproblems.values():
            persistent_solver = is_persistent(sub._solver_plugin)
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    if ndn_i in self._modeler_fixed_nonants:
                        continue
                    if xvar in s._mpisppy_data.all_surrogate_nonants:
                        continue
                    this_expected_rc = abs_reduced_costs[ci]
                    update_var = False
                    if np.isnan(this_expected_rc):
                        # is nan, variable is not converged in LP-LR
                        if xvar.fixed:
                            xvar.unfix()
                            update_var = True
                            raw_fixed_this_iter -= 1
                            if self.debug and self.opt.cylinder_rank == 0:
                                print(f"unfixing var {xvar.name}; not converged in LP-LR")
                    else: # not nan, variable is converged in LP-LR
                        if xvar.fixed:
                            if (this_expected_rc <= target):
                                xvar.unfix()
                                update_var = True
                                raw_fixed_this_iter -= 1
                                if self.debug and self.opt.cylinder_rank == 0:
                                    print(f"unfixing var {xvar.name}; reduced cost is zero/below target in LP-LR")
                        else:
                            xb = s._mpisppy_model.xbars[ndn_i].value
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
        self._heuristic_fixed_vars += raw_fixed_this_iter / len(self.opt.local_scenarios)
        if self.opt.cylinder_rank == 0 and self.verbose:
            print(f"Total unique vars fixed by heuristic: {int(round(self._heuristic_fixed_vars))}/{self.nonant_length}")
