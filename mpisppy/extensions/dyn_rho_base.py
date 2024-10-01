###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# A dynamic rho base class that assumes a member function compute_and_update_rho
# As of Oct 2024, it only mainly provides services and needs a better mid_iter
# Children should call the parent for post_iter0


import os
import numpy as np

import mpisppy.extensions.extension
from mpisppy.utils.wtracker import WTracker
from mpisppy import global_toc

# for trapping numpy warnings
import warnings

class Dyn_Rho_extension_base(mpisppy.extensions.extension.Extension):
    """
    This extension enables dynamic rho, e.g. gradient or sensitivity based.
    
    Args:
       opt (PHBase object): gives the problem
       cfg (Config object): config object
    
    """
    def __init__(self, opt, comm=None):
        super().__init__(opt)
        self.cylinder_rank = self.opt.cylinder_rank

        self.primal_conv_cache = []
        self.dual_conv_cache = []
        self.wt = WTracker(self.opt)

    def _display_rho_values(self):
        for sname, scenario in self.opt.local_scenarios.items():
            rho_list = [scenario._mpisppy_model.rho[ndn_i]._value
                      for ndn_i, _ in scenario._mpisppy_data.nonant_indices.items()]
            print(sname, 'rho values: ', rho_list[:5])
            break

    def _display_W_values(self):
        for (sname, scenario) in self.opt.local_scenarios.items():
            W_list = [w._value for w in scenario._mpisppy_model.W.values()]
            print(sname, 'W values: ', W_list)
            break


    def _update_rho_primal_based(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            curr_conv, last_conv = self.primal_conv_cache[-1], self.primal_conv_cache[-2]
            try:
                primal_diff =  np.abs((last_conv - curr_conv) / last_conv)
            except Warning:
                if self.cylinder_rank == 0:
                    print(f"dyn_rho extension reports {last_conv=} {curr_conv=} - no rho updates recommended")
                return False
            return (primal_diff <= self.cfg.dynamic_rho_primal_thresh)

    def _update_rho_dual_based(self):
        curr_conv, last_conv = self.dual_conv_cache[-1], self.dual_conv_cache[-2]
        dual_diff =  np.abs((last_conv - curr_conv) / last_conv) if last_conv != 0 else 0
        #print(f'{dual_diff =}')
        return (dual_diff <= self.cfg.dynamic_rho_dual_thresh)

    def _update_recommended(self):
        return (self.cfg.dynamic_rho_primal_crit and self._update_rho_primal_based()) or \
               (self.cfg.dynamic_rho_dual_crit and self._update_rho_dual_based())

    def pre_iter0(self):
        pass

    def post_iter0(self):
        self.primal_conv_cache.append(self.opt.convergence_diff())
        self.dual_conv_cache.append(self.wt.W_diff())

    def miditer(self):
        self.primal_conv_cache.append(self.opt.convergence_diff())
        self.dual_conv_cache.append(self.wt.W_diff())
        if self.opt._PHIter == 1:
            self.grad_object.write_grad_cost()
        if self.opt._PHIter == 1 or self._update_recommended():
            self.grad_object.write_grad_rho()
            rho_setter_kwargs = self.opt.options['rho_setter_kwargs'] \
                                if 'rho_setter_kwargs' in self.opt.options \
                                   else dict()

            sum_rho = 0.0
            num_rhos = 0
            for sname, scenario in self.opt.local_scenarios.items():
                rholist = self.rho_setter(scenario, **rho_setter_kwargs)
                for (vid, rho) in rholist:
                    (ndn, i) = scenario._mpisppy_data.varid_to_nonant_index[vid]
                    scenario._mpisppy_model.rho[(ndn, i)] = rho
                    sum_rho += rho
                    num_rhos += 1

            rho_avg = sum_rho / num_rhos
            
            global_toc(f"Rho values recomputed - average rank 0 rho={rho_avg}")

    def enditer(self):
        pass

    def post_everything(self):
        pass
