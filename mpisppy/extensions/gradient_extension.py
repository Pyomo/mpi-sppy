# Copyright 2023 by U. Naepels and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

import os
import time
import numpy as np
import pyomo.environ as pyo

import mpisppy.MPI as MPI
import mpisppy.extensions.extension
import mpisppy.utils.gradient as grad
import mpisppy.utils.find_rho as find_rho
import mpisppy.utils.sputils as sputils
import mpisppy.convergers.norms_and_residuals as norms
from mpisppy.utils.wtracker import WTracker
from mpisppy import global_toc


class Gradient_rho_extension(mpisppy.extensions.extension.Extension):
    """
    This extension makes PH use gradient-rho and the corresponding rho setter.
    
    Args:
       opt (PHBase object): gives the problem
       cfg (Config object): config object
    
    Attributes:
       grad_object (Find_Grad object): gradient object
    
    """
    def __init__(self, opt, comm=None):
        super().__init__(opt)
        self.cylinder_rank = self.opt.cylinder_rank
        self.cfg = opt.options["gradient_rho_extension_options"]["cfg"]
        self.cfg_args_cache = {'grad_cost_file': self.cfg.grad_cost_file,
                               'grad_rho_file': self.cfg.grad_rho_file,
                               'grad_rho_path': self.cfg.grad_rho_path,
                               'grad_rho_setter': self.cfg.grad_rho_setter}
        self.cfg.grad_cost_file = './_temp_grad_cost_file.csv'
        self.cfg.grad_rho_file = './_temp_grad_rho_file.csv'
        self.cfg.grad_rho_path = './_temp_grad_rho_file.csv'
        self.grad_object = grad.Find_Grad(opt, self.cfg)
        self.rho_setter = find_rho.Set_Rho(self.cfg).rho_setter
        self.prev_primal_norm, self.curr_primal_norm = 0, 0
        self.wt = WTracker(self.opt)


## preliminary functions

    def update_rho(self):
        self.grad_object.write_grad_rho()
        rho_setter_kwargs = self.opt.options['rho_setter_kwargs'] \
                            if 'rho_setter_kwargs' in self.opt.options \
                            else dict()
        for sname, scenario in self.opt.local_scenarios.items():
            rholist = self.rho_setter(scenario, **rho_setter_kwargs)
            for (vid, rho) in rholist:
                (ndn, i) = scenario._mpisppy_data.varid_to_nonant_index[vid]
                scenario._mpisppy_model.rho[(ndn, i)] = rho
        if self.cfg.get("grad_display_rho", True):
            self.display_rho_values()


    def _rho_primal_crit(self):
        primal_thresh = self.cfg.grad_primal_thresh
        self.prev_primal_norm = self.curr_primal_norm
        self.curr_primal_norm = norms.scaled_primal_metric(self.opt)
        norm_diff = np.abs(self.curr_primal_norm - self.prev_primal_norm)
        #print(f'{norm_diff =}')
        return (norm_diff <= primal_thresh)

    def _rho_dual_crit(self):
        dual_thresh = self.cfg.grad_dual_thresh
        self.wt.grab_local_Ws()
        dual_norm = norms.scaled_dual_metric(self.opt, self.wt.local_Ws, self.opt._PHIter)
        #print(f'{dual_norm =}')
        return (dual_norm <= dual_thresh)

    def _rho_primal_dual_crit(self):
        pd_thresh = self.cfg.grad_pd_thresh
        self.wt.grab_local_xbars()
        primal_resid = norms.primal_residuals_norm(self.opt)
        dual_resid = norms.dual_residuals_norm(self.opt, self.wt.local_xbars, self.opt._PHIter)
        resid_rel_norm = np.divide(dual_resid, primal_resid, out=np.zeros_like(primal_resid))
        #print(f'{resid_rel_norm =}')
        return (resid_rel_norm <= pd_thresh)

    def display_rho_values(self):
        for sname, scenario in self.opt.local_scenarios.items():
            rho_list = [scenario._mpisppy_model.rho[ndn_i]._value
                      for ndn_i, _ in scenario._mpisppy_data.nonant_indices.items()]
            print(sname, 'rho values: ', rho_list[:5])
            break

    def display_W_values(self):
        for (sname, scenario) in self.opt.local_scenarios.items():
            W_list = [w._value for w in scenario._mpisppy_model.W.values()]
            print(sname, 'W values: ', W_list)
            break


## extension functions

    def pre_iter0(self):
        pass

    def post_iter0(self):
        global_toc("Using gradient-based rho setter")
        self.wt.grab_local_Ws()
        self.wt.grab_local_xbars()
        self.curr_primal_norm = 0
        self.display_rho_values()

    def miditer(self):
        if self.opt._PHIter == 1:
            self.grad_object.write_grad_cost()
        if self._rho_dual_crit(): # or _rho_primal_crit, _rho_primal_dual_crit...
            self.update_rho()

    def enditer(self):
        pass

    def post_everything(self):
        if self.cylinder_rank == 0 and os.path.exists(self.cfg.grad_rho_file):
            os.remove(self.cfg.grad_rho_file)
        if self.cylinder_rank == 0 and os.path.exists(self.cfg.grad_cost_file):
            os.remove(self.cfg.grad_cost_file)
        self.cfg.grad_cost_file = self.cfg_args_cache['grad_cost_file']
        self.cfg.grad_rho_file = self.cfg_args_cache['grad_rho_file']
        self.cfg.grad_rho_path = self.cfg_args_cache['grad_rho_path']



