###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import os

import mpisppy.extensions.dyn_rho_base
import mpisppy.utils.gradient as grad
import mpisppy.utils.find_rho as find_rho
from mpisppy import global_toc


class Gradient_extension(mpisppy.extensions.dyn_rho_base.Dyn_Rho_extension_base):
    """
    This extension makes PH use gradient-rho and the corresponding rho setter.
    
    Args:
       opt (PHBase object): gives the problem
       cfg (Config object): config object
    
    Attributes:
       grad_object (Find_Grad object): gradient object
    
    """
    def __init__(self, opt, comm=None):
        super().__init__(opt, comm=comm)

        self.cfg = opt.options["gradient_extension_options"]["cfg"]
        # This is messy because we want to be able to use or give rhos as requested.
        # (e.g., if the user gave us an input file, use that)
        # TBD: stop using files
        # TBD: restore the rho_setter?
        self.cfg_args_cache = {'rho_file_in': self.cfg.rho_file_in,
                               'grad_rho_file_out': self.cfg.grad_rho_file_out}
        if self.cfg.get('grad_cost_file_out', ifmissing="") == "":
            self.cfg.grad_cost_file_out = './_temp_grad_cost_file.csv'
#        else:
#            self.cfg_args_cache["grad_cost_file_out"] = self.cfg.grad_cost_file_out
        if self.cfg.get('grad_cost_file_in', ifmissing="") == "":
            self.cfg.grad_cost_file_in = self.cfg.grad_cost_file_out  # write then read
        # from the perspective of this extension, we really should not have both
        if self.cfg.get('rho_file_in', ifmissing="") == "":
            if self.cfg.get("grad_rho_file_out", ifmissing="") == "":
                # we don't have either, but will write then read
                self.cfg.grad_rho_file_out = './_temp_rho_file.csv'
                self.cfg.rho_file_in = self.cfg.grad_rho_file_out  
            else:
                # we don't have an in, but we have an out, still write then read
                self.cfg.rho_file_in = self.cfg.grad_rho_file_out

        self.grad_object = grad.Find_Grad(opt, self.cfg)
        self.rho_setter = find_rho.Set_Rho(self.cfg).rho_setter

    def pre_iter0(self):
        pass

    def post_iter0(self):
        global_toc("Using gradient-based rho setter")
        self.update_caches()

    def miditer(self):
        self.update_caches()
        if self.opt._PHIter == 1:
            self.grad_object.write_grad_cost()
        if self.opt._PHIter == 1 or self._update_recommended():
            self.grad_object.write_grad_rho()
            rho_setter_kwargs = self.opt.options['rho_setter_kwargs'] \
                                if 'rho_setter_kwargs' in self.opt.options \
                                   else dict()

            # sum/num is a total hack
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
        # if we are using temp files, deal with it
        if self.cylinder_rank == 0 and os.path.exists(self.cfg.rho_file_in)\
           and self.cfg.rho_file_in != self.cfg_args_cache['rho_file_in']:
             os.remove(self.cfg.rho_file_in)
             self.cfg.rho_file_in = self.cfg_args_cache['rho_file_in']  # namely ""
        if self.cylinder_rank == 0 and os.path.exists(self.cfg.grad_cost_file_out) and \
           self.cfg.get('grad_cost_file_out', ifmissing="") == "":
             os.remove(self.cfg.grad_cost_file_out)
             self.cfg.grad_cost_file_out = self.cfg_args_cache['grad_cost_file_out']
