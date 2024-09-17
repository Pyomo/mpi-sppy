###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Runs PH with smaller rhos in order to compute a lower bound
# By default rescale the hub "default-rho" values (multiply by rf) but there are other options to set rho:
# - set rho manually (use _hack_set_rho)
# - use rho rescale factors written in a json file
# - use gradient-based rho (to be tested)

import json
import mpisppy.cylinders.spoke
import mpisppy.utils.find_rho as find_rho
import mpisppy.utils.gradient as grad
from mpisppy.utils.wtracker import WTracker

class PhOuterBound(mpisppy.cylinders.spoke.OuterBoundSpoke):
    """Updates its own W and x using its own rho.
    """
    converger_spoke_char = 'B'

    def ph_ob_prep(self):
        # Scenarios are created here
        self.opt.PH_Prep(attach_prox=True)
        self.opt._reenable_W()
        self.opt._create_solvers()
        if "ph_ob_rho_rescale_factors_json" in self.opt.options and\
            self.opt.options["ph_ob_rho_rescale_factors_json"] is not None:
            with open(self.opt.options["ph_ob_rho_rescale_factors_json"], "r") as fin:
                din = json.load(fin)
            self.rho_rescale_factors = {int(i): float(din[i]) for i in din}
        else:
            self.rho_rescale_factors = None

        # use gradient rho
        self.use_gradient_rho = False
        if "ph_ob_gradient_rho" in self.opt.options:
            assert self.opt.options["ph_ob_gradient_rho"]["cfg"] is not None, "You need to give a cfg to use gradient rho."
            self.use_gradient_rho = True
            print("PH Outer Bounder uses an iterative gradient-based rho setter")
            self.cfg = self.opt.options["ph_ob_gradient_rho"]["cfg"]
            if "rho_denom" in  self.opt.options["ph_ob_gradient_rho"]:
                self.cfg.grad_rho_denom = self.opt.options["ph_ob_gradient_rho"]["rho_denom"]
            self.cfg.grad_cost_file_out = '_temp_grad_cost_file_ph_ob.csv'
            self.cfg.grad_rho_file_out = '_temp_grad_rho_file_ph_ob.csv'  # out??? xxxx tbd
            # the xhat used here is the same as in the hub
            self.grad_object = grad.Find_Grad(self.opt, self.cfg)
            self.rho_setter = find_rho.Set_Rho(self.cfg).rho_setter
            self.grad_object.write_grad_cost()


    def _phsolve(self, iternum):
        verbose = self.opt.options['verbose']
        teeme = False
        if "tee-rank0-solves" in self.opt.options and self.opt.cylinder_rank == 0:
            teeme = self.opt.options['tee-rank0-solves']

        self.opt.solve_loop(
            solver_options=self.opt.current_solver_options,
            dtiming=False,
            gripe=True,
            tee=teeme,
            verbose=verbose
        )

    def _phboundsolve(self, iternum):
        self.opt._disable_prox()
        verbose = self.opt.options['verbose']
        teeme = False
        if "tee-rank0-solves" in self.opt.options and self.opt.cylinder_rank == 0:
            teeme = self.opt.options['tee-rank0-solves']
        self.opt.solve_loop(
            solver_options=self.opt.current_solver_options,
            dtiming=False,
            gripe=True,
            tee=teeme,
            verbose=verbose
        )
        self.opt._reenable_prox()
        # Compute the resulting bound
        return self.opt.Ebound(verbose)


    def _rescale_rho(self, rf):
        # IMPORTANT: the scalings accumulate.
        # E.g., 0.5 then 2.0 gets you back where you started.
        for (sname, scenario) in self.opt.local_scenarios.items():
            for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                scenario._mpisppy_model.rho[ndn_i] *= rf

    def _display_rho_values(self):
        for sname, scenario in self.opt.local_scenarios.items():
            rho_list = [scenario._mpisppy_model.rho[ndn_i]._value
                        for ndn_i in scenario._mpisppy_data.nonant_indices.keys()]
            print(sname, 'PH OB rho values: ', rho_list[:5])
            break

    def _display_W_values(self):
        for (sname, scenario) in self.opt.local_scenarios.items():
            W_list = [w._value for w in scenario._mpisppy_model.W.values()]
            print(sname, 'W values: ', W_list)
            break

    def _display_xbar_values(self):
        for (sname, scenario) in self.opt.local_scenarios.items():
            xbar_list = [scenario._mpisppy_model.xbars[ndn_i]._value
                         for ndn_i in scenario._mpisppy_data.nonant_indices.keys()]
            print(sname, 'PH OB xbar values: ', xbar_list)
            break

    def _set_gradient_rho(self):
        self.grad_object.write_grad_rho()
        rho_setter_kwargs = self.opt.options['rho_setter_kwargs'] \
                            if 'rho_setter_kwargs' in self.opt.options \
                            else dict()
        for sname, scenario in self.opt.local_scenarios.items():
            rholist = self.rho_setter(scenario, **rho_setter_kwargs)
            for (vid, rho) in rholist:
                (ndn, i) = scenario._mpisppy_data.varid_to_nonant_index[vid]
                scenario._mpisppy_model.rho[(ndn, i)] = rho

    def _hack_set_rho(self):
        # HACK to set specific rho values
        rhoval = 0.002
        for sname, scenario in self.opt.local_scenarios.items():
            for ndn_i in scenario._mpisppy_data.nonant_indices.keys():
                scenario._mpisppy_model.rho[ndn_i] = rhoval

    def _update_weights_and_solve(self, iternum):
        verbose = self.opt.options["verbose"]
        self.opt.Compute_Xbar(verbose=verbose)
        self.opt.Update_W(verbose=verbose)
        # see if rho should be rescaled
        if self.use_gradient_rho and iternum == 0:
            self._set_gradient_rho()
            self._display_rho_values()
        if self.rho_rescale_factors is not None\
           and iternum in self.rho_rescale_factors:
            self._rescale_rho(self.rho_rescale_factors[iternum])
        bound = self._phboundsolve(iternum) # keep it before calling _phsolve
        self._phsolve(iternum)
        return bound

    def main(self):
        self.ph_ob_prep()
        self._rescale_rho(self.opt.options["ph_ob_initial_rho_rescale_factor"] )

        self.trivial_bound = self._phsolve(0)
        self.bound = self.trivial_bound
        self.opt.current_solver_options = self.opt.iterk_solver_options

        self.B_iter = 1
        self.opt.B_iter = self.B_iter
        wtracker = WTracker(self.opt)

        while not self.got_kill_signal():
            # because of aph, do not check for new data, just go for it
            self.bound = self._update_weights_and_solve(self.B_iter)
            self.B_iter += 1
            self.opt.B_iter = self.B_iter
            wtracker.grab_local_Ws()

    def finalize(self):
        '''
        Do one final lagrangian pass with the final
        PH weights. Useful for when PH convergence
        and/or iteration limit is the cause of termination
        '''
        self.final_bound = self._update_weights_and_solve(self.B_iter)
        self.bound = self.final_bound
        return self.final_bound
