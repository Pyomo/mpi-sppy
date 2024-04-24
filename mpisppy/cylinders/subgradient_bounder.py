# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import mpisppy.cylinders.spoke
from mpisppy.cylinders.lagrangian_bounder import _LagrangianMixin

class SubgradientOuterBound(_LagrangianMixin, mpisppy.cylinders.spoke.OuterBoundSpoke):

    converger_spoke_char = 'G'

    def main(self):
        # The rho_setter should be attached to the opt object
        rho_setter = None
        if hasattr(self.opt, 'rho_setter'):
            rho_setter = self.opt.rho_setter
        verbose = self.opt.options['verbose']

        self.lagrangian_prep()

        # update rho / alpha
        if self.opt.options.get('subgradient_rho_multiplier') is not None:
            rf = self.opt.options['subgradient_rho_multiplier']
            for scenario in self.opt.local_scenarios.values():
                for ndn_i in scenario._mpisppy_model.rho:
                    scenario._mpisppy_model.rho[ndn_i] *= rf

        self.trivial_bound = self.lagrangian()

        self.opt.current_solver_options = self.opt.iterk_solver_options

        self.bound = self.trivial_bound

        while not self.got_kill_signal():
            # compute a subgradient step
            self.opt.Compute_Xbar(verbose)
            self.opt.Update_W(verbose)
            bound = self.lagrangian()
            if bound is not None:
                self.bound = bound
