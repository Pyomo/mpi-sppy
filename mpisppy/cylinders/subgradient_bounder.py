###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import mpisppy.cylinders.spoke
from mpisppy.cylinders.lagrangian_bounder import _LagrangianMixin

class SubgradientOuterBound(_LagrangianMixin, mpisppy.cylinders.spoke.OuterBoundSpoke):

    converger_spoke_char = 'G'

    def main(self):
        extensions = self.opt.extensions is not None
        verbose = self.opt.options['verbose']

        self.lagrangian_prep()

        if extensions:
            self.opt.extobject.pre_iter0()
        self.dk_iter = 1
        self.trivial_bound = self.lagrangian()
        if extensions:
            self.opt.extobject.post_iter0()

        self.bound = self.trivial_bound
        if extensions:
            self.opt.extobject.post_iter0_after_sync()

        self.opt.current_solver_options = self.opt.iterk_solver_options

        # update rho / alpha
        if self.opt.options.get('subgradient_rho_multiplier') is not None:
            rf = self.opt.options['subgradient_rho_multiplier']
            for scenario in self.opt.local_scenarios.values():
                for ndn_i in scenario._mpisppy_model.rho:
                    scenario._mpisppy_model.rho[ndn_i] *= rf

        while not self.got_kill_signal():
            # compute a subgradient step
            self.opt.Compute_Xbar(verbose)
            self.opt.Update_W(verbose)
            if extensions:
                self.opt.extobject.miditer()
            bound = self.lagrangian()
            if extensions:
                self.opt.extobject.enditer()
            if bound is not None:
                self.bound = bound
            if extensions:
                self.opt.extobject.enditer_after_sync()
