###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import mpisppy.cylinders.spoke

class SubgradientOuterBound(mpisppy.cylinders.spoke.OuterBoundSpoke):

    converger_spoke_char = 'G'

    def update_rho(self):
        rho_factor = self.opt.options.get("subgradient_rho_multiplier", 1.0)
        if rho_factor == 1.0:
            return
        for s in self.opt.local_scenarios.values():
            for rho in s._mpisppy_model.rho.values():
                rho._value = rho_factor * rho._value

    def main(self):
        # setup, PH Iter0
        if self.opt.options.get("smoothed", 0) != 0:
            raise RuntimeError("Cannnot use smoothing with Subgradient algorithm")
        attach_prox = False
        self.opt.PH_Prep(attach_prox=attach_prox, attach_smooth = 0)
        trivial_bound = self.opt.Iter0()
        if self.opt._can_update_best_bound():
            self.opt.best_bound_obj_val = trivial_bound

        # update the rho
        self.update_rho()

        # rest of PH
        self.opt.iterk_loop()

        return self.opt.conv, None, trivial_bound

    def sync(self):
        if self.opt.best_bound_obj_val is None:
            return

        # Tell the hub about the most recent bound
        self.send_bound(self.opt.best_bound_obj_val)

        # Update the nonant bounds, if possible
        self.receive_nonant_bounds()

    def finalize(self):
        if self.opt.best_bound_obj_val is None:
            return

        # Tell the hub about the most recent bound
        self.send_bound(self.opt.best_bound_obj_val)
        self.final_bound = self.opt.best_bound_obj_val

        return self.final_bound
