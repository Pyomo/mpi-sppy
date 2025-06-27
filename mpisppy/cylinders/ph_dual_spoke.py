###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from mpisppy.cylinders.spwindow import Field
from mpisppy.cylinders.spoke import Spoke
from mpisppy.cylinders.hub import PHHub


class _PHDualSpokeBase(Spoke, PHHub):

    send_fields = (*Spoke.send_fields, Field.DUALS,) 
    receive_fields = (*Spoke.receive_fields, )

    def send_boundsout(self):
        # overwrite PHHub.send_boundsout (not a hub)
        return

    def update_rho(self):
        rho_factor = self.opt.options.get("rho_factor", 1.0)
        if rho_factor == 1.0:
            return
        for s in self.opt.local_scenarios.values():
            for rho in s._mpisppy_model.rho.values():
                rho._value = rho_factor * rho._value

    def main(self):
        # setup, PH Iter0
        smoothed = self.options.get('smoothed', 0)
        attach_prox = True
        self.opt.PH_Prep(attach_prox=attach_prox, attach_smooth = smoothed)
        trivial_bound = self.opt.Iter0()
        if self.opt._can_update_best_bound():
            self.opt.best_bound_obj_val = trivial_bound

        # update the rho
        self.update_rho()

        # rest of PH
        self.opt.iterk_loop()

        return self.opt.conv, None, trivial_bound


class PHDualSpoke(_PHDualSpokeBase):

    @property
    def nonant_field(self):
        return None

    def send_nonants(self):
        # overwrite PHHub.send_nonants (don't send anything)
        return
