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

import pyomo.environ as pyo

class RelaxedPHSpoke(Spoke, PHHub):

    send_fields = (*Spoke.send_fields, Field.DUALS, Field.RELAXED_NONANT, )
    receive_fields = (*Spoke.receive_fields, )

    @property
    def nonant_field(self):
        return Field.RELAXED_NONANT

    def send_boundsout(self):
        # overwrite PHHub.send_boundsout (not a hub)
        return

    def update_rho(self):
        rho_factor = self.opt.options.get("relaxed_ph_rho_factor", 1.0)
        if rho_factor == 1.0:
            return
        for s in self.opt.local_scenarios.values():
            for rho in s._mpisppy_model.rho.values():
                rho._value = rho_factor * rho._value

    def main(self):
        # relax the integers
        integer_relaxer = pyo.TransformationFactory("core.relax_integer_vars")
        for s in self.opt.local_scenarios.values():
            integer_relaxer.apply_to(s)

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
