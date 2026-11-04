###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import pyomo.environ as pyo

from mpisppy.cylinders.ph_dual_spoke import _PHDualSpokeBase
from mpisppy.cylinders.spwindow import Field


class RelaxedPHSpoke(_PHDualSpokeBase):

    send_fields = (*_PHDualSpokeBase.send_fields, Field.RELAXED_NONANTS_VALS, )
    receive_fields = (*_PHDualSpokeBase.receive_fields, )

    @property
    def nonant_field(self):
        return Field.RELAXED_NONANTS_VALS

    def main(self):
        # relax the integers
        integer_relaxer = pyo.TransformationFactory("core.relax_integer_vars")
        for s in self.opt.local_scenarios.values():
            integer_relaxer.apply_to(s)

        return super().main()
