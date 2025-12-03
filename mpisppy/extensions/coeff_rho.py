###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import mpisppy.extensions.extension

from mpisppy.utils.sputils import nonant_cost_coeffs


class CoeffRho(mpisppy.extensions.extension.Extension):
    """
    Determine rho as a linear function of the objective coefficient
    """

    def __init__(self, ph):
        self.ph = ph
        self.multiplier = 1.0
        if (
            "coeff_rho_options" in ph.options
            and "multiplier" in ph.options["coeff_rho_options"]
        ):
            self.multiplier = ph.options["coeff_rho_options"]["multiplier"]

    def post_iter0(self):
        for s in self.ph.local_scenarios.values():
            cc = nonant_cost_coeffs(s)
            for ndn_i, rho in s._mpisppy_model.rho.items():
                if cc[ndn_i] != 0:
                    rho._value = abs(cc[ndn_i]) * self.multiplier
                # if self.ph.cylinder_rank==0:
                #     nv = s._mpisppy_data.nonant_indices[ndn_i] # var_data object
                #     print(ndn_i,nv.getname(),cc[ndn_i],rho._value)

        if self.ph.cylinder_rank == 0:
            print("Rho values updated by CoeffRho Extension")
