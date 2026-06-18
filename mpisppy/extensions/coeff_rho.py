###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from mpisppy import global_toc
import mpisppy.extensions.extension

from mpisppy.utils.sputils import nonant_cost_coeffs
from mpisppy.utils.rho_utils import report_zero_rho_fallback


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
        # remembers the last reported zero-coefficient count to avoid log spam
        self._rho_report_state = {}

    def post_iter0(self):
        # nonants with a zero objective coefficient yield no meaningful rho from
        # this heuristic; they keep the (positive) default rho. We report rather
        # than silently substituting; see issue #560.
        zero_coeff = set()
        for s in self.ph.local_scenarios.values():
            cc = nonant_cost_coeffs(s)
            for ndn_i, rho in s._mpisppy_model.rho.items():
                if cc[ndn_i] != 0:
                    rho._value = abs(cc[ndn_i]) * self.multiplier
                else:
                    zero_coeff.add(ndn_i)
                # if self.ph.cylinder_rank==0:
                #     nv = s._mpisppy_data.nonant_indices[ndn_i] # var_data object
                #     print(ndn_i,nv.getname(),cc[ndn_i],rho._value)

        report_zero_rho_fallback(self.ph, "CoeffRho", len(zero_coeff),
                                 self.ph.options.get("defaultPHrho"), self._rho_report_state)
        global_toc("Rho values updated by CoeffRho Extension", self.ph.cylinder_rank == 0)
