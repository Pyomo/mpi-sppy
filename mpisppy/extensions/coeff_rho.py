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
from mpisppy.utils.rho_utils import assign_rho_with_fallback


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
        # nonants with a (near-)zero objective coefficient yield no meaningful
        # rho from this heuristic; they fall back to the positive default rho.
        # We report the fallback rather than substituting silently; see issue #560.
        default_rho = self.ph.options.get("defaultPHrho")
        coeffs = {s: nonant_cost_coeffs(s) for s in self.ph.local_scenarios.values()}
        assign_rho_with_fallback(
            self.ph, default_rho, "CoeffRho", self._rho_report_state,
            lambda s, ndn_i: abs(coeffs[s][ndn_i]) * self.multiplier)
        global_toc("Rho values updated by CoeffRho Extension", self.ph.cylinder_rank == 0)
