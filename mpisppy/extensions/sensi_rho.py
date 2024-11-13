###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from mpisppy import global_toc
import mpisppy.extensions.dyn_rho_base
from mpisppy.utils.nonant_sensitivities import nonant_sensitivies


class _SensiRhoBase(mpisppy.extensions.dyn_rho_base.Dyn_Rho_extension_base):
    def __init__(self, ph, comm=None):
        super().__init__(ph, comm=comm)
        # we'll set a minimum rho value to be the default rho
        self._minimum_rho = ph.options["defaultPHrho"]

    def get_nonant_sensitivites(self):
        """
        Method should return a dict of dicts [s][ndn_i]
        of sensitivity values for each scenario, node, and
        index.
        """
        raise NotImplementedError()

    def compute_and_update_rho(self):
        ph = self.ph

        # dict of dicts [s][ndn_i]
        nonant_sensis = self.get_nonant_sensitivites()

        for s in ph.local_scenarios.values():
            xbars = s._mpisppy_model.xbars
            for ndn_i, rho in s._mpisppy_model.rho.items():
                nv = s._mpisppy_data.nonant_indices[ndn_i]  # var_data object
                val = abs(nonant_sensis[s][ndn_i]) / max(1, abs(nv._value - xbars[ndn_i]._value))
                val *= self.multiplier
                if val < self._minimum_rho:
                    rho._value = self._minimum_rho
                else:
                    rho._value = val
                # if ph.cylinder_rank == 0:
                #     print(f"{s.name=}, {nv.name=}, {rho.value=}")

        rhomax = self._compute_rho_max(ph)
        for s in ph.local_scenarios.values():
            xbars = s._mpisppy_model.xbars
            for ndn_i, rho in s._mpisppy_model.rho.items():
                rho._value = rhomax[ndn_i]
                # if ph.cylinder_rank == 0:
                #     nv = s._mpisppy_data.nonant_indices[ndn_i]  # var_data object
                #     print(f"{s.name=}, {nv.name=}, {rho.value=}")

        if ph.cylinder_rank == 0:
            print(f"Rho values updated by {self.__class__.__name__} Extension")

    def miditer(self):
        self.update_caches()

        if self._update_recommended():
            self.compute_and_update_rho()
            sum_rho = 0.0
            num_rhos = 0   # could be computed...
            for sname, s in self.opt.local_scenarios.items():
                for ndn_i, nonant in s._mpisppy_data.nonant_indices.items():
                    sum_rho += s._mpisppy_model.rho[ndn_i]._value
                    num_rhos += 1
            rho_avg = sum_rho / num_rhos
            global_toc(f"Rho values recomputed - average rank 0 rho={rho_avg}")


class SensiRho(_SensiRhoBase):
    """
    Rho determination algorithm using nonant sensitivities
    """

    def __init__(self, ph, comm=None):
        super().__init__(ph, comm=comm)
        self.ph = ph

        self.multiplier = 1.0

        if (
            "sensi_rho_options" in ph.options
            and "multiplier" in ph.options["sensi_rho_options"]
        ):
            self.multiplier = ph.options["sensi_rho_options"]["multiplier"]
        self.cfg = ph.options["sensi_rho_options"]["cfg"]

    def get_nonant_sensitivites(self):
        # dict of dicts [s][ndn_i]
        nonant_sensis = {}
        for k, s in self.ph.local_subproblems.items():
            nonant_sensis[s] = nonant_sensitivies(s, self.ph)
        return nonant_sensis

    def post_iter0(self):
        global_toc("Using sensitivity rho setter")
        self.update_caches()
        self.compute_and_update_rho()

