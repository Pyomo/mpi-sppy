###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import numpy as np
from mpisppy import global_toc
import mpisppy.extensions.dyn_rho_base
import mpisppy.MPI as MPI
from mpisppy.utils.nonant_sensitivities import nonant_sensitivies


class SensiRho(mpisppy.extensions.dyn_rho_base.Dyn_Rho_extension_base):
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

    @staticmethod
    def _compute_rho_min_max(ph, npop, mpiop, start):
        local_nodenames = []
        local_xmaxmin = {}
        global_xmaxmin = {}

        for k, s in ph.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                if node.name not in local_nodenames:
                    ndn = node.name
                    local_nodenames.append(ndn)
                    nlen = nlens[ndn]

                    local_xmaxmin[ndn] = start * np.ones(nlen, dtype="d")
                    global_xmaxmin[ndn] = np.zeros(nlen, dtype="d")

        for k, s in ph.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            rho = s._mpisppy_model.rho
            for node in s._mpisppy_node_list:
                ndn = node.name
                xmaxmin = local_xmaxmin[ndn]

                xmaxmin_partial = np.fromiter(
                    (rho[ndn,i]._value for i, _ in enumerate(node.nonant_vardata_list)),
                    dtype="d",
                    count=nlens[ndn],
                )
                xmaxmin = npop(xmaxmin, xmaxmin_partial)
                local_xmaxmin[ndn] = xmaxmin

        for nodename in local_nodenames:
            ph.comms[nodename].Allreduce(
                [local_xmaxmin[nodename], MPI.DOUBLE],
                [global_xmaxmin[nodename], MPI.DOUBLE],
                op=mpiop,
            )

        xmaxmin_dict = {}
        for ndn, global_xmaxmin_dict in global_xmaxmin.items():
            for i, v in enumerate(global_xmaxmin_dict):
                xmaxmin_dict[ndn, i] = v

        return xmaxmin_dict

    @staticmethod
    def _compute_rho_avg(ph):
        local_nodenames = []
        local_avg = {}
        global_avg = {}

        for k, s in ph.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            rho = s._mpisppy_model.rho
            for node in s._mpisppy_node_list:
                if node.name not in local_nodenames:
                    ndn = node.name
                    local_nodenames.append(ndn)
                    nlen = nlens[ndn]

                    local_avg[ndn] = np.zeros(nlen, dtype="d")
                    global_avg[ndn] = np.zeros(nlen, dtype="d")

        for k, s in ph.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            rho = s._mpisppy_model.rho
            for node in s._mpisppy_node_list:
                ndn = node.name

                local_rhos = np.fromiter(
                        (rho[ndn,i]._value for i, _ in enumerate(node.nonant_vardata_list)),
                        dtype="d",
                        count=nlens[ndn],
                    )
                # print(f"{k=}, {local_rhos=}, {s._mpisppy_probability=}, {s._mpisppy_data.prob_coeff[ndn]=}")
                # TODO: is this the right thing, or should it be s._mpisppy_probability?
                local_rhos *= s._mpisppy_data.prob_coeff[ndn]

                local_avg[ndn] += local_rhos

        for nodename in local_nodenames:
            ph.comms[nodename].Allreduce(
                [local_avg[nodename], MPI.DOUBLE],
                [global_avg[nodename], MPI.DOUBLE],
                op=MPI.SUM,
            )

        rhoavg_dict = {}
        for ndn, global_rhoavg_dict in global_avg.items():
            for i, v in enumerate(global_rhoavg_dict):
                rhoavg_dict[ndn, i] = v

        return rhoavg_dict

    @staticmethod
    def _compute_rho_max(ph):
        return SensiRho._compute_rho_min_max(ph, np.maximum, MPI.MAX, -np.inf)

    @staticmethod
    def _compute_rho_min(ph):
        return SensiRho._compute_rho_min_max(ph, np.minimum, MPI.MIN, np.inf)

    def compute_and_update_rho(self):
        ph = self.ph

        nonant_sensis = dict()  # dict of dicts [s][ndn_i]
        for k, s in ph.local_subproblems.items():
            nonant_sensis[s] = nonant_sensitivies(s, ph)
                
        for s in ph.local_scenarios.values():
            xbars = s._mpisppy_model.xbars
            for ndn_i, rho in s._mpisppy_model.rho.items():
                nv = s._mpisppy_data.nonant_indices[ndn_i]  # var_data object
                rho._value = abs(nonant_sensis[s][ndn_i]) / max(1, abs(nv._value - xbars[ndn_i]._value))
                rho._value *= self.multiplier
                # if ph.cylinder_rank == 0:
                #     print(f"{s.name=}, {nv.name=}, {rho.value=}")

        rhoavg = self._compute_rho_avg(ph)
        for s in ph.local_scenarios.values():
            xbars = s._mpisppy_model.xbars
            for ndn_i, rho in s._mpisppy_model.rho.items():
                rho._value = rhoavg[ndn_i]
                # if ph.cylinder_rank == 0:
                #     nv = s._mpisppy_data.nonant_indices[ndn_i]  # var_data object
                #     print(f"{s.name=}, {nv.name=}, {rho.value=}")

        if ph.cylinder_rank == 0:
            print("Rho values updated by SensiRho Extension")

    def pre_iter0(self):
        pass

    def post_iter0(self):
        global_toc("Using sensi-rho rho setter")
        super().post_iter0()        
        self.compute_and_update_rho()
        
    def miditer(self):
        self.primal_conv_cache.append(self.opt.convergence_diff())
        self.dual_conv_cache.append(self.wt.W_diff())

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

    def enditer(self):
        pass

    def post_everything(self):
        pass
