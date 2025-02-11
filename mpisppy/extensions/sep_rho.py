###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import mpisppy.extensions.dyn_rho_base
import numpy as np
import mpisppy.MPI as MPI
from mpisppy import global_toc
from mpisppy.utils.sputils import nonant_cost_coeffs


class SepRho(mpisppy.extensions.dyn_rho_base.Dyn_Rho_extension_base):
    """
    Rho determination algorithm "SEP" from
    Progressive hedging innovations for a class of stochastic mixed-integer
        resource allocation problems
    Jean-Paul Watson, David L. Woodruff, Compu Management Science, 2011
    DOI 10.1007/s10287-010-0125-4
    """

    def __init__(self, ph, comm=None):
        super().__init__(ph, comm=comm)
        self.ph = ph

        self.multiplier = 1.0

        if (
            "sep_rho_options" in ph.options
            and "multiplier" in ph.options["sep_rho_options"]
        ):
            self.multiplier = ph.options["sep_rho_options"]["multiplier"]
        self.cfg = ph.options["sep_rho_options"]["cfg"]

        self._nonant_cost_coeffs = {}

    def _compute_primal_residual_norm(self, ph):
        local_nodenames = []
        local_primal_residuals = {}
        global_primal_residuals = {}

        for k, s in ph.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                if node.name not in local_nodenames:
                    ndn = node.name
                    local_nodenames.append(ndn)
                    nlen = nlens[ndn]

                    local_primal_residuals[ndn] = np.zeros(nlen, dtype="d")
                    global_primal_residuals[ndn] = np.zeros(nlen, dtype="d")

        for k, s in ph.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            xbars = s._mpisppy_model.xbars
            for node in s._mpisppy_node_list:
                ndn = node.name
                primal_residuals = local_primal_residuals[ndn]

                unweighted_primal_residuals = np.fromiter(
                    (
                        abs(v._value - xbars[ndn, i]._value)
                        for i, v in enumerate(node.nonant_vardata_list)
                    ),
                    dtype="d",
                    count=nlens[ndn],
                )
                primal_residuals += s._mpisppy_probability * unweighted_primal_residuals

        for nodename in local_nodenames:
            ph.comms[nodename].Allreduce(
                [local_primal_residuals[nodename], MPI.DOUBLE],
                [global_primal_residuals[nodename], MPI.DOUBLE],
                op=MPI.SUM,
            )

        primal_resid = {}
        for ndn, global_primal_resid in global_primal_residuals.items():
            for i, v in enumerate(global_primal_resid):
                primal_resid[ndn, i] = v

        return primal_resid

    @staticmethod
    def _compute_min_max(ph, npop, mpiop, start):
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
            for node in s._mpisppy_node_list:
                ndn = node.name
                xmaxmin = local_xmaxmin[ndn]

                xmaxmin_partial = np.fromiter(
                    (v._value for v in node.nonant_vardata_list),
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
    def _compute_xmax(ph):
        return SepRho._compute_min_max(ph, np.maximum, MPI.MAX, -np.inf)

    @staticmethod
    def _compute_xmin(ph):
        return SepRho._compute_min_max(ph, np.minimum, MPI.MIN, np.inf)

    def nonant_cost_coeffs(self, s):
        if s not in self._nonant_cost_coeffs:
            if hasattr(s._mpisppy_data, "nonant_cost_coeffs"):
                self._nonant_cost_coeffs[s] = s._mpisppy_data.nonant_cost_coeffs
            else:
                self._nonant_cost_coeffs[s] = nonant_cost_coeffs(s)
        return self._nonant_cost_coeffs[s]

    def _compute_and_update_rho(self):
        ph = self.ph
        primal_resid = self._compute_primal_residual_norm(ph)
        xmax = self._compute_xmax(ph)
        xmin = self._compute_xmin(ph)

        for s in ph.local_scenarios.values():
            cc = self.nonant_cost_coeffs(s)
            for ndn_i, rho in s._mpisppy_model.rho.items():
                if cc[ndn_i] != 0:
                    nv = s._mpisppy_data.nonant_indices[ndn_i]  # var_data object
                    if nv.is_integer():
                        rho._value = abs(cc[ndn_i]) / (xmax[ndn_i] - xmin[ndn_i] + 1)
                    else:
                        rho._value = abs(cc[ndn_i]) / max(1, primal_resid[ndn_i])
                    rho._value *= self.multiplier

    def compute_and_update_rho(self):
        self._compute_and_update_rho()
        sum_rho = 0.0
        num_rhos = 0   # could be computed...
        for sname, s in self.opt.local_scenarios.items():
            for ndn_i, nonant in s._mpisppy_data.nonant_indices.items():
                sum_rho += s._mpisppy_model.rho[ndn_i]._value
                num_rhos += 1
        rho_avg = sum_rho / num_rhos
        global_toc(f"Rho values recomputed - average rank 0 rho={rho_avg}")
        
    def pre_iter0(self):
        pass

    def post_iter0(self):
        global_toc("Using sep-rho rho setter")
        self.update_caches()
        self.compute_and_update_rho()

    def miditer(self):
        self.update_caches()
        if self._update_recommended():
            self.compute_and_update_rho()

    def enditer(self):
        pass

    def post_everything(self):
        pass
