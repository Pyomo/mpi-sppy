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
from pyomo.core.expr.calculus.derivatives import differentiate
import pyomo.environ as pyo
import mpisppy.MPI as MPI
from mpisppy import global_toc
from mpisppy.utils.sputils import nonant_cost_coeffs


class GradRho(mpisppy.extensions.dyn_rho_base.Dyn_Rho_extension_base):
    """
    Gradient-based rho from
    Gradient-baed rho Parameter for Progressive Hedging
    U. Naepels, David L. Woodruff, 2023
    """

    def __init__(self, ph):
        cfg = ph.options["grad_rho_options"]["cfg"]
        super().__init__(ph, cfg)
        self.ph = ph
        self.alpha = cfg.grad_order_stat
        assert (self.alpha >= 0 and self.alpha <= 1), f"For grad_order_stat 0 is the min, 0.5 the average, 1 the max; {alpha=} is invalid."
        
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
        return GradRho._compute_min_max(ph, np.maximum, MPI.MAX, -np.inf)

    @staticmethod
    def _compute_xmin(ph):
        return GradRho._compute_min_max(ph, np.minimum, MPI.MIN, np.inf)

    def _nonant_grad_cost(self, s):
        grads = np.array([-differentiate(list(s.component_data_objects(
                    ctype=pyo.Objective, active=True, descend_into=True
                ))[0], wrt=var) for ndn_i, var in s._mpisppy_data.nonant_indices.items()])
        
        return grads

    def _compute_and_update_rho(self):
        ph = self.ph
        primal_resid = self._compute_primal_residual_norm(ph)
        xmax = self._compute_xmax(ph)
        xmin = self._compute_xmin(ph)
        prob_list = [s._mpisppy_data.prob_coeff["ROOT"]
                     for s in self.ph.local_scenarios.values()]

        costs = {s: self._nonant_grad_cost(s) for s in ph.local_scenarios.values()}

        w = dict()
        for s in self.ph.local_scenarios.values():
            w[s] = np.array([s._mpisppy_model.W[ndn_i]._value
                             for ndn_i in s._mpisppy_data.nonant_indices])

        ccs = {s: costs[s]  for s in ph.local_scenarios.values()}
        k0, s0 = list(self.ph.local_scenarios.items())[0]
        local_cc = {ndn_i: [cc_list[ndn_i[1]] for _, cc_list in ccs.items()]
                        for ndn_i, var in s0._mpisppy_data.nonant_indices.items()}


        vcnt = len(local_cc)  # variable count
        cc_mins = np.empty(vcnt, dtype='d')  # global
        cc_maxes = np.empty(vcnt, dtype='d')
        cc_means = np.empty(vcnt, dtype='d')

        local_cc_mins = np.fromiter((min(cc_vals) for cc_vals in local_cc.values()), dtype='d')
        local_cc_maxes = np.fromiter((max(cc_vals) for cc_vals in local_cc.values()), dtype='d')
        local_prob = np.sum(prob_list)
        local_wgted_means = np.fromiter((np.dot(cc_vals, prob_list) * local_prob for cc_vals in local_cc.values()), dtype='d')

        self.ph.comms["ROOT"].Allreduce([local_cc_mins, MPI.DOUBLE],
                                               [cc_mins, MPI.DOUBLE],
                                               op=MPI.MIN)
        self.ph.comms["ROOT"].Allreduce([local_cc_maxes, MPI.DOUBLE],
                                               [cc_maxes, MPI.DOUBLE],
                                               op=MPI.MAX)

        self.ph.comms["ROOT"].Allreduce([local_wgted_means, MPI.DOUBLE],
                                               [cc_means, MPI.DOUBLE],
                                               op=MPI.SUM)
        if self.alpha == 0.5:
            cc = {ndn_i: float(cc_mean) for ndn_i, cc_mean in zip(local_cc.keys(), cc_means)}
        elif self.alpha == 0.0:
            cc = {ndn_i: float(cc_min) for ndn_i, cc_min in zip(local_cc.keys(), cc_mins)}
        elif self.alpha == 1.0:
            cc = {ndn_i: float(cc_max) for ndn_i, cc_max in zip(local_cc.keys(), cc_maxes)}
        elif self.alpha < 0.5:
            cc = {ndn_i: float(cc_min + alpha * 2 * (cc_mean - cc_min))\
                    for ndn_i, cc_min, cc_mean in zip(local_cc.keys(), cc_mins, cc_means)}
        elif self.alpha > 0.5:
            cc = {ndn_i: float(2 * cc_mean - cc_max) + alpha * 2 * (cc_max - cc_mean)\
                    for ndn_i, cc_mean, cc_max in zip(local_cc.keys(), cc_means, cc_maxes)}
        else:
            raise RuntimeError("Coding error.")

        for s in ph.local_scenarios.values():    
            for ndn_i, rho in s._mpisppy_model.rho.items():
                if cc[ndn_i] != 0:
                    nv = s._mpisppy_data.nonant_indices[ndn_i]  # var_data object
                    if nv.is_integer():
                        rho._value = abs(cc[ndn_i]) / (xmax[ndn_i] - xmin[ndn_i] + 1)
                    else:
                        rho._value = abs(cc[ndn_i]) / max(1e-6, primal_resid[ndn_i])

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
        global_toc("Using grad-rho rho setter")
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