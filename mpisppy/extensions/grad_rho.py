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
from pyomo.core.expr.calculus.derivatives import Modes
import pyomo.environ as pyo
import mpisppy.MPI as MPI
from mpisppy import global_toc
import mpisppy.utils.sputils as sputils
from mpisppy.cylinders.spwindow import Field

class GradRho(mpisppy.extensions.dyn_rho_base.Dyn_Rho_extension_base):
    """
    Gradient-based rho from
    Gradient-based rho Parameter for Progressive Hedging
    U. Naepels, David L. Woodruff, 2023

    Includes modifications to extend scenario-based denominators
    to multi-stage problems and calculation of gradients from objective
    expressions on the fly. 
    A. Asger, B. Knueven, 2025
    """

    def __init__(self, opt):
        cfg = opt.options["grad_rho_options"]["cfg"]
        super().__init__(opt, cfg)
        self.opt = opt
        self.alpha = cfg.grad_order_stat
        assert (self.alpha >= 0 and self.alpha <= 1), f"For grad_order_stat 0 is the min, 0.5 the average, 1 the max; {self.alpha=} is invalid."
        self.multiplier = 1.0

        if (
            cfg.grad_rho_multiplier
        ):
            self.multiplier = cfg.grad_rho_multiplier

        self.eval_at_xhat = cfg.eval_at_xhat
        self.indep_denom = cfg.indep_denom
    
    def _scen_dep_denom(self, s):
        """ Computes scenario dependent denominator for grad rho calculation.

        Args:
           s (Pyomo Concrete Model): scenario

        Returns:
           scen_dep_denom (numpy array): denominator

        """

        scen_dep_denom = {}

        xbars = s._mpisppy_model.xbars

        for ndn_i, v in s._mpisppy_data.nonant_indices.items():
            scen_dep_denom[ndn_i] = abs(v._value - xbars[ndn_i]._value)
            
        denom_max = max(scen_dep_denom.values())

        for ndn_i, v in s._mpisppy_data.nonant_indices.items():
            if scen_dep_denom[ndn_i] <= self.opt.E1_tolerance:
                scen_dep_denom[ndn_i] = max(denom_max, self.opt.E1_tolerance)
        
        return scen_dep_denom

    def _scen_indep_denom(self):
        """ Computes scenario independent denominator for grad rho calculation.

        Returns:
           scen_indep_denom (numpy array): denominator

        """
        opt = self.opt
        local_nodenames = []
        local_denoms = {}
        global_denoms = {}

        for k, s in opt.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                if node.name not in local_nodenames:
                    ndn = node.name
                    local_nodenames.append(ndn)
                    nlen = nlens[ndn]

                    local_denoms[ndn] = np.zeros(nlen, dtype="d")
                    global_denoms[ndn] = np.zeros(nlen, dtype="d")

        for k, s in opt.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            xbars = s._mpisppy_model.xbars
            for node in s._mpisppy_node_list:
                ndn = node.name
                denoms = local_denoms[ndn]

                unweighted_denoms = np.fromiter(
                    (
                        abs(v._value - xbars[ndn, i]._value)
                        for i, v in enumerate(node.nonant_vardata_list)
                    ),
                    dtype="d",
                    count=nlens[ndn],
                )
                denoms += s._mpisppy_data.prob_coeff[ndn] * unweighted_denoms

        for nodename in local_nodenames:
            opt.comms[nodename].Allreduce(
                [local_denoms[nodename], MPI.DOUBLE],
                [global_denoms[nodename], MPI.DOUBLE],
                op=MPI.SUM,
            )

        scen_indep_denom = {}
        for ndn, global_denom in global_denoms.items():
            for i, v in enumerate(global_denom):
                scen_indep_denom[ndn, i] = v

        return scen_indep_denom

    def _get_grad_exprs(self):
        """ Grabs and caches the gradient expressions for each scenario's objective (without proximal term). """ 

        self.grad_exprs = dict()

        for s in self.opt.local_scenarios.values():
            self.grad_exprs[s] = differentiate(sputils.find_active_objective(s),
                                wrt_list=s._mpisppy_data.nonant_indices.values(),
                                mode=Modes.reverse_symbolic,
                                )

            self.grad_exprs[s] = {ndn_i : self.grad_exprs[s][i] for i, ndn_i in enumerate(s._mpisppy_data.nonant_indices)}
            
        return 

    def _eval_grad_exprs(self, s, xhat):
        """ Evaluates the gradient expressions of the objectives for scenario s at xhat (if available) or the current values. """
        
        ci = 0
        grads = {}

        if self.eval_at_xhat:
            if True not in np.isnan(self.best_xhat_buf.value_array()):
                for ndn_i, var in s._mpisppy_data.nonant_indices.items():
                    var.value = xhat[ci]
                    ci += 1

        for ndn_i, var in s._mpisppy_data.nonant_indices.items():
            grads[ndn_i] = pyo.value(self.grad_exprs[s][ndn_i])

        return grads

    def _compute_and_update_rho(self):
        """ Computes and sets rhos for each scenario and each variable based on scenario dependence of
        the denominator in rho calculation.
        """

        opt = self.opt
        local_scens = opt.local_scenarios.values()

        if self.indep_denom:
            grad_denom = self._scen_indep_denom()
            loc_denom = {s: grad_denom for s in local_scens}
        else:
            loc_denom = {s: self._scen_dep_denom(s)
                           for s in opt.local_scenarios.values()}

        costs = {s: self._eval_grad_exprs(s, self.best_xhat_buf.value_array())
                     for s in opt.local_scenarios.values()}

        local_nodenames = []
        local_rhos = {}
        local_rho_mins = {}
        local_rho_maxes = {}
        local_rho_means = {}
        global_rho_mins = {}
        global_rho_maxes = {}
        global_rho_means = {}

        for k, s in opt.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                if node.name not in local_nodenames:
                    ndn = node.name
                    local_nodenames.append(ndn)
                    nlen = nlens[ndn]
                    
                    local_rhos[ndn] = np.zeros(nlen, dtype="d")
                    local_rho_mins[ndn] = np.zeros(nlen, dtype="d")
                    local_rho_maxes[ndn] = np.zeros(nlen, dtype="d")
                    local_rho_means[ndn] = np.zeros(nlen, dtype="d")
                    global_rho_mins[ndn] = np.zeros(nlen, dtype="d")
                    global_rho_maxes[ndn] = np.zeros(nlen, dtype="d")
                    global_rho_means[ndn] = np.zeros(nlen, dtype="d")

        for k, s in opt.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                ndn = node.name
                rhos = local_rhos[ndn]
                rho_mins = local_rho_mins[ndn]
                rho_maxes = local_rho_maxes[ndn]
                rho_means = local_rho_means[ndn]

                rhos = np.fromiter(
                    (
                        abs(costs[s][ndn, i]/loc_denom[s][ndn, i])
                        for i, v in enumerate(node.nonant_vardata_list)
                    ),
                    dtype="d",
                    count=nlens[ndn],
                )

                np.minimum(rho_mins, rhos, out=rho_mins, where=(s._mpisppy_data.prob_coeff[ndn] > 0))
                np.maximum(rho_maxes, rhos, out=rho_maxes, where=(s._mpisppy_data.prob_coeff[ndn] > 0))
                rho_means += s._mpisppy_data.prob_coeff[ndn] * rhos

        for nodename in local_nodenames:
            opt.comms[nodename].Allreduce(
                [local_rho_mins[nodename], MPI.DOUBLE],
                [global_rho_mins[nodename], MPI.DOUBLE],
                op=MPI.MIN,
            )

            opt.comms[nodename].Allreduce(
                [local_rho_maxes[nodename], MPI.DOUBLE],
                [global_rho_maxes[nodename], MPI.DOUBLE],
                op=MPI.MAX,
            )

            opt.comms[nodename].Allreduce(
                [local_rho_means[nodename], MPI.DOUBLE],
                [global_rho_means[nodename], MPI.DOUBLE],
                op=MPI.SUM,
            )

        if self.alpha == 0.5:
            rhos = {(ndn, i): float(v) for ndn, rho_mean in global_rho_means.items() for i, v in enumerate(rho_mean)}
        elif self.alpha == 0.0:
            rhos = {(ndn, i): float(v) for ndn, rho_min in global_rho_mins.items() for i, v in enumerate(rho_min)}
        elif self.alpha == 1.0:
            rhos = {(ndn, i): float(v) for ndn, rho_max in global_rho_maxes.items() for i, v in enumerate(rho_max)}
        elif self.alpha < 0.5:
            rhos = {(ndn, i): float(min_v + self.alpha * 2 * (mean_v - min_v))
                    for ndn in global_rho_mins.keys()
                    for i, (min_v, mean_v) in enumerate(zip(global_rho_mins[ndn], global_rho_means[ndn]))}
        elif self.alpha > 0.5:
            rhos = {(ndn, i): float(2 * mean_v - max_v + self.alpha * 2 * (max_v - mean_v))
                    for ndn in global_rho_maxes.keys()
                    for i, (max_v, mean_v) in enumerate(zip(global_rho_maxes[ndn], global_rho_means[ndn]))}
        else:
            raise RuntimeError("Coding error.")

        for s in opt.local_scenarios.values():    
            for ndn_i, rho in s._mpisppy_model.rho.items():
                if rhos[ndn_i] != 0:
                    rho._value = self.multiplier*rhos[ndn_i]

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

    def iter0_post_solver_creation(self):
        pass

    def post_iter0(self):
        global_toc("Using grad-rho rho setter")
        self.update_caches()
        self._get_grad_exprs()
        self.compute_and_update_rho()

    def miditer(self):
        self.update_caches()
        self.opt.spcomm.get_receive_buffer(
            self.best_xhat_buf,
            Field.BEST_XHAT,
            self.best_xhat_spoke_index,
        )
        if self._update_recommended():
            self.compute_and_update_rho()

    def enditer(self):
        pass

    def post_everything(self):
        pass
    
    def register_receive_fields(self):
        spcomm = self.opt.spcomm
        best_xhat_ranks = spcomm.fields_to_ranks[Field.BEST_XHAT]
        assert len(best_xhat_ranks) == 1
        index = best_xhat_ranks[0]

        self.best_xhat_spoke_index = index

        self.best_xhat_buf = spcomm.register_recv_field(
            Field.BEST_XHAT,
            self.best_xhat_spoke_index,
        )

        return