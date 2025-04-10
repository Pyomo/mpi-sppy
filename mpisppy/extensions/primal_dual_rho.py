###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import numpy as np
import mpisppy.extensions.extension
import mpisppy.MPI as MPI
from mpisppy import global_toc

class PrimalDualRho(mpisppy.extensions.extension.Extension):
    """ Increase / decrease rho globally based on primal-dual metrics
        Primal convergence is measured as weighted sum over all scenarios s
        p_{s} * ||x_{s} - \bar{x}||_1.
        Dual convergence is measured as
        rho * ||\bar{x}_{t} - \bar{x}_{t-1}||_1
    """

    def __init__(self, ph):

        super().__init__(ph)

        self.options = ph.options.get('primal_dual_rho_options', {})
        self._verbose = self.options.get('verbose', False)
        self._ph = ph
        self.update_threshold = self.options.get('rho_update_threshold', 2)
        self.prev_xbars = None
        self._rank = self._ph.cylinder_rank

    def _get_xbars(self):
        """
        Get the current xbar values from the local scenarios
        Returns:
            xbars (dict): dictionary of xbar values indexed by
                          (decision node name, index)
        """
        xbars = {}
        for s in self._ph.local_scenarios.values():
            for ndn_i, xbar in s._mpisppy_model.xbars.items():
                xbars[ndn_i] = xbar.value
            break
        return xbars

    def _compute_primal_convergence(self):
        """
        Compute the primal convergence metric
        Returns:
            global_sum_diff (float): primal convergence metric
        """
        local_sum_diff = np.zeros(1)
        global_sum_diff = np.zeros(1)
        for _, s in self._ph.local_scenarios.items():
            # we iterate over decision nodes instead of
            # s._mpisppy_data.nonant_indices to use numpy
            for node in s._mpisppy_node_list:
                ndn = node.name
                nlen = s._mpisppy_data.nlens[ndn]
                x_bars = np.fromiter((s._mpisppy_model.xbars[ndn,i]._value
                                      for i in range(nlen)), dtype='d')

                nonants_array = np.fromiter(
                    (v._value for v in node.nonant_vardata_list),
                    dtype='d', count=nlen)
                _l1 = np.abs(x_bars - nonants_array)

                # invariant to prob_coeff being a scalar or array
                prob = s._mpisppy_data.prob_coeff[ndn] * np.ones(nlen)
                local_sum_diff[0] += np.dot(prob, _l1)

        self._ph.comms["ROOT"].Allreduce(local_sum_diff, global_sum_diff, op=MPI.SUM)
        return global_sum_diff[0]

    def _compute_dual_residual(self):
        """ Compute the dual residual

        Returns:
           global_diff (float): difference between to consecutive x bars

        """
        local_sum_diff = np.zeros(1)
        global_sum_diff = np.zeros(1)
        for s in self._ph.local_scenarios.values():
            for node in s._mpisppy_node_list:
                ndn = node.name
                nlen = s._mpisppy_data.nlens[ndn]
                rhos = np.fromiter((s._mpisppy_model.rho[ndn,i]._value
                                    for i in range(nlen)), dtype='d')
                xbars = np.fromiter((s._mpisppy_model.xbars[ndn,i]._value
                                        for i in range(nlen)), dtype='d')
                prev_xbars = np.fromiter((self.prev_xbars[ndn,i]
                                            for i in range(nlen)), dtype='d')

                local_sum_diff[0] += np.sum(rhos * np.abs(xbars - prev_xbars))

        self._ph.comms["ROOT"].Allreduce(local_sum_diff, global_sum_diff, op=MPI.SUM)
        return global_sum_diff[0]

    def miditer(self):
        if self.prev_xbars is None:
            self.prev_xbars = self._get_xbars()
            return
        if hasattr(self._ph, "_swap_nonant_vars"):
            self._ph._swap_nonant_vars()
        primal_gap = self._compute_primal_convergence()
        dual_gap = self._compute_dual_residual()
        self.prev_xbars = self._get_xbars()
        if self._verbose:
            global_toc(f"{primal_gap=}, {dual_gap=}", self._ph.cylinder_rank==0)

        increase_rho = primal_gap > self.update_threshold * dual_gap
        decrease_rho = dual_gap > self.update_threshold * primal_gap

        if increase_rho:
            global_toc(f"{primal_gap=:.4e}, {dual_gap=:.4e}, increasing all rhos by factor {self.update_threshold}", self._ph.cylinder_rank==0)
            for s in self._ph.local_scenarios.values():
                for rho in s._mpisppy_model.rho.values():
                    rho._value *= self.update_threshold
        if decrease_rho:
            global_toc(f"{primal_gap=:.4e}, {dual_gap=:.4e}, decreasing all rhos by factor {self.update_threshold}", self._ph.cylinder_rank==0)
            for s in self._ph.local_scenarios.values():
                for rho in s._mpisppy_model.rho.values():
                    rho._value /= self.update_threshold

        if hasattr(self._ph, "_swap_nonant_vars_back"):
            self._ph._swap_nonant_vars_back()
