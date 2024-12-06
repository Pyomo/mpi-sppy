###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import mpisppy.convergers.converger
from mpisppy import MPI
from mpisppy.extensions.phtracker import TrackedData

class PrimalDualConverger(mpisppy.convergers.converger.Converger):
    """ Convergence checker for the primal-dual metrics.
        Primal convergence is measured as weighted sum over all scenarios s
        p_{s} * ||x_{s} - \bar{x}||_1.
        Dual convergence is measured as
        rho * ||\bar{x}_{t} - \bar{x}_{t-1}||_1
    """
    def __init__(self, ph):
        """ Initialization method for the PrimalDualConverger class."""
        super().__init__(ph)

        self.options = ph.options.get('primal_dual_converger_options', {})
        self._verbose = self.options.get('verbose', False)
        self._ph = ph
        self.convergence_threshold = self.options.get('tol', 1)
        self.tracking = self.options.get('tracking', False)
        self.prev_xbars = self._get_xbars()
        self._rank = self._ph.cylinder_rank

        if self.tracking and self._rank == 0:
            # if phtracker is set up, save the results in the phtracker/hub folder
            if 'phtracker_options' in self._ph.options:
                tracker_options = self._ph.options["phtracker_options"]
                cylinder_name = tracker_options.get(
                    "cylinder_name", type(self._ph.spcomm).__name__)
                results_folder = tracker_options.get(
                    "results_folder", "results")
                results_folder = os.path.join(results_folder, cylinder_name)
            else:
                results_folder = self.options.get('results_folder', 'results')
            self.tracker = TrackedData('pd', results_folder, plot=True, verbose=self._verbose)
            os.makedirs(results_folder, exist_ok=True)
            self.tracker.initialize_fnames(name=self.options.get('pd_fname', None))
            self.tracker.initialize_df(['iteration', 'primal_gap', 'dual_gap'])

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

    def is_converged(self):
        """ check for convergence
        Args:
            self (object): create by prep

        Returns:
           converged?: True if converged, False otherwise
        """

        primal_gap = self._compute_primal_convergence()
        dual_gap = self._compute_dual_residual()
        self.prev_xbars = self._get_xbars()
        ret_val = max(primal_gap, dual_gap) <= self.convergence_threshold

        if self._verbose and self._rank == 0:
            print(f"primal gap = {round(primal_gap, 5)}, dual gap = {round(dual_gap, 5)}")

            if ret_val:
                print("Dual convergence check passed")
            else:
                print("Dual convergence check failed "
                      f"(requires primal + dual gaps) <= {self.convergence_threshold}")
        if self.tracking and self._rank == 0:
            self.tracker.add_row([self._ph._PHIter, primal_gap, dual_gap])
            self.tracker.write_out_data()
        return ret_val

    def plot_results(self):
        """
        Plot the results of the convergence checks
        by reading in csv file and plotting
        """
        plot_fname = self.tracker.plot_fname
        conv_data = pd.read_csv(self.tracker.fname)

        # Create a log-scale plot
        plt.semilogy(conv_data['iteration'], conv_data['primal_gap'], label='Primal Gap')
        plt.semilogy(conv_data['iteration'], conv_data['dual_gap'], label='Dual Gap')

        plt.xlabel('Iteration')
        plt.ylabel('Convergence Metric')
        plt.legend()
        plt.savefig(plot_fname)
        plt.close()

    def post_everything(self):
        '''
        Reading the convergence data and plotting the results
        '''
        if self.tracking and self._rank == 0:
            self.plot_results()