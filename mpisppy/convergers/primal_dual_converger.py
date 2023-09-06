import math
import numpy as np
import mpisppy.convergers.converger
from mpisppy import MPI

class PrimalDualConverger(mpisppy.convergers.converger.Converger):
    """ Convergence checker for the primal-dual metrics.
        Primal convergence is measured as weighted sum over all scenarios s
        p_{s} * ||x_{s} - \bar{x}||_1.
        Dual convergence is measured as 
        rho * ||\bar{x}_{t} - \bar{x}_{t}||_1 * r_{s}
    """
    def __init__(self, ph):
        """ Initialization method for the PrimalDualConverger class."""
        super().__init__(ph)
        if 'primal_dual_converger_options' in ph.options and \
                'verbose' in ph.options['primal_dual_converger_options'] and \
                ph.options['primal_dual_converger_options']['verbose']:
            self._verbose = True
        else:
            self._verbose = False
        self._ph = ph
        self.convergence_threshold = ph.options['primal_dual_converger_options']\
            ['tol']

        self.prev_xbars = self._get_xbars()

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
        ret_val = primal_gap + dual_gap <= self.convergence_threshold

        if self._verbose and self._ph.cylinder_rank == 0:
            print(f"primal gap = {round(primal_gap, 5)}, dual gap = {round(dual_gap, 5)}")

            if ret_val:
                print("Dual convergence check passed")
            else:
                print("Dual convergence check failed "
                      f"(requires primal + dual gaps) <= {self.convergence_threshold}")

        return ret_val
