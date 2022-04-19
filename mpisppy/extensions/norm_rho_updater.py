# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

# Adapted from the PySP extension adaptive_rho_converger, authored by Gabriel Hackebeil

import math
import mpisppy.extensions.extension

import numpy as np
import mpisppy.MPI as MPI

_norm_rho_defaults = { 'convergence_tolerance' : 1e-4,
                           'rho_decrease_multiplier' : 2.0,
                           'rho_increase_multiplier' : 2.0,
                           'primal_dual_difference_factor' : 100.,
                           'iterations_converged_before_decrease' : 0,
                           'rho_converged_decrease_multiplier' : 1.1,
                           'rho_update_stop_iterations' : None,
                           'verbose' : False,
}

_attr_to_option_name_map = {
    '_tol': 'convergence_tolerance',
    '_rho_decrease' : 'rho_decrease_multiplier',
    '_rho_increase' : 'rho_increase_multiplier',
    '_primal_dual_difference_factor' : 'primal_dual_difference_factor',
    '_required_converged_before_decrease' : 'iterations_converged_before_decrease',
    '_rho_converged_residual_decrease' : 'rho_converged_decrease_multiplier',
    '_stop_iter_rho_update' : 'rho_update_stop_iterations',
    '_verbose' : 'verbose',
}

class NormRhoUpdater(mpisppy.extensions.extension.Extension):

    def __init__(self, ph):

        self.ph = ph
        self.norm_rho_options = \
            ph.options['norm_rho_options'] if 'norm_rho_options' in ph.options else dict()

        self._set_options()
        self._prev_avg = {}
        self.ph._mpisppy_norm_rho_update_inuse = True  # allow NormRhoConverger
        
    def _set_options(self):
        options = self.norm_rho_options
        for attr_name, opt_name in _attr_to_option_name_map.items():
            setattr(self, attr_name, options[opt_name] if opt_name in options else _norm_rho_defaults[opt_name])

    def _snapshot_avg(self, ph):
        for s in ph.local_scenarios.values():
            for ndn_i, xbar in s._mpisppy_model.xbars.items():
                self._prev_avg[ndn_i] = xbar.value

    def _compute_primal_residual_norm(self, ph):

        local_nodenames = []
        local_primal_residuals = {}
        global_primal_residuals = {}

        for k,s in ph.local_scenarios.items():
            nlens = s._mpisppy_data.nlens        
            for node in s._mpisppy_node_list:
                if node.name not in local_nodenames:

                    ndn = node.name
                    local_nodenames.append(ndn)
                    nlen = nlens[ndn]

                    local_primal_residuals[ndn] = np.zeros(nlen, dtype='d')
                    global_primal_residuals[ndn] = np.zeros(nlen, dtype='d')

        for k,s in ph.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            xbars = s._mpisppy_model.xbars
            for node in s._mpisppy_node_list:
                ndn = node.name
                primal_residuals = local_primal_residuals[ndn]

                unweighted_primal_residuals = \
                        np.fromiter((abs(v._value - xbars[ndn,i]._value) for i,v in enumerate(node.nonant_vardata_list)),
                                    dtype='d', count=nlens[ndn] )
                primal_residuals += s._mpisppy_probability * unweighted_primal_residuals

        for nodename in local_nodenames:
            ph.comms[nodename].Allreduce(
                [local_primal_residuals[nodename], MPI.DOUBLE],
                [global_primal_residuals[nodename], MPI.DOUBLE],
                op=MPI.SUM)

        primal_resid = {}
        for ndn, global_primal_resid in global_primal_residuals.items():
            for i, v in enumerate(global_primal_resid):
                primal_resid[ndn,i] = v

        return primal_resid

    def _compute_dual_residual_norm(self, ph):
        dual_resid = {}
        for s in ph.local_scenarios.values():
            for ndn_i in s._mpisppy_data.nonant_indices:
                dual_resid[ndn_i] = s._mpisppy_model.rho[ndn_i].value * \
                        math.fabs( s._mpisppy_model.xbars[ndn_i].value - self._prev_avg[ndn_i] )
        return dual_resid

    def pre_iter0(self):
        pass

    def post_iter0(self):
        pass

    def miditer(self):

        ph = self.ph
        ph_iter = ph._PHIter
        if self._stop_iter_rho_update is not None and \
                (ph_iter > self._stop_iter_rho_update):
            return
        if not self._prev_avg:
            self._snapshot_avg(ph)
        else:
            primal_residuals = self._compute_primal_residual_norm(ph)
            dual_residuals = self._compute_dual_residual_norm(ph)
            self._snapshot_avg(ph)
            primal_dual_difference_factor = self._primal_dual_difference_factor
            first = True
            first_scenario = True
            for s in ph.local_scenarios.values():
                for ndn_i, rho in s._mpisppy_model.rho.items():
                    primal_resid = primal_residuals[ndn_i]
                    dual_resid = dual_residuals[ndn_i]

                    action = None
                    if (primal_resid > primal_dual_difference_factor*dual_resid) and (primal_resid > self._tol):
                        rho._value *= self._rho_increase
                        action = "Increasing"
                    elif (dual_resid > primal_dual_difference_factor*primal_resid) and (dual_resid > self._tol):
                        if ph_iter >= self._required_converged_before_decrease:
                            rho._value /= self._rho_decrease
                            action = "Decreasing"
                    elif (primal_resid < self._tol) and (dual_resid < self._tol):
                        rho._value /= self._rho_converged_residual_decrease
                        action = "Converged, Decreasing"
                    if self._verbose and ph.cylinder_rank == 0 and action is not None:
                        if first:
                            first = False
                            first_line = ("Updating rho values:\n%21s %40s %16s %16s %16s"
                                          % ("Action",
                                             "Variable",
                                             "Primal Residual",
                                             "Dual Residual",
                                             "New Rho"))
                            print(first_line)
                        if first_scenario:
                            print("%21s %40s %16g %16g %16g"
                                  % (action, s._mpisppy_data.nonant_indices[ndn_i].name,
                                     primal_resid, dual_resid, rho.value))
                first_scenario = False

    def enditer(self):
        pass

    def post_everything(self):
        pass
