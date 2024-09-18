###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import numpy as np

from mpisppy import MPI


############################################################################


def scaled_primal_metric(PHB):
    """
    Compute the scaled primal convergence metric with Euclidean norm.
    """
    local_sum = np.zeros(1)
    global_sum = np.zeros(1)
    for s in PHB.local_scenarios.values():
        for node in s._mpisppy_node_list:
            ndn = node.name
            nlen = s._mpisppy_data.nlens[ndn]
            xbars = np.fromiter((s._mpisppy_model.xbars[ndn,i]._value
                                  for i in range(nlen)), dtype='d')
            nonants_array = np.fromiter((v._value for v in node.nonant_vardata_list),
                                        dtype='d', count=nlen)
            scaled_norm = np.divide(np.abs(nonants_array - xbars), xbars, out=np.zeros_like(xbars))
            if not s._mpisppy_data.has_variable_probability:
                local_sum[0] += s._mpisppy_data.prob_coeff[ndn] * np.sum(np.multiply(scaled_norm, scaled_norm))
            else:
                # rarely-used overwrite in the event of variable probability
                prob_array = np.fromiter((s._mpisppy_data.prob_coeff[ndn_i[0]][ndn_i[1]]
                                          for ndn_i in s._mpisppy_data.nonant_indices
                                          if ndn_i[0] == ndn),
                                         dtype='d', count=nlen)
                ### TBD: check this!!
                local_sum[0] += np.dot(prob_array, scaled_norm**2)
    PHB.comms["ROOT"].Allreduce(local_sum, global_sum, op=MPI.SUM)
    return np.sqrt(global_sum[0])




def scaled_dual_metric(PHB, w_cache, curr_iter):
    """
    Compute the Compute the scaled norm of the difference between to consecutive Ws.
    """
    local_sum = np.zeros(1)
    global_sum = np.zeros(1)
    for sname, s in PHB.local_scenarios.items():
        current_w = np.array(w_cache[curr_iter][sname])
        prev_w = np.array(w_cache[curr_iter-1][sname])
        # np.seterr(invalid='ignore')
        scaled_norm = np.divide(np.abs(current_w - prev_w), np.abs(current_w), out=np.zeros_like(current_w))
        for node in s._mpisppy_node_list:
            ndn = node.name
            nlen = s._mpisppy_data.nlens[ndn]
            if not s._mpisppy_data.has_variable_probability:
                local_sum[0] += s._mpisppy_data.prob_coeff[ndn] * np.sum(np.multiply(scaled_norm, scaled_norm))
            else:
                # rarely-used overwrite in the event of variable probability
                prob_array = np.fromiter((s._mpisppy_data.prob_coeff[ndn_i[0]][ndn_i[1]]
                                          for ndn_i in s._mpisppy_data.nonant_indices
                                          if ndn_i[0] == ndn),
                                         dtype='d', count=nlen)
                # tbd: does this give use the squared norm?
                local_sum[0] += np.dot(prob_array, scaled_norm**2)

    PHB.comms["ROOT"].Allreduce(local_sum, global_sum, op=MPI.SUM)
    return np.sqrt(global_sum[0])




def primal_residuals_norm(PHB):
    """
    Compute the scaled primal residuals Euclidean norm.
    """
    local_sum = np.zeros(1)
    global_sum = np.zeros(1)
    for s in PHB.local_scenarios.values():
        for node in s._mpisppy_node_list:
            ndn = node.name
            nlen = s._mpisppy_data.nlens[ndn]
            xbars = np.fromiter((s._mpisppy_model.xbars[ndn,i]._value
                                  for i in range(nlen)), dtype='d')
            nonants_array = np.fromiter((v._value for v in node.nonant_vardata_list),
                                        dtype='d', count=nlen)
            resid = np.abs(nonants_array - xbars)
            if not s._mpisppy_data.has_variable_probability:
                local_sum[0] += s._mpisppy_data.prob_coeff[ndn] * np.sum(np.multiply(resid, resid))
            else:
                # rarely-used overwrite in the event of variable probability
                prob_array = np.fromiter((s._mpisppy_data.prob_coeff[ndn_i[0]][ndn_i[1]]
                                          for ndn_i in s._mpisppy_data.nonant_indices
                                          if ndn_i[0] == ndn),
                                         dtype='d', count=nlen)
                local_sum[0] += np.dot(prob_array, resid)
    PHB.comms["ROOT"].Allreduce(local_sum, global_sum, op=MPI.SUM)
    return np.sqrt(global_sum[0])



def dual_residuals_norm(PHB, xbar_cache, curr_iter):
    """
    Compute the scaled primal residuals Euclidean norm.
    """
    local_sum = np.zeros(1)
    global_sum = np.zeros(1)
    for sname, s in PHB.local_scenarios.items():
        current_xbar = np.array(xbar_cache[curr_iter][sname])
        prev_xbar = np.array(xbar_cache[curr_iter-1][sname])
        for node in s._mpisppy_node_list:
            ndn = node.name
            nlen = s._mpisppy_data.nlens[ndn]
            rhos = np.fromiter((s._mpisppy_model.rho[(ndn, i)]._value for i in range(nlen)),
                               dtype='d', count=nlen)
            resid = np.multiply(rhos, np.abs(current_xbar - prev_xbar))
            if not s._mpisppy_data.has_variable_probability:
                local_sum[0] += s._mpisppy_data.prob_coeff[ndn] * np.sum(np.multiply(resid, resid))
            else:
                # rarely-used overwrite in the event of variable probability
                prob_array = np.fromiter((s._mpisppy_data.prob_coeff[ndn_i[0]][ndn_i[1]]
                                          for ndn_i in s._mpisppy_data.nonant_indices
                                          if ndn_i[0] == ndn),
                                         dtype='d', count=nlen)
                local_sum[0] += np.dot(prob_array, resid)
    PHB.comms["ROOT"].Allreduce(local_sum, global_sum, op=MPI.SUM)
    return np.sqrt(global_sum[0])




