###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import numpy as np

import pyomo.environ as pyo
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU

import mpisppy.extensions.extension
import mpisppy.MPI as MPI
from mpisppy.utils.kkt.interface import InteriorPointInterface


class SensiRho(mpisppy.extensions.extension.Extension):
    """
    Rho determination algorithm using nonant sensitivities
    """

    def __init__(self, ph):
        self.ph = ph

        self.multiplier = 1.0

        if (
            "sensi_rho_options" in ph.options
            and "multiplier" in ph.options["sensi_rho_options"]
        ):
            self.multiplier = ph.options["sensi_rho_options"]["multiplier"]

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

    def pre_iter0(self):
        pass

    def post_iter0(self):
        ph = self.ph

        # first, solve the subproblems with Ipopt,
        # and gather sensitivity information
        ipopt = pyo.SolverFactory("ipopt")
        nonant_sensis = {}
        for k, s in ph.local_subproblems.items():
            solution_cache = pyo.ComponentMap()
            for var in s.component_data_objects(pyo.Var):
                solution_cache[var] = var._value
            relax_int = pyo.TransformationFactory('core.relax_integer_vars')
            relax_int.apply_to(s)

            assert hasattr(s, "_relaxed_integer_vars")

            # add the needed suffixes / remove later
            s.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            s.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            s.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

            results = ipopt.solve(s)
            pyo.assert_optimal_termination(results)

            kkt_builder = InteriorPointInterface(s)
            kkt_builder.set_barrier_parameter(1e-9)
            kkt_builder.set_bounds_relaxation_factor(1e-8)
            #rhs = kkt_builder.evaluate_primal_dual_kkt_rhs()
            #print(f"{rhs}")
            #print(f"{rhs.flatten()}")
            kkt = kkt_builder.evaluate_primal_dual_kkt_matrix()

            # print(f"{kkt=}")
            # could do better than SuperLU
            kkt_lu = ScipyLU()
            # always regularize equality constraints
            kkt_builder.regularize_equality_gradient(kkt=kkt, coef=-1e-8, copy_kkt=False)
            kkt_lu.do_numeric_factorization(kkt, raise_on_error=True)

            grad_vec = np.zeros(kkt.shape[1])
            grad_vec[0:kkt_builder._nlp.n_primals()] = kkt_builder._nlp.evaluate_grad_objective()

            grad_vec_kkt_inv = kkt_lu._lu.solve(grad_vec, "T")

            for scenario_name in s.scen_list:
                nonant_sensis[scenario_name] = {}
                rho = ph.local_scenarios[scenario_name]._mpisppy_model.rho
                for ndn_i, v in ph.local_scenarios[scenario_name]._mpisppy_data.nonant_indices.items():
                    var_idx = kkt_builder._nlp._vardata_to_idx[v]
 
                    y_vec = np.zeros(kkt.shape[0])
                    y_vec[var_idx] = 1.0

                    x_denom = y_vec.T @ kkt_lu._lu.solve(y_vec)
                    x = (-1 / x_denom)
                    e_x = x * y_vec

                    sensitivity = grad_vec_kkt_inv @ -e_x
                    # print(f"df/d{v.name}: {sensitivity:.2e}, ∂f/∂{v.name}: {grad_vec[var_idx]:.2e}, "
                    #       f"rho {v.name}: {ph.local_scenarios[scenario_name]._mpisppy_model.rho[ndn_i]._value:.2e}, ",
                    #       f"value: {v._value:.2e}"
                    #       )

                    rho[ndn_i]._value = abs(sensitivity)

            relax_int.apply_to(s, options={"undo":True})
            assert not hasattr(s, "_relaxed_integer_vars")
            del s.ipopt_zL_out
            del s.ipopt_zU_out
            del s.dual
            for var, val in solution_cache.items():
                var._value = val

        for s in ph.local_scenarios.values():
            xbars = s._mpisppy_model.xbars
            for ndn_i, rho in s._mpisppy_model.rho.items():
                nv = s._mpisppy_data.nonant_indices[ndn_i]  # var_data object
                rho._value = rho._value / max(1, abs(nv._value - xbars[ndn_i]._value))
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

    def miditer(self):
        pass

    def enditer(self):
        pass

    def post_everything(self):
        pass
