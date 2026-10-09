###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

# A dynamic rho base class that assumes a member function compute_and_update_rho
# As of Nov 2024, it mainly provides services

import numpy as np
import mpisppy.MPI as MPI

import mpisppy.extensions.extension
from mpisppy.utils.w_utils.wtracker import WTracker

# for trapping numpy warnings
import warnings

class Dyn_Rho_extension_base(mpisppy.extensions.extension.Extension):
    """
    This extension enables dynamic rho, e.g. gradient or sensitivity based.
    
    Args:
       opt (PHBase object): gives the problem
       cfg (Config object): config object
    
    """
    def __init__(self, opt, cfg):
        super().__init__(opt)
        self.cylinder_rank = self.opt.cylinder_rank

        self.primal_conv_cache = []
        self.dual_conv_cache = []
        self.wt = WTracker(self.opt)
        self.cfg = cfg

    def _display_rho_values(self):
        for sname, scenario in self.opt.local_scenarios.items():
            rho_list = [scenario._mpisppy_model.rho[ndn_i]._value
                      for ndn_i, _ in scenario._mpisppy_data.nonant_indices.items()]
            print(sname, 'rho values: ', rho_list[:5])
            break

    def _display_W_values(self):
        for (sname, scenario) in self.opt.local_scenarios.items():
            W_list = [w._value for w in scenario._mpisppy_model.W.values()]
            print(sname, 'W values: ', W_list)
            break


    def _update_rho_primal_based(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            # dlw Nov 2024: Is this lagged one iteration (that is not horrible)
            
            curr_conv, last_conv = self.primal_conv_cache[-1], self.primal_conv_cache[-2]
            try:
                primal_diff =  np.abs((last_conv - curr_conv) / last_conv)
            except Warning:
                if self.cylinder_rank == 0:
                    print(f"dyn_rho extension reports {last_conv=} {curr_conv=} - no rho updates recommended")
                return False
            return (primal_diff <= self.cfg.dynamic_rho_primal_thresh)

    def _update_rho_dual_based(self):
        if len(self.dual_conv_cache) < 4:
            # first two entries will be 0 by construction, so ignore
            return False
        # dlw Nov 2024: Is this lagged by one iteration (that is not horrible)
        curr_conv, last_conv = self.dual_conv_cache[-1], self.dual_conv_cache[-2]
        dual_diff =  np.abs((last_conv - curr_conv) / last_conv) if last_conv != 0 else 0
        return (dual_diff <= self.cfg.dynamic_rho_dual_thresh)

    def _update_recommended(self):
        return (hasattr(self.cfg, "dynamic_rho_primal_crit") and self.cfg.dynamic_rho_primal_crit and self._update_rho_primal_based()) or \
               (hasattr(self.cfg, "dynamic_rho_dual_crit") and self.cfg.dynamic_rho_dual_crit and self._update_rho_dual_based())

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
        return Dyn_Rho_extension_base._compute_rho_min_max(ph, np.maximum, MPI.MAX, -np.inf)

    @staticmethod
    def _compute_rho_min(ph):
        return Dyn_Rho_extension_base._compute_rho_min_max(ph, np.minimum, MPI.MIN, np.inf)

    def update_caches(self):
        self.primal_conv_cache.append(self.opt.convergence_diff())
        self.dual_conv_cache.append(self.wt.W_diff())

    def pre_iter0(self):
        pass

    def post_iter0(self):
        raise NotImplementedError

    def miditer(self):
        raise NotImplementedError

    def enditer(self):
        pass

    def checkpoint_state(self):
        """The convergence histories, and the two W snapshots W_diff reads.

        `_update_recommended` decides whether to recompute rho this iteration
        by comparing the last two entries of each cache, so a resumed run with
        empty caches cannot make that decision the same way -- and, for the
        dual criterion, declines to update at all until four iterations have
        accumulated again.

        The `WTracker` entries are not a nicety: without them `--sep-rho`,
        `--sensi-rho` and `--grad-rho` all died with a bare `KeyError` at the
        first iteration after a resume. `W_diff` indexes `local_Ws` at its own
        `ph_iter` and at `ph_iter - 1` -- and note that it reads `ph_iter`
        *before* refreshing it, so those are the two iterations before the one
        it is being called for. It also evaluates `local_Ws[0]` on the way
        through, which only means anything at iteration 0 (it is what makes
        the first difference zero).

        So exactly three entries are carried, and they are the three that
        call will read. Not the whole tracker: `local_Ws` grows by one entry
        per iteration -- scenarios times nonants of floats each -- so saving
        all of it would make the checkpoint grow with the length of the run to
        carry history nothing reads. If a future consumer of this tracker
        needs a window rather than a single step back, this is the line that
        has to grow with it.
        """
        if not self.primal_conv_cache and not self.dual_conv_cache:
            return None
        wanted = {0, self.wt.ph_iter, self.wt.ph_iter - 1}
        local_Ws = {k: v for k, v in self.wt.local_Ws.items() if k in wanted}
        return {
            "primal_conv_cache": list(self.primal_conv_cache),
            "dual_conv_cache": list(self.dual_conv_cache),
            "wtracker_local_Ws": local_Ws,
            "wtracker_ph_iter": self.wt.ph_iter,
        }

    def restore_state(self, state):
        self.primal_conv_cache = list(state["primal_conv_cache"])
        self.dual_conv_cache = list(state["dual_conv_cache"])
        self.wt.local_Ws.update(state["wtracker_local_Ws"])
        self.wt.ph_iter = state["wtracker_ph_iter"]

    def post_everything(self):
        pass
