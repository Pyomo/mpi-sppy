###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Indepedent Lagrangian that takes x values as input and
# updates its own W.

import json
import csv
import numpy as np
import mpisppy.cylinders.spoke
from mpisppy.cylinders.lagrangian_bounder import _LagrangianMixin

class LagrangerOuterBound(_LagrangianMixin, mpisppy.cylinders.spoke.OuterBoundNonantSpoke):
    """Indepedent Lagrangian that takes x values as input and updates its own W.
    """
    converger_spoke_char = 'A'

    def lagrangian_prep(self):
        # Scenarios are created here
        super().lagrangian_prep()
        if "lagranger_rho_rescale_factors_json" in self.opt.options and\
            self.opt.options["lagranger_rho_rescale_factors_json"] is not None:
            with open(self.opt.options["lagranger_rho_rescale_factors_json"], "r") as fin:
                din = json.load(fin)
            self.rho_rescale_factors = {int(i): float(din[i]) for i in din}
        else:
            self.rho_rescale_factors = None
        # side-effect is needed: create the nonant_cache
        self.opt._save_nonants()

    def _lagrangian(self, iternum):
        # see if rho should be rescaled
        if self.rho_rescale_factors is not None\
           and iternum in self.rho_rescale_factors:
            self._rescale_rho(self.rho_rescale_factors[iternum])
        return self.lagrangian()

    def _rescale_rho(self,rf):
        # IMPORTANT: the scalings accumulate.
        # E.g., 0.5 then 2.0 gets you back where you started.
        for (sname, scenario) in self.opt.local_scenarios.items():
            for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                scenario._mpisppy_model.rho[ndn_i] *= rf

    def _write_W_and_xbar(self, iternum):
        if self.opt.options.get("lagranger_write_W", False):
            w_fname = 'lagranger_w_vals.csv'
            with open(w_fname, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(['#iteration number', iternum])
            mpisppy.utils.wxbarutils.write_W_to_file(self.opt, w_fname,
                                                     sep_files=False)
        if self.opt.options.get("lagranger_write_xbar", False):
            xbar_fname = 'lagranger_xbar_vals.csv'
            with open(xbar_fname, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(['#iteration number', iternum])
            mpisppy.utils.wxbarutils.write_xbar_to_file(self.opt, xbar_fname)

    def _update_weights_and_solve(self, iternum):
        extensions = self.opt.extensions is not None
        # Work with the nonants that we have (and we might not have any yet).
        # Only update if the nonants are not nan:
        #   otherwise xbar / w will be undefined!
        if not np.isnan(self.localnonants[0]):
            self.opt._put_nonant_cache(self.localnonants)
            self.opt._restore_nonants()
        verbose = self.opt.options["verbose"]
        self.opt.Compute_Xbar(verbose=verbose)
        self.opt.Update_W(verbose=verbose)
        if extensions:
            self.opt.extobject.miditer()
        ## writes Ws here
        self._write_W_and_xbar(iternum)
        return self._lagrangian(iternum)

    def main(self):
        extensions = self.opt.extensions is not None

        self.lagrangian_prep()

        if extensions:
            self.opt.extobject.pre_iter0()
        self.A_iter = 1
        self.trivial_bound = self._lagrangian(0)
        if extensions:
            self.opt.extobject.post_iter0()

        self.bound = self.trivial_bound
        if extensions:
            self.opt.extobject.post_iter0_after_sync()

        self.opt.current_solver_options = self.opt.iterk_solver_options

        while not self.got_kill_signal():
            # because of aph, do not check for new data, just go for it
            bound = self._update_weights_and_solve(self.A_iter)
            if extensions:
                self.opt.extobject.enditer()
            if bound is not None:
                self.bound = bound
            if extensions:
                self.opt.extobject.enditer_after_sync()
            self.A_iter += 1
