# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Indepedent Lagrangian that takes x values as input and
# updates its own W.

import time
import mpisppy.cylinders.spoke

class LagrangerOuterBound(mpisppy.cylinders.spoke.OuterBoundNonantSpoke):

    converger_spoke_char = 'A'

    def lagrangian_prep(self):
        verbose = self.opt.options['verbose']
        # Scenarios are created here
        self.opt.PH_Prep(attach_prox=False)
        self.opt._reenable_W()
        self.opt.subproblem_creation(verbose)
        self.opt._create_solvers()
        if "lagranger_rho_rescale_factors_json" in self.opt.options and\
            self.opt.options["lagranger_rho_rescale_factors_json"] is not None:
            with open(args.lagranger_rho_rescale_factors_json, "r") as fin:
                din = json.load(fin)
            self.rho_rescale_factors = {int(i): din[i] for i in din}
        else:
            self.rho_rescale_factors = None
        # side-effect is needed: create the nonant_cache
        self.opt._save_nonants()

    def _lagrangian(self, iternum):
        verbose = self.opt.options['verbose']
        # see if rho should be rescaled
        if self.rho_rescale_factors is not None\
           and iternum in self.rho_rescale_factors:
            _rescale_rho(self.rho_rescale_factors[iternum])
        teeme = False
        if "tee-rank0-solves" in self.opt.options and self.opt.rank == self.opt.rank0:
            teeme = self.opt.options['tee-rank0-solves']

        self.opt.solve_loop(
            solver_options=self.opt.current_solver_options,
            dtiming=False,
            gripe=True,
            tee=teeme,
            verbose=verbose
        )

        # Compute the resulting bound
        return self.opt.Ebound(verbose)


    def _rescale_rho(rf):
        # IMPORTANT: the scalings accumulate.
        # E.g., 0.5 then 2.0 gets you back where you started.
        for (sname, scenario) in self.local_scenarios.items():
            for ndn_i, xvar in scenario._nonant_indexes.items():
                scenario._PHrho[ndn_i] *= rf
        
    
    def _update_weights_and_solve(self, iternum):
        # Work with the nonants that we have (and we might not have any yet).
        self.opt._put_nonant_cache(self.localnonants)
        self.opt._restore_nonants()
        verbose = self.opt.options["verbose"]
        self.opt.Compute_Xbar(verbose=verbose)
        self.opt.Update_W(verbose=verbose)
        return self._lagrangian(iternum)

    def main(self):
        # The rho_setter should be attached to the opt object
        rho_setter = None
        if hasattr(self.opt, 'rho_setter'):
            rho_setter = self.opt.rho_setter

        self.lagrangian_prep()

        self.trivial_bound = self._lagrangian(0)

        self.bound = self.trivial_bound

        self.A_iter = 1
        while not self.got_kill_signal():
            # because of aph, do not check for new data, just go for it
            self.bound = self._update_weights_and_solve(self.A_iter)
            self.A_iter += 1

    def finalize(self):
        '''
        Do one final lagrangian pass with the final
        PH weights. Useful for when PH convergence
        and/or iteration limit is the cause of termination
        '''
        self.final_bound = self._update_weights_and_solve(self.A_iter)
        self.bound = self.final_bound
        return self.final_bound
