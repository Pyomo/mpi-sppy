# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import mpisppy.cylinders.spoke

class LagrangianOuterBound(mpisppy.cylinders.spoke.OuterBoundWSpoke):

    converger_spoke_char = 'L'

    def lagrangian_prep(self):
        verbose = self.opt.options['verbose']
        # Split up PH_Prep? Prox option is important for APH.
        # Seems like we shouldn't need the Lagrangian stuff, so attach_prox=False
        # Scenarios are created here
        self.opt.PH_Prep(attach_prox=False)
        self.opt._reenable_W()
        self.opt.subproblem_creation(verbose)
        self.opt._create_solvers()

    # This signature changed April 2020 (delete these two comments in Aug 2020)
    #def lagrangian(self, PH_extensions=None, PH_converger=None, rho_setter=None):
    def lagrangian(self):
        verbose = self.opt.options['verbose']
        teeme = False
        if "tee-rank0-solves" in self.opt.options:
            teeme = self.opt.options['tee-rank0-solves']

        self.opt.solve_loop(
            solver_options=self.opt.current_solver_options,
            dtiming=False,
            gripe=True,
            tee=teeme,
            verbose=verbose
        )
        ''' DTM (dlw edits): This is where PHBase Iter0 checks for scenario
            probabilities that don't sum to one and infeasibility and
            will send a kill signal if needed. For now we are relying
            on the fact that the OPT thread is solving the same
            models, and hence would detect both of those things on its
            own--the Lagrangian spoke doesn't need to check again.  '''

        # Compute the resulting bound, checking to be sure
        # the weights came from the same PH iteration
        serial_number = self.get_serial_number()
        bound, extra_sums  = self.opt.Ebound(verbose, extra_sum_terms=[serial_number])

        serial_number_sum = int(round(extra_sums[0]))

        total = int(self.intracomm.Get_size())*serial_number
        if total == serial_number_sum:
            return bound
        elif self.rank_intra == 0:
            print("WARNING: Lagrangian spokes out of snyc, consider changing mpisppy.cylinders.SPOKE_SLEEP_TIME")
        return None

    def _set_weights_and_solve(self):
        self.opt.W_from_flat_list(self.localWs) # Sets the weights
        return self.lagrangian()

    def main(self):
        # The rho_setter should be attached to the opt object
        rho_setter = None
        if hasattr(self.opt, 'rho_setter'):
            rho_setter = self.opt.rho_setter

        self.lagrangian_prep()

        self.trivial_bound = self.lagrangian()

        self.bound = self.trivial_bound

        dk_iter = 1
        while not self.got_kill_signal():
            if self.new_Ws:
                bound = self._set_weights_and_solve()
                if bound is not None:
                    self.bound = bound
            dk_iter += 1

    def finalize(self):
        '''
        Do one final lagrangian pass with the final
        PH weights. Useful for when PH convergence
        and/or iteration limit is the cause of termination
        '''
        self.final_bound = self._set_weights_and_solve()
        self.bound = self.final_bound
        return self.final_bound
