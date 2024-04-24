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
        self.opt._create_solvers()

    def lagrangian(self):
        verbose = self.opt.options['verbose']
        # This is sort of a hack, but might help folks:
        if "ipopt" in self.opt.options["solver_name"]:
            print("\n WARNING: An ipopt solver will not give outer bounds\n")
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
        return self.opt.Ebound(verbose)

    def _set_weights_and_solve(self):
        self.opt.W_from_flat_list(self.localWs) # Sets the weights
        return self.lagrangian()

    def main(self):
        # The rho_setter should be attached to the opt object
        rho_setter = None
        if hasattr(self.opt, 'rho_setter'):
            rho_setter = self.opt.rho_setter

        self.lagrangian_prep()

        self.dk_iter = 1
        self.trivial_bound = self.lagrangian()

        self.opt.current_solver_options = self.opt.iterk_solver_options

        self.bound = self.trivial_bound

        while not self.got_kill_signal():
            if self.new_Ws:
                bound = self._set_weights_and_solve()
                if bound is not None:
                    self.bound = bound
                self.dk_iter += 1

    def finalize(self):
        self.final_bound = self.bound
        if self.opt.extensions is not None and \
            hasattr(self.opt.extobject, 'post_everything'):
            self.opt.extobject.post_everything()
        return self.final_bound
