# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Try a particular scenario as xhat. This is mainly for regression testing.
# DLW, Jan 2023
# IMPORTANT: as of Jan 2023, not tested on multi-stage


import mpisppy.utils.sputils as sputils
import mpisppy.extensions.xhatbase
import mpisppy.phbase as phbase

class XhatXbar(mpisppy.extensions.xhatbase.XhatBase):
    """
    Args:
        spo (SPOpt object): the calling object
    """
    def __init__(self, spo):
        super().__init__(spo)
        self.options = spo.options["xhat_xbar_options"]
        self.solver_options = self.options["xhat_solver_options"]
        self.keep_solution = True
        if ('keep_solution' in self.options) and (not self.options['keep_solution']):
            self.keep_solution = False

    def _fix_nonants_xhat(self):
        """ Fix the Vars subject to non-anticipativity at given values.
            Loop over the scenarios to restore, but loop over subproblems
            to alert persistent solvers.
        Args:
            cache (ndn dict of list or numpy vector): values at which to fix
        WARNING: 
            We are counting on Pyomo indices not to change order between
            when the cache_list is created and used.
        NOTE:
            You probably want to call _save_nonants right before calling this
            copy/pasted from phabse _fix_nonants
        """
        for k,s in self.opt.local_scenarios.items():
            persistent_solver = None
            if (sputils.is_persistent(s._solver_plugin)):
                persistent_solver = s._solver_plugin

            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                ndn = node.name
                for i in range(nlens[ndn]): 
                    this_vardata = node.nonant_vardata_list[i]
                    if this_vardata.is_integer() or this_vardata.is_binary():
                        this_vardata._value = round(s._mpisppy_model.xbars[(ndn,i)]._value)
                    else:
                        this_vardata._value = s._mpisppy_model.xbars[(ndn,i)]._value

                    this_vardata.fix()
                    if persistent_solver is not None:
                        persistent_solver.update_var(this_vardata)

            
    #==========
    def xhat_tryit(self,
                   verbose=False,
                   restore_nonants=True):
        """Use xbar to set the nonants; round integers

        Args:
            verbose (boolean): controls debugging output
            restore_nonants (boolean): put back the nonants
        Returns:
            xhatobjective (float or None): the objective function
                or None if one could not be obtained.
        """
        def _vb(msg):
            if verbose and self.cylinder_rank == 0:
                print("  xhat_xbar: " + msg)

        obj = None
        sname = None

        _vb("Enter XhatXbar.xhat_tryit")

        _vb("   Solver options="+str(self.solver_options))

        # This might be an extension for a Xhat_Eval object or a PH object, so we will assume the worst
        phbase._Compute_Xbar(self.opt)
        self._fix_nonants_xhat() # (BTW: for all local scenarios)

        # NOTE: for APH we may need disable_pyomo_signal_handling
        self.opt.solve_loop(solver_options=self.solver_options,
                           #dis_W=True, dis_prox=True,
                           verbose=verbose,
                            tee=False)

        infeasP = self.opt.infeas_prob()
        if infeasP != 0.:
            # restoring does no harm
            # if this solution is infeasible
            self.opt._restore_nonants()
            return None
        else:
            if verbose and src_rank == self.cylinder_rank:
                print("   Feasible xhat found at xbar")
            obj = self.opt.Eobjective(verbose=verbose)
            if restore_nonants:
                self._restore_nonants()

        if obj is None:
            _vb("Infeasible")
        else:
            _vb("Feasible, returning " + str(obj))

        return obj

    def pre_iter0(self):
        pass

    def post_iter0(self):
        # a little bit silly
        self.comms = self.opt.comms
        
    def miditer(self):
        pass

    def enditer(self):
        pass

    def post_everything(self):
        # if we're keeping the solution, we *do not* restore the nonants
        restore_nonants = not self.keep_solution
        self.opt.disable_W_and_prox()
        obj = self.xhat_tryit(verbose=self.verbose,
                              restore_nonants=restore_nonants)
        self.opt.reenable_W_and_prox()
        # to make available to tester
        self._xhat_xbar_obj_final = obj
