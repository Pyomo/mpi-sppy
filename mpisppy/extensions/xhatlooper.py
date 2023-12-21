# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# look for an xhat. 
# Written to be the only extension or called from an extension "manager."
# DLW, Jan 2019

import mpisppy.extensions.xhatbase

class XhatLooper(mpisppy.extensions.xhatbase.XhatBase):
    """
    Args:
        opt (SPBase object): problem that we are bounding
        rank (int): mpi process rank of currently running process
    """
    def __init__(self, ph):
        super().__init__(ph)
        self.options = ph.options["xhat_looper_options"]
        self.solver_options = self.options["xhat_solver_options"]
        self._xhat_looper_obj_final = None
        self.keep_solution = True
        if ('keep_solution' in self.options) and (not self.options['keep_solution']):
            self.keep_solution = False

    #==========
    def xhat_looper(self,
                    scen_limit=1,
                    seed=None,
                    verbose=False,
                    restore_nonants=True):
        """Loop over some number of the global scenarios; if your rank has
        the chosen guy, bcast, if not, recieve the bcast. In any event, fix the vars
        at the bcast values and see if it is feasible. If so, stop and 
        leave the nonants fixed.

        Args:
            scen_limit (int): number of scenarios to try
            seed (int): if none, loop starting at first scen; o.w. randomize
            verbose (boolean): controls debugging output
            restore_nonants (bool): if True, restores the nonants to their original
                                    values in all scenarios. If False, leaves the
                                    nonants as they are in the tried scenario
                                    You want true; False would be a bug!
        Returns:
            xhojbective (float or None), sname (string): the objective function
                or None if one could not be obtained.
        NOTE:
            If options has an append_file_name, write to it
            Also attach the resulting bound to the object
        """
        def _vb(msg):
            if verbose and self.cylinder_rank == 0:
                print ("    rank {} xhat_looper: {}".\
                       format(self.cylinder_rank,msg))
        obj = None
        sname = None
        snumlists = dict()
        llim = min(scen_limit, len(self.opt.all_scenario_names))
        _vb("Enter xhat_looper to try "+str(llim)+" scenarios.")
        # The tedious task of collecting the tree information for
        # local scenario tree nodes (maybe move to the constructor)
        for k, s in self.opt.local_scenarios.items():
            for nnode in s._mpisppy_node_list:
                ndn = nnode.name
                nsize = self.comms[ndn].size
                if seed is None:
                    snumlists[ndn] = [i % nsize for i in range(llim)]
                else:
                    print ("need a random permutation in snumlist xxxx quitting")
                    quit()
        
        self.opt._save_nonants() # to cache for use in fixing
        # for the moment (dec 2019) treat two-stage as special
        if len(snumlists) == 1:
            for snum in snumlists["ROOT"]:
                sname = self.opt.all_scenario_names[snum]
                _vb("Trying scenario "+sname)
                _vb("   Solver options="+str(self.solver_options))
                snamedict = {"ROOT": sname}
                obj = self._try_one(snamedict,
                                    solver_options=self.solver_options,
                                    verbose=False,
                                    restore_nonants=restore_nonants)
                if obj is None:
                    _vb("    Infeasible")
                else:
                    _vb("    Feasible, returning " + str(obj))
                    break
        else:
            raise RuntimeError("xhatlooper cannot do multi-stage")            

        if "append_file_name" in self.options and self.opt.cylinder_rank == 0:
            with open(self.options["append_file_name"], "a") as f:
                f.write(", "+str(obj))

        self.xhatlooper_obj = obj
        return obj, snamedict

    def pre_iter0(self):
        if self.opt.multistage:
            raise RuntimeError("xhatlooper cannot do multi-stage")            

    def post_iter0(self):
        # a little bit silly
        self.comms = self.opt.comms
        
    def miditer(self):
        pass

    def enditer(self):
        pass

    def post_everything(self):
        restore_nonants = not self.keep_solution

        self.opt.disable_W_and_prox()
        obj, snamedict = self.xhat_looper(
            scen_limit=self.options["scen_limit"],
            verbose=self.verbose,
            restore_nonants=restore_nonants,
        )
        self.opt.reenable_W_and_prox()
        # "secret menu" way to see the value in a script
        self._xhat_looper_obj_final = obj
        self.xhat_common_post_everything("xhatlooper", obj, snamedict, restore_nonants)
