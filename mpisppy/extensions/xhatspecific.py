###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Try a particular scenario as xhat. This is mainly for regression testing.
# DLW, Feb 2019
# This extension uses options["xhat_scenario_dict"] (keys are node names)

import mpisppy.extensions.xhatbase

class XhatSpecific(mpisppy.extensions.xhatbase.XhatBase):
    """
    Args:
        spo (SPOpt object): the calling object
        rank (int): mpi process rank of currently running process
    """
    def __init__(self, spo):
        super().__init__(spo)
        self.options = spo.options["xhat_specific_options"]
        self.solver_options = self.options["xhat_solver_options"]
        self.keep_solution = True
        if ('keep_solution' in self.options) and (not self.options['keep_solution']):
            self.keep_solution = False

    #==========
    def xhat_tryit(self,
                   xhat_scenario_dict,
                   verbose=False,
                   restore_nonants=True):
        """If your rank has
        the chosen guy, bcast, if not, recieve the bcast. In any event, fix the vars
        at the bcast values and see if it is feasible. 

        Args:
            xhat_scenario_dict (string): keys are nodes; values are scen names
            verbose (boolean): controls debugging output
        Returns:
            xhatobjective (float or None): the objective function
                or None if one could not be obtained.
        """
        def _vb(msg):
            if verbose and self.cylinder_rank == 0:
                print("  xhat_specific: " + msg)

        obj = None

        _vb("Enter XhatSpecific.xhat_tryit to try: "+str(xhat_scenario_dict))

        _vb("   Solver options="+str(self.solver_options))
        obj = self._try_one(xhat_scenario_dict,
                            solver_options=self.solver_options,
                            verbose=False,
                            restore_nonants=restore_nonants)
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
        xhat_scenario_dict = self.options["xhat_scenario_dict"]
        obj = self.xhat_tryit(xhat_scenario_dict,
                              verbose=self.verbose,
                              restore_nonants=restore_nonants)
        self.opt.reenable_W_and_prox()
        # to make available to tester
        self._xhat_specific_obj_final = obj
        self.xhat_common_post_everything("xhat specified scenario", obj, xhat_scenario_dict, restore_nonants)
