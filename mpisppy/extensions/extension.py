###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' A template for creating PH_extension.py files
    NOTE: we pass in the ph object, so extensions can wreck everything
    if they want to!
    Written:  DLW Jan 2019
    Modified: DTM Aug 2019

    NOTE: The return values of all non-constructor methods are ignored
'''

class Extension:
    """ Abstract base class for extensions to general SPOpt/SPCommunicator objects.
    """
    def __init__(self, spopt_object):
        self.opt = spopt_object

    def setup_hub(self):
        '''
        Method called when the Hub SPCommunicator is set up (if used)

        Returns
        -------
        None
        '''
        pass

    def initialize_spoke_indices(self):
        '''
        Method called when the Hub SPCommunicator initializes its spoke indices

        Returns
        -------
        None
        '''
        pass

    def sync_with_spokes(self):
        '''
        Method called when the Hub SPCommunicator syncs with spokes

        Returns
        -------
        None
        '''
        pass

    def pre_solve(self, subproblem):
        '''
        Method called before every subproblem solve

        Inputs
        ------
        subproblem : Pyomo subproblem (could be a scenario or bundle)

        Returns
        -------
        None
        '''
        pass

    def post_solve(self, subproblem, results):
        '''
        Method called after every subproblem solve

        Inputs
        ------
        subproblem : Pyomo subproblem (could be a scenario or bundle)
        results : Pyomo results object from initial solve or None if solve failed

        Returns
        -------
        results : Pyomo results objects from most recent solve
        '''
        return results

    def pre_solve_loop(self):
        ''' Method called before every solve loop within
            mpisppy.spot.SPOpt.solve_loop()
        '''
        pass

    def post_solve_loop(self):
        ''' Method called after every solve loop within
            mpisppy.spot.SPOpt.solve_loop()
        '''
        pass

    def pre_iter0(self):
        ''' When this method is called, all scenarios have been created, and
            the dual/prox terms have been attached to the objective, but the
            solvers have not yet been created.
        '''
        pass

    def iter0_post_solver_creation(self):
        ''' When this method is called, PH iteration 0 has been initiated and 
            all solver objects have been created.
        '''
        pass

    def post_iter0(self):
        ''' Method called after the first PH iteration.
            When this method is called, one call to solve_loop() has been
            completed, and we have ensured that none of the models are
            infeasible. The rho_setter, if present, has not yet been applied.
        '''
        pass

    def post_iter0_after_sync(self):
        ''' Method called after the first PH iteration, after the
            synchronization of sending messages between cylinders
            has completed.
        '''
        pass

    def miditer(self):
        ''' Method called after x-bar has been computed and the dual weights
            have been updated, but before solve_loop().
            If a converger is present, this method is called between the
            convergence_value() method and the is_converged() method.
        '''
        pass

    def enditer(self):
        ''' Method called after the solve_loop(), but before the next x-bar and
            weight update.
        '''
        pass

    def enditer_after_sync(self):
        ''' Method called after the solve_loop(), after the
            synchronization of sending messages between cylinders
            has completed.
        '''
        pass

    def post_everything(self):
        ''' Method called after the termination of the algorithm.
            This method is called after the scenario_denouement, if a
            denouement is present. This function will not begin on any rank
            within self.opt.mpicomm until the scenario_denouement has completed
            on all other ranks.
        '''
        pass


class MultiExtension(Extension):
    """ Container for all the extension classes we are using.
        Also grabs ph and rank, so ad hoc calls (e.g., lagrangian) can use them.
    """
    def __init__(self, ph, ext_classes):
        super().__init__(ph)
        self.extdict = dict()

        # Construct multiple extension objects
        for constr in ext_classes:
            name = constr.__name__
            self.extdict[name] = constr(ph)

    def setup_hub(self):
        for lobject in self.extdict.values():
            lobject.setup_hub()

    def initialize_spoke_indices(self):
        for lobject in self.extdict.values():
            lobject.initialize_spoke_indices()

    def sync_with_spokes(self):
        for lobject in self.extdict.values():
            lobject.sync_with_spokes()

    def pre_solve(self, subproblem):
        for lobject in self.extdict.values():
            lobject.pre_solve(subproblem)

    def post_solve(self, subproblem, results):
        for lobject in self.extdict.values():
            results = lobject.post_solve(subproblem, results)
        return results

    def pre_solve_loop(self):
        for lobject in self.extdict.values():
            lobject.pre_solve_loop()

    def post_solve_loop(self):
        for lobject in self.extdict.values():
            lobject.post_solve_loop()

    def pre_iter0(self):
        for lobject in self.extdict.values():
            lobject.pre_iter0()

    def iter0_post_solver_creation(self):
        for lobject in self.extdict.values():
            lobject.iter0_post_solver_creation()        

    def post_iter0(self):
        for lobject in self.extdict.values():
            lobject.post_iter0()

    def post_iter0_after_sync(self):
        for lobject in self.extdict.values():
            lobject.post_iter0_after_sync()

    def miditer(self):
        for lobject in self.extdict.values():
            lobject.miditer()

    def enditer(self):
        for lobject in self.extdict.values():
            lobject.enditer()

    def enditer_after_sync(self):
        for lobject in self.extdict.values():
            lobject.enditer_after_sync()

    def post_everything(self):
        for lobject in self.extdict.values():
            lobject.post_everything()
