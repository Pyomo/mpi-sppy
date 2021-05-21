# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
''' A template for creating PH_extension.py files
    NOTE: we pass in the ph object, so extensions can wreck everything
    if they want to!
    Written:  DLW Jan 2019
    Modified: DTM Aug 2019

    NOTE: The return values of all non-constructor methods are ignored
'''

class Extension:
    """ Abstract base class for extensions to general SPOpt objects.
    """
    def __init__(self, spopt_object):
        self.opt = spopt_object

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

    def pre_iter0(self):
        ''' Method called at the end of PH_Prep().
            When this method is called, all scenarios have been created, and
            the dual/prox terms have been attached to the objective, but the
            solvers have not yet been created.
        '''
        pass

    def post_iter0(self):
        ''' Method called after the first PH iteration.
            When this method is called, one call to solve_loop() has been
            completed, and we have ensured that none of the models are
            infeasible. The rho_setter, if present, has not yet been applied.
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

    def pre_iter0(self):
        for lobject in self.extdict.values():
            lobject.pre_iter0()
                                        
    def post_iter0(self):
        for lobject in self.extdict.values():
            lobject.post_iter0()

    def miditer(self):
        for lobject in self.extdict.values():
            lobject.miditer()

    def enditer(self):
        for lobject in self.extdict.values():
            lobject.enditer()

    def post_everything(self):
        for lobject in self.extdict.values():
            lobject.post_everything()
