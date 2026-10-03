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

    def register_send_fields(self):
        '''
        Method called by the Hub SPCommunicator to get any fields that the extension
        will make available to spokes.
        '''
        return

    def register_receive_fields(self):
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

    def maybe_checkpoint(self):
        ''' Method called at each point where the algorithm's state is
            coherent enough to be checkpointed: after the solve and after
            every enditer() on the hub, and once per pass through an
            xhatter spoke's main loop plus once more when the spoke
            finalizes.

            Unlike the hooks above, this one is called by the algorithm
            drivers directly rather than being one hook among many, so what
            a checkpoint contains does not depend on the order extensions
            were attached in. It exists because the hub and the xhatter
            spokes have differently shaped loops but need the same
            checkpoint writer; see mpisppy/extensions/checkpointer.py and
            doc/designs/checkpointing_design.md.
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

    #: Set True on an extension that has been looked at and found to keep no
    #: state of its own across a stop and resume -- no history, no counter, no
    #: record of what it has already changed. It is the other way of answering
    #: the question ``checkpoint_state`` answers, and a resumed run names every
    #: attached extension that has answered neither, because an extension that
    #: keeps state and carries none takes a different action at the next
    #: iteration than the uninterrupted run would.
    #:
    #: Deliberately not inherited: the check reads the class's own ``__dict__``,
    #: so a subclass that adds state to a stateless parent is named rather than
    #: quietly covered by its parent's answer. Declare it on each class.
    checkpoint_stateless = False

    def checkpoint_state(self):
        ''' Return this extension's own state as picklable plain data, or
            None if it has none worth carrying across a stop and resume.

            State that lives on a *scenario model* needs nothing here: the
            checkpoint dills the models, so it comes back on its own and
            comes back consistent with the variable values and fixedness it
            pairs with. This hook is for state on the extension *object*,
            which no model carries and which is therefore lost outright
            unless it is written down.

            It matters for any extension whose behavior depends on what it
            did in earlier iterations -- a rho updater comparing against the
            previous xbar, a fixer counting how long a variable has been
            converged, a slammer remembering what it already pinned. Such an
            extension that starts fresh on a resumed run does not merely lose
            a statistic; it takes a different action at the next iteration
            than the uninterrupted run would have, and the two runs diverge
            from there.

            Return plain data (dicts, lists, numbers, strings, tuples), not
            Pyomo objects: a resume replaces every model, so a saved vardata
            addresses something that no longer exists. Key by ``(ndn, i)``
            nonant index or by variable *name*, as the rest of the checkpoint
            code does.

            Called by the Checkpointer at each checkpoint point, which is
            always the end of a completed iteration. See
            doc/designs/checkpointing_design.md section 5.5.
        '''
        return None

    def restore_state(self, state):
        ''' Restore what checkpoint_state() returned, on a resumed run.

            Called once, near the end of Iter0 -- deliberately *after*
            pre_iter0 and post_iter0 have run, because those hooks are where
            extensions rebuild their bookkeeping from the models, and
            restoring first would simply have it overwritten.

            ``state`` is whatever this extension's own checkpoint_state()
            produced; entries are matched to extensions by class name, so an
            extension that was not attached when the checkpoint was written
            is never called.

            Not collective. Every restore runs inside an agreement across
            the cylinder's ranks (checkpointing.run_agreed), which is what
            keeps a restore that fails on one rank from stranding the others
            in the next collective -- and an MPI call in here would be that
            collective, reached by some ranks and not others.
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

    def register_send_fields(self):
        for lobject in self.extdict.values():
            lobject.register_send_fields()

    def register_receive_fields(self):
        for lobject in self.extdict.values():
            lobject.register_receive_fields()

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

    def maybe_checkpoint(self):
        for lobject in self.extdict.values():
            lobject.maybe_checkpoint()

    def post_everything(self):
        for lobject in self.extdict.values():
            lobject.post_everything()


class EFExtension:
    """ Abstract base class for extensions to general EF objects when used with generic_cylinders
    """

    
    def __init__(self, ef_object):
        self.ef = ef_object


    def pre_solve(self):
        """
        called after EF creation, before passing the model to the solver
        """
        pass

    def post_solve(self, results):
        """
        called after solver returns
        """
        return results

    def get_objective_value(self, obj_val, **kwargs):
        """
        for performing anything before returning the objective value (e.g. excluding the value of soft
        penalties)
        """
        return obj_val    

class EFMultiExtension(EFExtension):
    """ Container for all the extension classes we are using, for EF objects when used with generic_cylinders
    """
    def __init__(self, ef_object, ext_classes):
        super().__init__(ef_object)
        self.extdict = dict()

        # Construct multiple extension objects
        for constr in ext_classes:
            name = constr.__name__
            self.extdict[name] = constr(ef_object)
   
    def pre_solve(self):
        """
        called after EF creation, before passing the model to the solver
        """
        for lobject in self.extdict.values():
            lobject.pre_solve()

    def post_solve(self, results):
        """
        called after solver returns
        """
        for lobject in self.extdict.values():
            results = lobject.post_solve(results)
        return results

    def get_objective_value(self, obj_val, **kwargs):
        """
        for performing anything before returning the objective value (e.g. excluding the value of soft
        penalties)
        """
        for lobject in self.extdict.values():
            obj_val = lobject.get_objective_value(obj_val, **kwargs)
        return obj_val
