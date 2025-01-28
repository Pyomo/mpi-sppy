###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import time
import logging
import math

import numpy as np
import mpisppy.MPI as MPI

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.spopt

from mpisppy.utils.prox_approx import ProxApproxManager
from mpisppy import global_toc

# decorator snarfed from stack overflow - allows per-rank profile output file generation.
def profile(filename=None, comm=MPI.COMM_WORLD):
    pass

logger = logging.getLogger('PHBase')
logger.setLevel(logging.WARN)

#======================

def _Compute_Xbar(opt, verbose=False):
    """ Gather xbar and x squared bar for each node in the list and
    distribute the values back to the scenarios.

    Args:
        opt (phbase or xhat_eval object): object with the local scenarios
        verbose (boolean):
            If True, prints verbose output.
    """

    """
    Note:
        Each scenario knows its own probability and its nodes.
    Note:
        The scenario only "sends a reduce" to its own node's comms so even
        though the rank is a member of many comms, the scenario won't
        contribute to the wrong node.
    Note:
        As of March 2019, we concatenate xbar and xsqbar into one long
        vector to make it easier to use the current asynch code.
    """

    nodenames = [] # to transmit to comms
    local_concats = {}   # keys are tree node names
    global_concats =  {} # values are concat of xbar and xsqbar

    # we need to accumulate all local contributions before the reduce
    for k,s in opt.local_scenarios.items():
        nlens = s._mpisppy_data.nlens
        for node in s._mpisppy_node_list:
            if node.name not in nodenames:
                ndn = node.name
                nodenames.append(ndn)
                mylen = 2*nlens[ndn]

                local_concats[ndn] = np.zeros(mylen, dtype='d')
                global_concats[ndn] = np.zeros(mylen, dtype='d')

    # compute the local xbar and sqbar (put the sq in the 2nd 1/2 of concat)
    for k,s in opt.local_scenarios.items():
        nlens = s._mpisppy_data.nlens
        for node in s._mpisppy_node_list:
            ndn = node.name
            nlen = nlens[ndn]

            xbars = local_concats[ndn][:nlen]
            xsqbars = local_concats[ndn][nlen:]

            nonants_array = np.fromiter((v._value for v in node.nonant_vardata_list),
                                        dtype='d', count=nlen)

            probs = s._mpisppy_data.prob_coeff[ndn] * np.ones(nlen)
            xbars += probs * nonants_array
            xsqbars += probs * nonants_array**2

    # compute node xbar values(reduction)
    for nodename in nodenames:
        opt.comms[nodename].Allreduce(
            [local_concats[nodename], MPI.DOUBLE],
            [global_concats[nodename], MPI.DOUBLE],
            op=MPI.SUM)

    # set the xbar and xsqbar in all the scenarios
    for k,s in opt.local_scenarios.items():
        logger.debug('  top of assign xbar loop for {} on rank {}'.\
                     format(k, opt.cylinder_rank))
        nlens = s._mpisppy_data.nlens
        for node in s._mpisppy_node_list:
            ndn = node.name
            nlen = nlens[ndn]

            xbars = global_concats[ndn][:nlen]
            xsqbars = global_concats[ndn][nlen:]

            for i in range(nlen):
                s._mpisppy_model.xbars[(ndn,i)]._value = xbars[i]
                s._mpisppy_model.xsqbars[(ndn,i)]._value = xsqbars[i]
                if verbose: # and opt.cylinder_rank == 0:
                    print ("cylinder rank, scen, node, var, xbar:",
                           opt.cylinder_rank, k, ndn, node.nonant_vardata_list[i].name,
                           pyo.value(s._mpisppy_model.xbars[(ndn,i)]))

def _Compute_Wbar(opt, verbose=False, repair=True):
    """ Seldom used (mainly for diagnostics); gather  Wbar for each node.

    Args:
        opt (phbase or xhat_eval object): object with the local scenarios
        verbose (boolean):
            If True, prints verbose output.
        repair (boolean):
            If True, normalize the W values so EW = 0

    """
    nodenames = [] # to transmit to comms
    local_concats = {}   # keys are tree node names
    global_concats =  {} # values are concat of xbar and xsqbar

    # we need to accumulate all local contributions before the reduce
    for k,s in opt.local_scenarios.items():
        nlens = s._mpisppy_data.nlens
        for node in s._mpisppy_node_list:
            if node.name not in nodenames:
                ndn = node.name
                nodenames.append(ndn)
                mylen = nlens[ndn]

                local_concats[ndn] = np.zeros(mylen, dtype='d')
                global_concats[ndn] = np.zeros(mylen, dtype='d')

    # compute the local Wbar
    for k,s in opt.local_scenarios.items():
        nlens = s._mpisppy_data.nlens
        for node in s._mpisppy_node_list:
            ndn = node.name
            nlen = nlens[ndn]

            Wbars = local_concats[ndn][:nlen]

            # s._mpisppy_data.nonant_indices.keys() indexes the W Param
            Wnonants_array = np.fromiter((pyo.value(s._mpisppy_model.W[idx]) for idx in s._mpisppy_data.nonant_indices if idx[0] == ndn),
                                        dtype='d', count=nlen)
            probs = s._mpisppy_data.prob_coeff[ndn] * np.ones(nlen)
            Wbars += probs * Wnonants_array

    # compute node xbar values(reduction)
    for nodename in nodenames:
        opt.comms[nodename].Allreduce(
            [local_concats[nodename], MPI.DOUBLE],
            [global_concats[nodename], MPI.DOUBLE],
            op=MPI.SUM)

    # check the Wbar
    for k,s in opt.local_scenarios.items():
        logger.debug('  top of Wbar loop for {} on rank {}'.\
                     format(k, opt.cylinder_rank))
        nlens = s._mpisppy_data.nlens
        for node in s._mpisppy_node_list:
            ndn = node.name
            nlen = nlens[ndn]

            Wbars = global_concats[ndn][:nlen]

            for i in range(nlen):
                if abs(Wbars[i]) > opt.E1_tolerance and opt.cylinder_rank == 0:
                    print(f"EW={Wbars[i]} (should be zero) for {node.nonant_vardata_list[i].name}")
                    if repair:
                        print(f"   repairing in {k}")
                        s._mpisppy_model.W[(ndn,i)]._value -= Wbars[i]
 

#======================

class PHBase(mpisppy.spopt.SPOpt):
    """ Base class for all PH-based algorithms.

        Based on mpi4py (but should run with, or without, mpi)
        EVERY INDEX IS ZERO-BASED! (Except stages, which are one based).

        Node names other than ROOT, although strings, must be a number or end
        in a number because mpi4py comms need a number. PH using a smart
        referencemodel that knows how to make its own tree nodes and just wants
        a trailing number in the scenario name. Assume we have only non-leaf
        nodes.

        To check for rank 0 use self.cylinder_rank == 0.

        Attributes:
            local_scenarios (dict):
                Dictionary mapping scenario names (strings) to scenarios (Pyomo
                conrete model objects). These are only the scenarios managed by
                the current rank (not all scenarios in the entire model).
            comms (dict):
                Dictionary mapping node names (strings) to MPI communicator
                objects.
            local_scenario_names (list):
                List of local scenario names (strings). Should match the keys
                of the local_scenarios dict.
            current_solver_options (dict): from options, but callbacks might
                Dictionary of solver options provided in options. Note that
                callbacks could change these options.

        Args:
            options (dict):
                Options for the PH algorithm.
            all_scenario_names (list):
                List of all scenario names in the model (strings).
            scenario_creator (callable):
                Function which take a scenario name (string) and returns a
                Pyomo Concrete model with some things attached.
            scenario_denouement (callable, optional):
                Function which does post-processing and reporting.
            all_nodenames (list, optional):
                List of all node name (strings). Can be `None` for two-stage
                problems.
            mpicomm (MPI comm, optional):
                MPI communicator to use between all scenarios. Default is
                `MPI.COMM_WORLD`.
            scenario_creator_kwargs (dict, optional):
                Keyword arguments passed to `scenario_creator`.
            extensions (object, optional):
                PH extension object.
            extension_kwargs (dict, optional):
                Keyword arguments to pass to the extensions.
            ph_converger (object, optional):
                PH converger object.
            rho_setter (callable, optional):
                Function to set rho values throughout the PH algorithm.
            variable_probability (callable, optional):
                Function to set variable specific probabilities.
            cfg (config object, optional?)  controls (mainly from user)
                (Maybe this should move up to spbase)

    """
    def __init__(
        self,
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement=None,
        all_nodenames=None,
        mpicomm=None,
        scenario_creator_kwargs=None,
        extensions=None,
        extension_kwargs=None,
        ph_converger=None,
        rho_setter=None,
        variable_probability=None,
    ):
        self._PHIter = 0  # moved to _init June 2024 so extensions can see it
        """ PHBase constructor. """
        super().__init__(
            options,
            all_scenario_names,
            scenario_creator,
            scenario_denouement=scenario_denouement,
            all_nodenames=all_nodenames,
            mpicomm=mpicomm,
            extensions=extensions,
            extension_kwargs=extension_kwargs,
            scenario_creator_kwargs=scenario_creator_kwargs,
            variable_probability=variable_probability,
        )
        global_toc("Initializing PHBase")

        # Note that options can be manipulated from outside on-the-fly.
        # self.options (from super) will archive the original options.
        self.options = options
        self.Ag = options.get("Ag", None)  # The Agnostic Object
        
        self.options_check()
        self.ph_converger = ph_converger
        self.rho_setter = rho_setter

        self.iter0_solver_options = options["iter0_solver_options"]
        self.iterk_solver_options = options["iterk_solver_options"]
        self.current_solver_options = self.iter0_solver_options

        # flags to complete the invariant
        self.convobject = None  # PH converger
        self.attach_xbars()


    def Compute_Xbar(self, verbose=False):
        """ Gather xbar and x squared bar for each node in the list and
        distribute the values back to the scenarios.

        Args:
            verbose (boolean):
                If True, prints verbose output.
        """
        _Compute_Xbar(self, verbose=verbose)


    def Update_W(self, verbose):
        """ Update the dual weights during the PH algorithm.

        Args:
            verbose (bool):
                If True, displays verbose output during update.
        """
        # Assumes the scenarios are up to date
        for k,s in self.local_scenarios.items():
            for ndn_i, nonant in s._mpisppy_data.nonant_indices.items():

                ##if nonant._value == None:
                ##    print(f"***_value is None for nonant var {nonant.name}")

                xdiff = nonant._value \
                        - s._mpisppy_model.xbars[ndn_i]._value
                s._mpisppy_model.W[ndn_i]._value += pyo.value(s._mpisppy_model.rho[ndn_i]) * xdiff
                if verbose and self.cylinder_rank == 0:
                    print ("rank, node, scen, var, W", ndn_i[0], k,
                           self.cylinder_rank, nonant.name,
                           pyo.value(s._mpisppy_model.W[ndn_i]))
            # Special code for variable probabilities to mask W; rarely used.
            if s._mpisppy_data.has_variable_probability:
                for ndn_i in s._mpisppy_data.nonant_indices:
                    (lndn, li) = ndn_i
                    s._mpisppy_model.W[ndn_i] *= s._mpisppy_data.prob0_mask[lndn][li]


    def Update_z(self, verbose):
        """ Update the smoothing variable z during the PH algorithm.

        Args:
            verbose (bool):
                If True, displays verbose output during update.
        """
        # Assumes the scenarios are up to date
        for k,s in self.local_scenarios.items():
            for ndn_i, nonant in s._mpisppy_data.nonant_indices.items():

                xzdiff = nonant._value \
                        - s._mpisppy_model.z[ndn_i]._value
                s._mpisppy_model.z[ndn_i]._value += pyo.value(s._mpisppy_model.beta[ndn_i]) * xzdiff
                if verbose and self.cylinder_rank == 0:
                    print ("rank, node, scen, var, z", ndn_i[0], k,
                           self.cylinder_rank, nonant.name,
                           pyo.value(s._mpisppy_model.z[ndn_i]))
    

    def convergence_diff(self):
        """ Compute the convergence metric ||x_s - \\bar{x}||_1 / num_scenarios.

            Returns:
                float:
                    The convergence metric ||x_s - \\bar{x}||_1 / num_scenarios.

        """
        # Every scenario has its own node list, with a vardata list
        global_diff = np.zeros(1)
        local_diff = np.zeros(1)
        varcount = 0
        for k,s in self.local_scenarios.items():
            for ndn_i, nonant in s._mpisppy_data.nonant_indices.items():
                xval = nonant._value
                xdiff = xval - s._mpisppy_model.xbars[ndn_i]._value
                local_diff[0] += abs(xdiff)
                varcount += 1
        local_diff[0] /= varcount

        self.comms["ROOT"].Allreduce(local_diff, global_diff, op=MPI.SUM)

        return global_diff[0] / self.n_proc


    def _populate_W_cache(self, cache, padding):
        """ Copy the W values for nonants *for all local scenarios*
        Args:
            cache (np vector) to receive the W's for all local scenarios (for sending)

        NOTE: This is not the same as the nonant Vars because it puts all local W
              values into the same cache and the cache is *not* attached to the scenario.

        """
        ci = 0 # Cache index
        for model in self.local_scenarios.values():
            if (ci + len(model._mpisppy_data.nonant_indices)) >= len(cache):
                tlen = len(model._mpisppy_data.nonant_indices) * len(self.local_scenarios)
                raise RuntimeError("W cache length mismatch detected by "
                                   f"{self.__class__.__name__} that has "
                                   f"total W len {tlen} but passed cache len-1={len(cache)-1}; "
                                   f"len(nonants)={len(model._mpisppy_data.nonant_indices)}")
            for ix in model._mpisppy_data.nonant_indices:
                cache[ci] = pyo.value(model._mpisppy_model.W[ix])
                ci += 1
        assert(ci == len(cache) - padding)  # the other cylinder will fail above


    def W_from_flat_list(self, flat_list):
        """ Set the dual weight values (Ws) for all local scenarios from a
        flat list.

        Args:
            flat_list (list):
                One-dimensional list of dual weights.

        Warning:
            We are counting on Pyomo indices not to change order between list
            creation and use.
        """
        ci = 0 # Cache index
        for model in self.local_scenarios.values():
            for ndn_i in model._mpisppy_data.nonant_indices:
                model._mpisppy_model.W[ndn_i].value = flat_list[ci]
                ci += 1

    def _use_rho_setter(self, verbose):
        """ set rho values using a function self.rho_setter
        that gives us a list of (id(vardata), rho)]
        """
        if self.rho_setter is None:
            return
        didit = 0
        skipped = 0
        rho_setter_kwargs = self.options['rho_setter_kwargs'] \
                            if 'rho_setter_kwargs' in self.options \
                            else dict()
        for sname, scenario in self.local_scenarios.items():
            rholist = self.rho_setter(scenario, **rho_setter_kwargs)
            for (vid, rho) in rholist:
                (ndn, i) = scenario._mpisppy_data.varid_to_nonant_index[vid]
                scenario._mpisppy_model.rho[(ndn, i)] = rho
            didit += len(rholist)
            skipped += len(scenario._mpisppy_data.varid_to_nonant_index) - didit
        if verbose and self.cylinder_rank == 0:
            print ("rho_setter set",didit,"and skipped",skipped)


    def _disable_prox(self):
        for k, scenario in self.local_scenarios.items():
            scenario._mpisppy_model.prox_on = 0
            if self.Ag is not None:
                self.Ag.callout_agnostic({"scenario": scenario})


    def _disable_W(self):
        # It would be odd to disable W and not prox.
        # TODO: we should eliminate this method
        #       probably not mathematically useful
        for scenario in self.local_scenarios.values():
            scenario._mpisppy_model.W_on = 0
            if self.Ag is not None:
                self.Ag.callout_agnostic({"scenario": scenario})


    def disable_W_and_prox(self):
        self._disable_W()
        self._disable_prox()


    def _reenable_prox(self):
        for k, scenario in self.local_scenarios.items():
            scenario._mpisppy_model.prox_on = 1
            if self.Ag is not None:
                self.Ag.callout_agnostic({"scenario": scenario})


    def _reenable_W(self):
        # TODO: we should eliminate this method
        for k, scenario in self.local_scenarios.items():
            scenario._mpisppy_model.W_on = 1
            if self.Ag is not None:
                self.Ag.callout_agnostic({"scenario": scenario})


    def reenable_W_and_prox(self):
        self._reenable_W()
        self._reenable_prox()


    def post_solve_bound(self, solver_options=None, verbose=False):
        ''' Compute a bound Lagrangian bound using the existing weights.

        Args:
            solver_options (dict, optional):
                Options for these solves.
            verbose (boolean, optional):
                If True, displays verbose output. Default False.

        Returns:
            float:
                An outer bound on the optimal objective function value.

        Note:
            This function overwrites current variable values. This is only
            suitable for use at the end of the solves, or if you really know
            what you are doing.  It is not suitable as a general, per-iteration
            Lagrangian bound solver.
        '''
        if (self.cylinder_rank == 0):
            print('Warning: Lagrangian bounds might not be correct in certain '
                  'cases where there are integers not subject to '
                  'non-anticipativity and those integers do not reach integrality.')
        if (verbose and self.cylinder_rank == 0):
            print('Beginning post-solve Lagrangian bound computation')

        if (self.W_disabled):
            self._reenable_W()
        self._disable_prox()

        # Fixed variables can lead to an invalid lower bound
        self._restore_original_fixedness()

        # If dis_prox=True, they are enabled at the end, and Ebound returns
        # the incorrect value (unless you explicitly disable them again)
        self.solve_loop(solver_options=solver_options,
                        dis_prox=False, # Important
                        gripe=True,
                        tee=False,
                        verbose=verbose)

        bound = self.Ebound(verbose)

        # A half-hearted attempt to restore the state
        self._reenable_prox()

        if (verbose and self.cylinder_rank == 0):
            print(f'Post-solve Lagrangian bound: {bound:.4f}')
        return bound


    def solve_loop(self, solver_options=None,
                   use_scenarios_not_subproblems=False,
                   dtiming=False,
                   dis_W=False,
                   dis_prox=False,
                   gripe=False,
                   disable_pyomo_signal_handling=False,
                   tee=False,
                   verbose=False,
                   need_solution=True):
        """ Loop over `local_subproblems` and solve them in a manner
        dicated by the arguments.

        In addition to changing the Var values in the scenarios, this function
        also updates the `_PySP_feas_indictor` to indicate which scenarios were
        feasible/infeasible.

        Args:
            solver_options (dict, optional):
                The scenario solver options.
            use_scenarios_not_subproblems (boolean, optional):
                If True, solves individual scenario problems, not subproblems.
                This distinction matters when using bundling. Default is False.
            dtiming (boolean, optional):
                If True, reports solve timing information. Default is False.
            dis_W (boolean, optional):
                If True, duals weights (Ws) are disabled before solve, then
                re-enabled after solve. Default is False.
            dis_prox (boolean, optional):
                If True, prox terms are disabled before solve, then
                re-enabled after solve. Default is False.
            gripe (boolean, optional):
                If True, output a message when a solve fails. Default is False.
            disable_pyomo_signal_handling (boolean, optional):
                True for asynchronous PH; ignored for persistent solvers.
                Default False.
            tee (boolean, optional):
                If True, displays solver output. Default False.
            verbose (boolean, optional):
                If True, displays verbose output. Default False.
            need_solution (boolean, optional):
                If True, raises an exception if a solution is not available.
                Default True
        """

        """ Developer notes:

        This function assumes that every scenario already has a
        `_solver_plugin` attached.

        I am not sure what happens with solver_options None for a persistent
        solver. Do options persist?

        set_objective takes care of W and prox changes.
        """
        if dis_W and dis_prox:
            self.disable_W_and_prox()
        elif dis_W:
            self._disable_W()
        elif dis_prox:
            self._disable_prox()

        if self._prox_approx and (not self.prox_disabled):
            self._update_prox_approx()

        super().solve_loop(
            solver_options,
            use_scenarios_not_subproblems,
            dtiming,
            gripe,
            disable_pyomo_signal_handling,
            tee,
            verbose,
            need_solution,
        )

        if dis_W and dis_prox:
            self.reenable_W_and_prox()
        elif dis_W:
            self._reenable_W()
        elif dis_prox:
            self._reenable_prox()


    def _update_prox_approx(self):
        """
        update proximal term approximation by potentially
        adding a linear cut near each current xvar value

        NOTE: This is badly inefficient for bundles, but works
        """
        tol = self.prox_approx_tol
        for sn, s in self.local_scenarios.items():
            persistent_solver = (s._solver_plugin if sputils.is_persistent(s._solver_plugin) else None)
            #print(f"total number of proximal cuts: {len(s._mpisppy_model.xsqvar_cuts)}")
            for prox_approx_manager in s._mpisppy_data.xsqvar_prox_approx.values():
                prox_approx_manager.check_tol_add_cut(tol, persistent_solver)


    def attach_Ws_and_prox(self):
        """ Attach the dual and prox terms to the models in `local_scenarios`.
        """
        for (sname, scenario) in self.local_scenarios.items():
            # these are bound by index to the vardata list at the node
            scenario._mpisppy_model.W = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                        initialize=0.0,
                                        mutable=True)

            # create ph objective terms, but disabled
            scenario._mpisppy_model.W_on = pyo.Param(initialize=0, mutable=True, within=pyo.Binary)

            scenario._mpisppy_model.prox_on = pyo.Param(initialize=0, mutable=True, within=pyo.Binary)

            # note that rho is per var and scenario here
            scenario._mpisppy_model.rho = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                        mutable=True,
                                        default=self.options["defaultPHrho"])
            if self.Ag is not None:
                self.Ag.callout_agnostic({"sname":sname, "scenario":scenario})
            
    
    def attach_smoothing(self):
        """ Attach the smoothing terms to the models in `local_scenarios`.
        """
        for (sname, scenario) in self.local_scenarios.items():
            scenario._mpisppy_model.z = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                        initialize=0.0,
                                        mutable=True)

            scenario._mpisppy_model.p = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                        mutable=True,
                                        default=self.options["defaultPHp"])
            
            scenario._mpisppy_model.beta = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                        mutable=True,
                                        default=self.options["defaultPHbeta"])
    

    @property
    def W_disabled(self):
        assert hasattr(self.local_scenarios[self.local_scenario_names[0]]._mpisppy_model, 'W_on')
        return not bool(self.local_scenarios[self.local_scenario_names[0]]._mpisppy_model.W_on.value)


    @property
    def prox_disabled(self):
        assert hasattr(self.local_scenarios[self.local_scenario_names[0]]._mpisppy_model, 'prox_on')
        return not bool(self.local_scenarios[self.local_scenario_names[0]]._mpisppy_model.prox_on.value)
    

    def attach_PH_to_objective(self, add_duals, add_prox, add_smooth=0):
        """ Attach dual weight and prox terms to the objective function of the
        models in `local_scenarios`.

        Args:
            add_duals (boolean):
                If True, adds dual weight (Ws) to the objective.
            add_prox (boolean):
                If True, adds the prox term to the objective.
        """

        if ('linearize_binary_proximal_terms' in self.options):
            lin_bin_prox = self.options['linearize_binary_proximal_terms']
        else:
            lin_bin_prox = False

        if ('linearize_proximal_terms' in self.options):
            self._prox_approx = self.options['linearize_proximal_terms']
            if 'proximal_linearization_tolerance' in self.options:
                self.prox_approx_tol = self.options['proximal_linearization_tolerance']
            else:
                self.prox_approx_tol = 1.e-1
            # The proximal approximation code now checks the tolerance based on the x-coordinates
            # as opposed to the y-coordinates. Therefore, we will use the square root of the
            # y-coordinate tolerance.
            self.prox_approx_tol = math.sqrt(self.prox_approx_tol)
        else:
            self._prox_approx = False

        for (sname, scenario) in self.local_scenarios.items():
            """Attach the dual and prox terms to the objective.
            """
            if ((not add_duals) and (not add_prox)):
                return
            objfct = self.saved_objectives[sname]
            is_min_problem = objfct.is_minimizing()

            xbars = scenario._mpisppy_model.xbars

            if self._prox_approx:
                # set-up pyomo IndexVar, but keep it sparse
                # since some nonants might be binary
                # Define the first cut to be _xsqvar >= 0
                scenario._mpisppy_model.xsqvar = pyo.Var(scenario._mpisppy_data.nonant_indices, dense=False, bounds=(0, None))
                scenario._mpisppy_model.xsqvar_cuts = pyo.Constraint(scenario._mpisppy_data.nonant_indices, pyo.Integers)
                scenario._mpisppy_data.xsqvar_prox_approx = {}
                try:
                    scenario._mpisppy_data.nonant_cost_coeffs = sputils.nonant_cost_coeffs(scenario)
                except sputils.NonLinearProblemFound:
                    raise RuntimeError("The proximal term approximation can only be used with a linear objective function")
            else:
                scenario._mpisppy_model.xsqvar = None
                scenario._mpisppy_data.xsqvar_prox_approx = False

            ph_term = 0
            # Dual term (weights W)
            if (add_duals):
                scenario._mpisppy_model.WExpr = pyo.Expression(expr=\
                        sum(scenario._mpisppy_model.W[ndn_i] * xvar \
                            for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items()) )
                ph_term += scenario._mpisppy_model.W_on * scenario._mpisppy_model.WExpr

            # Prox term (quadratic)
            if (add_prox):
                prox_expr = 0.
                smooth_expr = 0.
                for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                    # expand (x - xbar)**2 to (x**2 - 2*xbar*x + xbar**2)
                    # x**2 is the only qradratic term, which might be
                    # dealt with differently depending on user-set options
                    if xvar.is_binary() and (lin_bin_prox or self._prox_approx):
                        xvarsqrd = xvar
                    elif self._prox_approx:
                        xvarsqrd = scenario._mpisppy_model.xsqvar[ndn_i]
                        scenario._mpisppy_data.xsqvar_prox_approx[ndn_i] = \
                                ProxApproxManager(scenario, xvar, ndn_i)
                    else:
                        xvarsqrd = xvar**2
                    prox_expr += (scenario._mpisppy_model.rho[ndn_i] / 2.0) * \
                                 (xvarsqrd - 2.0 * xbars[ndn_i] * xvar + xbars[ndn_i]**2)
                    
                    # Computing smoothing term (quadratic)
                    if (add_smooth):
                        smooth_expr += (scenario._mpisppy_model.p[ndn_i] / 2.0) * \
                                    (xvarsqrd - 2.0 * scenario._mpisppy_model.z[ndn_i] * xvar \
                                    + scenario._mpisppy_model.z[ndn_i]**2)

                scenario._mpisppy_model.ProxExpr = pyo.Expression(expr=prox_expr)
                ph_term += scenario._mpisppy_model.prox_on * scenario._mpisppy_model.ProxExpr

                if (add_smooth):
                    # Adding smoothing term
                    scenario._mpisppy_model.SmoothExpr = pyo.Expression(expr=smooth_expr)
                    ph_term += scenario._mpisppy_model.prox_on * scenario._mpisppy_model.SmoothExpr

            if (is_min_problem):
                objfct.expr += ph_term
            else:
                objfct.expr -= ph_term
            
            if self.Ag is not None:
                self.Ag.callout_agnostic({"sname":sname, "scenario":scenario,
                                          "add_duals": add_duals, "add_prox": add_prox})



    def PH_Prep(
        self,
        attach_duals=True,
        attach_prox=True,
        attach_smooth=0
    ):
        """ Set up PH objectives (duals and prox terms), and prepare
        extensions, if available.

        Args:
            add_duals (boolean, optional):
                If True, adds dual weight (Ws) to the objective. Default True.
            add_prox (boolean, optional):
                If True, adds prox terms to the objective. Default True.
            attach_smooth (int, optional):
                If 0, no smoothing; if 1, p_value is used; if 2, p_ratio is used. 

        Note:
            This function constructs an Extension object if one was specified
            at the time the PH object was created. 
        """

        self.attach_Ws_and_prox()
        if attach_smooth:
            self.attach_smoothing()
        self.attach_PH_to_objective(attach_duals, attach_prox, attach_smooth)

    
    def options_check(self):
        """ Check whether the options in the `options` attribute are
        acceptable.

        Required options are

        - solver_name (string): The name of the solver to use.
        - PHIterLimit (int): The maximum number of PH iterations to execute.
        - defaultPHrho (float): The default value of rho (penalty parameter) to
          use for PH.
        - convthresh (float): The convergence tolerance of the PH algorithm.
        - verbose (boolean): Flag indicating whether to display verbose output.
        - display_progress (boolean): Flag indicating whether to display
          information about the progression of the algorithm.
        - iter0_solver_options (dict): Dictionary of solver options to use on
          the first solve loop.
        - iterk_solver_options (dict): Dictionary of solver options to use on
          subsequent solve loops (after iteration 0).

        """
        required = [
            "solver_name", "PHIterLimit", "defaultPHrho",
            "convthresh", "verbose", "display_progress",
        ]
        self._options_check(required, self.options)
        # Display timing and display convergence detail are special for no good reason.
        if "display_timing" not in self.options:
            self.options["display_timing"] = False
        if "display_convergence_detail" not in self.options:
            self.options["display_convergence_detail"] = False
        # Smoothed is optional, not required
        if "smoothed" not in self.options:
            self.options["smoothed"] = 0
        # time_limit is optional, not required
        if "time_limit" not in self.options:
            self.options["time_limit"] = None


    def Iter0(self):
        """ Create solvers and perform the initial PH solve (with no dual
        weights or prox terms).

        This function quits() if the scenario probabilities do not sum to one,
        or if any of the scenario subproblems are infeasible. It also calls the
        `post_iter0` method of any extensions, and uses the rho setter (if
        present) after the inital solve.

        Returns:
            float:
                The so-called "trivial bound", i.e., the objective value of the
                stochastic program with the nonanticipativity constraints
                removed.
        """

        if (self.extensions is not None):
            self.extobject.pre_iter0()
        
        verbose = self.options["verbose"]
        dprogress = self.options["display_progress"]
        dtiming = self.options["display_timing"]
        dconvergence_detail = self.options["display_convergence_detail"]
        smooth_type = self.options["smoothed"]
        have_extensions = self.extensions is not None
        have_converger = self.ph_converger is not None

        def _vb(msg):
            if verbose and self.cylinder_rank == 0:
                print("(rank0)", msg)

        self._PHIter = 0
        self._save_original_nonants()

        global_toc("Creating solvers")
        self._create_solvers()

        if (self.extensions is not None):
            self.extobject.iter0_post_solver_creation()        

        teeme = ("tee-rank0-solves" in self.options
                 and self.options['tee-rank0-solves']
                 and self.cylinder_rank == 0
                 )

        if self.options["verbose"]:
            print ("About to call PH Iter0 solve loop on rank={}".format(self.cylinder_rank))
        global_toc("Entering solve loop in PHBase.Iter0")

        self.solve_loop(solver_options=self.current_solver_options,
                        dtiming=dtiming,
                        gripe=True,
                        tee=teeme,
                        verbose=verbose)

        if self.options["verbose"]:
            print ("PH Iter0 solve loop complete on rank={}".format(self.cylinder_rank))

        self._update_E1()  # Apologies for doing this after the solves...
        if (abs(1 - self.E1) > self.E1_tolerance):
            raise RuntimeError(f"Total probability of scenarios was {self.E1};  E1_tolerance = ", self.E1_tolerance)
        feasP = self.feas_prob()
        if feasP != self.E1:
            raise RuntimeError(f"Infeasibility detected; E_feas={feasP}, E1={self.E1}")

        """
        with open('mpi.out-{}'.format(rank), 'w') as fd:
            for sname in self.local_scenario_names:
                fd.write('*** {} ***\n'.format(sname))
        """
        #global_toc('Rank: {} - Building and solving models 0th iteration'.format(rank), True)

        #global_toc('Rank: {} - assigning rho'.format(rank), True)

        if have_extensions:
            self.extobject.post_iter0()

        if self.spcomm is not None:
            self.spcomm.sync()

        if have_extensions:
            self.extobject.post_iter0_after_sync()

        if self.rho_setter is not None:
            if self.cylinder_rank == 0:
                self._use_rho_setter(verbose)
            else:
                self._use_rho_setter(False)

        ## If ratio: Add reset p according to rho
        if smooth_type == 2:
            for _, scenario in self.local_scenarios.items():
                for ndn_i, _ in scenario._mpisppy_data.nonant_indices.items():
                        scenario._mpisppy_model.p[ndn_i] *= scenario._mpisppy_model.rho[ndn_i] 

        if have_converger:
            # Call the constructor of the converger object
            self.convobject = self.ph_converger(self)
        #global_toc('Rank: {} - Before iter loop'.format(self.cylinder_rank), True)
        self.conv = None

        self.trivial_bound = self.Ebound(verbose)

        if dprogress and self.cylinder_rank == 0:
            print("")
            print("After PH Iteration",self._PHIter)
            print("Trivial bound =", self.trivial_bound)
            print("PHBase Convergence Metric =",self.conv)
            print("Elapsed time: %6.2f" % (time.perf_counter() - self.start_time))

        if dconvergence_detail:
            self.report_var_values_at_rank0(header="Convergence detail:", fixed_vars=False)

        self.reenable_W_and_prox()

        self.current_solver_options = self.options["iterk_solver_options"]

        return self.trivial_bound


    def iterk_loop(self):
        """ Perform all PH iterations after iteration 0.

        This function terminates if any of the following occur:

        1. The maximum number of iterations is reached.
        2. The user specifies a converger, and the `is_converged()` method of
           that converger returns True.
        3. The hub tells it to terminate.
        4. The user does not specify a converger, and the default convergence
           criteria are met (i.e. the convergence value falls below the
           user-specified threshold).

        Args: None

        """
        verbose = self.options["verbose"]
        have_extensions = self.extensions is not None
        have_converger = self.ph_converger is not None
        dprogress = self.options["display_progress"]
        dtiming = self.options["display_timing"]
        dconvergence_detail = self.options["display_convergence_detail"]
        smoothed = bool(self.options["smoothed"])
        self.conv = None

        max_iterations = int(self.options["PHIterLimit"])

        for self._PHIter in range(1, max_iterations+1):
            iteration_start_time = time.time()

            if dprogress:
                global_toc(f"Initiating PH Iteration {self._PHIter}\n", self.cylinder_rank == 0)

            # Compute xbar
            #global_toc('Rank: {} - Before Compute_Xbar'.format(self.cylinder_rank), True)
            self.Compute_Xbar(verbose)
            #global_toc('Rank: {} - After Compute_Xbar'.format(self.cylinder_rank), True)

            # update the weights
            self.Update_W(verbose)
            #global_toc('Rank: {} - After Update_W'.format(self.cylinder_rank), True)

            if smoothed:
                self.Update_z(verbose)
            
            self.conv = self.convergence_diff()
            #global_toc('Rank: {} - After convergence_diff'.format(self.cylinder_rank), True)
            if have_extensions:
                self.extobject.miditer()

            # The hub object takes precedence
            # over the converger, such that
            # the spokes will always have the
            # latest data, even at termination
            if have_converger:
                if self.convobject.is_converged():
                    global_toc("User-supplied converger determined termination criterion reached", self.cylinder_rank == 0)
                    break
            if self.conv is not None:
                if self.conv < self.options["convthresh"]:
                    global_toc("Convergence metric=%f dropped below user-supplied threshold=%f" % (self.conv, self.options["convthresh"]), self.cylinder_rank == 0)
                    break
            if self.options["time_limit"] is not None:
                time_to_stop = self.allreduce_or( (time.perf_counter() - self.start_time) >= self.options["time_limit"] )
                if time_to_stop:
                    global_toc(f"Time limit {self.options['time_limit']} seconds reached.", self.cylinder_rank == 0)
                    break

            teeme = (
                "tee-rank0-solves" in self.options
                 and self.options["tee-rank0-solves"]
                and self.cylinder_rank == 0
            )

            self.solve_loop(
                solver_options=self.current_solver_options,
                dtiming=dtiming,
                gripe=True,
                disable_pyomo_signal_handling=False,
                tee=teeme,
                verbose=verbose
            )

            if have_extensions:
                self.extobject.enditer()

            if self.spcomm is not None:
                self.spcomm.sync()
                if self.spcomm.is_converged():
                    global_toc("Cylinder convergence", self.cylinder_rank == 0)
                    break

            if have_extensions:
                self.extobject.enditer_after_sync()

            if dprogress and self.cylinder_rank == 0:
                print("")
                print("After PH Iteration",self._PHIter)
                print("Scaled PHBase Convergence Metric=",self.conv)
                print("Iteration time: %6.2f" % (time.time() - iteration_start_time))
                print("Elapsed time:   %6.2f" % (time.perf_counter() - self.start_time))

            if dconvergence_detail:
                self.report_var_values_at_rank0(header="Convergence detail:", fixed_vars=False)

        else: # no break, (self._PHIter == max_iterations)
            # NOTE: If we return for any other reason things are reasonably in-sync.
            #       due to the convergence check. However, here we return we'll be
            #       out-of-sync because of the solve_loop could take vasty different
            #       times on different threads. This can especially mess up finalization.
            #       As a guard, we'll put a barrier here.
            self.mpicomm.Barrier()
            global_toc("Reached user-specified limit=%d on number of PH iterations" % max_iterations, self.cylinder_rank == 0)


    def post_loops(self, extensions=None):
        """ Call scenario denouement methods, and report the expected objective
        value.

        Args:
            extensions (object, optional):
                PH extension object.
        Returns:
            float:
                Pretty useless weighted, proxed objective value.
        """
        verbose = self.options["verbose"]
        have_extensions = extensions is not None
        dprogress = self.options["display_progress"]
        dtiming = self.options["display_timing"]

        # for reporting sanity
        self.mpicomm.Barrier()

        if self.cylinder_rank == 0 and dprogress:
            print("")
            print("Invoking scenario reporting functions, if applicable")
            print("")

        if self.scenario_denouement is not None:
            for sname,s in self.local_scenarios.items():
                self.scenario_denouement(self.cylinder_rank, sname, s)

        self.mpicomm.Barrier()

        if self.cylinder_rank == 0 and dprogress:
            print("")
            print("Invoking PH extension finalization, if applicable")
            print("")

        if have_extensions:
            self.extobject.post_everything()

        if self.ph_converger is not None and hasattr(self.ph_converger, 'post_everything'):
            self.convobject.post_everything()

        Eobj = self.Eobjective(verbose)

        self.mpicomm.Barrier()

        if dprogress and self.cylinder_rank == 0:
            print("")
            print("Current ***weighted*** E[objective] =", Eobj)
            print("")

        if dtiming and self.cylinder_rank == 0:
            print("")
            print("Cumulative execution time=%5.2f" % (time.perf_counter()-self.start_time))
            print("")

        return Eobj


    def attach_xbars(self):
        """ Attach xbar and xbar^2 Pyomo parameters to each model in
        `local_scenarios`.
        """
        for scenario in self.local_scenarios.values():
            scenario._mpisppy_model.xbars = pyo.Param(
                scenario._mpisppy_data.nonant_indices.keys(), initialize=0.0, mutable=True
            )
            scenario._mpisppy_model.xsqbars = pyo.Param(
                scenario._mpisppy_data.nonant_indices.keys(), initialize=0.0, mutable=True
            )


if __name__ == "__main__":
    print ("No main for PHBase")
