# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import inspect
import collections
import time
import datetime as dt
import logging
import math

import numpy as np
import pyomo.environ as pyo
import mpi4py.MPI as mpi

import mpisppy.utils.sputils as sputils
import mpisppy.utils.listener_util.listener_util as listener_util
import mpisppy.spbase

from pyomo.opt import SolverFactory, SolutionStatus, TerminationCondition
from mpisppy.utils.sputils import find_active_objective
from mpisppy.utils.prox_approx import ProxApproxManager

from mpisppy import global_toc

# decorator snarfed from stack overflow - allows per-rank profile output file generation.
def profile(filename=None, comm=mpi.COMM_WORLD):
    pass

logger = logging.getLogger('PHBase')
logger.setLevel(logging.WARN)

class PHBase(mpisppy.spbase.SPBase):
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
            current_solver_options (dict): from PHoptions, but callbacks might
                Dictionary of solver options provided in PHoptions. Note that
                callbacks could change these options.

        Args:
            PHoptions (dict): 
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
            PH_extensions (object, optional):
                PH extension object.
            PH_extension_kwargs (dict, optional):
                Keyword arguments to pass to the PH_extensions.
            PH_converger (object, optional):
                PH converger object.
            rho_setter (callable, optional):
                Function to set rho values throughout the PH algorithm.
            variable_probability (callable, optional):
                Function to set variable specific probabilities.

    """
    def __init__(
        self,
        PHoptions,
        all_scenario_names,
        scenario_creator,
        scenario_denouement=None,
        all_nodenames=None,
        mpicomm=None,
        scenario_creator_kwargs=None,
        PH_extensions=None,
        PH_extension_kwargs=None,
        PH_converger=None,
        rho_setter=None,
        variable_probability=None,
    ):
        """ PHBase constructor. """
        super().__init__(
            PHoptions,
            all_scenario_names,
            scenario_creator,
            scenario_denouement=scenario_denouement,
            all_nodenames=all_nodenames,
            mpicomm=mpicomm,
            scenario_creator_kwargs=scenario_creator_kwargs,
            variable_probability=variable_probability,
        )

        global_toc("Initializing PHBase")

        # Note that options can be manipulated from outside on-the-fly.
        # self.options (from super) will archive the original options.
        self.PHoptions = PHoptions
        self.options_check()
        self.PH_extensions = PH_extensions
        self.PH_extension_kwargs = PH_extension_kwargs 
        self.PH_converger = PH_converger
        self.rho_setter = rho_setter

        self.iter0_solver_options = PHoptions["iter0_solver_options"]
        self.iterk_solver_options = PHoptions["iterk_solver_options"]
        # flags to complete the invariant
        self.W_disabled = None   # will be set by Prep
        self.prox_disabled = None
        self.convobject = None  # PH converger
        self.attach_xbars()

        if (self.PH_extensions is not None):
            if self.PH_extension_kwargs is None:
                self.extobject = self.PH_extensions(self)
            else:
                self.extobject = self.PH_extensions(
                    self, **self.PH_extension_kwargs
                )

    def Compute_Xbar(self, verbose=False):
        """ Gather xbar and x squared bar for each node in the list and
        distribute the values back to the scenarios.

        Args:
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
        for k,s in self.local_scenarios.items():
            nlens = s._mpisppy_data.nlens        
            for node in s._mpisppy_node_list:
                if node.name not in nodenames:
                    ndn = node.name
                    nodenames.append(ndn)
                    mylen = 2*nlens[ndn]

                    local_concats[ndn] = np.zeros(mylen, dtype='d')
                    global_concats[ndn] = np.zeros(mylen, dtype='d')

        # compute the local xbar and sqbar (put the sq in the 2nd 1/2 of concat)
        for k,s in self.local_scenarios.items():
            nlens = s._mpisppy_data.nlens        
            for node in s._mpisppy_node_list:
                ndn = node.name
                nlen = nlens[ndn]

                xbars = local_concats[ndn][:nlen]
                xsqbars = local_concats[ndn][nlen:]

                nonants_array = np.fromiter( (v._value for v in node.nonant_vardata_list),
                                             dtype='d', count=nlen )
                xbars += s._mpisppy_data.prob_coeff[ndn] * nonants_array
                xsqbars += s._mpisppy_data.prob_coeff[ndn] * nonants_array**2

        # compute node xbar values(reduction)
        for nodename in nodenames:
            self.comms[nodename].Allreduce(
                [local_concats[nodename], mpi.DOUBLE],
                [global_concats[nodename], mpi.DOUBLE],
                op=mpi.SUM)

        # set the xbar and xsqbar in all the scenarios
        for k,s in self.local_scenarios.items():
            logger.debug('  top of assign xbar loop for {} on rank {}'.\
                         format(k, self.cylinder_rank))
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                ndn = node.name
                nlen = nlens[ndn]

                xbars = global_concats[ndn][:nlen]
                xsqbars = global_concats[ndn][nlen:]

                for i in range(nlen):
                    s._mpisppy_model.xbars[(ndn,i)]._value = xbars[i]
                    s._mpisppy_model.xsqbars[(ndn,i)]._value = xsqbars[i]
                    if verbose and self.cylinder_rank == 0:
                        print ("rank, scen, node, var, xbar:",
                               self.cylinder_rank, k, ndn, node.nonant_vardata_list[i].name,
                               pyo.value(s._mpisppy_model.xbars[(ndn,i)]))


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
                    # Requiring a vector for every tree node? (should we?)
                    # if type(s._mpisppy_data.w_coeff[lndn]) is not float:
                    s._mpisppy_model.W[ndn_i] *= s._mpisppy_data.w_coeff[lndn][li]


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

        self.comms["ROOT"].Allreduce(local_diff, global_diff, op=mpi.SUM)

        return global_diff[0] / self.n_proc
           

    def Eobjective(self, verbose=False):
        """ Compute the expected objective function across all scenarios.

        Note: 
            Assumes the optimization is done beforehand,
            therefore DOES NOT CHECK FEASIBILITY or NON-ANTICIPATIVITY!
            This method uses whatever the current value of the objective
            function is.

        Args:
            verbose (boolean, optional):
                If True, displays verbose output. Default False.

        Returns:
            float:
                The expected objective function value
        """
        local_Eobjs = []
        for k,s in self.local_scenarios.items():
            if self.bundling:
                objfct = self.saved_objs[k]
            else:
                objfct = find_active_objective(s)
            local_Eobjs.append(s._mpisppy_probability * pyo.value(objfct))
            if verbose:
                print ("caller", inspect.stack()[1][3])
                print ("E_Obj Scenario {}, prob={}, Obj={}, ObjExpr={}"\
                       .format(k, s._mpisppy_probability, pyo.value(objfct), objfct.expr))

        local_Eobj = np.array([math.fsum(local_Eobjs)])
        global_Eobj = np.zeros(1)
        self.mpicomm.Allreduce(local_Eobj, global_Eobj, op=mpi.SUM)

        return global_Eobj[0]

    def Ebound(self, verbose=False, extra_sum_terms=None):
        """ Compute the expected outer bound across all scenarios.

        Note: 
            Assumes the optimization is done beforehand.
            Uses whatever bound is currently  attached to the subproblems.

        Args:
            verbose (boolean):
                If True, displays verbose output. Default False.
            extra_sum_terms: (None or iterable)
                If iterable, additional terms to put in the floating-point
                sum reduction

        Returns:
            float:
                The expected objective outer bound.
        """
        local_Ebounds = []
        for k,s in self.local_subproblems.items():
            logger.debug("  in loop Ebound k={}, rank={}".format(k, self.cylinder_rank))
            local_Ebounds.append(s._mpisppy_probability * s._mpisppy_data.outer_bound)
            if verbose:
                print ("caller", inspect.stack()[1][3])
                print ("E_Bound Scenario {}, prob={}, bound={}"\
                       .format(k, s._mpisppy_probability, s._mpisppy_data.outer_bound))

        if extra_sum_terms is not None:
            local_Ebound_list = [math.fsum(local_Ebounds)] + list(extra_sum_terms)
        else:
            local_Ebound_list = [math.fsum(local_Ebounds)]

        local_Ebound = np.array(local_Ebound_list)
        global_Ebound = np.zeros(len(local_Ebound_list))
        
        self.mpicomm.Allreduce(local_Ebound, global_Ebound, op=mpi.SUM)

        if extra_sum_terms is None:
            return global_Ebound[0]
        else:
            return global_Ebound[0], global_Ebound[1:]

    def avg_min_max(self, compstr):
        """ Can be used to track convergence progress.

        Args:
            compstr (str): 
                The name of the Pyomo component. Should not be indexed.

        Returns:
            tuple: 
                Tuple containing

                avg (float): 
                    Average across all scenarios.
                min (float):
                    Minimum across all scenarios.
                max (float):
                    Maximum across all scenarios.

        Note:
            Not user-friendly. If you give a bad compstr, it will just crash.
        """
        firsttime = True
        localavg = np.zeros(1, dtype='d')
        localmin = np.zeros(1, dtype='d')
        localmax = np.zeros(1, dtype='d')
        globalavg = np.zeros(1, dtype='d')
        globalmin = np.zeros(1, dtype='d')
        globalmax = np.zeros(1, dtype='d')

        v_cuid = pyo.ComponentUID(compstr)

        for k,s in self.local_scenarios.items():

            compv = pyo.value(v_cuid.find_component_on(s))

            
            ###compv = pyo.value(getattr(s, compstr))
            localavg[0] += s._mpisppy_probability * compv  
            if compv < localmin[0] or firsttime:
                localmin[0] = compv
            if compv > localmax[0] or firsttime:
                localmax[0] = compv
            firsttime = False

        self.comms["ROOT"].Allreduce([localavg, mpi.DOUBLE],
                                     [globalavg, mpi.DOUBLE],
                                     op=mpi.SUM)
        self.comms["ROOT"].Allreduce([localmin, mpi.DOUBLE],
                                     [globalmin, mpi.DOUBLE],
                                     op=mpi.MIN)
        self.comms["ROOT"].Allreduce([localmax, mpi.DOUBLE],
                                     [globalmax, mpi.DOUBLE],
                                     op=mpi.MAX)
        return (float(globalavg[0]),
                float(globalmin[0]),
                float(globalmax[0]))

    def _save_original_nonants(self):
        """ Save the current value of the nonanticipative variables.
            
        Values are saved in the `_PySP_original_nonants` attribute. Whether
        the variable was fixed is stored in `_PySP_original_fixedness`.
        """
        for k,s in self.local_scenarios.items():
            if hasattr(s,"_PySP_original_fixedness"):
                print ("ERROR: Attempt to replace original nonants")
                raise
            if not hasattr(s._mpisppy_data,"nonant_cache"):
                # uses nonant cache to signal other things have not
                # been created 
                # TODO: combine cache creation (or something else)
                clen = len(s._mpisppy_data.nonant_indices)
                s._mpisppy_data.original_fixedness = [None] * clen
                s._mpisppy_data.original_nonants = np.zeros(clen, dtype='d')

            for ci, xvar in enumerate(s._mpisppy_data.nonant_indices.values()):
                s._mpisppy_data.original_fixedness[ci]  = xvar.is_fixed()
                s._mpisppy_data.original_nonants[ci]  = xvar._value

    def _restore_original_nonants(self):
        """ Restore nonanticipative variables to their original values.
            
        This function works in conjunction with _save_original_nonants. 
        
        We loop over the scenarios to restore variables, but loop over
        subproblems to alert persistent solvers.

        Warning: 
            We are counting on Pyomo indices not to change order between save
            and restoration. THIS WILL NOT WORK ON BUNDLES (Feb 2019) but
            hopefully does not need to.
        """
        for k,s in self.local_scenarios.items():

            persistent_solver = None
            if not self.bundling:
                if (sputils.is_persistent(s._solver_plugin)):
                    persistent_solver = s._solver_plugin
            else:
                print("restore_original_nonants called for a bundle")
                raise

            for ci, vardata in enumerate(s._mpisppy_data.nonant_indices.values()):
                vardata._value = s._mpisppy_data.original_nonants[ci]
                vardata.fixed = s._mpisppy_data.original_fixedness[ci]
                if persistent_solver != None:
                    persistent_solver.update_var(vardata)

    def _save_nonants(self):
        """ Save the values and fixedness status of the Vars that are
        subject to non-anticipativity.

        Note:
            Assumes nonant_cache is on the scenarios and can be used
            as a list, or puts it there.
        Warning: 
            We are counting on Pyomo indices not to change order before the
            restoration. We also need the Var type to remain stable.
        Note:
            The value cache is np because it might be transmitted
        """
        for k,s in self.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            if not hasattr(s._mpisppy_data,"nonant_cache"):
                clen = sum(nlens[ndn] for ndn in nlens)
                s._mpisppy_data.nonant_cache = np.zeros(clen, dtype='d')
                s._mpisppy_data.fixedness_cache = [None for _ in range(clen)]

            for ci, xvar in enumerate(s._mpisppy_data.nonant_indices.values()):
                s._mpisppy_data.nonant_cache[ci]  = xvar._value
                s._mpisppy_data.fixedness_cache[ci]  = xvar.is_fixed()

    def _restore_nonants(self):
        """ Restore nonanticipative variables to their original values.
            
        This function works in conjunction with _save_nonants. 
        
        We loop over the scenarios to restore variables, but loop over
        subproblems to alert persistent solvers.

        Warning: 
            We are counting on Pyomo indices not to change order between save
            and restoration. THIS WILL NOT WORK ON BUNDLES (Feb 2019) but
            hopefully does not need to.
        """
        for k,s in self.local_scenarios.items():

            persistent_solver = None
            if (sputils.is_persistent(s._solver_plugin)):
                persistent_solver = s._solver_plugin

            for ci, vardata in enumerate(s._mpisppy_data.nonant_indices.values()):
                vardata._value = s._mpisppy_data.nonant_cache[ci]
                vardata.fixed = s._mpisppy_data.fixedness_cache[ci]

                if persistent_solver is not None:
                    persistent_solver.update_var(vardata)

    def _fix_nonants(self, cache):
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
        """
        for k,s in self.local_scenarios.items():

            persistent_solver = None
            if (sputils.is_persistent(s._solver_plugin)):
                persistent_solver = s._solver_plugin

            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                ndn = node.name
                if ndn not in cache:
                    raise RuntimeError("Could not find {} in {}"\
                                       .format(ndn, cache))
                if cache[ndn] is None:
                    raise RuntimeError("Empty cache for scen={}, node={}".format(k, ndn))
                if len(cache[ndn]) != nlens[ndn]:
                    raise RuntimeError("Needed {} nonant Vars for {}, got {}"\
                                       .format(nlens[ndn], ndn, len(cache[ndn])))
                for i in range(nlens[ndn]): 
                    this_vardata = node.nonant_vardata_list[i]
                    this_vardata._value = cache[ndn][i]
                    this_vardata.fix()
                    if persistent_solver is not None:
                        persistent_solver.update_var(this_vardata)

                            
    def _restore_original_fixedness(self):
        # We are going to hack a little to get the original fixedness, but current values
        # (We are assuming that algorithms are not fixing anticipative vars; but if they
        # do, they had better put their fixedness back to its correct state.)
        self._save_nonants()
        for k,s in self.local_scenarios.items():        
            for ci, _ in enumerate(s._mpisppy_data.nonant_indices):
                s._mpisppy_data.fixedness_cache[ci] = s._mpisppy_data.original_fixedness[ci]
        self._restore_nonants()

        
    def _populate_W_cache(self, cache):
        """ Copy the W values for noants *for all local scenarios*
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
        assert(ci == len(cache) - 1)  # the other cylinder will fail above

    def _put_nonant_cache(self, cache):
        """ Put the value in the cache for noants *for all local scenarios*
        Args:
            cache (np vector) to receive the nonant's for all local scenarios

        """
        ci = 0 # Cache index
        for sname, model in self.local_scenarios.items():
            if model._mpisppy_data.nonant_cache is None:
                raise RuntimeError(f"Rank {self.global_rank} Scenario {sname}"
                                   " nonant_cache is None"
                                   " (call _save_nonants first?)")
            for i,_ in enumerate(model._mpisppy_data.nonant_indices):
                assert(ci < len(cache))
                model._mpisppy_data.nonant_cache[i] = cache[ci]
                ci += 1

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

    def _update_E1(self):
        """ Add up the probabilities of all scenarios using a reduce call.
            then attach it to the PH object as a float.
        """
        localP = np.zeros(1, dtype='d')
        globalP = np.zeros(1, dtype='d')

        for k,s in self.local_scenarios.items():
            localP[0] +=  s._mpisppy_probability

        self.mpicomm.Allreduce([localP, mpi.DOUBLE],
                           [globalP, mpi.DOUBLE],
                           op=mpi.SUM)

        self.E1 = float(globalP[0])

    def feas_prob(self):
        """ Compute the total probability of all feasible scenarios.

        This function can be used to check whether all scenarios are feasible
        by comparing the return value to one.
        
        Note:
            This function assumes the scenarios have a boolean
            `_mpisppy_data.scenario_feasible` attribute.

        Returns:
            float:
                Sum of the scenario probabilities over all feasible scenarios.
                This value equals E1 if all scenarios are feasible.
        """

        # locals[0] is E_feas and locals[1] is E_1
        locals = np.zeros(1, dtype='d')
        globals = np.zeros(1, dtype='d')

        for k,s in self.local_scenarios.items():
            if s._mpisppy_data.scenario_feasible:
                locals[0] += s._mpisppy_probability

        self.mpicomm.Allreduce([locals, mpi.DOUBLE],
                           [globals, mpi.DOUBLE],
                           op=mpi.SUM)

        return float(globals[0])

    def infeas_prob(self):
        """ Sum the total probability for all infeasible scenarios.

        Note:
            This function assumes the scenarios have a boolean
            `_mpisppy_data.scenario_feasible` attribute.

        Returns:
            float:
                Sum of the scenario probabilities over all infeasible scenarios.
                This value equals 0 if all scenarios are feasible.
        """

        locals = np.zeros(1, dtype='d')
        globals = np.zeros(1, dtype='d')

        for k,s in self.local_scenarios.items():
            if not s._mpisppy_data.scenario_feasible:
                locals[0] += s._mpisppy_probability

        self.mpicomm.Allreduce([locals, mpi.DOUBLE],
                           [globals, mpi.DOUBLE],
                           op=mpi.SUM)

        return float(globals[0])

    def _use_rho_setter(self, verbose):
        """ set rho values using a function self.rho_setter
        that gives us a list of (id(vardata), rho)]
        """
        if self.rho_setter is None:
            return
        didit = 0
        skipped = 0
        rho_setter_kwargs = self.PHoptions['rho_setter_kwargs'] \
                            if 'rho_setter_kwargs' in self.PHoptions \
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
        self.prox_disabled = True
        for k, scenario in self.local_scenarios.items():
            for (ndn, i) in scenario._mpisppy_data.nonant_indices:
                scenario._mpisppy_model.prox_on[(ndn,i)]._value = 0

    def _disable_W_and_prox(self):
        self.prox_disabled = True
        self.W_disabled = True
        for k, scenario in self.local_scenarios.items():
            for (ndn, i) in scenario._mpisppy_data.nonant_indices:
                scenario._mpisppy_model.prox_on[(ndn,i)]._value = 0
                scenario._mpisppy_model.w_on[(ndn,i)]._value = 0

    def _disable_W(self):
        # It would be odd to disable W and not prox.
        self.W_disabled = True
        for scenario in self.local_scenarios.values():
            for (ndn, i) in scenario._mpisppy_data.nonant_indices:
                scenario._mpisppy_model.w_on[ndn,i]._value = 0

    def _reenable_prox(self):
        self.prox_disabled = False        
        for k, scenario in self.local_scenarios.items():
            for (ndn, i) in scenario._mpisppy_data.nonant_indices:
                scenario._mpisppy_model.prox_on[(ndn,i)]._value = 1

    def _reenable_W_and_prox(self):
        self.prox_disabled = False
        self.W_disabled = False
        for k, scenario in self.local_scenarios.items():
            for (ndn, i) in scenario._mpisppy_data.nonant_indices:
                scenario._mpisppy_model.prox_on[(ndn,i)]._value = 1
                scenario._mpisppy_model.w_on[(ndn,i)]._value = 1

    def _reenable_W(self):
        self.W_disabled = False
        for k, scenario in self.local_scenarios.items():
            for (ndn, i) in scenario._mpisppy_data.nonant_indices:
                scenario._mpisppy_model.w_on[(ndn,i)]._value = 1

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

    def FormEF(self, scen_dict, EF_name=None):
        """ Make the EF for a list of scenarios. 
        
        This function is mainly to build bundles. To build (and solve) the
        EF of the entire problem, use the EF class instead.

        Args:
            scen_dict (dict): 
                Subset of local_scenarios; the scenarios to put in the EF. THe
                dictionary maps sccneario names (strings) to scenarios (Pyomo
                concrete model objects).
            EF_name (string, optional):
                Name for the resulting EF model.

        Returns:
            :class:`pyomo.environ.ConcreteModel`: 
                The EF with explicit non-anticipativity constraints.

        Raises:
            RuntimeError:
                If the `scen_dict` is empty, or one of the scenarios in
                `scen_dict` is not owned locally (i.e. is not in
                `local_scenarios`).

        Note: 
            We attach a list of the scenario names called _PySP_subsecen_names
        Note:
            We deactivate the objective on the scenarios.
        Note:
            The scenarios are sub-blocks, so they naturally get the EF solution
            Also the EF objective references Vars and Parms on the scenarios
            and hence is automatically updated when the scenario
            objectives are. THIS IS ALL CRITICAL to bundles.
            xxxx TBD: ask JP about objective function transmittal to persistent solvers
        Note:
            Objectives are scaled (normalized) by _mpisppy_probability
        """
        if len(scen_dict) == 0:
            raise RuntimeError("Empty scenario list for EF")

        if len(scen_dict) == 1:
            sname, scenario_instance = list(scen_dict.items())[0]
            if EF_name is not None:
                print ("WARNING: EF_name="+EF_name+" not used; singleton="+sname)
                print ("MAJOR WARNING: a bundle of size one encountered; if you try to compute bounds it might crash (Feb 2019)")
            return scenario_instance

        # The individual scenario instances are sub-blocks of the binding
        # instance. Needed to facilitate bundles + persistent solvers
        if not hasattr(self, "saved_objs"): # First bundle
             self.saved_objs = dict()

        for sname, scenario_instance in scen_dict.items():
            if sname not in self.local_scenarios:
                raise RuntimeError("EF scen not in local_scenarios="+sname)
            self.saved_objs[sname] = find_active_objective(scenario_instance)

        EF_instance = sputils._create_EF_from_scen_dict(scen_dict, EF_name=EF_name,
                        nonant_for_fixed_vars=False)
        return EF_instance

    def solve_one(self, solver_options, k, s,
                  dtiming=False,
                  gripe=False,
                  tee=False,
                  verbose=False,
                  disable_pyomo_signal_handling=False):
        """ Solve one subproblem.

        Args:
            solver_options (dict or None): 
                The scenario solver options.
            k (str): 
                Subproblem name.
            s (ConcreteModel with appendages): 
                The subproblem to solve.
            dtiming (boolean, optional): 
                If True, reports timing values. Default False.
            gripe (boolean, optional):
                If True, outputs a message when a solve fails. Default False.
            tee (boolean, optional):
                If True, displays solver output. Default False.
            verbose (boolean, optional):
                If True, displays verbose output. Default False.
            disable_pyomo_signal_handling (boolean, optional):
                True for asynchronous PH; ignored for persistent solvers.
                Default False.

        Returns:
            float:
                Pyomo solve time in seconds.
        """


        def _vb(msg): 
            if verbose and self.cylinder_rank == 0:
                print ("(rank0) " + msg)
        
        # if using a persistent solver plugin,
        # re-compile the objective due to changed weights and x-bars
        if (sputils.is_persistent(s._solver_plugin)):
            set_objective_start_time = time.time()

            active_objective_datas = list(s.component_data_objects(
                pyo.Objective, active=True, descend_into=True))
            if len(active_objective_datas) > 1:
                raise RuntimeError('Multiple active objectives identified '
                                   'for scenario {sn}'.format(sn=s._name))
            elif len(active_objective_datas) < 1:
                raise RuntimeError('Could not find any active objectives '
                                   'for scenario {sn}'.format(sn=s._name))
            else:
                s._solver_plugin.set_objective(active_objective_datas[0])

            if dtiming:

                set_objective_time = time.time() - set_objective_start_time

                all_set_objective_times = self.mpicomm.gather(set_objective_time,
                                                          root=0)
                if self.cylinder_rank == 0:
                    print("Set objective times (seconds):")
                    print("\tmin=%4.2f mean=%4.2f max=%4.2f" %
                          (np.mean(all_set_objective_times),
                           np.mean(all_set_objective_times),
                           np.max(all_set_objective_times)))

        solve_start_time = time.time()
        if (solver_options):
            _vb("Using sub-problem solver options="
                + str(solver_options))
            for option_key,option_value in solver_options.items():
                s._solver_plugin.options[option_key] = option_value

        solve_keyword_args = dict()
        if self.cylinder_rank == 0:
            if tee is not None and tee is True:
                solve_keyword_args["tee"] = True
        if (sputils.is_persistent(s._solver_plugin)):
            solve_keyword_args["save_results"] = False
        elif disable_pyomo_signal_handling:
            solve_keyword_args["use_signal_handling"] = False

        try:
            results = s._solver_plugin.solve(s,
                                             **solve_keyword_args,
                                             load_solutions=False)
            solver_exception = None
        except Exception as e:
            results = None
            solver_exception = e

        if self.PH_extensions is not None:
            results = self.extobject.post_solve(s, results)

        pyomo_solve_time = time.time() - solve_start_time
        if (results is None) or (len(results.solution) == 0) or \
                (results.solution(0).status == SolutionStatus.infeasible) or \
                (results.solver.termination_condition == TerminationCondition.infeasible) or \
                (results.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded) or \
                (results.solver.termination_condition == TerminationCondition.unbounded):

            s._mpisppy_data.scenario_feasible = False

            if gripe:
                name = self.__class__.__name__
                if self.spcomm:
                    name = self.spcomm.__class__.__name__
                print (f"[{name}] Solve failed for scenario {s.name}")
                if results is not None:
                    print ("status=", results.solver.status)
                    print ("TerminationCondition=",
                           results.solver.termination_condition)

            if solver_exception is not None:
                raise solver_exception

        else:
            if sputils.is_persistent(s._solver_plugin):
                s._solver_plugin.load_vars()
            else:
                s.solutions.load_from(results)
            if self.is_minimizing:
                s._mpisppy_data.outer_bound = results.Problem[0].Lower_bound
            else:
                s._mpisppy_data.outer_bound = results.Problem[0].Upper_bound
            s._mpisppy_data.scenario_feasible = True
        # TBD: get this ready for IPopt (e.g., check feas_prob every time)
        # propogate down
        if self.bundling: # must be a bundle
            for sname in s._ef_scenario_names:
                 self.local_scenarios[sname]._mpisppy_data.scenario_feasible\
                     = s._mpisppy_data.scenario_feasible
        return pyomo_solve_time
    
    
    def solve_loop(self, solver_options=None,
                   use_scenarios_not_subproblems=False,
                   dtiming=False,
                   dis_W=False,
                   dis_prox=False,
                   gripe=False,
                   disable_pyomo_signal_handling=False,
                   tee=False,
                   verbose=False):
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
        """

        """ Developer notes:

        This function assumes that every scenario already has a
        `_solver_plugin` attached.

        I am not sure what happens with solver_options None for a persistent
        solver. Do options persist?

        set_objective takes care of W and prox changes.
        """
        def _vb(msg): 
            if verbose and self.cylinder_rank == 0:
                print ("(rank0) " + msg)
        _vb("Entering solve_loop function.")
        if dis_W and dis_prox:
            self._disable_W_and_prox()
        elif dis_W:
            self._disable_W()
        elif dis_prox:
            self._disable_prox()
        logger.debug("  early solve_loop for rank={}".format(self.cylinder_rank))

        if self._prox_approx and (not self.prox_disabled):
            self._update_prox_approx()
        # note that when there is no bundling, scenarios are subproblems
        if use_scenarios_not_subproblems:
            s_source = self.local_scenarios
        else:
            s_source = self.local_subproblems
        for k,s in s_source.items():
            logger.debug("  in loop solve_loop k={}, rank={}".format(k, self.cylinder_rank))
            if tee:
                print(f"Tee solve for {k} on global rank {self.global_rank}")
            pyomo_solve_time = self.solve_one(solver_options, k, s,
                                              dtiming=dtiming,
                                              verbose=verbose,
                                              tee=tee,
                                              gripe=gripe,
                disable_pyomo_signal_handling=disable_pyomo_signal_handling
            )

        if dtiming:
            all_pyomo_solve_times = self.mpicomm.gather(pyomo_solve_time, root=0)
            if self.cylinder_rank == 0:
                print("Pyomo solve times (seconds):")
                print("\tmin=%4.2f mean=%4.2f max=%4.2f" %
                      (np.min(all_pyomo_solve_times),
                      np.mean(all_pyomo_solve_times),
                      np.max(all_pyomo_solve_times)))

        if dis_W and dis_prox:
            self._reenable_W_and_prox()
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
            scenario._mpisppy_model.w_on = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                        initialize=0.0,
                                        mutable=True)
            self.W_disabled = True
            scenario._mpisppy_model.prox_on = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                        initialize=0.0,
                                        mutable=True)
            self.prox_disabled = True
            # note that rho is per var and scenario here
            scenario._mpisppy_model.rho = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                        mutable=True,
                                        default=self.PHoptions["defaultPHrho"])

    def attach_PH_to_objective(self, add_duals=True, add_prox=False):
        """ Attach dual weight and prox terms to the objective function of the
        models in `local_scenarios`.

        Args:
            add_duals (boolean, optional):
                If True, adds dual weight (Ws) to the objective. Default True.
            add_prox (boolean, optional):
                If True, adds the prox term to the objective. Default True.
        """

        if ('linearize_binary_proximal_terms' in self.PHoptions):
            lin_bin_prox = self.PHoptions['linearize_binary_proximal_terms']
        else:
            lin_bin_prox = False

        if ('linearize_proximal_terms' in self.PHoptions):
            self._prox_approx = self.PHoptions['linearize_proximal_terms']
            if 'proximal_linearization_tolerance' in self.PHoptions:
                self.prox_approx_tol = self.PHoptions['proximal_linearization_tolerance']
            else:
                self.prox_approx_tol = 1.e-1
            if 'initial_proximal_cut_count' in self.PHoptions:
                initial_prox_cuts = self.PHoptions['initial_proximal_cut_count']
            else:
                initial_prox_cuts = 2
        else:
            self._prox_approx = False

        for (sname, scenario) in self.local_scenarios.items():
            """Attach the dual and prox terms to the objective.
            """
            if ((not add_duals) and (not add_prox)):
                return
            objfct = find_active_objective(scenario)
            is_min_problem = objfct.is_minimizing()

            xbars = scenario._mpisppy_model.xbars

            if self._prox_approx:
                # set-up pyomo IndexVar, but keep it sparse
                # since some nonants might be binary
                # Define the first cut to be _xsqvar >= 0
                scenario._mpisppy_model.xsqvar = pyo.Var(scenario._mpisppy_data.nonant_indices, dense=False,
                                            within=pyo.NonNegativeReals)
                scenario._mpisppy_model.xsqvar_cuts = pyo.Constraint(scenario._mpisppy_data.nonant_indices, pyo.Integers)
                scenario._mpisppy_data.xsqvar_prox_approx = {}
            else:
                scenario._mpisppy_model.xsqvar = None
                scenario._mpisppy_data.xsqvar_prox_approx = False

            for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                ph_term = 0
                # Dual term (weights W)
                if (add_duals):
                    ph_term += \
                        scenario._mpisppy_model.w_on[ndn_i] * scenario._mpisppy_model.W[ndn_i] * xvar
                # Prox term (quadratic)
                if (add_prox):
                    # expand (x - xbar)**2 to (x**2 - 2*xbar*x + xbar**2)
                    # x**2 is the only qradratic term, which might be
                    # dealt with differently depending on user-set options
                    if xvar.is_binary() and (lin_bin_prox or self._prox_approx):
                        xvarsqrd = xvar
                    elif self._prox_approx:
                        xvarsqrd = scenario._mpisppy_model.xsqvar[ndn_i]
                        scenario._mpisppy_data.xsqvar_prox_approx[ndn_i] = \
                                ProxApproxManager(xvar, xvarsqrd, scenario._mpisppy_model.xsqvar_cuts, ndn_i, initial_prox_cuts)
                    else:
                        xvarsqrd = xvar**2
                    ph_term += scenario._mpisppy_model.prox_on[ndn_i] * \
                        (scenario._mpisppy_model.rho[ndn_i] / 2.0) * \
                        (xvarsqrd - 2.0 * xbars[ndn_i] * xvar + xbars[ndn_i]**2)
                if (is_min_problem):
                    objfct.expr += ph_term
                else:
                    objfct.expr -= ph_term

    def PH_Prep(
        self, 
        attach_duals=True,
        attach_prox=True,
    ):
        """ Set up PH objectives (duals and prox terms), and prepare
        extensions, if available.

        Args:
            add_duals (boolean, optional):
                If True, adds dual weight (Ws) to the objective. Default True.
            add_prox (boolean, optional):
                If True, adds prox terms to the objective. Default True.

        Note:
            This function constructs an Extension object if one was specified
            at the time the PH object was created. It also calls the
            `pre_iter0` method of the Extension object.
        """

        self.current_solver_options = self.PHoptions["iter0_solver_options"]

        self.attach_Ws_and_prox()
        self.attach_PH_to_objective(add_duals=attach_duals,
                                    add_prox=attach_prox)

        if (self.PH_extensions is not None):
            self.extobject.pre_iter0()

    def options_check(self):
        """ Check whether the options in the `PHoptions` attribute are
        acceptable.

        Required options are

        - solvername (string): The name of the solver to use.
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
            "solvername", "PHIterLimit", "defaultPHrho", 
            "convthresh", "verbose", "display_progress", 
            "iter0_solver_options", "iterk_solver_options"
        ]
        self._options_check(required, self.PHoptions)
        # Display timing and display convergence detail are special for no good reason.
        if "display_timing" not in self.PHoptions:
            self.PHoptions["display_timing"] = False
        if "display_convergence_detail" not in self.PHoptions:
            self.PHoptions["display_convergence_detail"] = False
       

    def subproblem_creation(self, verbose=False):
        """ Create local subproblems (not local scenarios).

        If bundles are specified, this function creates the bundles.
        Otherwise, this function simply copies pointers to the already-created
        `local_scenarios`.

        Args:
            verbose (boolean, optional):
                If True, displays verbose output. Default False.
        """
        self.local_subproblems = dict()
        if self.bundling:
            rank_local = self.cylinder_rank
            for bun in self.names_in_bundles[rank_local]:
                sdict = dict()
                bname = "rank" + str(self.cylinder_rank) + "bundle" + str(bun)
                for sname in self.names_in_bundles[rank_local][bun]:
                    if (verbose and self.cylinder_rank==0):
                        print ("bundling "+sname+" into "+bname)
                    sdict[sname] = self.local_scenarios[sname]
                self.local_subproblems[bname] = self.FormEF(sdict, bname)
                self.local_subproblems[bname].scen_list = \
                    self.names_in_bundles[rank_local][bun]
                self.local_subproblems[bname]._mpisppy_probability = \
                                    sum(s._mpisppy_probability for s in sdict.values())
        else:
            for sname, s in self.local_scenarios.items():
                self.local_subproblems[sname] = s
                self.local_subproblems[sname].scen_list = [sname]

    def _create_solvers(self):

        for sname, s in self.local_subproblems.items(): # solver creation
            s._solver_plugin = SolverFactory(self.PHoptions["solvername"])

            if (sputils.is_persistent(s._solver_plugin)):

                if (self.PHoptions["display_timing"]):
                    set_instance_start_time = time.time()

                # this loop is required to address the sitution where license
                # token servers become temporarily over-subscribed / non-responsive
                # when large numbers of ranks are in use.

                # these parameters should eventually be promoted to a non-PH
                # general class / location. even better, the entire retry
                # logic can be encapsulated in a sputils.py function.
                MAX_ACQUIRE_LICENSE_RETRY_ATTEMPTS = 5
                LICENSE_RETRY_SLEEP_TIME = 2 # in seconds
        
                num_retry_attempts = 0
                while True:
                    try:
                        s._solver_plugin.set_instance(s)
                        if num_retry_attempts > 0:
                            print("Acquired solver license (call to set_instance() for scenario=%s) after %d retry attempts" % (sname, num_retry_attempts))
                        break
                    # pyomo presently has no general way to trap a license acquisition
                    # error - so we're stuck with trapping on "any" exception. not ideal.
                    except:
                        if num_retry_attempts == 0:
                            print("Failed to acquire solver license (call to set_instance() for scenario=%s) after first attempt" % (sname))
                        else:
                            print("Failed to acquire solver license (call to set_instance() for scenario=%s) after %d retry attempts" % (sname, num_retry_attempts))
                        if num_retry_attempts == MAX_ACQUIRE_LICENSE_RETRY_ATTEMPTS:
                            raise RuntimeError("Failed to acquire solver license - call to set_instance() for scenario=%s failed after %d retry attempts" % (sname, num_retry_attempts))
                        else:
                            print("Sleeping for %d seconds before re-attempting" % LICENSE_RETRY_SLEEP_TIME)
                            time.sleep(LICENSE_RETRY_SLEEP_TIME)
                            num_retry_attempts += 1

                if (self.PHoptions["display_timing"]):
                    set_instance_time = time.time() - set_instance_start_time
                    all_set_instance_times = self.mpicomm.gather(set_instance_time,
                                                                 root=0)
                    if self.cylinder_rank == 0:
                        print("Set instance times:")
                        print("\tmin=%4.2f mean=%4.2f max=%4.2f" %
                              (np.min(all_set_instance_times),
                               np.mean(all_set_instance_times),
                               np.max(all_set_instance_times)))

            ## if we have bundling, attach
            ## the solver plugin to the scenarios
            ## as well to avoid some gymnastics
            if self.bundling:
                for scen_name in s.scen_list:
                    scen = self.local_scenarios[scen_name]
                    scen._solver_plugin = s._solver_plugin

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
        
        verbose = self.PHoptions["verbose"]
        dprogress = self.PHoptions["display_progress"]
        dtiming = self.PHoptions["display_timing"]
        dconvergence_detail = self.PHoptions["display_convergence_detail"]        
        have_extensions = self.PH_extensions is not None
        have_converger = self.PH_converger is not None

        def _vb(msg):
            if verbose and self.cylinder_rank == 0:
                print("(rank0)", msg)

        self._PHIter = 0
        self._save_original_nonants()

        global_toc("Creating solvers")
        self._create_solvers()
        
        teeme = ("tee-rank0-solves" in self.PHoptions
                 and self.PHoptions['tee-rank0-solves']
                 and self.cylinder_rank == 0
                 )
            
        if self.PHoptions["verbose"]:
            print ("About to call PH Iter0 solve loop on rank={}".format(self.cylinder_rank))
        global_toc("Entering solve loop in PHBase.Iter0")

        self.solve_loop(solver_options=self.current_solver_options,
                        dtiming=dtiming,
                        gripe=True,
                        tee=teeme,
                        verbose=verbose)
        
        if self.PHoptions["verbose"]:
            print ("PH Iter0 solve loop complete on rank={}".format(self.cylinder_rank))
        
        self._update_E1()  # Apologies for doing this after the solves...
        if (abs(1 - self.E1) > self.E1_tolerance):
            if self.cylinder_rank == 0:
                print("ERROR")
                print("Total probability of scenarios was ", self.E1)
                print("E1_tolerance = ", self.E1_tolerance)
            quit()
        feasP = self.feas_prob()
        if feasP != self.E1:
            if self.cylinder_rank == 0:
                print("ERROR")
                print("Infeasibility detected; E_feas, E1=", feasP, self.E1)
            quit()

        """
        with open('mpi.out-{}'.format(rank), 'w') as fd:
            for sname in self.local_scenario_names:
                fd.write('*** {} ***\n'.format(sname))
        """
        #global_toc('Rank: {} - Building and solving models 0th iteration'.format(rank), True)

        #global_toc('Rank: {} - assigning rho'.format(rank), True)

        if have_extensions:
            self.extobject.post_iter0()

        if self.rho_setter is not None:
            if self.cylinder_rank == 0:
                self._use_rho_setter(verbose)
            else:
                self._use_rho_setter(False)

        converged = False
        if have_converger:
            # Call the constructor of the converger object
            self.convobject = self.PH_converger(self)
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
            self.report_var_values_at_rank0(header="Convergence detail:")            

        self._reenable_W_and_prox()

        self.current_solver_options = self.PHoptions["iterk_solver_options"]

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
        verbose = self.PHoptions["verbose"]
        have_extensions = self.PH_extensions is not None
        have_converger = self.PH_converger is not None
        dprogress = self.PHoptions["display_progress"]
        dtiming = self.PHoptions["display_timing"]
        dconvergence_detail = self.PHoptions["display_convergence_detail"]
        self.conv = None

        max_iterations = int(self.PHoptions["PHIterLimit"])

        for self._PHIter in range(1, max_iterations+1):
            iteration_start_time = time.time()

            if dprogress:
                global_toc(f"\nInitiating PH Iteration {self._PHIter}\n", self.cylinder_rank == 0)

            # Compute xbar
            #global_toc('Rank: {} - Before Compute_Xbar'.format(self.cylinder_rank), True)
            self.Compute_Xbar(verbose)
            #global_toc('Rank: {} - After Compute_Xbar'.format(self.cylinder_rank), True)

            # update the weights        
            self.Update_W(verbose)
            #global_toc('Rank: {} - After Update_W'.format(self.cylinder_rank), True)

            self.conv = self.convergence_diff()
            #global_toc('Rank: {} - After convergence_diff'.format(self.cylinder_rank), True)
            if have_extensions:
                self.extobject.miditer()

            # The hub object takes precedence 
            # over the converger, such that
            # the spokes will always have the
            # latest data, even at termination
            if self.spcomm is not None:
                self.spcomm.sync()
                if self.spcomm.is_converged():
                    global_toc("Cylinder convergence", self.cylinder_rank == 0)
                    break    
            if have_converger:
                if self.convobject.is_converged():
                    converged = True
                    global_toc("User-supplied converger determined termination criterion reached", self.cylinder_rank == 0)
                    break
            elif self.conv is not None:
                if self.conv < self.PHoptions["convthresh"]:
                    converged = True
                    global_toc("Convergence metric=%f dropped below user-supplied threshold=%f" % (self.conv, self.PHoptions["convthresh"]), self.cylinder_rank == 0)
                    break

            teeme = (
                "tee-rank0-solves" in self.PHoptions
                 and self.PHoptions["tee-rank0-solves"]
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

            if dprogress and self.cylinder_rank == 0:
                print("")
                print("After PH Iteration",self._PHIter)
                print("Scaled PHBase Convergence Metric=",self.conv)
                print("Iteration time: %6.2f" % (time.time() - iteration_start_time))
                print("Elapsed time:   %6.2f" % (time.perf_counter() - self.start_time))

            if dconvergence_detail:
                self.report_var_values_at_rank0(header="Convergence detail:")                

            if (self._PHIter == max_iterations):
                global_toc("Reached user-specified limit=%d on number of PH iterations" % max_iterations, self.cylinder_rank == 0)

    def post_loops(self, PH_extensions=None):
        """ Call scenario denouement methods, and report the expected objective
        value.

        Args:
            PH_extensions (object, optional):
                PH extension object.
        Returns:
            float:
                Pretty useless weighted, proxed objective value.
        """
        verbose = self.PHoptions["verbose"]
        have_extensions = PH_extensions is not None
        dprogress = self.PHoptions["display_progress"]
        dtiming = self.PHoptions["display_timing"]

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
