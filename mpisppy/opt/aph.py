###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# APH

import numpy as np
import math
import collections
import time
import logging
import mpisppy.MPI as mpi
import pyomo.environ as pyo
import mpisppy.utils.listener_util.listener_util as listener_util
import mpisppy.phbase as ph_base
import mpisppy.utils.sputils as sputils

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

logging.basicConfig(level=logging.CRITICAL, # level=logging.CRITICAL, DEBUG
            format='(%(threadName)-10s) %(message)s',
            )

EPSILON = 1e-5  # for, e.g., fractions of ranks

"""
Delete this comment block; dlw May 2019
- deal with "waiting out" a negative tau
"""

""" APH started by DLW, March 2019.
Based on "Algorithm 2: Asynchronous projective hedging (APH) -- Algorithm 1
specialize to the setup S1-S4" from 
"Asynchronous Projective Hedging for Stochastic Programming" 
http://www.optimization-online.org/DB_HTML/2018/10/6895.html
(note: there are therefore some notation changes from PySP1)
Note: we deviate from the paper's notation in the use of i and k
(i is used here as an arbitrary index, usually into the nonants at a node and
k is often used as the "key" (i.e., scenario name) for the local scenarios)
"""

class APH(ph_base.PHBase):
    """
    Args:
        options (dict): PH options
        all_scenario_names (list): all scenario names
        scenario_creator (fct): returns a concrete model with special things
        scenario_denouement (fct): for post processing and reporting
        all_node_names (list of str): non-leaf node names
        scenario_creator_kwargs (dict): keyword arguments passed to
            `scenario_creator`.

    Attributes (partial list):
        local_scenarios (dict of scenario objects): concrete models with 
              extra data, key is name
        comms (dict): keys are node names values are comm objects.
        scenario_name_to_rank (dict): all scenario names
        local_scenario_names (list): names of locals 
        current_solver_options (dict): from options; callbacks might change
        synchronizer (object): asynch listener management
        scenario_creator_kwargs (dict): keyword arguments passed to
            `scenario_creator`.

    """
    def setup_Lens(self):
        """ We need to know the lengths of c-style vectors for listener_util
        """
        self.Lens = collections.OrderedDict({"FirstReduce": {},
                                            "SecondReduce": {}})

        for sname, scenario in self.local_scenarios.items():
            for node in scenario._mpisppy_node_list:
                self.Lens["FirstReduce"][node.name] \
                    = 3 * len(node.nonant_vardata_list)
                self.Lens["SecondReduce"][node.name] = 0 # only use root?
        self.Lens["FirstReduce"]["ROOT"] += self.n_proc  # for time of update
        # tau, phi, pusqnorm, pvsqnorm, pwsqnorm, pzsqnorm, secs
        self.Lens["SecondReduce"]["ROOT"] += 6 + self.n_proc
        

    #============================
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
        super().__init__(
            options,
            all_scenario_names,
            scenario_creator,
            scenario_denouement,
            mpicomm=mpicomm,
            all_nodenames=all_nodenames,
            scenario_creator_kwargs=scenario_creator_kwargs,
            extensions=extensions,
            extension_kwargs=extension_kwargs,
            ph_converger=ph_converger,
            rho_setter=rho_setter,
            variable_probability=variable_probability,
        )

        self.phis = {} # phi values, indexed by scenario names
        self.tau_summand = 0  # place holder for iteration 1 reduce
        self.phi_summand = 0
        self.global_tau = 0
        self.global_phi = 0
        self.global_pusqnorm = 0 # ... may be out of date...
        self.global_pvsqnorm = 0
        self.global_pwsqnorm = 0
        self.global_pzsqnorm = 0
        self.local_pwsqnorm = 0
        self.local_pzsqnorm = 0
        self.conv = None
        self.use_lag = options.get("APHuse_lag", False)
        self.APHgamma = options.get("APHgamma", 1)
        assert(self.APHgamma > 0)
        self.use_dynamic_gamma = options.get("use_dynamic_gamma", False)
        if self.use_dynamic_gamma:
            print('**** dynamic gamma is True so watch out!')
        self.shelf_life = options.get("shelf_life", 99)  # 99 is intended to be large
        self.round_robin_dispatch = options.get("round_robin_dispatch", False)
        # TBD: use a property decorator for nu to enforce 0 < nu < 2

        # ESR June, 2023
        self.nu = options.get("APHnu", 1)
        # Note June, 2023: Hack for nu
        self.use_hack_for_nu = options.get("use_hack_for_nu", False)
        if self.use_hack_for_nu:
            print('**** you are using the hack for nu so be careful!')

        assert 0 < self.nu and self.nu < 2
        self.dispatchrecord = dict()   # for local subproblems sname: (iter, phi)
        # plot_trace_prefix or filename will indicate output is needed
        self.plot_trace_prefix = options.get("APHplot_trace_prefix") if self.cylinder_rank == 0 else None
        self.conv_trace_filename = None if self.plot_trace_prefix is None else f"{self.plot_trace_prefix}_dyngam_{self.use_dynamic_gamma}"\
            +f"_hack_nu_{self.use_hack_for_nu}_nu_{self.nu}"\
            +".csv"
        if self.plot_trace_prefix is not None:
            with open(self.conv_trace_filename, "w") as fil:
                fil.write("iter,conv,gamma,nu,theta,punorm,pvnorm\n")
            for k,s in self.local_scenarios.items():
                with open(f"trace_{k}_{self.conv_trace_filename}", "w") as fil:
                    fil.write("iter,obj fct")
                    for (ndn,i), xvar in s._mpisppy_data.nonant_indices.items():
                        fil.write(f",{xvar.name} x,{xvar.name} z, {xvar.name} w")
                    fil.write("\n")

    #============================
    def setup_dispatchrecord(self):
        # Start with a small number for iteration to randomize fist dispatch.
        for sname in self.local_subproblems:
            r = np.random.rand()
            self.dispatchrecord[sname] = [(r,0)]


    #============================
    def Update_y(self, dlist, verbose):
        # compute the new y (or set to zero if it is iter 1)
        # iter 1 is iter 0 post-solves when seen from the paper
        # dlist is used only after iter0 (it has the dispatched scen names)

        slist = [d[0] for d in dlist]  # just the names
        if self._PHIter != 1:
            for k,s in self.local_scenarios.items():
                if (not self.bundling and k in slist) \
                   or (self.bundling and s._mpisppy_data.bundlename in slist):
                    for (ndn,i), xvar in s._mpisppy_data.nonant_indices.items():
                        if not self.use_lag:
                            z_touse = s._mpisppy_model.z[(ndn,i)]._value
                            W_touse = pyo.value(s._mpisppy_model.W[(ndn,i)])
                        else:
                            z_touse = s._mpisppy_model.z_foropt[(ndn,i)]._value
                            W_touse = pyo.value(s._mpisppy_model.W_foropt[(ndn,i)])
                        # pyo.value vs. _value ??
                        # NOTE: W_touse and z_touse are coming from
                        # the previous iteration
                        s._mpisppy_model.y[(ndn,i)]._value = W_touse \
                                              + pyo.value(s._mpisppy_model.rho[(ndn,i)]) \
                                              * (xvar._value - z_touse) #Eq.25
                        if verbose and self.cylinder_rank == 0:
                            print ("node, scen, var, y", ndn, k,
                                   self.cylinder_rank, xvar.name,
                                   pyo.value(s._mpisppy_model.y[(ndn,i)]))
                        # Special code for variable probabilities to mask y; rarely used.
                        if s._mpisppy_data.has_variable_probability:
                            s._mpisppy_model.y[(ndn,i)]._value *= s._mpisppy_data.prob0_mask[ndn][i]
        else:
            for k,s in self.local_scenarios.items():
                for (ndn,i), xvar in s._mpisppy_data.nonant_indices.items():
                    s._mpisppy_model.y[(ndn,i)]._value = 0
            if verbose and self.cylinder_rank == 0:
                print ("All y=0 for iter1")



    #============================
    def compute_phis_summand(self):
        # update phis, return summand (variable_probability is already resolved)
        summand = 0.0
        for k,s in self.local_scenarios.items():
            self.phis[k] = 0.0
            for (ndn,i), xvar in s._mpisppy_data.nonant_indices.items():
                # Step 16, phi
                self.phis[k] += (pyo.value(s._mpisppy_model.z[(ndn,i)]) - xvar._value) \
                    *(pyo.value(s._mpisppy_model.W[(ndn,i)]) - pyo.value(s._mpisppy_model.y[(ndn,i)]))
            self.phis[k] *= pyo.value(s._mpisppy_probability)
            summand += self.phis[k]
        return summand

    #============================***********=========

    def _calculate_APHgamma(self, synchro):
        """This function calculates a gamma value that accounts for the value
        of ||u||^2 and ||v||^2 for each scenario for each iteration

        The side effects are that we store the previous iteration's v and u norms

        1. gamma should be monotonic?
        2. we need to do a reduction to get one gamma OR better use the global norms
        3. gamma should always (probably) be positive?
        4. Maybe we should be looking into scaled_vterm and scaled_uterm
        The global norms might be zero, so think about how to avoid
        using it when they are zero

        """
        
        uk = self.global_pusqnorm
        vk = self.global_pvsqnorm
        
        # Note June, 2023: We are waiting until we get values greater
        # than 0 for the norms. Iteration 3 is arbitrary
        if self._PHIter <= 3:
            gamma = self.APHgamma
            self.uk1 = self.global_pusqnorm
            self.vk1 = self.global_pvsqnorm
        else:
            if vk <= 0 or uk <= 0:
                gamma = self.APHgamma
            else:
                uk1 = self.uk1
                vk1 = self.vk1
                # Note June, 2023: vk1 and uk1 should be going down
                v_term = ((vk1 - vk) / vk) # use vk1 in denominator?
                u_term = ((uk1 - uk) / uk) # use uk1 in denominator?
                if v_term <= 0 or u_term <= 0:
                    # print('v_term=', v_term, 'u_term=', u_term, 'vk1=', vk1, 'vk=', vk, 'uk1=', uk1, 'uk=', uk)
                    gamma = self.APHgamma
                else:
                    gamma = (
                        v_term / u_term # gamma value gets 
                        # vk / uk
                        # (vk / zk) / (uk / wk) # use scaled v and u
                    )
                    self.uk1 = uk
                    self.vk1 = vk
                    
        self.APHgamma = gamma

        return self.APHgamma


    def listener_side_gig(self, synchro):
        """ Called by the listener after the first reduce.
        First, see if there are enough xbar contributions to proceed.
        If there are, then compute tau and phi.
        NOTE: it gets the synchronizer as an arg but self already has it.
        [WIP]
        We are going to disable the side_gig on self if we
        updated tau and phi.
        Massive side-effects: e.g., update xbar etc.

        Iter 1 (iter 0) in the paper is special: the v := u, which is a little
        complicated because we only compute y-bar.        

        """
        # This does unsafe things, so it can only be called when the worker is
        # in a tight loop that respects the data lock.

        verbose = self.options["verbose"]
        # See if we have enough xbars to proceed (need not be perfect)
        self.synchronizer._unsafe_get_global_data("FirstReduce",
                                                  self.node_concats)
        self.synchronizer._unsafe_get_global_data("SecondReduce",
                                                  self.node_concats)
        # last_phi_tau_update_time
        # (the last time this side-gig did the calculations)
        # We are going to see how many rank's xbars have been computed
        # since then. If enough (determined by frac_needed), the do the calcs.
        # The six is because the reduced data (e.g. phi) are in the first 6.
        lptut =  np.max(self.node_concats["SecondReduce"]["ROOT"][6:])

        logging.debug('   +++ debug enter listener_side_gig on cylinder_rank {} last phi update {}'\
              .format(self.cylinder_rank, lptut))

        xbarin = 0 # count ranks (close enough to be a proxy for scenarios)
        for cr in range(self.n_proc):
            backdist = self.n_proc - cr  # how far back into the vector
            ##logging.debug('      *side_gig* cr {} on rank {} time {}'.\
            ##    format(cr, self.cylinder_rank,
            ##        self.node_concats["FirstReduce"]["ROOT"][-backdist]))
            if  self.node_concats["FirstReduce"]["ROOT"][-backdist] \
                > lptut:
                xbarin += 1

        fracin = xbarin/self.n_proc + EPSILON
        if  fracin < self.options["async_frac_needed"]:
            # We have not really "done" the side gig.
            logging.debug('  ^ debug not good to go listener_side_gig on cylinder_rank {}; xbarin={}; fracin={}'\
                  .format(self.cylinder_rank, xbarin, fracin))
            return

        # If we are still here, we have enough to do the calculations
        logging.debug('^^^ debug good to go  listener_side_gig on cylinder_rank {}; xbarin={}'\
              .format(self.cylinder_rank, xbarin))
        if verbose and self.cylinder_rank == 0:
            print ("(%d)" % xbarin)
            
        # set the xbar, xsqbar, and ybar in all the scenarios
        for k,s in self.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            for (ndn,i) in s._mpisppy_data.nonant_indices:
                s._mpisppy_model.xbars[(ndn,i)]._value \
                    = self.node_concats["FirstReduce"][ndn][i]
                s._mpisppy_model.xsqbars[(ndn,i)]._value \
                    = self.node_concats["FirstReduce"][ndn][nlens[ndn]+i]
                s._mpisppy_model.ybars[(ndn,i)]._value \
                    = self.node_concats["FirstReduce"][ndn][2*nlens[ndn]+i]

                if verbose and self.cylinder_rank == 0:
                    print ("rank, scen, node, var, xbar:",
                           self.cylinder_rank,k,ndn,s._mpisppy_data.nonant_indices[ndn,i].name,
                           pyo.value(s._mpisppy_model.xbars[(ndn,i)]))

        # There is one tau_summand for the rank; global_tau is out of date when
        # we get here because we could not compute it until the averages were.
        # vk is just going to be ybar directly
        if not hasattr(self, "uk"):
            # indexed by sname and nonant index [sname][(ndn,i)]
            self.uk = {sname: dict() for sname in self.local_scenarios.keys()} 
        self.local_pusqnorm = 0  # local summand for probability weighted sqnorm
        self.local_pvsqnorm = 0
        new_tau_summand = 0  # for this rank
        for sname,s in self.local_scenarios.items():
            scen_usqnorm = 0.0
            scen_vsqnorm = 0.0
            nlens = s._mpisppy_data.nlens

            if not s._mpisppy_data.has_variable_probability:

                for (ndn,i), xvar in s._mpisppy_data.nonant_indices.items():
                    self.uk[sname][(ndn,i)] = xvar._value \
                        - pyo.value(s._mpisppy_model.xbars[(ndn,i)]) # Eq.27
                    # compute the usqnorm and vsqnorm (squared L2 norms)
                    scen_usqnorm += (self.uk[sname][(ndn,i)] \
                                     * self.uk[sname][(ndn,i)])
                    scen_vsqnorm += (pyo.value(s._mpisppy_model.ybars[(ndn,i)]) \
                                     * pyo.value(s._mpisppy_model.ybars[(ndn,i)]))
            else:
                # In the unlikely event of variable probability, do it
                # over again
                for (ndn,i), xvar in s._mpisppy_data.nonant_indices.items():
                    if s._mpisppy_data.prob0_mask[ndn][i] != 0:
                        self.uk[sname][(ndn,i)] = (
                            xvar._value \
                            - pyo.value(s._mpisppy_model.xbars[(ndn,i)]) # Eq.27
                        )
                    else:
                        self.uk[sname][(ndn,i)] = 0
                    # compute the usqnorm and vsqnorm (squared L2 norms)
                    scen_usqnorm += (self.uk[sname][(ndn,i)] \
                                     * self.uk[sname][(ndn,i)])
                    scen_vsqnorm += (pyo.value(s._mpisppy_model.ybars[(ndn,i)]) \
                                     * pyo.value(s._mpisppy_model.ybars[(ndn,i)]))
                   
            # Note by DLW April 2023: You need to move the probs up for multi-stage
            if not s._mpisppy_data.has_variable_probability:
                # NOTE: The p is not true but we avoid changing the
                # code in other places because pusqnorm and pvsqnorm
                # are used everywhere
                self.local_pusqnorm += scen_usqnorm
                self.local_pvsqnorm += scen_vsqnorm
            else:
                self.local_pusqnorm += pyo.value(s._mpisppy_probability) * scen_usqnorm  # prob first done
                self.local_pvsqnorm += pyo.value(s._mpisppy_probability) * scen_vsqnorm  # prob first done

            if self.use_dynamic_gamma:
                gamma = self._calculate_APHgamma(synchro) # update APHgamma
                print('dynamic gamma=', gamma, 'i=', i, 'sname=', sname)
            
            # I don't think s._mpisppy_dat.has_variable_probability is needed here
            new_tau_summand += (
                pyo.value(s._mpisppy_probability) \
                * (scen_usqnorm + scen_vsqnorm/self.APHgamma)
            )
            
        # tauk is the expectation of the sum sum of squares; update for this calc
        logging.debug('  in side-gig, old global_tau={}'.format(self.global_tau))
        logging.debug('  in side-gig, old summand={}'.format(self.tau_summand))
        logging.debug('  in side-gig, new summand={}'.format(new_tau_summand))
        self.global_tau = self.global_tau - self.tau_summand + new_tau_summand
        self.tau_summand = new_tau_summand # make available for the next reduce
        logging.debug('  in side-gig, new global_tau={}'.format(self.global_tau))

        # now we can get the local contribution to the phi_sum 
        if self.global_tau <= 0:
            logging.debug('  *** Negative tau={} on rank {}'\
                          .format(self.global_tau, self.cylinder_rank))
        self.phi_summand = self.compute_phis_summand()

        # prepare for the reduction that will take place after this side-gig
        # (this is where the 6 comes from)
        self.local_concats["SecondReduce"]["ROOT"][0] = self.tau_summand
        self.local_concats["SecondReduce"]["ROOT"][1] = self.phi_summand
        self.local_concats["SecondReduce"]["ROOT"][2] = self.local_pusqnorm
        self.local_concats["SecondReduce"]["ROOT"][3] = self.local_pvsqnorm
        self.local_concats["SecondReduce"]["ROOT"][4] = self.local_pwsqnorm
        self.local_concats["SecondReduce"]["ROOT"][5] = self.local_pzsqnorm
        # we have updated our summands and the listener will do a reduction
        secs_so_far = time.perf_counter() - self.start_time
        # Put in a time only for this rank, so the "sum" is really a report
        self.local_concats["SecondReduce"]["ROOT"][6+self.cylinder_rank] = secs_so_far
        # This is run by the listener, so don't tell the worker you have done
        # it until you are sure you have.
        self.synchronizer._unsafe_put_local_data("SecondReduce",
                                                 self.local_concats)
        self.synchronizer.enable_side_gig = False  # we did it
        logging.debug(' exit side_gid on rank {}'.format(self.cylinder_rank))
        
    #============================
    def Compute_Averages(self, verbose=False):
        """Gather ybar, xbar and x squared bar for each node 
           and also distribute the values back to the scenarios.
           Compute the tau summand from self and distribute back tauk
           (tau_k is a scalar and special with respect to synchronizing).
           Compute the phi summand and reduce it.

        Args:
          verbose (boolean): verbose output

        note: this is a long routine because we need a reduce before
              we can do more calcs that need another reduce and I want
              to keep the reduce calls together.
        NOTE: see compute_xbar for more notes.
        note: DLW: think about multi-stage harder (March 2019); e.g. tau and phi

        """
        if not hasattr(self, "local_concats"):
            nodenames = [] # avoid repeated work
            self.local_concats = {"FirstReduce": {}, # keys are tree node names
                             "SecondReduce": {}}
            self.node_concats = {"FirstReduce": {}, # concat of xbar and xsqbar
                             "SecondReduce": {}} 

            # accumulate & concatenate all local contributions before the reduce

            # create the c-style storage for the concats
            for k,s in self.local_scenarios.items():
                nlens = s._mpisppy_data.nlens        
                for node in s._mpisppy_node_list:
                    if node.name not in nodenames:
                        ndn = node.name
                        nodenames.append(ndn)
                        mylen = self.Lens["FirstReduce"][ndn]
                        self.local_concats["FirstReduce"][ndn]\
                            = np.zeros(mylen, dtype='d')
                        self.node_concats["FirstReduce"][ndn]\
                            = np.zeros(mylen, dtype='d')
            # second reduce is tau and phi
            mylen = self.Lens["SecondReduce"]["ROOT"]
            self.local_concats["SecondReduce"]["ROOT"]\
                = np.zeros(mylen, dtype='d') 
            self.node_concats["SecondReduce"]["ROOT"]\
                = np.zeros(mylen, dtype='d')
        else: # concats are here, just zero them out. 
            # We zero them so we can use an accumulator in the next loop and
            # that seems to be OK.
            nodenames = []
            for k,s in self.local_scenarios.items():
                nlens = s._mpisppy_data.nlens        
                for node in s._mpisppy_node_list:
                    if node.name not in nodenames:
                        ndn = node.name
                        nodenames.append(ndn)
                        self.local_concats["FirstReduce"][ndn].fill(0)
                        self.node_concats["FirstReduce"][ndn].fill(0)                    
            self.local_concats["SecondReduce"]["ROOT"].fill(0)
            self.node_concats["SecondReduce"]["ROOT"].fill(0)

        # Compute the locals and concat them for the first reduce, which includes xbar.
        # We don't need to lock here because the direct buffers are only accessed
        # by compute_global_data.
        for k,s in self.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                ndn = node.name
                for i in range(nlens[node.name]):
                    v_value = node.nonant_vardata_list[i]._value
                    self.local_concats["FirstReduce"][node.name][i] += \
                        (s._mpisppy_probability / node.uncond_prob) * v_value
                    logging.debug("  rank= {} scen={}, i={}, v_value={}".\
                                  format(global_rank, k, i, v_value))
                    self.local_concats["FirstReduce"][node.name][nlens[ndn]+i]\
                        += (s._mpisppy_probability / node.uncond_prob) * v_value * v_value
                    self.local_concats["FirstReduce"][node.name][2*nlens[ndn]+i]\
                        += (s._mpisppy_probability / node.uncond_prob) \
                           * pyo.value(s._mpisppy_model.y[(node.name,i)])
                    # print('test1', 'i:', i, 'mpisppy_prob', s._mpisppy_probability, 'uncond_prob', node.uncond_prob)
                    if s._mpisppy_data.has_variable_probability:
                        # re-do in the unlikely event of variable probabilities xxx TBD: check for multi-stage
                        ##prob = s._mpisppy_data.prob_coeff[ndn_i[0]][ndn_i[1]]
                        prob = s._mpisppy_data.prob_coeff[ndn][i]
                        self.local_concats["FirstReduce"][node.name][i] += \
                            (prob / node.uncond_prob) * v_value
                        self.local_concats["FirstReduce"][node.name][nlens[ndn]+i]\
                            += (prob / node.uncond_prob) * v_value * v_value
                        # for variable probability, ybar is really ysum!!!
                        self.local_concats["FirstReduce"][node.name][2*nlens[ndn]+i]\
                            += pyo.value(s._mpisppy_model.y[(node.name,i)])
                        # print('test2', 'i:', i, 'prob', prob, 'uncond_prob', node.uncond_prob)

        # record the time
        secs_sofar = time.perf_counter() - self.start_time
        # only this rank puts a time for this rank, so the sum is a report
        self.local_concats["FirstReduce"]["ROOT"][3*nlens["ROOT"]+self.cylinder_rank] \
            = secs_sofar
        logging.debug('Compute_Averages at secs_sofar {} on rank {}'\
                      .format(secs_sofar, self.cylinder_rank))
                    
        self.synchronizer.compute_global_data(self.local_concats,
                                              self.node_concats,
                                              enable_side_gig = True,
                                              rednames = ["FirstReduce"],
                                              keep_up = True)
        # The lock is something to worry about here.
        while self.synchronizer.global_quitting == 0 \
              and self.synchronizer.enable_side_gig:
            # Other ranks could be reporting, so keep looking for them.
            self.synchronizer.compute_global_data(self.local_concats,
                                                  self.node_concats)
            if not self.synchronizer.enable_side_gig:
                logging.debug(' did side gig break on rank {}'.format(self.cylinder_rank))
                break
            else:
                logging.debug('   gig wait sleep on rank {}'.format(self.cylinder_rank))
                if verbose and self.cylinder_rank == 0:
                    print ('s'),
                time.sleep(self.options["async_sleep_secs"])

        # (if the listener still has the lock, compute_global_will wait for it)
        self.synchronizer.compute_global_data(self.local_concats,
                                              self.node_concats)
        # We  assign the global xbar, etc. as side-effect in the side gig, btw
        self.global_tau = self.node_concats["SecondReduce"]["ROOT"][0]
        self.global_phi = self.node_concats["SecondReduce"]["ROOT"][1]
        self.global_pusqnorm = self.node_concats["SecondReduce"]["ROOT"][2]
        self.global_pvsqnorm = self.node_concats["SecondReduce"]["ROOT"][3]
        self.global_pwsqnorm = self.node_concats["SecondReduce"]["ROOT"][4]
        self.global_pzsqnorm = self.node_concats["SecondReduce"]["ROOT"][5]

        logging.debug('Assigned global tau {} and phi {} on rank {}'\
                      .format(self.global_tau, self.global_phi, self.cylinder_rank))
 
    #============================
    def Update_theta_zw(self, verbose):
        """
        Compute and store theta, then update z and w and update
        the probability weighted norms.
        """
        if self.global_tau <= 0:
            logging.debug('|tau {}, rank {}'.format(self.global_tau, self.cylinder_rank))
            self.theta = 0 # Step 17
        elif self.global_phi <= 0:
            logging.debug('|phi {}, rank {}'.format(self.global_phi, self.cylinder_rank))
            self.theta = 0
        else:
            punorm = math.sqrt(self.global_pusqnorm)
            pvnorm = math.sqrt(self.global_pvsqnorm)
            if self.use_hack_for_nu:
                # these vals control the additive hacking below
                nu_val = 0.1
                rho_val = 0.1
            else:
                nu_val = 0
                rho_val = 0
            if self._PHIter <= 3:
                self.nu = 1 - nu_val
            else:
                self.nu = 1 + nu_val
            self.theta = self.global_phi * self.nu / self.global_tau # Step 16
            # print(f'nu={self.nu}')
            for k,s in self.local_scenarios.items():
                for (ndn,i), xvar in s._mpisppy_data.nonant_indices.items():
                    if punorm <= pvnorm:
                        factor = 1 - rho_val
                    else:
                        factor = 1 + rho_val
                    s._mpisppy_model.rho[(ndn,i)] = pyo.value(s._mpisppy_model.rho[(ndn,i)]) * factor
                    # print(f'rho={pyo.value(s._mpisppy_model.rho[(ndn,i)])}')
                    
        logging.debug('Iter {} assigned theta {} on rank {}'\
                      .format(self._PHIter, self.theta, self.cylinder_rank))

        oldpw = self.local_pwsqnorm
        oldpz = self.local_pzsqnorm
        self.local_pwsqnorm = 0
        self.local_pzsqnorm = 0
        # v is just ybar
        for k,s in self.local_scenarios.items():
            probs = pyo.value(s._mpisppy_probability)
            for (ndn, i) in s._mpisppy_data.nonant_indices:
                Wupdate = self.theta * self.uk[k][(ndn,i)]
                Ws = pyo.value(s._mpisppy_model.W[(ndn,i)]) + Wupdate # Step 19, Algorithm 2
                # Special code for variable probabilities to mask W; rarely used.
                if s._mpisppy_data.has_variable_probability:
                    Ws *= s._mpisppy_data.prob0_mask[ndn][i]

                s._mpisppy_model.W[(ndn,i)] = Ws
                self.local_pwsqnorm += probs * Ws * Ws
                # iter 1 is iter 0 post-solves when seen from the paper
                if self._PHIter != 1: # Step 18, Algorithm 2
                    # NOTE: for variable probability, ybar was computed as a sum!!!!
                    zs = pyo.value(s._mpisppy_model.z[(ndn,i)])\
                     + self.theta * pyo.value(s._mpisppy_model.ybars[(ndn,i)])/self.APHgamma
                else:
                    zs = pyo.value(s._mpisppy_model.xbars[(ndn,i)])

                # Special code for variable probabilities to mask W; rarely used.
                if s._mpisppy_data.has_variable_probability:
                    zs *= s._mpisppy_data.prob0_mask[ndn][i]

                s._mpisppy_model.z[(ndn,i)] = zs 
                self.local_pzsqnorm += probs * zs * zs
                logging.debug("rank={}, scen={}, i={}, Ws={}, zs={}".\
                              format(global_rank, k, i, Ws, zs))
        # ? so they will be there next time? (we really need a third reduction)
        self.local_concats["SecondReduce"]["ROOT"][4] = self.local_pwsqnorm
        self.local_concats["SecondReduce"]["ROOT"][5] = self.local_pzsqnorm
        # The values we just computed can't be in the global yet, so update here
        self.global_pwsqnorm += (self.local_pwsqnorm - oldpw)
        self.global_pzsqnorm += (self.local_pzsqnorm - oldpz)
                
    #============================
    def Compute_Convergence(self, verbose=False):
        """
        The convergence metric is the sqrt of the sum of
        probability weighted unorm scaled by the probability weighted w norm
        probability weighted vnorm scaled by the probability weighted z norm
 
        Returns:
            update self.conv if appropriate
        """
        # dlw to dlw, April 2019: wnorm and znorm are in update_zw;
        # the u and v should be in the side gig.
        # you need a reduction on all the norms!!

        punorm = math.sqrt(self.global_pusqnorm)
        pwnorm = math.sqrt(self.global_pwsqnorm)
        pvnorm = math.sqrt(self.global_pvsqnorm)
        pznorm = math.sqrt(self.global_pzsqnorm)

        if pwnorm > 0 and pznorm > 0:
            self.conv = punorm / pwnorm + pvnorm / pznorm

        logging.debug('self.conv={} self.global_pusqnorm={} self.global_pwsqnorm={} self.global_pvsqnorm={} self.global_pzsqnorm={})'\
                      .format(self.conv, self.global_pusqnorm, self.global_pwsqnorm, self.global_pvsqnorm, self.global_pzsqnorm))
        # allow a PH converger, mainly for mpisspy to get xhat from a wheel conv
        if hasattr(self, "ph_convobject") and self.ph_convobject is not None:
            phc = self.ph_convobject(self, self.cylinder_rank, self.n_proc)
            logging.debug("PH converger called (returned {})".format(phc))

        if self.conv_trace_filename is not None:
            with open(self.conv_trace_filename, "a") as fil:
                fil.write(f"{self._PHIter},{self.conv},{self.APHgamma},{self.nu},{self.theta},{punorm},{pvnorm}\n")


    #==========
    def _update_foropt(self, dlist):
        # dlist is a list of subproblem names that were dispatched
        assert self.use_lag
        """
        if not self.bundling:
            phidict = self.phis
        else:
            phidict = {k: self.phis[self.local_subproblems[k].scen_list[0]]}
        """
        if not self.bundling:
            for dl in dlist:
                scenario = self.local_scenarios[dl[0]]
                for (ndn,i), xvar in scenario._mpisppy_data.nonant_indices.items():
                    scenario._mpisppy_model.z_foropt[(ndn,i)] = scenario._mpisppy_model.z[(ndn,i)]
                    scenario._mpisppy_model.W_foropt[(ndn,i)] = scenario._mpisppy_model.W[(ndn,i)]
        else:
            for dl in dlist:
                for sname in self.local_subproblems[dl[0]].scen_list:
                    scenario = self.local_scenarios[sname]
                    for (ndn,i), xvar in scenario._mpisppy_data.nonant_indices.items():
                        scenario._mpisppy_model.z_foropt[(ndn,i)] = scenario._mpisppy_model.z[(ndn,i)]
                        scenario._mpisppy_model.W_foropt[(ndn,i)] = scenario._mpisppy_model.W[(ndn,i)]


    #====================================================================
    def APH_solve_loop(self, solver_options=None,
                       use_scenarios_not_subproblems=False,
                       dtiming=False,
                       gripe=False,
                       disable_pyomo_signal_handling=False,
                       tee=False,
                       verbose=False,
                       dispatch_frac=1):
        """See phbase.solve_loop. Loop over self.local_subproblems and solve
            them in a manner dicated by the arguments. In addition to
            changing the Var values in the scenarios, update
            _PySP_feas_indictor for each.

        Args:
            solver_options (dict or None): the scenario solver options
            use_scenarios_not_subproblems (boolean): for use by bounds
            dtiming (boolean): indicates that timing should be reported
            gripe (boolean): output a message if a solve fails
            disable_pyomo_signal_handling (boolean): set to true for asynch, 
                                                     ignored for persistent solvers.
            tee (boolean): show solver output to screen if possible
            verbose (boolean): indicates verbose output
            dispatch_frac (float): fraction to send out for solution based on phi

        Returns:
            dlist (list of (str, float): (dispatched name, phi )
        """
        #==========
        def _vb(msg): 
            if verbose and self.cylinder_rank == 0:
                print ("(cylinder rank {}) {}".format(self.cylinder_rank, msg))
        _vb("Entering solve_loop function.")


        if use_scenarios_not_subproblems:
            s_source = self.local_scenarios
            phidict = self.phis
        else:
            s_source = self.local_subproblems
            if not self.bundling:
                phidict = self.phis
            else:
                phidict = {k: self.phis[self.local_subproblems[k].scen_list[0]] for k in s_source.keys()}
        # dict(sorted(phidict.items(), key=lambda item: item[1]))
        # sortedbyphi = {k: v for k, v in sorted(phidict.items(), key=lambda item: item[1])}


        #========
        def _dispatch_list(scnt):
            # Return the list of scnt (subproblems,phi) 
            # pairs for dispatch.
            # There is an option to allow for round-robin for research purposes.
            # NOTE: intermediate lists are created to help with verification.
            # reminder: dispatchrecord is sname:[(iter,phi)...]
            if self.round_robin_dispatch:
                # TBD: check this sort
                sortedbyI = {k: v for k, v in sorted(self.dispatchrecord.items(), 
                                                     key=lambda item: item[1][-1])}
                _vb("  sortedbyI={}.format(sortedbyI)")
                # There is presumably a pythonic way to do this...
                retval = list()
                i = 0
                for k,v in sortedbyI.items():
                    retval.append((k, phidict[k]))  # sname, phi
                    i += 1
                    if i >= scnt:
                        return retval
                raise RuntimeError(f"bad scnt={scnt} in _dispatch_list;"
                                   f" len(sortedbyI)={len(sortedbyI)}")
            else:
                # Not doing round robin
                # k is sname
                tosort = [(k, -max(self.dispatchrecord[k][-1][0], self.shelf_life-1), phidict[k])\
                          for k in self.dispatchrecord.keys()]
                sortedlist = sorted(tosort, key=lambda element: (element[1], element[2]))
                retval = [(sortedlist[k][0], sortedlist[k][2]) for k in range(scnt)]
                # TBD: See if there were enough w/negative phi values and warn.
                # TBD: see if shelf-life is hitting and warn
                return retval


        # body of APH_solve_loop fct starts hare
        logging.debug("  early APH solve_loop for rank={}".format(self.cylinder_rank))

        scnt = max(1, round(len(self.dispatchrecord) * dispatch_frac))
        dispatch_list = _dispatch_list(scnt)
        _vb("dispatch list before dispath: {}".format(dispatch_list))
        pyomo_solve_times = list()
        for dguy in dispatch_list:
            k = dguy[0]   # name of who to dispatch
            p = dguy[1]   # phi
            s = s_source[k]
            self.dispatchrecord[k].append((self._PHIter, p))
            _vb("dispatch k={}; phi={}".format(k, p))
            logging.debug("  in APH solve_loop rank={}, k={}, phi={}".\
                          format(self.cylinder_rank, k, p))
            # the lower lever dtiming does a gather
            pyomo_solve_times.append(self.solve_one(solver_options, k, s,
                                              dtiming=False,
                                              verbose=verbose,
                                              tee=tee,
                                              gripe=gripe,
                disable_pyomo_signal_handling=disable_pyomo_signal_handling
            ))

        if dtiming:
            print("Pyomo solve times (seconds):")
            print("\trank=,%d, n=,%d, min=,%4.2f, mean=,%4.2f, max=,%4.2f" %
                  (self.global_rank,
                   len(pyomo_solve_times),
                   np.min(pyomo_solve_times),
                   np.mean(pyomo_solve_times),
                   np.max(pyomo_solve_times)))

        return dispatch_list

    #========
    def _print_conv_detail(self):
        print("Convergence Metric=",self.conv)
        punorm = math.sqrt(self.global_pusqnorm)
        pwnorm = math.sqrt(self.global_pwsqnorm)
        pvnorm = math.sqrt(self.global_pvsqnorm)
        pznorm = math.sqrt(self.global_pzsqnorm)
        print(f'   punorm={punorm} pwnorm={pwnorm} pvnorm={pvnorm} pznorm={pznorm}')
        if pwnorm > 0 and pznorm > 0:
            print(f"    scaled U term={punorm / pwnorm}; scaled V term={pvnorm / pznorm}")
        else:
            print("    ! convergence metric cannot be computed due to zero-divide")


    #========
    def display_details(self, msg):
        """Ouput as much as you can about the current state"""
        print(f"hello {msg}")
        print(f"*** global rank {global_rank} display details: {msg}")
        print(f"zero-based iteration number {self._PHIter}")
        self._print_conv_detail()
        print(f"phi={self.global_phi}, nu={self.nu}, tau={self.global_tau} so theta={self.theta}")
        print(f"{'Nonants for':19} {'x':8} {'z':8} {'W':8} {'u':8} {'y':8}")
        for k,s in self.local_scenarios.items():
            print(f"   Scenario {k}")
            for (ndn,i), xvar in s._mpisppy_data.nonant_indices.items():
                print(f"   {(ndn,i)} {float(xvar._value):9.3} "
                      f"{float(s._mpisppy_model.z[(ndn,i)]._value):9.3}"
                      f"{float(s._mpisppy_model.W[(ndn,i)]._value):9.3}"
                      f"{float(self.uk[k][(ndn,i)]):9.3}"
                      f"{float(s._mpisppy_model.y[(ndn,i)]._value):9.3}")
        ph_base._Compute_Wbar(self)

        if self.plot_trace_prefix is not None:
            for k,s in self.local_scenarios.items():
                objval = pyo.value(self.saved_objectives[k])
                with open(f"trace_{k}_{self.conv_trace_filename}", "a") as fil:
                    fil.write(f"{self._PHIter},{objval}")
                    for (ndn,i), xvar in s._mpisppy_data.nonant_indices.items():
                        fil.write(f",{xvar._value},{s._mpisppy_model.z[(ndn,i)]._value},{s._mpisppy_model.W[(ndn,i)]._value}")
                    fil.write("\n")


    #====================================================================
    def APH_iterk(self, spcomm):
        """ Loop for the main iterations (called by synchronizer).

        Args:
        spcomm (SPCommunitator object): to communicate intra and inter

        Updates: 
            self.conv (): APH convergence

        """
        logging.debug('==== enter iterk on rank {}'.format(self.cylinder_rank))
        verbose = self.options["verbose"]
        have_extensions = self.extensions is not None

        # We have the "bottom of the loop at the top"
        # so we need a dlist to get the ball rolling (it might not be used)
        dlist = [(sn, 0.0) for sn in self.local_scenario_names]
        
        # put dispatch_frac on the object so extensions can modify it
        self.dispatch_frac = self.options["dispatch_frac"]\
                             if "dispatch_frac" in self.options else 1

        have_converger = self.ph_converger is not None
        dprogress = self.options["display_progress"]
        dtiming = self.options["display_timing"]
        ddetail = "display_convergence_detail" in self.options and\
            self.options["display_convergence_detail"]
        self.conv = None
        # The notion of an iteration is unclear
        # we enter after the iteration 0 solves, so do updates first
        for self._PHIter in range(1, self.options["PHIterLimit"]+1):
            if self.synchronizer.global_quitting:
                break
            iteration_start_time = time.time()

            if dprogress and self.cylinder_rank == 0:
                print("")
                print ("Initiating APH Iteration",self._PHIter)
                print("")

            self.Update_y(dlist, verbose)
            # Compute xbar, etc
            logging.debug('pre Compute_Averages on rank {}'.format(self.cylinder_rank))
            self.Compute_Averages(verbose)
            logging.debug('post Compute_Averages on rank {}'.format(self.cylinder_rank))
            if self.global_tau <= 0:
                logging.critical('***tau is 0 on rank {}'.format(self.cylinder_rank))

            # Apr 2019 dlw: If you want the convergence crit. to be up to date,
            # do this as a listener side-gig and add another reduction.
            self.Update_theta_zw(verbose)
            self.Compute_Convergence()  # updates conv
            phisum = self.compute_phis_summand() # post-step phis for dispatch
            logging.debug('phisum={} after step on {}'.format(phisum, self.cylinder_rank))

            # ORed checks for convergence
            if spcomm is not None and type(spcomm) is not mpi.Intracomm:
                spcomm.sync_with_spokes()
                logging.debug('post sync_with_spokes on rank {}'.format(self.cylinder_rank))
                if spcomm.is_converged():
                    break    
            if have_converger:
                if self.convobject.is_converged():
                    if self.cylinder_rank == 0:
                        print("User-supplied converger determined termination criterion reached")
                    break
            if ddetail:
                self.display_details("pre-solve loop (everything is updated from prev iter)")
            # slight divergence from PH, where mid-iter is before conv
            if have_extensions:
                self.extobject.miditer()
            
            teeme = ("tee-rank0-solves" in self.options) \
                 and (self.options["tee-rank0-solves"]
                      and self.cylinder_rank == 0)
            # Let the solve loop deal with persistent solvers & signal handling
            # Aug2020 switch to a partial loop xxxxx maybe that is enough.....
            # Aug2020 ... at least you would get dispatch
            # Oct 2021: still need full dispatch in iter 1 (as well as iter 0)
            # TBD: ? restructure so iter 1 can have partial dispatch
            if self._PHIter == 1:
                savefrac = self.dispatch_frac
                self.dispatch_frac = 1   # to get a decent w for everyone
            logging.debug('pre APH_solve_loop on rank {}'.format(self.cylinder_rank))
            dlist = self.APH_solve_loop(solver_options = \
                                        self.current_solver_options,
                                        dtiming=dtiming,
                                        gripe=True,
                                        disable_pyomo_signal_handling=True,
                                        tee=teeme,
                                        verbose=verbose,
                                        dispatch_frac=self.dispatch_frac)

            logging.debug('post APH_solve_loop on rank {}'.format(self.cylinder_rank))
            if self._PHIter == 1:
                 self.dispatch_frac = savefrac
            if have_extensions:
                self.extobject.enditer()

            if dprogress and self.cylinder_rank == 0:
                print("")
                print("After APH Iteration",self._PHIter)
                if not ddetail:
                    self._print_conv_detail()
                print("Iteration time: %6.2f" \
                      % (time.time() - iteration_start_time))
                print("Elapsed time:   %6.2f" \
                      % (time.perf_counter() - self.start_time))
            if self.use_lag:
                self._update_foropt(dlist)

        logging.debug('Setting synchronizer.quitting on rank %d' % self.cylinder_rank)
        self.synchronizer.quitting = 1

    #====================================================================
    def APH_main(self, spcomm=None, finalize=True):

        """Execute the APH algorithm.
        Args:
            spcomm (SPCommunitator object): for intra or inter communications
            finalize (bool, optional, default=True):
                        If True, call self.post_loops(), if False, do not,
                        and return None for Eobj

        Returns:
            conv, Eobj, trivial_bound: 
                        The first two CANNOT BE EASILY INTERPRETED. 
                        Eobj is the expected,  weighted objective with 
                        proximal term. It is not directly useful.
                        The trivial bound is computed after iter 0
        NOTE:
            You need an xhat finder either in denoument or in an extension.
        """
        # Prep needs to be before iter 0 for bundling
        # (It could be split up)
        logging.debug('enter aph main on cylinder_rank {}'.format(self.cylinder_rank))
        self.PH_Prep(attach_duals=False, attach_prox=False)

        # Begin APH-specific Prep
        for sname, scenario in self.local_scenarios.items():    
            # ys is plural of y
            scenario._mpisppy_model.y = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                     initialize = 0.0,
                                     mutable = True)
            scenario._mpisppy_model.ybars = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                        initialize = 0.0,
                                        mutable = True)
            scenario._mpisppy_model.z = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                     initialize = 0.0,
                                     mutable = True)
            # lag: we will support lagging back only to the last solve
            # IMPORTANT: pyomo does not support a second reference so no:
            # scenario._mpisppy_model.z_foropt = scenario._mpisppy_model.z
            
            if self.use_lag:
                scenario._mpisppy_model.z_foropt = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                         initialize = 0.0,
                                         mutable = True)
                scenario._mpisppy_model.W_foropt = pyo.Param(scenario._mpisppy_data.nonant_indices.keys(),
                                         initialize = 0.0,
                                         mutable = True)
                
            objfct = self.saved_objectives[sname]
                
            if self.use_lag:
                assert not scenario._mpisppy_data.has_variable_probability
                   
                for (ndn,i), xvar in scenario._mpisppy_data.nonant_indices.items():
                    # proximal term
                    objfct.expr +=  scenario._mpisppy_model.prox_on * \
                        (scenario._mpisppy_model.rho[(ndn,i)] /2.0) * \
                        (xvar**2 - 2.0*xvar*scenario._mpisppy_model.z_foropt[(ndn,i)] + scenario._mpisppy_model.z_foropt[(ndn,i)]**2)                                            
                    # W term
                    objfct.expr +=  scenario._mpisppy_model.W_on * scenario._mpisppy_model.W_foropt[ndn,i] * xvar
            else:
                for (ndn,i), xvar in scenario._mpisppy_data.nonant_indices.items():
                    if not scenario._mpisppy_data.has_variable_probability:
                        # proximal term
                        objfct.expr +=  scenario._mpisppy_model.prox_on * \
                            (scenario._mpisppy_model.rho[(ndn,i)] /2.0) * \
                            (xvar**2 - 2.0*xvar*scenario._mpisppy_model.z[(ndn,i)] + scenario._mpisppy_model.z[(ndn,i)]**2)                        
                    else:
                        objfct.expr += scenario._mpisppy_data.prob0_mask[ndn][i] * \
                            scenario._mpisppy_model.prox_on * \
                            (scenario._mpisppy_model.rho[(ndn,i)] /2.0) * \
                            (xvar**2 - 2.0*xvar*scenario._mpisppy_model.z[(ndn,i)] + scenario._mpisppy_model.z[(ndn,i)]**2)                        

                    # W term
                    objfct.expr +=  scenario._mpisppy_model.W_on * scenario._mpisppy_model.W[ndn,i] * xvar

        # End APH-specific Prep

        trivial_bound = self.Iter0()
        if self._can_update_best_bound():
            self.best_bound_obj_val = trivial_bound

        self.setup_Lens()
        self.setup_dispatchrecord()

        sleep_secs = self.options["async_sleep_secs"]

        lkwargs = None  # nothing beyond synchro
        listener_gigs = {"FirstReduce": (self.listener_side_gig, lkwargs),
                         "SecondReduce": None}
        self.synchronizer = listener_util.Synchronizer(comms = self.comms,
                                                    Lens = self.Lens,
                                                    work_fct = self.APH_iterk,
                                                    rank = self.cylinder_rank,
                                                    sleep_secs = sleep_secs,
                                                    asynch = True,
                                                    listener_gigs = listener_gigs)
        args = [spcomm] if spcomm is not None else [fullcomm]
        kwargs = None  # {"extensions": extensions}
        self.synchronizer.run(args, kwargs)

        if finalize:
            Eobj = self.post_loops()
        else:
            Eobj = None

#        print(f"Debug: here's the dispatch record for rank={self.global_rank}")
#        for k,v in self.dispatchrecord.items():
#            print(k, v)
#            print()
#        print("End dispatch record")

        return self.conv, Eobj, trivial_bound

#************************************************************
if __name__ == "__main__":
    # hardwired by dlw for debugging
    import mpisppy.tests.examples.farmer as refmodel

    PHopt = {}
    PHopt["asynchronousPH"] = False # APH is *projective* and always APH
    PHopt["solver_name"] = "cplex"
    PHopt["PHIterLimit"] = 5
    PHopt["defaultPHrho"] = 1
    PHopt["APHgamma"] = 1
    PHopt["convthresh"] = 0.001
    PHopt["verbose"] = True
    PHopt["display_timing"] = True
    PHopt["display_progress"] = True
    # one way to set up options (never mind that this is not a MIP)
    PHopt["iter0_solver_options"] = sputils.option_string_to_dict("mipgap=0.01")
    # another way
    PHopt["iterk_solver_options"] = {"mipgap": 0.001}

    ScenCount = 3
    scenario_creator_kwargs = {'use_integer': False, "crops_multiplier": 1}
    all_scenario_names = list()
    for sn in range(ScenCount):
        all_scenario_names.append("scen"+str(sn))
    # end hardwire

    scenario_creator = refmodel.scenario_creator
    scenario_denouement = refmodel.scenario_denouement

    PHopt["async_frac_needed"] = 0.5
    PHopt["async_sleep_secs"] = 0.5
    aph = APH(
        PHopt,
        all_scenario_names,
        scenario_creator,
        scenario_denouement,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )


    """
    import xhatlooper
    PHopt["xhat_looper_options"] =  {"xhat_solver_options":\
                                     PHopt["iterk_solver_options"],
                                     "scen_limit": 3,
                                     "csvname": "looper.csv"}
    """
    conv, obj, bnd = aph.APH_main()

    if aph.cylinder_rank == 0:
        print ("E[obj] for converged solution (probably NOT non-anticipative)",
               obj)

    dopts = sputils.option_string_to_dict("mipgap=0.001")
    objbound = aph.post_solve_bound(solver_options=dopts, verbose=False)
    if (aph.cylinder_rank == 0):
        print ("**** Lagrangian objective function bound=",objbound)
        print ("(probably converged way too early, BTW)")
