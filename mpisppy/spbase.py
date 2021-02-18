# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# base class for hub and for spoke strata

import time
import logging
import weakref
import numpy as np
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils

from mpi4py import MPI

from mpisppy import global_toc

logger = logging.getLogger("SPBase")
logger.setLevel(logging.WARN)


class SPBase(object):
    """ Defines an interface to all strata (hubs and spokes)

        Args:
            options (dict): options
            all_scenario_names (list): all scenario names
            scenario_creator (fct): returns a concrete model with special things
            scenario_denouement (fct): for post processing and reporting
            all_nodenames (list): all non-leaf node names; can be None for 2 Stage
            mpicomm (MPI comm): if not given, use the global fullcomm
            cb_data (any): passed directly to instance callback                
            variable_probability (fct): returns a list of tuples of (id(var), prob)
                to set variable-specific probability (similar to PHBase.rho_setter).

        Attributes:
          local_scenarios (dict of scenario objects): concrete models with 
                extra data, key is name
          comms (dict): keys are node names values are comm objects.
          local_scenario_names (list): names of locals 
    """

    def __init__(
            self,
            options,
            all_scenario_names,
            scenario_creator,
            scenario_denouement=None,
            all_nodenames=None,
            mpicomm=None,
            cb_data=None,
            variable_probability=None,
            E1_tolerance=1e-5
    ):
        # TODO add missing and private attributes (JP)
        # TODO add a class attribute called ROOTNODENAME = "ROOT"
        # TODO? add decorators to the class attributes
        self.start_time = time.perf_counter()
        self.options = options
        self.all_scenario_names = all_scenario_names
        self.scenario_creator = scenario_creator
        self.scenario_denouement = scenario_denouement
        self.comms = dict()
        self.local_scenarios = dict()
        self.local_scenario_names = list()
        self.E1_tolerance = E1_tolerance  # probs must sum to almost 1
        self.names_in_bundles = None
        self.scenarios_constructed = False
        if all_nodenames is None:
            self.all_nodenames = ["ROOT"]
        elif "ROOT" in all_nodenames:
            self.all_nodenames = all_nodenames
        else:
            raise RuntimeError("'ROOT' must be in the list of node names")
        self.variable_probability = variable_probability
        self.multistage = (len(self.all_nodenames) > 1)

        # Set up MPI communicator and rank
        if mpicomm is not None:
            self.mpicomm = mpicomm
        else:
            self.mpicomm = MPI.COMM_WORLD
        self.cylinder_rank = self.mpicomm.Get_rank()
        self.n_proc = self.mpicomm.Get_size()
        self.global_rank = MPI.COMM_WORLD.Get_rank()

        global_toc("Initializing SPBase")

        if self.n_proc > len(self.all_scenario_names):
            raise RuntimeError("More ranks than scenarios")

        # Call various initialization methods
        if "branching_factors" in self.options:
            self.branching_factors = self.options["branching_factors"]
        else:
            self.branching_factors = [len(self.all_scenario_names)]
        self._calculate_scenario_ranks()
        if "bundles_per_rank" in self.options and self.options["bundles_per_rank"] > 0:
            self._assign_bundles()
            self.bundling = True
        else:
            self.bundling = False
        self._create_scenarios(cb_data)
        self._look_before_leap_all()
        self._compute_unconditional_node_probabilities()
        self._attach_nlens()
        self._attach_nonant_indices()
        self._attach_varid_to_nonant_index()
        self._create_communicators()
        self._verify_nonant_lengths()
        self._set_sense()
        self._use_variable_probability_setter()

        ## SPCommunicator object
        self._spcomm = None

    def _set_sense(self, comm=None):
        """ Check to confirm that all the models constructed by scenario_crator
            have the same sense (min v. max), and set self.is_minimizing
            accordingly.
        """
        is_min, clear = sputils._models_have_same_sense(self.local_scenarios)
        if not clear:
            raise RuntimeError(
                "All scenario models must have the same "
                "model sense (minimize or maximize)"
            )
        self.is_minimizing = is_min

        if self.n_proc <= 1:
            return

        # Check that all the ranks agree
        global_senses = self.mpicomm.gather(is_min, root=0)
        if self.cylinder_rank != 0:
            return
        sense = global_senses[0]
        clear = all(val == sense for val in global_senses)
        if not clear:
            raise RuntimeError(
                "All scenario models must have the same "
                "model sense (minimize or maximize)"
            )

    def _verify_nonant_lengths(self):
        local_node_nonant_lengths = {}   # keys are tree node names

        # we need to accumulate all local contributions before the reduce
        for k,s in self.local_scenarios.items():
            nlens = s._PySP_nlens
            for node in s._PySPnode_list:
                ndn = node.name
                mylen = nlens[ndn]
                if ndn not in local_node_nonant_lengths:
                    local_node_nonant_lengths[ndn] = mylen
                elif local_node_nonant_lengths[ndn] != mylen:
                    raise RuntimeError(f"Tree node {ndn} has different number of non-anticipative "
                            f"variables between scenarios {mylen} vs. {local_node_nonant_lengths[ndn]}")

        # compute node xbar values(reduction)
        for ndn, val in local_node_nonant_lengths.items():
            local_val = np.array([val], 'i')
            max_val = np.zeros(1, 'i')
            self.comms[ndn].Allreduce([local_val, MPI.INT],
                                      [max_val, MPI.INT],
                                      op=MPI.MAX)

            if val != int(max_val[0]):
                raise RuntimeError(f"Tree node {ndn} has different number of non-anticipative "
                        f"variables between scenarios {val} vs. max {max_val[0]}")


    def _calculate_scenario_ranks(self):
        """ Populate the following attributes
            1. self.scenario_names_to_rank (dict of dict):
                keys are comms (i.e., tree nodes); values are dicts with keys
                that are scenario names and values that are ranks within that comm

            2. self._rank_slices (list of lists)
                indices correspond to ranks in self.mpicomm and the values are a list
                of scenario indices
                rank -> list of scenario indices for that rank

            3. self._scenario_slices (list)
                indices are scenario indices and values are the rank of that scenario
                within self.mpicomm
                scenario index -> rank

            4. self._scenario_tree (instance of sputils._ScenTree)

            5. self.local_scenario_names (list)
               List of index names owned by the local rank

        """
        tree = sputils._ScenTree(self.branching_factors, self.all_scenario_names)

        self.scenario_names_to_rank, self._rank_slices, self._scenario_slices =\
                tree.scen_names_to_ranks(self.n_proc)
        self._scenario_tree = tree

        # list of scenario names owned locally
        self.local_scenario_names = [
            self.all_scenario_names[i] for i in self._rank_slices[self.cylinder_rank]
        ]


    def _assign_bundles(self):
        """ Create self.names_in_bundles, a dict of dicts
            
            self.names_in_bundles[rank number][bundle number] = 
                    list of scenarios in that bundle

        """
        scen_count = len(self.all_scenario_names)

        if self.options["verbose"] and self.cylinder_rank == 0:
            print("(rank0)", self.options["bundles_per_rank"], "bundles per rank")
        if self.n_proc * self.options["bundles_per_rank"] > scen_count:
            raise RuntimeError(
                "Not enough scenarios to satisfy the bundles_per_rank requirement"
            )

        # dict: rank number --> list of scenario names owned by rank
        names_at_rank = {
            curr_rank: [self.all_scenario_names[i] for i in slc]
            for (curr_rank, slc) in enumerate(self._rank_slices)
        }

        self.names_in_bundles = dict()
        num_bundles = self.options["bundles_per_rank"]

        for curr_rank in range(self.n_proc):
            scen_count = len(names_at_rank[curr_rank])
            avg = scen_count / num_bundles
            slices = [
                range(int(i * avg), int((i + 1) * avg)) for i in range(num_bundles)
            ]
            self.names_in_bundles[curr_rank] = {
                curr_bundle: [names_at_rank[curr_rank][i] for i in slc]
                for (curr_bundle, slc) in enumerate(slices)
            }

    def _create_scenarios(self, cb_data):
        """ Call the scenario_creator for every local scenario, and store the
            results in self.local_scenarios (dict indexed by scenario names).

            Notes:
                If a scenario probability is not specified as an attribute
                PySP_prob of the ConcreteModel returned by ScenarioCreator,
                this function automatically assumes uniform probabilities.
        """
        if self.scenarios_constructed:
            raise RuntimeError("Scenarios already constructed.")

        for sname in self.local_scenario_names:
            instance_creation_start_time = time.time()
            ### s = self.scenario_creator(sname, **scenario_creator_kwargs)
            s = self.scenario_creator(sname, node_names=None, cb_data=cb_data)
            if not hasattr(s, "PySP_prob"):
                s.PySP_prob = 1.0 / len(self.all_scenario_names)
            s._PySP_has_varprob = False  # Might be later set to True (but rarely)
            self.local_scenarios[sname] = s
            if "display_timing" in self.options and self.options["display_timing"]:
                instance_creation_time = time.time() - instance_creation_start_time
                all_instance_creation_times = self.mpicomm.gather(
                    instance_creation_time, root=0
                )
                if self.cylinder_rank == 0:
                    aict = all_instance_creation_times
                    print("Scenario instance creation times:")
                    print(f"\tmin={np.min(aict):4.2f} mean={np.mean(aict):4.2f} max={np.max(aict):4.2f}")
        self.scenarios_constructed = True

    def _attach_nonant_indices(self):
        for (sname, scenario) in self.local_scenarios.items():
            _nonant_indices = dict()
            nlens = scenario._PySP_nlens        
            for node in scenario._PySPnode_list:
                ndn = node.name
                for i in range(nlens[ndn]):
                    _nonant_indices[ndn,i] = node.nonant_vardata_list[i]
            scenario._nonant_indices = _nonant_indices

            
    def _attach_nlens(self):
        for (sname, scenario) in self.local_scenarios.items():
            # Things need to be by node so we can bind to the
            # indices of the vardata lists for the nodes.
            scenario._PySP_nlens = {
                node.name: len(node.nonant_vardata_list)
                for node in scenario._PySPnode_list
            }

            # NOTE: This only is used by extensions.xhatbase.XhatBase._try_one.
            #       If that is re-factored, we can remove it here.
            scenario._PySP_cistart = dict()
            sofar = 0
            for ndn, ndn_len in scenario._PySP_nlens.items():
                scenario._PySP_cistart[ndn] = sofar
                sofar += ndn_len

                
    def _attach_varid_to_nonant_index(self):
        """ Create a map from the id of nonant variables to their Pyomo index.
        """
        for (sname, scenario) in self.local_scenarios.items():
            # In order to support rho setting, create a map
            # from the id of vardata object back its _nonant_index.
            scenario._varid_to_nonant_index =\
                {id(var): ndn_i for ndn_i, var in scenario._nonant_indices.items()}
            

    def _create_communicators(self):

        # Create communicator objects, one for each node
        nonleafnodes = dict()
        for (sname, scenario) in self.local_scenarios.items():
            for node in scenario._PySPnode_list:
                nonleafnodes[node.name] = node  # might be assigned&reassigned

        # check the node names given by the scenarios
        for nodename in nonleafnodes:
            if nodename not in self.all_nodenames:
                raise RuntimeError(f"Tree node '{nodename}' not in all_nodenames list {self.all_nodenames}")

        # loop over all nodes and make the comms (split requires all ranks)
        # make sure we loop in the same order, so every rank iterate over
        # the nodelist
        for nodename in self.all_nodenames:
            if nodename == "ROOT":
                self.comms["ROOT"] = self.mpicomm
            elif nodename in nonleafnodes:
                nodenumber = sputils.extract_num(nodename)
                # IMPORTANT: See note in sputils._ScenTree.scen_names_to_ranks. Need to keep
                #            this split aligned with self.scenario_names_to_rank
                self.comms[nodename] = self.mpicomm.Split(color=nodenumber, key=self.cylinder_rank)
            else: # this rank is not included in the communicator
                self.mpicomm.Split(color=MPI.UNDEFINED, key=self.n_proc)

        ## ensure we've set things up correctly for all comms
        for nodename, comm in self.comms.items():
            scenario_names_to_comm_rank = self.scenario_names_to_rank[nodename]
            for sname, rank in scenario_names_to_comm_rank.items():
                if sname in self.local_scenarios:
                    assert rank == comm.Get_rank()

        ## ensure we've set things up correctly for all local scenarios
        for sname in self.local_scenarios:
            for nodename, comm in self.comms.items():
                scenario_names_to_comm_rank = self.scenario_names_to_rank[nodename]
                if sname in scenario_names_to_comm_rank:
                    assert comm.Get_rank() == scenario_names_to_comm_rank[sname]


    def _compute_unconditional_node_probabilities(self):
        """ calculates unconditional node probabilities and _PySP_prob_coeff
            and _PySP_W_coeff is set to a scalar 1 (used by variable_probability)"""
        for k,s in self.local_scenarios.items():
            root = s._PySPnode_list[0]
            root.uncond_prob = 1.0
            for parent,child in zip(s._PySPnode_list[:-1],s._PySPnode_list[1:]):
                child.uncond_prob = parent.uncond_prob * child.cond_prob
            if not hasattr(s, '_PySP_prob_coeff'):
                s._PySP_prob_coeff = dict()
                s._PySP_W_coeff = dict()
                for node in s._PySPnode_list:
                    s._PySP_prob_coeff[node.name] = (s.PySP_prob / node.uncond_prob)
                    s._PySP_W_coeff[node.name] = 1.0  # needs to be a float


    def _use_variable_probability_setter(self, verbose=False):
        """ set variable probability unconditional values using a function self.variable_probability
        that gives us a list of (id(vardata), probability)]
        ALSO set _PySP_W_coeff, which is a mask for W calculations (mask out zero probs)
        Note: We estimate that less than 0.01 of mpi-sppy runs will call this.
        """
        if self.variable_probability is None:
            return
        didit = 0
        skipped = 0
        variable_probability_kwargs = self.options['variable_probability_kwargs'] \
                            if 'variable_probability_kwargs' in self.options \
                            else dict()
        for sname, s in self.local_scenarios.items():
            variable_probability = self.variable_probability(s, **variable_probability_kwargs)
            s._PySP_has_varprob = True
            for (vid, prob) in variable_probability:
                ndn, i = s._varid_to_nonant_index[vid]
                # If you are going to do any variables at a node, you have to do all.
                if type(s._PySP_prob_coeff[ndn]) is float:  # not yet a vector
                    defprob = s._PySP_prob_coeff[ndn]
                    s._PySP_prob_coeff[ndn] = np.full(s._PySP_nlens[ndn], defprob, dtype='d')
                    s._PySP_W_coeff[ndn] = np.ones(s._PySP_nlens[ndn], dtype='d')
                s._PySP_prob_coeff[ndn][i] = prob
                if prob == 0:  # there's probably a way to do this in numpy...
                    s._PySP_W_coeff[ndn][i] = 0
            didit += len(variable_probability)
            skipped += len(s._varid_to_nonant_index) - didit
        if verbose and self.cylinder_rank == 0:
            print ("variable_probability set",didit,"and skipped",skipped)

        self._check_variable_probabilities_sum(verbose)

    def _check_variable_probabilities_sum(self, verbose):

        nodenames = [] # to transmit to comms
        local_concats = {}   # keys are tree node names
        global_concats =  {} # values sums of node conditional probabilities

        # we need to accumulate all local contributions before the reduce
        for k,s in self.local_scenarios.items():
            nlens = s._PySP_nlens
            for node in s._PySPnode_list:
                if node.name not in nodenames:
                    ndn = node.name
                    nodenames.append(ndn)
                    local_concats[ndn] = np.zeros(nlens[ndn], dtype='d')
                    global_concats[ndn] = np.zeros(nlens[ndn], dtype='d')

        # sum local conditional probabilities
        for k,s in self.local_scenarios.items():
            for node in s._PySPnode_list:
                ndn = node.name
                local_concats[ndn] += s._PySP_prob_coeff[ndn]

        # compute sum node conditional probabilities (reduction)
        for ndn in nodenames:
            self.comms[ndn].Allreduce(
                [local_concats[ndn], MPI.DOUBLE],
                [global_concats[ndn], MPI.DOUBLE],
                op=MPI.SUM)

        tol = self.E1_tolerance
        checked_nodes = list()
        # check sum node conditional probabilites are close to 1
        for k,s in self.local_scenarios.items():
            nlens = s._PySP_nlens
            for node in s._PySPnode_list:
                ndn = node.name
                if ndn not in checked_nodes:
                    if not np.allclose(global_concats[ndn], 1., atol=tol):
                        notclose = ~np.isclose(global_concats[ndn], 1., atol=tol)
                        indices = np.nonzero(notclose)[0]
                        bad_vars = [ s._nonant_indices[ndn,idx].name for idx in indices ]
                        badprobs = [ global_concats[ndn][idx] for idx in indices]
                        raise RuntimeError(f"Node {ndn}, variables {bad_vars} have respective"
                                           f" conditional probability sum {badprobs}"
                                           " which are not 1")
                    checked_nodes.append(ndn)

    # TODO: There is an issue (#60) to put these things on a block, but really we should
    # have two blocks: one for Pyomo objects and the other for lists and caches.
    def _look_before_leap(self, scen, addlist):
        """ utility to check before attaching something to the user's model
        """
        for attr in addlist:
            if hasattr(scen, attr):
                raise RuntimeError("Model already has `internal' attribute" + attr)

    def _look_before_leap_all(self):
        for (sname, scenario) in self.local_scenarios.items():
            self._look_before_leap(
                scenario,
                [
                    "_nonant_indices",
                    "_xbars",
                    "_xsqbars",
                    "_xsqvar",
                    "_xsqvar_cuts",
                    "_xsqvar_prox_approx",
                    "_Ws",
                    "_PySP_nlens",
                    "_PHrho",
                    "_PHtermon",
                    "_varid_to_nonant_index",
                    "_PHW_on",
                    "_PySP_nonant_cache",
                    "_PHprox_on",
                    "_PySP_fixedness_cache",
                    "_PySP_original_fixedness",
                    "_PySP_original_nonants",
                    "_zs",
                    "_ys",
                ],
            )

    def _options_check(self, required_options, given_options):
        """ Confirm that the specified list of options contains the specified
            list of required options. Raises a ValueError if anything is
            missing.
        """
        missing = [option for option in required_options if option not in given_options] 
        if missing:
            raise ValueError(f"Missing the following required options: {', '.join(missing)}")

    @property
    def spcomm(self):
        if self._spcomm is None:
            return None
        return self._spcomm()

    @spcomm.setter
    def spcomm(self, value):
        if self._spcomm is None:
            self._spcomm = weakref.ref(value)
        else:
            raise RuntimeError("SPBase.spcomm should only be set once")


    def gather_var_values_to_rank0(self):
        """ Gather the values of the nonanticipative variables to the root of
        the `mpicomm` for the cylinder

        Returns:
            dict or None:
                On the root (rank0), returns a dictionary mapping
                (scenario_name, variable_name) pairs to their values. On other
                ranks, returns None.
        """
        var_values = dict()
        for (sname, model) in self.local_scenarios.items():
            for node in model._PySPnode_list:
                for var in node.nonant_vardata_list:
                    var_values[sname, var.name] = pyo.value(var)

        result = self.mpicomm.gather(var_values, root=0)

        if (self.cylinder_rank == 0):
            result = {key: value
                for dic in result
                for (key, value) in dic.items()
            }
            return result
