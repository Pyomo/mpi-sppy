# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# base class for hub and for spoke strata

import os
import time
import logging
import weakref
import numpy as np
import re
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy import global_toc

from mpisppy import MPI

logger = logging.getLogger("SPBase")
logger.setLevel(logging.WARN)


class SPBase:
    """ Defines an interface to all strata (hubs and spokes)

        Args:
            options (dict): options
            all_scenario_names (list): all scenario names
            scenario_creator (fct): returns a concrete model with special things
            scenario_denouement (fct): for post processing and reporting
            all_nodenames (list): all node names (including leaves); can be None for 2 Stage
            mpicomm (MPI comm): if not given, use the global fullcomm
            scenario_creator_kwargs (dict): kwargs passed directly to
                scenario_creator.
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
            scenario_creator_kwargs=None,
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
            self._check_nodenames()
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

        self._calculate_scenario_ranks()
        if "bundles_per_rank" in self.options and self.options["bundles_per_rank"] > 0:
            self._assign_bundles()
            self.bundling = True
        else:
            self.bundling = False
        self._create_scenarios(scenario_creator_kwargs)
        self._look_and_leap()
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

        # for writers, if the appropriate
        # solution is loaded into the subproblems
        self.tree_solution_available = False
        self.first_stage_solution_available = False

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
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
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
                
    def _check_nodenames(self):
        for ndn in self.all_nodenames:
            if ndn != 'ROOT' and sputils.parent_ndn(ndn) not in self.all_nodenames:
                raise RuntimeError(f"all_nodenames is inconsistent:"
                                   f"The node {sputils.parent_ndn(ndn)}, parent of {ndn}, is missing.")


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
        tree = sputils._ScenTree(self.all_nodenames, self.all_scenario_names)

        self.scenario_names_to_rank, self._rank_slices, self._scenario_slices =\
                tree.scen_names_to_ranks(self.n_proc)
        self._scenario_tree = tree
        self.nonleaves = {node.name : node for node in tree.nonleaves}

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

    def _create_scenarios(self, scenario_creator_kwargs):
        """ Call the scenario_creator for every local scenario, and store the
            results in self.local_scenarios (dict indexed by scenario names).

            Notes:
                If a scenario probability is not specified as an attribute
                _mpisppy_probability of the ConcreteModel returned by ScenarioCreator,
                this function automatically assumes uniform probabilities.
        """
        if self.scenarios_constructed:
            raise RuntimeError("Scenarios already constructed.")

        if scenario_creator_kwargs is None:
            scenario_creator_kwargs = dict()

        for sname in self.local_scenario_names:
            instance_creation_start_time = time.time()
            s = self.scenario_creator(sname, **scenario_creator_kwargs)
            self.local_scenarios[sname] = s
            if self.multistage:
                #Checking that the scenario can have an associated leaf node in all_nodenames
                stmax = np.argmax([nd.stage for nd in s._mpisppy_node_list])
                if(s._mpisppy_node_list[stmax].name)+'_0' not in self.all_nodenames:
                    raise RuntimeError("The leaf node associated with this scenario is not on all_nodenames"
                        f"Its last non-leaf node {s._mpisppy_node_list[stmax].name} has no first child {s._mpisppy_node_list[stmax].name+'_0'}")
            
            if self.options.get("display_timing", False):
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
            nlens = scenario._mpisppy_data.nlens        
            for node in scenario._mpisppy_node_list:
                ndn = node.name
                for i in range(nlens[ndn]):
                    _nonant_indices[ndn,i] = node.nonant_vardata_list[i]
            scenario._mpisppy_data.nonant_indices = _nonant_indices
        self.nonant_length = len(_nonant_indices)


    def _attach_nlens(self):
        for (sname, scenario) in self.local_scenarios.items():
            # Things need to be by node so we can bind to the
            # indices of the vardata lists for the nodes.
            scenario._mpisppy_data.nlens = {
                node.name: len(node.nonant_vardata_list)
                for node in scenario._mpisppy_node_list
            }

            # NOTE: This only is used by extensions.xhatbase.XhatBase._try_one.
            #       If that is re-factored, we can remove it here.
            scenario._mpisppy_data.cistart = dict()
            sofar = 0
            for ndn, ndn_len in scenario._mpisppy_data.nlens.items():
                scenario._mpisppy_data.cistart[ndn] = sofar
                sofar += ndn_len

                
    def _attach_varid_to_nonant_index(self):
        """ Create a map from the id of nonant variables to their Pyomo index.
        """
        for (sname, scenario) in self.local_scenarios.items():
            # In order to support rho setting, create a map
            # from the id of vardata object back its _nonant_index.
            scenario._mpisppy_data.varid_to_nonant_index =\
                {id(var): ndn_i for ndn_i, var in scenario._mpisppy_data.nonant_indices.items()}
            

    def _create_communicators(self):

        # Create communicator objects, one for each node
        nonleafnodes = dict()
        for (sname, scenario) in self.local_scenarios.items():
            for node in scenario._mpisppy_node_list:
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
                #The position in all_nodenames is an integer unique id.
                nodenumber = self.all_nodenames.index(nodename)
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
                    if rank != comm.Get_rank():
                        raise RuntimeError(f"For the node {nodename}, the scenario {sname} has the rank {rank} from scenario_names_to_rank and {comm.Get_rank()} from its comm.")
                        
        ## ensure we've set things up correctly for all local scenarios
        for sname in self.local_scenarios:
            for nodename, comm in self.comms.items():
                scenario_names_to_comm_rank = self.scenario_names_to_rank[nodename]
                if sname in scenario_names_to_comm_rank:
                    if comm.Get_rank() != scenario_names_to_comm_rank[sname]:
                        raise RuntimeError(f"For the node {nodename}, the scenario {sname} has the rank {rank} from scenario_names_to_rank and {comm.Get_rank()} from its comm.")


    def _compute_unconditional_node_probabilities(self):
        """ calculates unconditional node probabilities and prob_coeff
            and _PySP_W_coeff is set to a scalar 1 (used by variable_probability)"""
        for k,s in self.local_scenarios.items():
            root = s._mpisppy_node_list[0]
            root.uncond_prob = 1.0
            for parent,child in zip(s._mpisppy_node_list[:-1],s._mpisppy_node_list[1:]):
                child.uncond_prob = parent.uncond_prob * child.cond_prob
            if not hasattr(s._mpisppy_data, 'prob_coeff'):
                s._mpisppy_data.prob_coeff = dict()
                s._mpisppy_data.w_coeff = dict()
                for node in s._mpisppy_node_list:
                    s._mpisppy_data.prob_coeff[node.name] = (s._mpisppy_probability / node.uncond_prob)
                    s._mpisppy_data.w_coeff[node.name] = 1.0  # needs to be a float


    def _use_variable_probability_setter(self, verbose=False):
        """ set variable probability unconditional values using a function self.variable_probability
        that gives us a list of (id(vardata), probability)]
        ALSO set _PySP_W_coeff, which is a mask for W calculations (mask out zero probs)
        Note: We estimate that less than 0.01 of mpi-sppy runs will call this.
        """
        if self.variable_probability is None:
            for s in self.local_scenarios.values():
                s._mpisppy_data.has_variable_probability = False
            return
        didit = 0
        skipped = 0
        variable_probability_kwargs = self.options['variable_probability_kwargs'] \
                            if 'variable_probability_kwargs' in self.options \
                            else dict()
        for sname, s in self.local_scenarios.items():
            variable_probability = self.variable_probability(s, **variable_probability_kwargs)
            s._mpisppy_data.has_variable_probability = True
            for (vid, prob) in variable_probability:
                ndn, i = s._mpisppy_data.varid_to_nonant_index[vid]
                # If you are going to do any variables at a node, you have to do all.
                if type(s._mpisppy_data.prob_coeff[ndn]) is float:  # not yet a vector
                    defprob = s._mpisppy_data.prob_coeff[ndn]
                    s._mpisppy_data.prob_coeff[ndn] = np.full(s._mpisppy_data.nlens[ndn], defprob, dtype='d')
                    s._mpisppy_data.w_coeff[ndn] = np.ones(s._mpisppy_data.nlens[ndn], dtype='d')
                s._mpisppy_data.prob_coeff[ndn][i] = prob
                if prob == 0:  # there's probably a way to do this in numpy...
                    s._mpisppy_data.w_coeff[ndn][i] = 0
            didit += len(variable_probability)
            skipped += len(s._mpisppy_data.varid_to_nonant_index) - didit
        if verbose and self.cylinder_rank == 0:
            print ("variable_probability set",didit,"and skipped",skipped)

        if 'do_not_check_variable_probabilities' in self.options\
           and not self.options['do_not_check_variable_probabilities']:
            self._check_variable_probabilities_sum(verbose)

    def is_zero_prob( self, scenario_model, var ):
        """
        Args:
            scenario_model : a value in SPBase.local_scenarios
            var : a Pyomo Var on the scenario_model

        Returns:
            True if the variable has 0 probability, False otherwise
        """
        if self.variable_probability is None:
            return False
        _mpisppy_data = scenario_model._mpisppy_data
        ndn, i = _mpisppy_data.varid_to_nonant_index[id(var)]
        if isinstance(_mpisppy_data.prob_coeff[ndn], np.ndarray):
            return float(_mpisppy_data.prob_coeff[ndn][i]) == 0.
        else:
            return False

    def _check_variable_probabilities_sum(self, verbose):

        nodenames = [] # to transmit to comms
        local_concats = {}   # keys are tree node names
        global_concats =  {} # values sums of node conditional probabilities

        # we need to accumulate all local contributions before the reduce
        for k,s in self.local_scenarios.items():
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                if node.name not in nodenames:
                    ndn = node.name
                    nodenames.append(ndn)
                    local_concats[ndn] = np.zeros(nlens[ndn], dtype='d')
                    global_concats[ndn] = np.zeros(nlens[ndn], dtype='d')

        # sum local conditional probabilities
        for k,s in self.local_scenarios.items():
            for node in s._mpisppy_node_list:
                ndn = node.name
                local_concats[ndn] += s._mpisppy_data.prob_coeff[ndn]

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
            nlens = s._mpisppy_data.nlens
            for node in s._mpisppy_node_list:
                ndn = node.name
                if ndn not in checked_nodes:
                    if not np.allclose(global_concats[ndn], 1., atol=tol):
                        notclose = ~np.isclose(global_concats[ndn], 1., atol=tol)
                        indices = np.nonzero(notclose)[0]
                        bad_vars = [ s._mpisppy_data.nonant_indices[ndn,idx].name for idx in indices ]
                        badprobs = [ global_concats[ndn][idx] for idx in indices]
                        raise RuntimeError(f"Node {ndn}, variables {bad_vars} have respective"
                                           f" conditional probability sum {badprobs}"
                                           " which are not 1")
                    checked_nodes.append(ndn)


    def _look_and_leap(self):
        for (sname, scenario) in self.local_scenarios.items():

            if not hasattr(scenario, "_mpisppy_data"):
                scenario._mpisppy_data = pyo.Block(name="For non-Pyomo mpi-sppy data")
            if not hasattr(scenario, "_mpisppy_model"):
                scenario._mpisppy_model = pyo.Block(name="For mpi-sppy Pyomo additions to the scenario model")

            if hasattr(scenario, "PySP_prob"):
                raise RuntimeError(f"PySP_prob is deprecated; use _mpisppy_probability")
            if not hasattr(scenario, "_mpisppy_probability"):
                prob = 1./len(self.all_scenario_names)
                if self.cylinder_rank == 0:
                    print(f"Did not find _mpisppy_probability, assuming uniform probability {prob}")
                scenario._mpisppy_probability = prob
            if not hasattr(scenario, "_mpisppy_node_list"):
                raise RuntimeError(f"_mpisppy_node_list not found on scenario {sname}")

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


    def gather_var_values_to_rank0(self, get_zero_prob_values=False):
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
            for node in model._mpisppy_node_list:
                for var in node.nonant_vardata_list:
                    var_name = var.name
                    if self.bundling:
                        dot_index = var_name.find('.')
                        assert dot_index >= 0
                        var_name = var_name[(dot_index+1):]
                    if (self.is_zero_prob(model, var)) and (not get_zero_prob_values):
                        var_values[sname, var_name] = None
                    else:
                        var_values[sname, var_name] = pyo.value(var)

        if self.n_proc == 1:
            return var_values

        result = self.mpicomm.gather(var_values, root=0)

        if (self.cylinder_rank == 0):
            result = {key: value
                for dic in result
                for (key, value) in dic.items()
            }
            return result


    def report_var_values_at_rank0(self, header="", print_zero_prob_values=False):
        """ Pretty-print the values and associated statistics for
        non-anticipative variables across all scenarios. """

        var_values = self.gather_var_values_to_rank0(get_zero_prob_values=print_zero_prob_values)

        if self.cylinder_rank == 0:

            if len(header) != 0:
                print(header)

            scenario_names = sorted(set(x for (x,y) in var_values))
            max_scenario_name_len = max(len(s) for s in scenario_names)
            variable_names = sorted(set(y for (x,y) in var_values))
            max_variable_name_len = max(len(v) for v in variable_names)
            # the "10" below is a reasonable minimum for floating-point output
            value_field_len = max(10, max_scenario_name_len)

            print("{0: <{width}s} | ".format("", width=max_variable_name_len), end='')
            for this_scenario in scenario_names:
                print("{0: ^{width}s} ".format(this_scenario, width=value_field_len), end='')
            print("")

            for this_var in variable_names:
                print("{0: <{width}} | ".format(this_var, width=max_variable_name_len), end='')
                for this_scenario in scenario_names:
                    this_var_value = var_values[this_scenario, this_var]
                    if (this_var_value == None) and (not print_zero_prob_values):
                        print("{0: ^{width}s}".format("-", width=value_field_len), end='')
                    else:
                        print("{0: {width}.4f}".format(this_var_value, width=value_field_len), end='')
                    print(" ", end='')
                print("")

    def write_first_stage_solution(self, file_name,
            first_stage_solution_writer=sputils.first_stage_nonant_writer):
        """ Writes the first-stage solution, if this object reports one available.

            Args:
                file_name: path of file to write first stage solution to
                first_stage_solution_writer (optional): custom first stage solution writer function
        """

        if not self.first_stage_solution_available:
            raise RuntimeError("No first stage solution available")
        if self.cylinder_rank == 0:
            dirname = os.path.dirname(file_name)
            if dirname != '':
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
            representative_scenario = self.local_scenarios[self.local_scenario_names[0]]
            first_stage_solution_writer(file_name, representative_scenario, self.bundling)

    def write_tree_solution(self, directory_name,
            scenario_tree_solution_writer=sputils.scenario_tree_solution_writer):
        """ Writes the tree solution, if this object reports one available.
            Raises a RuntimeError if it is not.

            Args:
                directory_name: directory to write tree solution to
                scenario_tree_solution_writer (optional): custom scenario solution writer function
        """
        if not self.tree_solution_available:
            raise RuntimeError("No tree solution available")
        if self.cylinder_rank == 0:
            os.makedirs(directory_name, exist_ok=True)
        self.mpicomm.Barrier()
        for scenario_name, scenario in self.local_scenarios.items():
            scenario_tree_solution_writer(directory_name, scenario_name, scenario, self.bundling)
