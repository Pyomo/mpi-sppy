# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# base class for hub and for spoke strata
# (DLW delete this comment): some of prep moved to init

import time
import logging
import weakref
import numpy as np
import datetime as dt
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils

from collections import OrderedDict
from mpi4py import MPI

from mpisppy import tt_timer

logger = logging.getLogger("PHBase")
logger.setLevel(logging.WARN)


class SPBase(object):
    """ Defines an interface to all strata (hubs and spokes)

        Args:
            options (dict): PH options
            all_scenario_names (list): all scenario names
            scenario_creator (fct): returns a concrete model with special things
            scenario_denouement (fct): for post processing and reporting
            all_nodenames (list): all node names; can be None for 2 Stage
            mpicomm (MPI comm): if not given, use the global fullcomm
            rank0 (int): The lowest global rank for this type of object
            cb_data (any): passed directly to instance callback                

        Attributes:
          local_scenarios (dict of scenario objects): concrete models with 
                extra data, key is name
          comms (dict): keys are node names values are comm objects.
          local_scenario_names (list): names of locals 

          NOTEs:
          = dropping current_solver_options to children if needed

    """

    def __init__(
        self,
        options,
        all_scenario_names,
        scenario_creator,
        scenario_denouement=None,
        all_nodenames=None,
        mpicomm=None,
        rank0=0,
        cb_data=None,
    ):
        self.startdt = dt.datetime.now()
        self.start_time = time.time()
        self.options = options
        self.all_scenario_names = all_scenario_names
        self.scenario_creator = scenario_creator
        self.scenario_denouement = scenario_denouement
        self.comms = dict()
        self.local_scenarios = dict()
        self.local_scenario_names = list()
        self.E1_tolerance = 1e-5  # probs must sum to almost 1
        self.names_in_bundles = None
        self.scenarios_constructed = False
        if all_nodenames is None:
            self.all_nodenames = ["ROOT"]
        elif "ROOT" in all_nodenames:
            self.all_nodenames = all_nodenames
        else:
            raise RuntimeError("'ROOT' must be in the list of node names")

        # Set up MPI communicator and rank
        if mpicomm is not None:
            self.mpicomm = mpicomm
        else:
            self.mpicomm = MPI.COMM_WORLD
        self.rank = self.mpicomm.Get_rank()
        self.n_proc = self.mpicomm.Get_size()
        self.rank0 = rank0
        self.rank_global = MPI.COMM_WORLD.Get_rank()

        if self.rank_global == 0:
            tt_timer.toc("Start SPBase.__init__" ,delta=False)

        # This doesn't seemed to be checked anywhere else
        if self.n_proc > len(self.all_scenario_names):
            raise RuntimeError("More ranks than scenarios")

        # Call various initialization methods
        if "branching_factors" in self.options:
            self.branching_factors = self.options["branching_factors"]
        else:
            self.branching_factors = [len(self.all_scenario_names)]
        self.calculate_scenario_ranks()
        self.attach_scenario_rank_maps()
        if "bundles_per_rank" in self.options and self.options["bundles_per_rank"] > 0:
            self.assign_bundles()
            self.bundling = True
        else:
            self.bundling = False
        self.create_scenarios(cb_data)
        self.look_before_leap_all()
        self.compute_unconditional_node_probabilities()
        self.attach_nlens()
        self.attach_nonant_indexes()
        self.create_communicators()
        self.set_sense()
        self.set_multistage()

        ## SPCommunicator object
        self._spcomm = None

    def set_sense(self, comm=None):
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
        global_senses = self.comms["ROOT"].gather(is_min, root=self.rank0)
        if self.rank != self.rank0:
            return
        sense = global_senses[0]
        clear = all(val == sense for val in global_senses)
        if not clear:
            raise RuntimeError(
                "All scenario models must have the same "
                "model sense (minimize or maximize)"
            )

    def calculate_scenario_ranks(self):
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

        """
        tree = sputils._ScenTree(self.branching_factors, self.all_scenario_names)

        self.scenario_names_to_rank, self._rank_slices, self._scenario_slices =\
                tree.scen_names_to_ranks(self.n_proc)
        self._scenario_tree = tree

    def attach_scenario_rank_maps(self):
        """ Populate the following attribute

             1. self.local_scenario_names (list)
                List of index names owned by the local rank

            Notes:
                Called within __init__

                Modified by dlw for companiondriver use: opt, lb, and ub each get
                a full set of scenarios. So name_to_rank gives the rank within
                the comm for the type (e.g., the rank within the comm for lb, 
                if called by an lb.)
        """
        scen_count = len(self.all_scenario_names)

        # list of scenario names owned locally
        self.local_scenario_names = [
            self.all_scenario_names[i] for i in self._rank_slices[self.rank]
        ]

    def assign_bundles(self):
        """ Create self.names_in_bundles, a dict of dicts
            
            self.names_in_bundles[rank number][bundle number] = 
                    list of scenarios in that bundle

        """
        scen_count = len(self.all_scenario_names)

        if self.options["verbose"] and self.rank == self.rank0:
            print("(rank0)", self.options["bundles_per_rank"], "bundles per rank")
        if self.n_proc * self.options["bundles_per_rank"] > scen_count:
            raise RuntimeError(
                "Not enough scenarios to satisfy the bundles_per_rank requirement"
            )
        slices = self._rank_slices

        # dict: rank number --> list of scenario names owned by rank
        names_at_rank = {
            curr_rank: [self.all_scenario_names[i] for i in slc]
            for (curr_rank, slc) in enumerate(slices)
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

    def create_scenarios(self, cb_data):
        """ Call the scenario_creator for every local scenario, and store the
            results in self.local_scenarios (dict indexed by scenario names).

            Notes:
                If a scenario probability is not specified as an attribute
                PySP_prob of the ConcreteModel returned by ScenarioCreator,
                this function automatically assumes uniform probabilities.
        """
        if self.scenarios_constructed:
            if self.rank == self.rank0:
                print("Warning: scenarios already constructed")
            return

        for sname in self.local_scenario_names:
            instance_creation_start_time = time.time()
            s = self.scenario_creator(sname, node_names=None, cb_data=cb_data)
            if not hasattr(s, "PySP_prob"):
                s.PySP_prob = 1.0 / len(self.all_scenario_names)
            self.local_scenarios[sname] = s
            if "display_timing" in self.options and self.options["display_timing"]:
                instance_creation_time = time.time() - instance_creation_start_time
                all_instance_creation_times = self.mpicomm.gather(
                    instance_creation_time, root=self.rank0
                )
                if self.rank == self.rank0:
                    aict = all_instance_creation_times
                    print("Scenario instance creation times:")
                    print(f"\tmin={np.min(aict):4.2f} mean={np.mean(aict):4.2f} max={np.max(aict):4.2f}")
        self.scenarios_constructed = True

    def attach_nonant_indexes(self):
        for (sname, scenario) in self.local_scenarios.items():
            _nonant_indexes = OrderedDict() #paranoia
            nlens = scenario._PySP_nlens        
            for node in scenario._PySPnode_list:
                ndn = node.name
                for i in range(nlens[ndn]):
                    _nonant_indexes[ndn,i] = node.nonant_vardata_list[i]
            scenario._nonant_indexes = _nonant_indexes

    def attach_nlens(self):
        for (sname, scenario) in self.local_scenarios.items():
            # Things need to be by node so we can bind to the
            # indexes of the vardata lists for the nodes.
            scenario._PySP_nlens = {
                node.name: len(node.nonant_vardata_list)
                for node in scenario._PySPnode_list
            }
            scenario._PySP_cistart = dict()
            sofar = 0
            for ndn, ndn_len in scenario._PySP_nlens.items():
                scenario._PySP_cistart[ndn] = sofar
                sofar += ndn_len

    def create_communicators(self):
        # If the scenarios have not been constructed yet, 
        # set up the one communicator we know this rank will have 
        # and return
        if not self.local_scenario_names:
            self.comms["ROOT"] = self.mpicomm
            return

        # Create communicator objects, one for each node
        nonleafnodes = dict()
        for (sname, scenario) in self.local_scenarios.items():
            for node in scenario._PySPnode_list:
                nonleafnodes[node.name] = node  # might be assigned&reassigned

        # check the node names given by the scenarios
        for nodename in nonleafnodes.keys():
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
                self.comms[nodename] = self.mpicomm.Split(color=nodenumber, key=self.rank)
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


    def compute_unconditional_node_probabilities(self):
        """ calculates unconditional node probabilities """
        for k,s in self.local_scenarios.items():
            root = s._PySPnode_list[0]
            root.uncond_prob = 1.0
            for parent,child in zip(s._PySPnode_list[:-1],s._PySPnode_list[1:]):
                child.uncond_prob = parent.uncond_prob * child.cond_prob


    def set_multistage(self):
        self.multistage = (len(self.all_nodenames) > 1)

    def _look_before_leap(self, scen, addlist):
        """ utility to check before attaching something to the user's model
        """
        for attr in addlist:
            if hasattr(scen, attr):
                raise RuntimeError("Model already has `internal' attribute" + attr)

    def look_before_leap_all(self):
        for (sname, scenario) in self.local_scenarios.items():
            # TBD (Feb 2019) Take the caches and lists off the scenario
            self._look_before_leap(
                scenario,
                [
                    "_nonant_indexes",
                    "_xbars",
                    "_xsqbars",
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
            scenario._PySP_nonant_cache = None
            scenario._PySP_fixedness_cache = None

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

    def get_var_value(self, scenario_name, variable_name, stage_number):
        """ Get the value of the specified variable.
        Args:
            scenario_name (str):
                Name of the scenario for which to select the variable.
            variable_name (str):
                Name of the variable as constructed by scenario_creator.
            stage_number (int):
                Stage of the variable (one-based; ROOT is stage 1).

        Returns:
            float:
                Value of the specified variable.

        Raises:
            ValueError:
                If no variable with the specified name can be found, or if more
                than one variable with the specified name is found
        """
        if stage_number != 1:
            raise NotImplementedError("Can only get root node variables for now")
        scenario_model = self.local_scenarios[scenario_name]
        variables = [
            var for node in scenario_model._PySPnode_list
            for var in node.nonant_vardata_list
            if var.name == f"{scenario_name}.{variable_name}"
        ]
        if len(variables) == 0:
            raise ValueError(
                f"Could not find a variable with name {variable_name} in scenario {scenario_name}"
            )
        elif len(variables) > 1:
            raise ValueError(
                f"Found {len(variables)} variables with name {variable_name} in scenario {scenario_name}"
            )
        return variables[0].value

    def gather_var_values_to_root(self):
        """ Gather the values of the nonanticipative variables to the root of
        `mpicomm`.

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

        result = self.mpicomm.gather(var_values, root=self.rank0)

        if (self.rank == self.rank0):
            result = {key: value
                for dic in result
                for (key, value) in dic.items()
            }
            return result
