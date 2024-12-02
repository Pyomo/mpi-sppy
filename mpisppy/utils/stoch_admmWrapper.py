###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
#creating the class stoch_admmWrapper
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
import mpisppy.scenario_tree as scenario_tree
import numpy as np

def _consensus_vars_number_creator(consensus_vars):
    """associates to each consensus vars the number of time it appears

    Args:
        consensus_vars (dict): dictionary which keys are the subproblems and values are the list of consensus variables 
        present in the subproblem

    Returns:
        consensus_vars_number (dict): dictionary whose keys are the consensus variables 
        and values are the number of subproblems the variable is linked to.
    """
    consensus_vars_number={}
    for subproblem in consensus_vars:
        for var_stage_tuple in consensus_vars[subproblem]:
            var = var_stage_tuple[0]
            if var not in consensus_vars_number: # instanciates consensus_vars_number[var]
                consensus_vars_number[var] = 0
            consensus_vars_number[var] += 1
    return consensus_vars_number

class Stoch_AdmmWrapper(): #add scenario_tree
    """ Defines an interface to all strata (hubs and spokes)

        Args:
            options (dict): options
            all_scenario_names (list): all scenario names
            scenario_creator (fct): returns a concrete model with special things
            consensus_vars (dict): dictionary which keys are the subproblems and values are the list of consensus variables 
            present in the subproblem
            n_cylinder (int): number of cylinders that will ultimately be used
            mpicomm (MPI comm): creates communication
            scenario_creator_kwargs (dict): kwargs passed directly to scenario_creator.
            verbose (boolean): if True gives extra debugging information

        Attributes:
          local_scenarios (dict of scenario objects): concrete models with 
                extra data, key is name
          local_scenario_names (list): names of locals 
    """
    def __init__(self,
            options,
            all_admm_stoch_subproblem_scenario_names,
            split_admm_stoch_subproblem_scenario_name,
            admm_subproblem_names,
            stoch_scenario_names,
            scenario_creator, #supplied by the user/ modeller, used only here
            consensus_vars,
            n_cylinders,
            mpicomm,
            scenario_creator_kwargs=None,
            verbose=None,
            BFs=None,
    ):
        assert len(options) == 0, "no options supported by stoch_admmWrapper"
        # We need local_scenarios
        self.local_admm_stoch_subproblem_scenarios = {}
        scen_tree = sputils._ScenTree(["ROOT"], all_admm_stoch_subproblem_scenario_names)
        assert mpicomm.Get_size() % n_cylinders == 0, \
            f"{mpicomm.Get_size()=} and {n_cylinders=}, but {mpicomm.Get_size() % n_cylinders=} should be 0"
        ranks_per_cylinder = mpicomm.Get_size() // n_cylinders
        
        scenario_names_to_rank, _rank_slices, _scenario_slices =\
                scen_tree.scen_names_to_ranks(ranks_per_cylinder)

        cylinder_rank = mpicomm.Get_rank() // n_cylinders
        
        # taken from spbase
        self.local_admm_stoch_subproblem_scenarios_names = [
            all_admm_stoch_subproblem_scenario_names[i] for i in _rank_slices[cylinder_rank]
        ]
        for sname in self.local_admm_stoch_subproblem_scenarios_names:
            s = scenario_creator(sname, **scenario_creator_kwargs)
            self.local_admm_stoch_subproblem_scenarios[sname] = s
        # we are not collecting instantiation time

        self.split_admm_stoch_subproblem_scenario_name = split_admm_stoch_subproblem_scenario_name
        self.consensus_vars = consensus_vars
        self.verbose = verbose
        self.consensus_vars_number = _consensus_vars_number_creator(consensus_vars)
        self.admm_subproblem_names = admm_subproblem_names
        self.stoch_scenario_names = stoch_scenario_names
        self.BFs = BFs
        self.number_admm_subproblems = len(self.admm_subproblem_names)
        self.all_nodenames = self.create_node_names(num_admm_subproblems=len(admm_subproblem_names), num_stoch_scens=len(stoch_scenario_names))
        self.assign_variable_probs(verbose=self.verbose)
    

    def create_node_names(self, num_admm_subproblems, num_stoch_scens):
        if self.BFs is not None: # already multi-stage problem initially
            self.BFs.append(num_admm_subproblems) # Adds the last stage with admm_subproblems
            all_nodenames = sputils.create_nodenames_from_branching_factors(self.BFs)
        else: # 2-stage problem initially
            all_node_names_0 = ["ROOT"]
            all_node_names_1 = ["ROOT_" + str(i) for i in range(num_stoch_scens)]
            all_node_names_2 = [parent + "_" + str(j) \
                                for parent in all_node_names_1 \
                                    for j in range(num_admm_subproblems)]
            all_nodenames = all_node_names_0 + all_node_names_1 + all_node_names_2
        return all_nodenames


    def var_prob_list(self, s):
        """Associates probabilities to variables and raises exceptions if the model doesn't match the dictionary consensus_vars

        Args:
            s (Pyomo ConcreteModel): scenario

        Returns:
            list: list of pairs (variables id (int), probabilities (float)). The order of variables is invariant with the scenarios.
                If the consensus variable is present in the scenario it is associated with a probability 1/#subproblem
                where it is present. Otherwise it has a probability 0.
        """
        return self.varprob_dict[s]


    def assign_variable_probs(self, verbose=False):
        self.varprob_dict = {}

        #we collect the consensus variables
        all_consensus_vars = {var_stage_tuple[0]: var_stage_tuple[1] for admm_subproblem_names in self.consensus_vars for var_stage_tuple in self.consensus_vars[admm_subproblem_names]}
        error_list1 = []
        error_list2 = []
        for sname,s in self.local_admm_stoch_subproblem_scenarios.items():
            if verbose:
                print(f"stoch_admmWrapper.assign_variable_probs is processing scenario: {sname}")
            admm_subproblem_name = self.split_admm_stoch_subproblem_scenario_name(sname)[0]
            # varlist[stage] will contain the variables at each stage
            depth = len(s._mpisppy_node_list)+1
            varlist = [[] for _ in range(depth)]

            assert hasattr(s,"_mpisppy_probability"), f"the scenario {sname} doesn't have any _mpisppy_probability attribute"
            if s._mpisppy_probability == "uniform": #if there is a two-stage scenario defined by sputils.attach_root_node
                s._mpisppy_probability = 1/len(self.stoch_scenario_names)

            self.varprob_dict[s] = list()
            for vstr in all_consensus_vars.keys():
                stage = all_consensus_vars[vstr]
                v = s.find_component(vstr)
                var_stage_tuple = vstr, stage
                if var_stage_tuple in self.consensus_vars[admm_subproblem_name]:
                    if v is not None:
                        # variables that should be on the model
                        if stage == depth: 
                            # The node is a stochastic scenario, the probability at this node has not yet been defined 
                            cond_prob = 1.
                        else:
                            prob_node = np.prod([s._mpisppy_node_list[ancestor_stage-1].cond_prob for ancestor_stage in range(1,stage+1)])
                            # conditional probability of the scenario at the node (without considering the leaves as probabilities)
                            cond_prob = s._mpisppy_probability/prob_node 
                        self.varprob_dict[s].append((id(v),cond_prob/(self.consensus_vars_number[vstr])))
                        # s._mpisppy_probability has not yet been divided by the number of subproblems, it the probability of the
                        # stochastic scenario
                    else:
                        error_list1.append((sname,vstr))
                else:
                    if v is None:
                        # This var will not be indexed but that might not matter??
                        # Going to replace the brackets
                        v2str = vstr.replace("[","__").replace("]","__") # To distinguish the consensus_vars fixed at 0
                        v = pyo.Var()
                        
                        ### Lines equivalent to setattr(s, v2str, v) without warning
                        #s.del_component(v2str) UNUSEFUL
                        s.add_component(v2str, v) 
                        #is the consensus variable should be earlier, then its not added in the good place, or is it?

                        v.fix(0)
                        self.varprob_dict[s].append((id(v),0))
                    else:
                        error_list2.append((sname,vstr))
                varlist[stage-1].append(v)

            # Create the new scenario tree node for admm_consensus
            assert hasattr(s,"_mpisppy_node_list"), f"the scenario {sname} doesn't have any _mpisppy_node_list attribute"
            parent = s._mpisppy_node_list[-1]
            admm_subproblem_name, stoch_scenario_name = self.split_admm_stoch_subproblem_scenario_name(sname)
            num_scen = self.stoch_scenario_names.index(stoch_scenario_name)
            if self.BFs is not None:
                node_num = num_scen % self.BFs[-2] #BFs has already been updated to include the last rank with the leaves
            else:
                node_num = num_scen
            node_name = parent.name + '_' + str(node_num) 

            #could be more efficient with a dictionary rather than index
            s._mpisppy_node_list.append(scenario_tree.ScenarioNode(
                node_name,
                1/self.number_admm_subproblems, #branching probability at this node
                parent.stage + 1, # The stage is the stage of the previous leaves
                pyo.Expression(expr=0), # The cost is spread on the branches which are the subproblems
                varlist[depth-1],
                s)
            )
            s._mpisppy_probability /= self.number_admm_subproblems

            # underscores have a special signification in the tree
            for stage in range(1, depth):
                #print(f"{stage, varlist[stage-1], sname=}")
                old_node = s._mpisppy_node_list[stage-1]
                s._mpisppy_node_list[stage-1] = scenario_tree.ScenarioNode(
                old_node.name,
                old_node.cond_prob,
                old_node.stage,
                old_node.cost_expression * self.number_admm_subproblems,
                varlist[stage-1],
                s)

            self.local_admm_stoch_subproblem_scenarios[sname] = s # I don't know whether this changes anything

        if len(error_list1) + len(error_list2) > 0:
            raise RuntimeError (f"for each pair (scenario, variable) of the following list, the variable appears"
                                f"in consensus_vars, but not in the model:\n {error_list1} \n"
                                f"for each pair (scenario, variable) of the following list, the variable appears "
                                f"in the model, but not in consensus var: \n {error_list2}")


    def admmWrapper_scenario_creator(self, admm_stoch_subproblem_scenario_name):
        scenario = self.local_admm_stoch_subproblem_scenarios[admm_stoch_subproblem_scenario_name]

        # Although every stage is already multiplied earlier, we must still multiply the overall objective function
        # Grabs the objective function and multiplies its value by the number of scenarios to compensate for the probabilities
        obj = sputils.find_active_objective(scenario)
        obj.expr = obj.expr * self.number_admm_subproblems

        return scenario