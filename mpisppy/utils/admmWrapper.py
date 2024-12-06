###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
#creating the class AdmmWrapper
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
from mpisppy import MPI
global_rank = MPI.COMM_WORLD.Get_rank()

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
        for var in consensus_vars[subproblem]:
            if var not in consensus_vars_number: # instanciates consensus_vars_number[var]
                consensus_vars_number[var] = 0
            consensus_vars_number[var] += 1
    for var in consensus_vars_number:
        if consensus_vars_number[var] == 1:
            print(f"The consensus variable {var} appears in a single subproblem")
    return consensus_vars_number

class AdmmWrapper():
    """ This class assigns variable probabilities and creates wrapper for the scenario creator

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
            all_scenario_names,
            scenario_creator, #supplied by the user/ modeller, used only here
            consensus_vars,
            n_cylinders,
            mpicomm,
            scenario_creator_kwargs=None,
            verbose=None,
    ):
        assert len(options) == 0, "no options supported by AdmmWrapper"
        # We need local_scenarios
        self.local_scenarios = {}
        scenario_tree = sputils._ScenTree(["ROOT"], all_scenario_names)
        assert mpicomm.Get_size() % n_cylinders == 0, \
            f"{mpicomm.Get_size()=} and {n_cylinders=}, but {mpicomm.Get_size() % n_cylinders=} should be 0"
        ranks_per_cylinder = mpicomm.Get_size() // n_cylinders 
        
        scenario_names_to_rank, _rank_slices, _scenario_slices =\
                scenario_tree.scen_names_to_ranks(ranks_per_cylinder)

        self.cylinder_rank = mpicomm.Get_rank() // n_cylinders
        #self.cylinder_rank = mpicomm.Get_rank() % ranks_per_cylinder
        self.all_scenario_names = all_scenario_names
        #taken from spbase
        self.local_scenario_names = [
            all_scenario_names[i] for i in _rank_slices[self.cylinder_rank]
        ]
        for sname in self.local_scenario_names:
            s = scenario_creator(sname, **scenario_creator_kwargs)
            self.local_scenarios[sname] = s
        #we are not collecting instantiation time

        self.consensus_vars = consensus_vars
        self.verbose = verbose
        self.consensus_vars_number = _consensus_vars_number_creator(consensus_vars)
        #check_consensus_vars(consensus_vars)
        self.assign_variable_probs(verbose=self.verbose)
        self.number_of_scenario = len(all_scenario_names)

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
        all_consensus_vars = {var_stage_tuple: None for admm_subproblem_names in self.consensus_vars for var_stage_tuple in self.consensus_vars[admm_subproblem_names]}
        error_list1 = []
        error_list2 = []
        for sname,s in self.local_scenarios.items():
            if verbose:
                print(f"AdmmWrapper.assign_variable_probs is processing scenario: {sname}")
            varlist = list()
            self.varprob_dict[s] = list()
            for vstr in all_consensus_vars.keys():
                v = s.find_component(vstr)
                if vstr in self.consensus_vars[sname]:
                    if v is not None:
                        #variables that should be on the model
                        self.varprob_dict[s].append((id(v),1/(self.consensus_vars_number[vstr])))
                    else:
                        error_list1.append((sname,vstr))
                else:
                    if v is None: 
                        # This var will not be indexed but that might not matter??
                        # Going to replace the brackets
                        v2str = vstr.replace("[","__").replace("]","__") # To distinguish the consensus_vars fixed at 0
                        v = pyo.Var()
                        
                        ### Lines equivalent to setattr(s, v2str, v) without warning
                        s.del_component(v2str)
                        s.add_component(v2str, v)

                        v.fix(0)
                        self.varprob_dict[s].append((id(v),0))
                    else:
                        error_list2.append((sname,vstr))
                if v is not None: #if None the error is trapped earlier
                    varlist.append(v)
            objfunc = sputils.find_active_objective(s)
            #this will overwrite the nonants already there
            sputils.attach_root_node(s, objfunc, varlist) 

        if len(error_list1) + len(error_list2) > 0:
            raise RuntimeError (f"for each pair (scenario, variable) of the following list, the variable appears"
                                f"in consensus_vars, but not in the model:\n {error_list1} \n"
                                f"for each pair (scenario, variable) of the following list, the variable appears "
                                f"in the model, but not in consensus var: \n {error_list2}")


    def admmWrapper_scenario_creator(self, sname):
        #this is the function the user will supply for all cylinders 
        assert sname in self.local_scenario_names, f"{global_rank=} {sname=} \n {self.local_scenario_names=}"
        #should probably be deleted as it takes time
        scenario = self.local_scenarios[sname]

        # Grabs the objective function and multiplies its value by the number of scenarios to compensate for the probabilities
        obj = sputils.find_active_objective(scenario)
        obj.expr = obj.expr * self.number_of_scenario

        return scenario
    