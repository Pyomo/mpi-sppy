# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
#Code for sampling for multistage problem

import numpy as np
import mpi4py.MPI as mpi
import importlib

import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgomator as amalgomator
import mpisppy.confidence_intervals.ciutils as ciutils

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()



class SampleSubtree():
    '''
    The Sample Subtree class is generating a scenario tree such that every
    scenario shares the same common nodes up to a starting stage t. These nodes
    are the first t nodes of a given scenario 'root_scen'.
    The aim of this class is to compute a feasible policy at stage t for 'root_scen'
    We take as an argument xhats, feasible policies for stages 1 to t-1, and 
    fix the nonants up to stage t-1 to the values given by 'xhats' for every scenario.
    
    The run() method is solving directly the Extensive Form. 
    After callling the method, one can fetch the feasible policy for stage t 
    by taking the attribute xhat_at_stage.

    Parameters
    ----------
    mname : str
        name of the module used to sample.
    xhats : list of lists
        List of nonanticipative feasible policies for stages 1 to t-1 for root_scen.
    root_scen : pyomo ConcreteModel
        A scenario. Every scenario in the subtree shares the same nodes
        for stages 1 to t with root_scen
    starting_stage : int
        Stage t>=1.
    BFs : list of int
        Branching factors for sample trees. 
        BFs[i] is the branching factor for stage i+2
    seed : int
        Seed to create scenarios.
    options : dict
        Arguments passed to the scenario creator.
    solvername : str, optional
        Solver name. The default is 'gurobi'.
    solver_options : dict, optional
        Solver options. The default is None.

    '''
    def __init__(self, mname, xhats, root_scen, starting_stage, BFs,
                 seed, options, solvername='gurobi', solver_options=None):

        self.refmodel = importlib.import_module(mname)
        #Checking that refmodel has all the needed attributes
        attrs = ["sample_tree_scen_creator","kw_creator","scenario_names_creator"]
        for attr in attrs:
            if not hasattr(self.refmodel, attr):
                raise RuntimeError(f"The construction of a sample subtree failed because the reference model has no {attr} function.")
        self.xhats = xhats
        self.root_scen = root_scen
        self.stage = starting_stage
        self.sampling_BFs = BFs[(self.stage-1):]
        self.numscens = np.prod(self.sampling_BFs)
        self.seed = seed
        self.options = options
        self.solvername = solvername
        self.solver_options = solver_options
        
        #Create an amalgomator object to solve the subtree problem
        self.create_amalgomator()

        
        
        
    def sample_creator(self,sname,**scenario_creator_kwargs):
        '''
        This method is similar to scenario_creator function, but for subtrees.
        Given a scenario names and kwargs, it creates a scenario from our subtree
        
        WARNING: The multistage model (aka refmodel) must contain a 
        sample_tree_scen_creator function
        
        '''
        s = self.refmodel.sample_tree_scen_creator(sname,
                                                   given_scenario=self.root_scen,
                                                   stage=self.stage,
                                                   sample_BFs=self.sampling_BFs,
                                                   seed = self.seed,
                                                   **scenario_creator_kwargs)
        nlens = {node.name: len(node.nonant_vardata_list) 
                 for node in s._mpisppy_node_list}
        
        #Fixing the xhats
        for k,ndn in enumerate(self.fixed_nodes):
            node = s._mpisppy_node_list[k]
            if len(self.xhats[k]) != nlens[ndn]:
                raise RuntimeError("xhats does not have the size"
                                   f"For stage {k+1}, xhats has {len(self.xhats[k])} nonant variables, but should have {nlens[ndn]} of them")
            for i in range(nlens[ndn]):
                #node.nonant_vardata_list[i].fix(self.xhats[k][i])
                node.nonant_vardata_list[i].value = self.xhats[k][i]
                node.nonant_vardata_list[i].fixed = True
        return s
        
    def create_amalgomator(self):
        '''
        This method attaches an Amalgomator object to a sample subtree.
        
        WARNING: sample_creator must be called before that.
        '''
        self.fixed_nodes = ["ROOT"+"_0"*i for i in range(self.stage-1)]
        self.scenario_creator = self.sample_creator
        
        ama_options = self.options.copy()
        ama_options['EF-mstage'] =True
        ama_options['EF_solver_name']= self.solvername
        if self.solver_options is not None:
            ama_options['EF_solver_options']= self.solver_options
        ama_options['num_scens'] = self.numscens
        ama_options['_mpisppy_probability'] = 1/self.numscens #Probably not used
        
        scen_names = self.refmodel.scenario_names_creator(self.numscens,
                                                          start=self.seed)
        denouement = self.refmodel.scenario_denouement if hasattr(self.refmodel, 'scenario_denouement') else None
        
        self.ama = amalgomator.Amalgomator(ama_options, scen_names,
                                           self.scenario_creator,
                                           self.refmodel.kw_creator,
                                           denouement,
                                           verbose = False)
    def run(self):
        #Running the Amalgomator and attaching the result to the SampleSubtree object
        self.ama.run()
        self.ef = self.ama.ef
        self.EF_Obj = self.ama.EF_Obj
        pseudo_root = "ROOT"+"_0"*(self.stage-1)
        self.xhat_at_stage =[self.ef.ref_vars[(pseudo_root,i)].value for i in range(self.ef._nlens[pseudo_root])]


def feasible_solution(mname,scenario,xhat_one,BFs,seed,options,
                      solvername="gurobi",solver_options=None):
    '''
    Given a scenario and a first-stage policy xhat_one, this method computes
    non-anticipative feasible policies for the following stages.

    '''
    if xhat_one is None:
        raise RuntimeError("Xhat_one can't be None for now")
    ciutils.is_sorted(scenario._mpisppy_node_list)
    nodenames = [node.name for node in scenario._mpisppy_node_list]
    num_stages = len(BFs)+1
    xhats = [xhat_one]
    for t in range(2,num_stages): #We do not compute xhat for the final stage
    
        subtree = SampleSubtree(mname, xhats, scenario, 
                                t, BFs, seed, options,
                                solvername, solver_options)
        subtree.run()
        xhats.append(subtree.xhat_at_stage)
        seed+=sputils.number_of_nodes(BFs[(t-1):])
    xhat_dict = {ndn:xhat for (ndn,xhat) in zip(nodenames,xhats)}
    return xhat_dict,seed

def walking_tree_xhats(mname,local_scenarios,xhat_one,BFs,seed,options,
                       solvername="gurobi", solver_options=None):
    """
    This methods takes a scenario tree (represented by a scenario list) as an input, 
    a first stage policy xhat_one and several settings, and computes 
    a feasible policy for every scenario, i.e. finds nonanticipative xhats 
    using the SampleSubtree class.
    We use a tree traversal approach, so that for every non-leaf node of
    the scenario tree we compute an associated sample tree only once.

    Parameters
    ----------
    mname : str
        name of the module used to sample.
    local_scenarios : dict of pyomo.ConcreteModel
        Scenarios forming the scenario tree.
    xhat_one : list or np.array of float
        A feasible and nonanticipative first stage policy.
    BFs : list of int
        Branching factors for sample trees. 
        BFs[i] is the branching factor for stage i+2.
    seed : int
        Starting seed to create scenarios.
    options : dict
        Arguments passed to the scenario creator.

    Returns
    -------
    xhats : dict
        Dict of values for the nonanticipative variable for every node.
        keys are node names and values are lists of nonant variables.
        
    NOTE: The local_scenarios do not need to form a regular tree (unbalanced tree are authorized)

    """
    if xhat_one is None:
        raise RuntimeError("Xhat_one can't be None for now")
        
    xhats = {'ROOT':xhat_one}
    
    #Special case if we only have one scenario
    if len(local_scenarios)==1:
        scen = list(local_scenarios.values())[0]
        res = feasible_solution(mname, scen, xhat_one, BFs, seed, options,
                                solvername=solvername,
                                solver_options=solver_options)
        return res
        
    
    for k,s in local_scenarios.items():
        scen_xhats = []
        ciutils.is_sorted(s._mpisppy_node_list)
        for node in s._mpisppy_node_list:
            if node.name in xhats:
               scen_xhats.append(xhats[node.name])
            else:
               subtree = SampleSubtree(mname, scen_xhats, s, 
                                       node.stage, BFs, seed, options,
                                       solvername, solver_options)
               subtree.run()
               xhat = subtree.xhat_at_stage
               
               seed+=sputils.number_of_nodes(BFs[(node.stage-1):])
               
               xhats[node.name] = xhat
               scen_xhats.append(xhat)
    return xhats, seed

               
            

if __name__ == "__main__":
    BFs = [3,2,4,4]
    num_scens = np.prod(BFs)
    mname = "mpisppy.tests.examples.aircond_submodels"
    
    ama_options = { "EF-mstage": True,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "BFs":BFs,
                    }
    #We use from_module to build easily an Amalgomator object
    ama = amalgomator.from_module(mname, ama_options,use_command_line=False)
    ama.run()
    
    # get the xhat
    xhat_one = sputils.nonant_cache_from_ef(ama.ef)['ROOT']
    
    #----------Find a feasible solution for a single scenario-------------
    scenario = ama.ef.scen0
    seed = sputils.number_of_nodes(BFs)
    options = dict() #We take default aircond options
    
    xhats,seed = feasible_solution(mname, scenario, xhat_one, BFs, seed, options)
    print(xhats)
    
    #----------Find feasible solutions for every scenario ------------
    
    #Fetching scenarios from EF
    scenarios = dict()
    for k in ama.ef._ef_scenario_names:
        #print(f"{k =}")
        scenarios[k] = getattr(ama.ef, k)
        s = scenarios[k]
        demands = [s.stage_models[t].Demand for t in s.T]
        #print(f"{demands =}")
    seed = sputils.number_of_nodes(BFs)
    options = dict() #We take default aircond options
    
    xhats,seed = walking_tree_xhats(mname,scenarios,xhat_one,BFs,seed,options,)
    print(xhats)
    

    
    
    
        
    
