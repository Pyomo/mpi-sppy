# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
#Code for sampling for multistage problem

import numpy as np
import mpisppy.MPI as mpi
import importlib

import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator
import mpisppy.confidence_intervals.ciutils as ciutils
from mpisppy import global_toc

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
    branching_factors : list of int
        Branching factors for sample trees. 
        branching_factors[i] is the branching factor for stage i+2
    seed : int
        Seed to create scenarios.
    cfg : Config
       To create scenario creator arguments.
    solver_name : str, optional
        Solver name. The default is None.
    solver_options : dict, optional
        Solver options. The default is None.

    '''
    def __init__(self, mname, xhats, root_scen, starting_stage, branching_factors,
                 seed, cfg, solver_name=None, solver_options=None):

        self.refmodel = importlib.import_module(mname)
        #Checking that refmodel has all the needed attributes
        attrs = ["sample_tree_scen_creator","kw_creator","scenario_names_creator"]
        for attr in attrs:
            if not hasattr(self.refmodel, attr):
                raise RuntimeError(f"The construction of a sample subtree failed because the reference model has no {attr} function.")
        self.xhats = xhats
        self.root_scen = root_scen
        self.stage = starting_stage
        self.sampling_branching_factors = branching_factors[(self.stage-1):]
        self.original_branching_factors = branching_factors
        self.numscens = np.prod(self.sampling_branching_factors)
        self.seed = seed
        self.cfg = cfg
        self.solver_name = solver_name
        self.solver_options = solver_options
        
        #Create an amalgamator object to solve the subtree problem
        self._create_amalgamator()

                
    def _sample_creator(self, sname, **scenario_creator_kwargs):
        '''
        This method is similar to scenario_creator function, but for subtrees.
        Given a scenario names and kwargs, it creates a scenario from our subtree
        
        WARNING: The multistage model (aka refmodel) must contain a 
        sample_tree_scen_creator function
        
        '''
        # gymnastics because options get passed around through a variety of paths
        s_c_k = scenario_creator_kwargs.copy()
        if "seed" in scenario_creator_kwargs:
            s_c_k.pop("seed", None)  # we want control over the seed here

        s = self.refmodel.sample_tree_scen_creator(sname,
                                                   given_scenario=self.root_scen,
                                                   stage=self.stage,
                                                   sample_branching_factors=self.sampling_branching_factors,
                                                   seed = self.seed,
                                                   **s_c_k)
        nlens = {node.name: len(node.nonant_vardata_list) 
                 for node in s._mpisppy_node_list}
        
        #Fixing the xhats
        for k,ndn in enumerate(self.fixed_nodes):
            node = s._mpisppy_node_list[k]
            if len(self.xhats[k]) != nlens[ndn]:
                raise RuntimeError("xhats does not have the right size "
                                   f"for stage {k+1}, xhats has {len(self.xhats[k])} nonant variables, but should have {nlens[ndn]} of them")
            for i in range(nlens[ndn]):
                #node.nonant_vardata_list[i].fix(self.xhats[k][i])
                node.nonant_vardata_list[i].value = self.xhats[k][i]
                node.nonant_vardata_list[i].fixed = True
        return s
        
    def _create_amalgamator(self):
        '''
        This method attaches an Amalgamator object to a sample subtree.
        Changes to self.cfg will be ephemeral.
        
        WARNING: sample_creator must be called before using for stages beyond 1
        '''
        self.fixed_nodes = ["ROOT"+"_0"*i for i in range(self.stage-1)]
        self.scenario_creator = self._sample_creator

        ###cfg = copy.deepcopy(self.cfg)
        cfg = self.cfg()   # ephemeral
        cfg.quick_assign('EF-mstage', domain=str, value=len(self.original_branching_factors) > 1)
        cfg.quick_assign('EF-2stage', domain=str, value=len(self.original_branching_factors) <= 1)
        cfg.quick_assign('EF_solver_name', domain=str, value=self.solver_name)
        if self.solver_options is not None:
            cfg.add_and_assign("solver_options", "solver options dict", dict, None, self.solver_options)
        cfg.quick_assign('num_scens', domain=int, value=self.numscens)
        if "start_seed" not in cfg:
            cfg.add_and_assign("start_seed", description="first seed", domain=int, default=None, value=self.seed)
        #### cfg['_mpisppy_probability'] = 1/self.numscens #Probably not used
        # TBD: better options flow (Dec 2021; working on it June 2022)
        if "branching_factors" not in cfg:
            raise RuntimeError("branching_factors is not in cfg")
            """
            if "kwargs" in ama_options:
                ama_options["branching_factors"] = ama_options["kwargs"]["branching_factors"]
            """
        scen_names = self.refmodel.scenario_names_creator(self.numscens,
                                                          start=self.seed)
        denouement = self.refmodel.scenario_denouement if hasattr(self.refmodel, 'scenario_denouement') else None
        
        self.ama = amalgamator.Amalgamator(cfg, scen_names,
                                           self.scenario_creator,
                                           self.refmodel.kw_creator,
                                           denouement,
                                           verbose = False)
    def run(self):
        #Running the Amalgamator and attaching the result to the SampleSubtree object
        #global_toc("Enter SampleSubTree run")
        self.ama.run()
        self.ef = self.ama.ef
        self.EF_Obj = self.ama.EF_Obj
        pseudo_root = "ROOT"+"_0"*(self.stage-1)
        self.xhat_at_stage =[self.ef.ref_vars[(pseudo_root,i)].value for i in range(self.ef._nlens[pseudo_root])]


def _feasible_solution(mname,scenario,xhat_one,branching_factors,seed,cfg,
                       solver_name=None, solver_options=None):
    '''
    Given a scenario and a first-stage policy xhat_one, this method computes
    non-anticipative feasible policies for the following stages.

    '''
    assert solver_name is not None
    if xhat_one is None:
        raise RuntimeError("Xhat_one can't be None for now")
    ciutils.is_sorted(scenario._mpisppy_node_list)
    nodenames = [node.name for node in scenario._mpisppy_node_list]
    num_stages = len(branching_factors)+1
    xhats = [xhat_one]
    for t in range(2,num_stages): #We do not compute xhat for the final stage
    
        subtree = SampleSubtree(mname, xhats, scenario, 
                                t, branching_factors, seed, cfg,
                                solver_name, solver_options)
        subtree.run()
        xhats.append(subtree.xhat_at_stage)
        seed+=sputils.number_of_nodes(branching_factors[(t-1):])
    xhat_dict = {ndn:xhat for (ndn,xhat) in zip(nodenames,xhats)}
    return xhat_dict,seed

def walking_tree_xhats(mname, local_scenarios, xhat_one,branching_factors, seed, cfg,
                       solver_name=None, solver_options=None):
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
    branching_factors : list of int
        Branching factors for sample trees. 
        branching_factors[i] is the branching factor for stage i+2.
    seed : int
        Starting seed to create scenarios.
    cfg : Config
        Parameters

    Returns
    -------
    xhats : dict
        Dict of values for the nonanticipative variable for every node.
        keys are node names and values are lists of nonant variables.
        
    NOTE: The local_scenarios do not need to form a regular tree (unbalanced trees are authorized)

    """
    assert xhat_one is not None, "Xhat_one can't be None for now"
        
    xhats = {'ROOT':xhat_one}
    
    #Special case if we only have one scenario
    if len(local_scenarios)==1:
        scen = list(local_scenarios.values())[0]
        res = _feasible_solution(mname, scen, xhat_one, branching_factors, seed, cfg,
                                solver_name=solver_name,
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
                                       node.stage, branching_factors, seed, cfg,
                                       solver_name, solver_options)
               subtree.run()
               xhat = subtree.xhat_at_stage

               # TBD: is this needed
               seed += sputils.number_of_nodes(branching_factors[(node.stage-1):])
               
               xhats[node.name] = xhat
               scen_xhats.append(xhat)

    return xhats, seed


if __name__ == "__main__":
    # This main program is provided to developers can run quick tests.
    import pyomo.common.config as pyofig
    from mpisppy.utils import config
    
    branching_factors = [3,2,4,4]
    num_scens = np.prod(branching_factors)
    solver_name = "cplex"
    mname = "mpisppy.tests.examples.aircond"

    ama_options = { "EF-mstage": True,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "branching_factors":branching_factors,
                    }

    
    cfg = config.Config()
    cfg.add_and_assign("EF_mstage", description="EF mstage (vs 2stage)", domain=bool, default=None, value=True)
    cfg.add_and_assign("num_scens", description="total scenarios", domain=int, default=None, value=num_scens)
    cfg.add_and_assign("_mpisppy_probability", description="unif prob", domain=float, default=None, value=1./num_scens)
    cfg.add_and_assign("branching_factors", description="branching factors", domain=pyofig.ListOf(int), default=None, value=branching_factors)
    cfg.add_and_assign("EF_solver_name", description="EF Solver", domain=str, default=None, value=solver_name)
    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module(mname, cfg, use_command_line=False)
    ama.run()
    
    # get the xhat
    xhat_one = sputils.nonant_cache_from_ef(ama.ef)['ROOT']
    
    #----------Find a feasible solution for a single scenario-------------
    scenario = ama.ef.scen0
    seed = sputils.number_of_nodes(branching_factors)
    
    xhats,seed = _feasible_solution(mname, scenario, xhat_one, branching_factors, seed, cfg, solver_name=solver_name)
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
    seed = sputils.number_of_nodes(branching_factors)
    
    xhats,seed = walking_tree_xhats(mname,scenarios,xhat_one,branching_factors,seed,cfg,solver_name=solver_name)
    print(xhats)
    

    
    
    
        
    
