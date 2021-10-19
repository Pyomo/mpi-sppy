#for use with amalgomator mmw testing, so __main__ should be 2-stage for now,
# and kw_creator also defaults to 2-stage
#ReferenceModel for full set of scenarios for AirCond; June 2021
# As of Oct 2021, aircond_submodels.py in the test directory has diverged
# (it includes startups, but that needs to be handled with more elegance.)

import pyomo.environ as pyo
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgomator as amalgomator
from mpisppy import global_toc

# Use this random stream:
aircondstream = np.random.RandomState()

def demands_creator(sname,branching_factors,start_seed, mudev, sigmadev, 
                    starting_d=200,root_name="ROOT"):
    
    max_d = 400
    min_d = 0

    if branching_factors is None:
        raise RuntimeError("scenario_creator for aircond needs branching_factors")
    scennum   = sputils.extract_num(sname)
    # Find the right path and the associated seeds (one for each node) using scennum
    prod = np.prod(branching_factors)
    s = int(scennum % prod)
    d = starting_d
    demands = [d]
    nodenames = [root_name]
    for bf in branching_factors:
        assert prod%bf == 0
        prod = prod//bf
        nodenames.append(str(s//prod))
        s = s%prod
    
    stagelist = [int(x) for x in nodenames[1:]]
    for t in range(1,len(nodenames)):
        aircondstream.seed(start_seed+sputils.node_idx(stagelist[:t],branching_factors))
        d = min(max_d,max(min_d,d+aircondstream.normal(mudev,sigmadev)))
        demands.append(d)
    
    return demands,nodenames


def StageModel_creator(time, demand, last_stage=False):
    model = pyo.ConcreteModel()
    model.T = [time]
    #Parameters
    #Demand
    model.Demand = demand
    #Inventoy Cost
    model.InventoryCost = -0.8 if last_stage else 0.5
    #Regular Capacity
    model.Capacity = 200
    #Regular Production Cost
    model.RegularProdCost = 1
    #Overtime Production Cost
    model.OvertimeProdCost = 3

    
    #Variables
    model.RegularProd = pyo.Var(domain=pyo.NonNegativeReals)
    
    model.OvertimeProd = pyo.Var(domain=pyo.NonNegativeReals)
    
    model.Inventory = pyo.Var(domain=pyo.NonNegativeReals)
    
    #Constraints
    def CapacityRule(m):
        return m.RegularProd<=m.Capacity
    model.MaximumCapacity = pyo.Constraint(rule=CapacityRule)
    
    #Objective
    def stage_objective(m):
        return m.RegularProdCost*m.RegularProd +\
            m.OvertimeProdCost*m.OvertimeProd +\
                m.InventoryCost*m.Inventory
    model.StageObjective = pyo.Objective(rule=stage_objective,sense=pyo.minimize)
    
    return model

#Assume that demands has been drawn before
def aircond_model_creator(demands):
    model = pyo.ConcreteModel()
    num_stages = len(demands)
    model.T = range(1,num_stages+1) #Stages 1,2,...,T
    
    #Parameters
    model.BeginInventory = 100
    
    #Creating stage models
    model.stage_models = {}
    for t in model.T:
        last_stage = (t==num_stages)
        model.stage_models[t] = StageModel_creator(t, demands[t-1],last_stage)  

    #Constraints
    
    def material_balance_rule(m,t):
        if t == 1:
            return(m.BeginInventory+m.stage_models[t].RegularProd+m.stage_models[t].OvertimeProd-m.stage_models[t].Inventory == m.stage_models[t].Demand)
        else:
            return(m.stage_models[t-1].Inventory+m.stage_models[t].RegularProd+m.stage_models[t].OvertimeProd-m.stage_models[t].Inventory == m.stage_models[t].Demand)
    model.MaterialBalance = pyo.Constraint(model.T, rule=material_balance_rule)
    
    #Objectives
    
    #Disactivating stage objectives
    for t in model.T:
        model.stage_models[t].StageObjective.deactivate()
    
    def stage_cost(stage_m):
        return  stage_m.RegularProdCost*stage_m.RegularProd +\
            stage_m.OvertimeProdCost*stage_m.OvertimeProd +\
                stage_m.InventoryCost*stage_m.Inventory
    def total_cost_rule(m):
        return(sum(stage_cost(m.stage_models[t]) for t in m.T))
    model.TotalCostObjective = pyo.Objective(rule = total_cost_rule,
                                             sense = pyo.minimize)
    
    #Last step
    for t in model.T:
        setattr(model, "stage_model_"+str(t), model.stage_models[t])
    
    return model


def MakeNodesforScen(model,nodenames,branching_factors,starting_stage=1):
    #Create all nonleaf nodes used by the scenario
    #Compatible with sample scenario creation
    TreeNodes = []
    for stage in model.T:

        nonant_list=[model.stage_models[stage].RegularProd,
                     model.stage_models[stage].OvertimeProd]
        """
        if model.start_ups:
            nonant_list.append(model.stage_models[stage].StartUp)
        """
        if stage ==1:
            ndn="ROOT"
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                       cond_prob=1.0,
                                                       stage=stage,
                                                       cost_expression=model.stage_models[stage].StageObjective,
                                                       scen_name_list=None, # Not maintained
                                                       nonant_list=nonant_list,
                                                       scen_model=model,
                                                       nonant_ef_suppl_list = [model.stage_models[stage].Inventory],
                                                       )
                             )
        elif stage <=starting_stage:
            parent_ndn = ndn
            ndn = parent_ndn+"_0" #Only one node per stage before starting stage
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                       cond_prob=1.0,
                                                       stage=stage,
                                                       cost_expression=model.stage_models[stage].StageObjective,
                                                       scen_name_list=None, # Not maintained
                                                       nonant_list=nonant_list,
                                                       scen_model=model,
                                                       nonant_ef_suppl_list = [model.stage_models[stage].Inventory],
                                                       parent_name = parent_ndn
                                                       )
                             )
        elif stage < max(model.T): #We don't add the leaf node
            parent_ndn = ndn
            ndn = parent_ndn+"_"+nodenames[stage-starting_stage]
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                       cond_prob=1.0/branching_factors[stage-starting_stage-1],
                                                       stage=stage,
                                                       cost_expression=model.stage_models[stage].StageObjective,
                                                       scen_name_list=None, # Not maintained
                                                       nonant_list=nonant_list,
                                                       scen_model=model,
                                                       nonant_ef_suppl_list = [model.stage_models[stage].Inventory],
                                                       parent_name = parent_ndn
                                                       )
                             )
    return(TreeNodes)

        
def scenario_creator(sname, branching_factors, num_scens=None,
                     mudev=0, sigmadev=40, start_seed=0):
    scennum   = sputils.extract_num(sname)
    BFs = branching_factors
    # Find the right path and the associated seeds (one for each node) using scennum
    prod = np.prod(BFs)
    s = int(scennum % prod)
    d = 200
    demands = [d]
    nodenames = ["ROOT"]
    
    for bf in BFs:
        assert prod%bf == 0
        prod = prod//bf
        nodenames.append(str(s//prod))
        s = s%prod
    
    stagelist = [int(x) for x in nodenames[1:]]
    for t in range(1,len(nodenames)):
        aircondstream.seed(start_seed+sputils.node_idx(stagelist[:t],BFs))
        d = min(400,max(0,d+aircondstream.normal(mudev,sigmadev)))
        demands.append(d)
    
    model = aircond_model_creator(demands)
    
    if num_scens is not None:
        model._mpisppy_probability = 1/num_scens
    
    #Constructing the nodes used by the scenario
    model._mpisppy_node_list = MakeNodesforScen(model, nodenames, BFs)
    
    return(model)
                

#=========
def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    """ Create a scenario within a sample tree. Mainly for multi-stage and simple for two-stage.
    Args:
        sname (string): scenario name to be created
        stage (int >=1 ): for stages > 1, fix data based on sname in earlier stages
        sample_branching_factors (list of ints): branching factors for the sample tree
        seed (int): To allow randome sampling (for some problems, it might be scenario offset)
        given_scenario (Pyomo concrete model): if not None, use this to get data for ealier stages
        scenario_creator_kwargs (dict): keyword args for the standard scenario creator funcion
    Returns:
        scenario (Pyomo concrete model): A scenario for sname with data in stages < stage determined
                                         by the arguments
    """
    # Finding demands from stage 1 to t
    if given_scenario is None:
        if stage == 1:
            past_demands = [200]
        else:
            raise RuntimeError(f"sample_tree_scen_creator for aircond needs a 'given_scenario' argument if the starting stage is greater than 1")
    else:
        past_demands = [given_scenario.stage_models[t].Demand for t in given_scenario.T if t<=stage]
    optional_things = ['mudev','sigmadev','start_ups']
    default_values = [0,40,False]
    for thing,value in zip(optional_things,default_values):
        if thing not in scenario_creator_kwargs:
            scenario_creator_kwargs[thing] = value
    
    #Finding demands for stages after t
    future_demands,nodenames = demands_creator(sname, sample_branching_factors, 
                                               start_seed = seed, 
                                               mudev = scenario_creator_kwargs['mudev'], 
                                               sigmadev = scenario_creator_kwargs['sigmadev'],
                                               starting_d=past_demands[stage-1],
                                               root_name='ROOT'+'_0'*(stage-1))
    
    demands = past_demands+future_demands[1:] #The demand at the starting stage is in both past and future demands
    
    model = aircond_model_creator(demands)
    
    model._mpisppy_probability = 1/np.prod(sample_branching_factors)
    
    #Constructing the nodes used by the scenario
    model._mpisppy_node_list = MakeNodesforScen(model, nodenames, sample_branching_factors,
                                                starting_stage=stage)
    
    return model
                

#=========
def scenario_names_creator(num_scens,start=None):
    # (only for Amalgomator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]
        

#=========
def inparser_adder(inparser):
    # (only for Amalgomator): add command options unique to farmer
    inparser.add_argument("--mu-dev",
                          help="average deviation of demand between two periods (default 0)",
                          dest="mudev",
                          type=float,
                          default=0.)
    inparser.add_argument("--sigma-dev",
                          help="average standard deviation of demands between two periods (default 40)",
                          dest="sigmadev",
                          type=float,
                          default=40.)
    inparser.add_argument("--start-seed",
                          help="random number seed (default 1134)",
                          dest="sigmadev",
                          type=int,
                          default=1134)

#=========
def kw_creator(options):

    def _kwarg(option_name, default = None, arg_name=None):
        # options trumps args
        retval = options.get(option_name)
        if retval is not None:
            return retval
        args = options.get('args')
        aname = option_name if arg_name is None else arg_name
        retval = getattr(args, aname) if hasattr(args, aname) else None
        retval = default if retval is None else retval
        return retval

    # (only for Amalgomator): linked to the scenario_creator and inparser_adder
    # for confidence intervals, we need to see if the values are in args
    BFs = _kwarg("branching_factors")
    mudev = _kwarg("mudev", 0.)
    sigmadev = _kwarg("sigmadev", 40.)
    start_seed = _kwarg("start_seed", 1134)
    kwargs = {"num_scens" : options['num_scens'] if 'num_scens' in options else None,
              "branching_factors": BFs,
              "mudev": mudev,
              "sigmadev": sigmadev,
              "start_seed": start_seed,
              }
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass
        
#============================
def xhat_generator_aircond(scenario_names, solvername="gurobi", solver_options=None,
                           BFs=[3,2,3], mudev = 0, sigmadev = 40, start_seed = 0):
    '''
    For sequential sampling.
    Takes scenario names as input and provide the best solution for the 
        approximate problem associated with the scenarios.
    Parameters
    ----------
    scenario_names: int
        Names of the scenario we use
    solvername: str, optional
        Name of the solver used. The default is "gurobi"
    solver_options: dict, optional
        Solving options. The default is None.
    BFs: list, optional
        Branching factors of the scenario 3. The default is [3,2,3] 
        (a 4 stage model with 18 different scenarios)
    mudev: float, optional
        The average deviation of demand between two stages; The default is 0.
    sigma_dev: float, optional
        The standard deviation from mudev for the demand difference between
        two stages. The default is 40.
    start_seed: int, optional
        The starting seed, used to create different sample scenario trees.
        The default is 0.

    Returns
    -------
    xhat: str
        A generated xhat, solution to the approximate problem induced by scenario_names.
        
    NOTE: This tool only works when the file is in mpisppy. In SPInstances, 
            you must change the from_module line.

    '''
    num_scens = len(scenario_names)
    
    ama_options = { "EF-mstage": True,
                    "EF_solver_name": solvername,
                    "EF_solver_options": solver_options,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "BFs":BFs,
                    "mudev":mudev,
                    "sigmadev":sigmadev
                    }
    #We use from_module to build easily an Amalgomator object
    ama = amalgomator.from_module("mpisppy.tests.examples.aircond_submodels",
                                  ama_options,use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return xhat

if __name__ == "__main__":
    bfs = [3]
    num_scens = np.prod(bfs) #To check with a full tree
    ama_options = { "EF-2stage": True,
                    "EF_solver_name": "gurobi_direct",
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "BFs":bfs,
                    "mudev":0,
                    "sigmadev":80,
                    "start":1
                    }
    refmodel = "aaircond" # WARNING: Change this in SPInstances
    #We use from_module to build easily an Amalgomator object
    ama = amalgomator.from_module(refmodel,
                                  ama_options,use_command_line=False)
    ama.run()
    print(f"inner bound=", ama.best_inner_bound)
    print(f"outer bound=", ama.best_outer_bound)

    sputils.ef_ROOT_nonants_npy_serializer(ama.ef, "aircond_root_nonants_temp.npy") 
        
