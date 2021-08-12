#for use with amalgomator mmw testing, so __main__ should be 2-stage for now,
# and kw_creator also defaults to 2-stage
#ReferenceModel for full set of scenarios for AirCond; June 2021

import pyomo.environ as pyo
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgomator as amalgomator
from mpisppy import global_toc

# Use this random stream:
aircondstream = np.random.RandomState()



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

def MakeNodesforScen(model,nodenames,branching_factors):
    #Create all nonleaf nodes used by the scenario
    TreeNodes = []
    for stage in model.T:
        if stage ==1:
            ndn="ROOT"
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                       cond_prob=1.0,
                                                       stage=stage,
                                                       cost_expression=model.stage_models[stage].StageObjective,
                                                       scen_name_list=None, # Not maintained
                                                       nonant_list=[model.stage_models[stage].RegularProd,
                                                                    model.stage_models[stage].OvertimeProd],
                                                       scen_model=model,
                                                       nonant_ef_suppl_list = [model.stage_models[stage].Inventory],
                                                       )
                             )
        elif stage < max(model.T): #We don't add the leaf node
            parent_ndn = ndn
            ndn = parent_ndn+"_"+nodenames[stage-1]
            TreeNodes.append(scenario_tree.ScenarioNode(name=ndn,
                                                       cond_prob=1.0/branching_factors[stage-2],
                                                       stage=stage,
                                                       cost_expression=model.stage_models[stage].StageObjective,
                                                       scen_name_list=None, # Not maintained
                                                       nonant_list=[model.stage_models[stage].RegularProd,
                                                                    model.stage_models[stage].OvertimeProd],
                                                       scen_model=model,
                                                       nonant_ef_suppl_list = [model.stage_models[stage].Inventory],
                                                       parent_name = parent_ndn
                                                       )
                             )
    return(TreeNodes)

        
def scenario_creator(sname, BFs, num_scens=None, mudev=0, sigmadev=40, start_seed=0):
    scennum   = sputils.extract_num(sname)
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
                          help="average deviation of demand between two periods",
                          dest="mudev",
                          type=float,
                          default=0.)
    inparser.add_argument("--sigma-dev",
                          help="average standard deviation of demands between two periods",
                          dest="sigmadev",
                          type=float,
                          default=40.)

#=========
def kw_creator(options):
    if 'branching_factors' in options:
        BFs = options['branching_factors']
    else:
        BFs = [3]
        
    # (only for Amalgomator): linked to the scenario_creator and inparser_adder
    kwargs = {"num_scens" : options['num_scens'] if 'num_scens' in options else None,
              "BFs" : BFs,
              "mudev" : options['mudev'] if 'mudev' in options else 0.,
              "sigmadev" : options['sigmadev'] if 'sigmadev' in options else 40.,
              "start_seed": options['start_seed'] if 'start_seed' in options else 0,
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
        